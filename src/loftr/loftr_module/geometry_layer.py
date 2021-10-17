import math
from typing import List

import cv2
import torch
import torch.nn as nn
import random
import numpy as np

from ..utils.coarse_matching import CoarseMatching


def get_matches(b_ids: torch.Tensor, i_ids: torch.Tensor, j_ids: torch.Tensor,
                hw0_c: torch.Tensor, hw1_c: torch.Tensor, bs: int) -> List[torch.Tensor]:
    # 每个 batch 得到 [M, 2, 2] 的匹配
    matches = []
    xy0 = torch.stack([
        torch.div(i_ids, hw0_c[1], rounding_mode='floor'),
        i_ids % hw0_c[1],
    ], dim=1)  # [M',2]
    xy1 = torch.stack([
        torch.div(j_ids, hw1_c[1], rounding_mode='floor'),
        j_ids % hw1_c[1],
    ], dim=1)  # [M',2]
    for b in range(bs):
        matches.append(torch.stack([xy0[b_ids == b], xy1[b_ids == b]], dim=1))  # [M, 2, 2]
    return matches


def get_anchors(matches: List[torch.Tensor], anchor_num: int,
                mconf: torch.Tensor = None, b_ids: torch.Tensor = None) -> torch.Tensor:
    # 返回 [bs, anchor_num, 2, 2]
    assert (mconf is None) or (b_ids is not None)
    bs = len(matches)
    anchors_by_b = []
    for b in range(bs):
        matches_b = matches[b]
        ms = matches_b.size(0)
        if mconf is None:  # 随机采样
            if ms >= anchor_num:
                anchors_select = random.sample([m for m in matches_b], k=anchor_num)
                anchors_b = torch.stack(anchors_select, dim=0)
            else:
                anchors_b_base = matches_b.repeat(anchor_num // ms, 1, 1)
                if anchor_num % ms == 0:
                    anchors_b = anchors_b_base
                else:
                    anchors_select = random.sample([m for m in matches_b], k=anchor_num % ms)
                    anchors_select = torch.stack(anchors_select, dim=0)
                    anchors_b = torch.cat([anchors_b_base, anchors_select], dim=0)
        else:  # 选择 mconf 最大的匹配
            mconf_b = mconf[b_ids == b]  # [M]
            if ms >= anchor_num:
                _, indices = torch.topk(mconf_b, k=anchor_num)
                anchors_b = matches_b[indices]
            else:
                anchors_b_base = matches_b.repeat(anchor_num // ms, 1, 1)
                if anchor_num % ms == 0:
                    anchors_b = anchors_b_base
                else:
                    _, indices = torch.topk(mconf_b, k=anchor_num % ms)
                    anchors_b = torch.cat([anchors_b_base, matches_b[indices]], dim=0)
        anchors_by_b.append(anchors_b)
    anchors = torch.stack(anchors_by_b, dim=0)
    return anchors


def get_warped_pos(pos: torch.Tensor, transform_matrix: torch.Tensor) -> torch.Tensor:
    # pos: [bs, L, 2]
    # transform_matrix: [bs, 3, 3]
    # return: [bs, L, 2]
    pos_in = torch.cat([pos, torch.ones([pos.size(0), pos.size(1), 1], dtype=pos.dtype, device=pos.device)], dim=2)
    pos_out = torch.einsum('bcd,bld->blc', transform_matrix, pos_in)
    pos_warped = pos_out[:, :, :2] / pos_out[:, :, 2:3]
    assert not torch.any(torch.isnan(pos_warped)), f'NaN detected! {pos_warped}'
    assert not torch.any(torch.isinf(pos_warped)), f'Inf detected! {pos_warped}'
    return pos_warped


# 计算每个点到 anchor 点的坐标差距
def calc_anchor_coord_dist_map(anchors_pos: torch.Tensor, point_pos: torch.Tensor) -> torch.Tensor:
    # anchors_pos: [bs, anchor_num, 2]
    # points_pos: [bs, h*w, 2]
    coord_dist = point_pos[:, :, None, :] - anchors_pos[:, None, :, :]
    return coord_dist  # [bs, h*w, anchor_num, 2]


class GeometryLayer(nn.Module):

    def __init__(self, config, feat_channels: int):
        super(GeometryLayer, self).__init__()
        self.anchor_num = config['anchor_num']
        self.matcher = CoarseMatching(config['matcher'])
        self.use_warp = config['use_warp']
        self.use_gt_matches_in_training = config['use_gt_matches_in_training']
        self.max_coord_dist = config['max_coord_dist']
        self.feat_channels = feat_channels
        self.conv = nn.Conv1d(in_channels=feat_channels + 2 * self.anchor_num, out_channels=feat_channels,
                              kernel_size=(1,), stride=(1,))
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def get_coord_dist_no_enough_matches(self, feat0: torch.Tensor, feat1: torch.Tensor):

        _device = feat0.device
        _dtype = feat0.dtype

        bs = feat0.size(0)
        l = feat0.size(1)
        s = feat0.size(1)

        img0_rest = torch.zeros(size=(bs, l, self.anchor_num * 2), dtype=_dtype, device=_device)
        img1_rest = torch.zeros(size=(bs, s, self.anchor_num * 2), dtype=_dtype, device=_device)

        return img0_rest, img1_rest

    def get_coord_dist(self, feat0, feat1, data, mask0=None, mask1=None):

        _device = feat0.device
        _dtype = feat0.dtype
        bs = feat0.size(0)
        hw0_c = data['hw0_c']
        hw1_c = data['hw1_c']

        assert hw0_c[0] != 0 and hw0_c[1] != 0 and hw1_c[0] != 0 and hw1_c[1] != 0

        data_copy = data.copy()
        self.matcher.forward(feat0, feat1, data_copy, mask0, mask1, only_find_matches=True)
        if self.matcher.match_type == 'sinkhorn':
            data.update({
                'conf_matrix_with_bin_a': data_copy['conf_matrix_with_bin']
            })
        else:
            data.update({
                'conf_matrix_a': data_copy['conf_matrix']
            })

        if self.training and self.use_gt_matches_in_training:
            # 训练时使用 ground truth 匹配
            b_ids_gt = data['spv_b_ids']
            i_ids_gt = data['spv_i_ids']
            j_ids_gt = data['spv_j_ids']
            matches = get_matches(b_ids_gt, i_ids_gt, j_ids_gt, hw0_c, hw1_c, bs=bs)  # bs x [M, 2, 2]
            anchors = get_anchors(matches, anchor_num=self.anchor_num)  # [bs, anchor_num, 2, 2]
        else:
            # 测试时使用 matcher 获得的匹配
            b_ids = data_copy['b_ids']
            i_ids = data_copy['i_ids']
            j_ids = data_copy['j_ids']
            mconf = data_copy['mconf']
            matches = get_matches(b_ids, i_ids, j_ids, hw0_c, hw1_c, bs=bs)
            anchors = get_anchors(matches, anchor_num=self.anchor_num, mconf=mconf, b_ids=b_ids)
        anchors = anchors.to(dtype=_dtype)

        feat0_x = torch.arange(0, hw0_c[0], dtype=_dtype, device=_device)
        feat0_y = torch.arange(0, hw0_c[1], dtype=_dtype, device=_device)
        feat0_pos = torch.stack(torch.meshgrid(feat0_x, feat0_y), dim=-1).view(-1, 2)  # [h*w, 2]
        feat0_pos = feat0_pos.as_strided(size=[bs, feat0_pos.size(0), 2], stride=[0, 2, 1])  # [bs, h*w, 2]
        if feat0.size() == feat1.size():
            feat1_pos = feat0_pos
        else:
            feat1_x = torch.arange(0, hw1_c[0], dtype=_dtype, device=_device)
            feat1_y = torch.arange(0, hw1_c[1], dtype=_dtype, device=_device)
            feat1_pos = torch.stack(torch.meshgrid(feat1_x, feat1_y), dim=-1).view(-1, 2)  # [h*w, 2]
            feat1_pos = feat1_pos.as_strided(size=[bs, feat1_pos.size(0), 2], stride=[0, 2, 1])  # [bs, h*w, 2]

        if self.use_warp:
            transform_matrix_by_b = []
            for b in range(bs):
                m = matches[b].detach().cpu().numpy()
                if m.shape[0] < 4:
                    transform_matrix_b = torch.eye(3, dtype=_dtype, device=_device)
                else:
                    if self.training and self.use_gt_matches_in_training:
                        # 训练时用最小二乘法
                        transform_matrix_b_np, _ = cv2.findHomography(srcPoints=m[:, 0, :], dstPoints=m[:, 1, :],
                                                                      method=0)
                    else:
                        # 测试时使用 RANSAC 算法
                        transform_matrix_b_np, _ = cv2.findHomography(srcPoints=m[:, 0, :], dstPoints=m[:, 1, :],
                                                                      method=cv2.RANSAC)
                    # 如果 transform_matrix 为 None，或者包含 inf nan，则忽略
                    if transform_matrix_b_np is None \
                            or np.any(np.isnan(transform_matrix_b_np)) \
                            or np.any(np.isinf(transform_matrix_b_np)):
                        transform_matrix_b = torch.eye(3, dtype=_dtype, device=_device)
                    else:
                        transform_matrix_b = torch.tensor(transform_matrix_b_np, dtype=_dtype, device=_device)
                transform_matrix_by_b.append(transform_matrix_b)
            transform_matrix = torch.stack(transform_matrix_by_b, dim=0)  # [bs, 3, 3]

            feat0_pos_warped = get_warped_pos(feat0_pos, transform_matrix)  # [bs, h*w, 2]
            anchors0_warped = get_warped_pos(anchors[:, :, 0, :], transform_matrix)  # [bs, anchor_num, 2]

            # img_0 的坐标点使用 warp 后的
            img0_coord_dist = calc_anchor_coord_dist_map(anchors0_warped, feat0_pos_warped)
            # 归一化
            img0_coord_dist[:, :, :, 0] /= hw1_c[0]  # 归一化也是用 img1 的尺度
            img0_coord_dist[:, :, :, 1] /= hw1_c[1]

            data.update({
                'anchors0_warped': anchors0_warped,
                'transform_matrix': transform_matrix,
            })

        else:

            img0_coord_dist = calc_anchor_coord_dist_map(anchors[:, :, 0, :], feat0_pos)
            # 归一化
            img0_coord_dist[:, :, :, 0] /= hw0_c[0]
            img0_coord_dist[:, :, :, 1] /= hw0_c[1]

            data.update({
                'anchors0': anchors[:, :, 0, :],
            })

        img1_coord_dist = calc_anchor_coord_dist_map(anchors[:, :, 1, :], feat1_pos)
        img1_coord_dist[:, :, :, 0] /= hw1_c[0]
        img1_coord_dist[:, :, :, 1] /= hw1_c[1]

        data.update({
            'anchors1': anchors[:, :, 1, :],
        })

        # FIXME: warp 方式导致 img0_coord_dist 过大，训练时出现 NaN
        # FIXME: 暂时使用裁剪坐标范围的方式缓解问题

        img0_coord_dist = torch.clamp(img0_coord_dist, -self.max_coord_dist, self.max_coord_dist)
        img1_coord_dist = torch.clamp(img1_coord_dist, -self.max_coord_dist, self.max_coord_dist)

        # 转化成极坐标
        # FIXME: 极坐标存在角度不连续的问题，暂时使用直角坐标
        # [bs, h*w, anchor_num]
        # img0_dist = torch.sqrt(torch.square(img0_coord_dist).sum(dim=-1)) / math.sqrt(2)
        # img1_dist = torch.sqrt(torch.square(img1_coord_dist).sum(dim=-1)) / math.sqrt(2)
        # img0_grad = img0_coord_dist[:, :, :, 0] / img0_coord_dist[:, :, :, 1]
        # img1_grad = img1_coord_dist[:, :, :, 0] / img1_coord_dist[:, :, :, 1]
        # torch.nan_to_num(img0_grad, nan=0)
        # torch.nan_to_num(img1_grad, nan=0)
        # img0_arc = torch.atan(img0_grad) / (math.pi / 2)
        # img1_arc = torch.atan(img1_grad) / (math.pi / 2)

        img0_coord_dist = img0_coord_dist.flatten(2, 3)
        img1_coord_dist = img1_coord_dist.flatten(2, 3)

        return img0_coord_dist, img1_coord_dist

    def forward(self, feat0, feat1, data, mask0=None, mask1=None):

        with torch.no_grad():
            img0_coord_dist, img1_coord_dist = self.get_coord_dist(feat0, feat1, data, mask0, mask1)

        feat0_in = torch.cat([feat0, img0_coord_dist], dim=2).transpose(1, 2)
        feat1_in = torch.cat([feat1, img1_coord_dist], dim=2).transpose(1, 2)

        feat0 = self.conv(feat0_in).transpose(1, 2)
        feat1 = self.conv(feat1_in).transpose(1, 2)

        return feat0, feat1
