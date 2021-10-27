import math
from typing import List, Tuple

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
from einops import rearrange, repeat

from ..utils.coarse_matching import CoarseMatching


# def get_matches(b_ids: torch.Tensor, i_ids: torch.Tensor, j_ids: torch.Tensor,
#                 hw0_c: torch.Tensor, hw1_c: torch.Tensor, bs: int) -> List[torch.Tensor]:
#     # 每个 batch 得到 [M, 2, 2] 的匹配
#     matches = []
#     xy0 = torch.stack([
#         torch.div(i_ids, hw0_c[1], rounding_mode='floor'),
#         i_ids % hw0_c[1],
#     ], dim=1)  # [M',2]
#     xy1 = torch.stack([
#         torch.div(j_ids, hw1_c[1], rounding_mode='floor'),
#         j_ids % hw1_c[1],
#     ], dim=1)  # [M',2]
#     for b in range(bs):
#         matches.append(torch.stack([xy0[b_ids == b], xy1[b_ids == b]], dim=1))  # [M, 2, 2]
#     return matches
#
#
# def get_anchors(matches: List[torch.Tensor], anchor_num: int,
#                 mconf: torch.Tensor = None, b_ids: torch.Tensor = None) -> torch.Tensor:
#     # 返回 [bs, anchor_num, 2, 2]
#     assert (mconf is None) or (b_ids is not None)
#     bs = len(matches)
#     anchors_by_b = []
#     for b in range(bs):
#         matches_b = matches[b]
#         ms = matches_b.size(0)
#         if ms == 0:
#             anchors_b = torch.zeros(size=(anchor_num, 2, 2), dtype=matches_b.dtype, device=matches_b.device)
#         elif mconf is None:  # 随机采样
#             if ms >= anchor_num:
#                 anchors_select = random.sample([m for m in matches_b], k=anchor_num)
#                 anchors_b = torch.stack(anchors_select, dim=0)
#             else:
#                 anchors_b_base = matches_b.repeat(anchor_num // ms, 1, 1)
#                 if anchor_num % ms == 0:
#                     anchors_b = anchors_b_base
#                 else:
#                     anchors_select = random.sample([m for m in matches_b], k=anchor_num % ms)
#                     anchors_select = torch.stack(anchors_select, dim=0)
#                     anchors_b = torch.cat([anchors_b_base, anchors_select], dim=0)
#         else:  # 选择 mconf 最大的匹配，非极大抑制
#             mconf_b = mconf[b_ids == b]  # [M]
#             if ms >= anchor_num:
#                 _, indices = torch.topk(mconf_b, k=anchor_num)
#                 anchors_b = matches_b[indices]
#             else:
#                 anchors_b_base = matches_b.repeat(anchor_num // ms, 1, 1)
#                 if anchor_num % ms == 0:
#                     anchors_b = anchors_b_base
#                 else:
#                     _, indices = torch.topk(mconf_b, k=anchor_num % ms)
#                     anchors_b = torch.cat([anchors_b_base, matches_b[indices]], dim=0)
#         anchors_by_b.append(anchors_b)
#     anchors = torch.stack(anchors_by_b, dim=0)
#     return anchors


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


# 预测表观特征和结构特征的权重
class WeightLayer(nn.Module):

    def __init__(self):
        super(WeightLayer, self).__init__()
        in_channels = 3
        out_channels = 1
        self.res_branch = nn.Sequential(
            nn.Linear(in_features=in_channels, out_features=in_channels),
            nn.LeakyReLU(inplace=True),
            nn.Linear(in_features=in_channels, out_features=in_channels),
        )
        self.head = nn.Sequential(
            nn.Linear(in_features=in_channels, out_features=out_channels),
            nn.Tanh(),
        )

    def forward(self, conf_matrix) -> Tuple[torch.Tensor, torch.Tensor]:
        # conf_matrix [N, L, S]
        # 使用最大值、标准差、信息熵这三个指标预测
        with torch.no_grad():
            max0, _ = torch.max(conf_matrix, dim=2)  # [N, L]
            max1, _ = torch.max(conf_matrix, dim=1)  # [N, S]
            std0 = torch.std(conf_matrix, dim=2)  # [N, L]
            std1 = torch.std(conf_matrix, dim=1)  # [N, S]
            conf_matrix_clamped = torch.clamp_min(conf_matrix, 1e-5)  # 避免 log 出现 nan
            entropy = -conf_matrix_clamped * torch.log(conf_matrix_clamped)
            entropy0 = torch.mean(entropy, dim=2)  # [N, L]
            entropy1 = torch.mean(entropy, dim=1)  # [N, S]
        feat0 = torch.stack([max0, std0, entropy0], dim=-1)  # [N, L, 3]
        feat1 = torch.stack([max1, std1, entropy1], dim=-1)  # [N, S, 3]
        weight0 = self.head(feat0 + self.res_branch(feat0))  # [N, L, 2]
        weight1 = self.head(feat1 + self.res_branch(feat1))  # [N, S, 2]
        return weight0, weight1


class GeometryLayer(nn.Module):

    def __init__(self, config, feat_channels: int):
        super(GeometryLayer, self).__init__()
        self.anchor_num = config['anchor_num']
        self.matcher = CoarseMatching(config['matcher'])
        self.use_warp = config['use_warp']
        self.use_gt_matches_in_training = config['use_gt_matches_in_training']
        self.max_coord_dist = config['max_coord_dist']
        self.use_nms = config['use_nms']
        self.nms_pooling_ks = config['nms_pooling_ks']
        self.feat_channels = feat_channels
        if self.use_nms:
            self.nms_pooling = nn.Sequential(
                nn.ConstantPad2d(padding=(0, 1, 0, 1), value=0),
                nn.MaxPool2d(kernel_size=self.nms_pooling_ks, stride=1),
            )
        self.geo_feat_linear = nn.Linear(in_features=3 * self.anchor_num, out_features=2 * self.anchor_num)
        self.weight_layer = WeightLayer()
        self.merge_linear = nn.Linear(in_features=self.feat_channels + 2 * self.anchor_num,
                                      out_features=self.feat_channels)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def get_matches(self, b_ids: torch.Tensor, i_ids: torch.Tensor, j_ids: torch.Tensor,
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

    def get_matches_nms(self, b_ids: torch.Tensor, i_ids: torch.Tensor, j_ids: torch.Tensor,
                        mconf: torch.Tensor,
                        hw0_c: torch.Tensor, hw1_c: torch.Tensor, bs: int) -> List[torch.Tensor]:
        conf0 = torch.zeros(size=(bs, hw0_c[0], hw0_c[1]), dtype=mconf.dtype, device=mconf.device)
        conf0_flatten = rearrange(conf0, 'b h w -> b (h w)')
        conf0_flatten[b_ids, i_ids] = mconf  # 有置信度的地方赋值，其他保持 0
        conf0 = rearrange(conf0_flatten, 'b (h w) -> b h w', h=hw0_c[0])
        conf0_nms = self.nms_pooling(conf0)
        is_matches = (conf0 > 0) & (conf0 == conf0_nms)
        is_matches_flatten = rearrange(is_matches, 'b h w -> b (h w)')
        b_ids_new, i_ids_new = torch.where(is_matches_flatten)
        j_id_new = []
        # 找到在右图的匹配点
        for b_new, i_new in zip(b_ids_new, i_ids_new):
            for b, i, j in zip(b_ids, i_ids, j_ids):
                if b_new == b and i_new == i:
                    j_id_new.append(j)
                    break
        j_id_new = torch.tensor(j_id_new, device=j_ids.device)
        matches = self.get_matches(b_ids_new, i_ids_new, j_id_new, hw0_c, hw1_c, bs)
        return matches

    def get_anchors(self, matches: List[torch.Tensor], anchor_num: int,
                    mconf: torch.Tensor = None, b_ids: torch.Tensor = None) -> torch.Tensor:
        # 返回 [bs, anchor_num, 2, 2]
        assert (mconf is None) or (b_ids is not None)
        bs = len(matches)
        anchors_by_b = []
        for b in range(bs):
            matches_b = matches[b]
            ms = matches_b.size(0)
            if ms == 0:
                anchors_b = torch.zeros(size=(anchor_num, 2, 2), dtype=matches_b.dtype, device=matches_b.device)
            elif mconf is None:  # 随机采样
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
            else:  # 选择 mconf 最大的匹配，非极大抑制
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

    def get_coord_dist_no_enough_matches(self, feat0: torch.Tensor, feat1: torch.Tensor):

        _device = feat0.device
        _dtype = feat0.dtype

        bs = feat0.size(0)
        l = feat0.size(1)
        s = feat0.size(1)

        img0_rest = torch.zeros(size=(bs, l, self.anchor_num * 2), dtype=_dtype, device=_device)
        img1_rest = torch.zeros(size=(bs, s, self.anchor_num * 2), dtype=_dtype, device=_device)

        return img0_rest, img1_rest

    def get_coord_dist(self, data, feat0, feat1):

        b_ids = data['b_ids']
        i_ids = data['i_ids']
        j_ids = data['j_ids']
        mconf = data['mconf']

        hw0_c = data['hw0_c']
        hw1_c = data['hw1_c']

        _device = feat0.device
        _dtype = feat0.dtype
        bs = feat0.size(0)

        assert hw0_c[0] != 0 and hw0_c[1] != 0 and hw1_c[0] != 0 and hw1_c[1] != 0

        if self.training and self.use_gt_matches_in_training:
            # 训练时使用 ground truth 匹配
            b_ids_gt = data['spv_b_ids']
            i_ids_gt = data['spv_i_ids']
            j_ids_gt = data['spv_j_ids']
            matches = self.get_matches(b_ids_gt, i_ids_gt, j_ids_gt, hw0_c, hw1_c, bs=bs)  # bs x [M, 2, 2]
            anchors = self.get_anchors(matches, anchor_num=self.anchor_num)  # [bs, anchor_num, 2, 2]
        else:
            # 测试时使用 matcher 获得的匹配
            if self.use_nms:
                matches = self.get_matches_nms(b_ids, i_ids, j_ids, mconf, hw0_c, hw1_c, bs=bs)  # 使用 NMS
                anchors = self.get_anchors(matches, anchor_num=self.anchor_num)
            else:
                matches = self.get_matches(b_ids, i_ids, j_ids, hw0_c, hw1_c, bs=bs)
                anchors = self.get_anchors(matches, anchor_num=self.anchor_num, mconf=mconf, b_ids=b_ids)

        anchors = anchors.to(dtype=_dtype)

        feat0_x = torch.arange(0, hw0_c[0], dtype=_dtype, device=_device)
        feat0_y = torch.arange(0, hw0_c[1], dtype=_dtype, device=_device)
        feat0_pos = torch.stack(torch.meshgrid(feat0_x, feat0_y), dim=-1).view(-1, 2)  # [h*w, 2]
        feat0_pos = feat0_pos.as_strided(size=[bs, feat0_pos.size(0), 2],
                                         stride=(0,) + feat0_pos.stride())  # [bs, h*w, 2]
        if feat0.size() == feat1.size():
            feat1_pos = feat0_pos
        else:
            feat1_x = torch.arange(0, hw1_c[0], dtype=_dtype, device=_device)
            feat1_y = torch.arange(0, hw1_c[1], dtype=_dtype, device=_device)
            feat1_pos = torch.stack(torch.meshgrid(feat1_x, feat1_y), dim=-1).view(-1, 2)  # [h*w, 2]
            feat1_pos = feat1_pos.as_strided(size=[bs, feat1_pos.size(0), 2],
                                             stride=(0,) + feat1_pos.stride())  # [bs, h*w, 2]

        if self.use_warp:

            raise NotImplementedError

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

            # FIXME: warp 方式导致 img0_coord_dist 过大，训练时出现 NaN
            # FIXME: 暂时使用裁剪坐标范围的方式缓解问题

            img0_coord_dist = torch.clamp(img0_coord_dist, -self.max_coord_dist, self.max_coord_dist)

        else:

            img0_coord_dist = calc_anchor_coord_dist_map(anchors[:, :, 0, :], feat0_pos)

        img1_coord_dist = calc_anchor_coord_dist_map(anchors[:, :, 1, :], feat1_pos)

        return img0_coord_dist, img1_coord_dist, anchors

    def forward(self, feat0, feat1, data, mask0=None, mask1=None):

        data_copy = data.copy()

        self.matcher.forward(feat0, feat1, data_copy, mask0, mask1, only_find_matches=True)

        conf_matrix = data_copy['conf_matrix']  # 用于生成监督信号

        if self.matcher.match_type == 'sinkhorn':
            if 'conf_matrix_with_bin_i' not in data:
                data['conf_matrix_with_bin_i'] = []
            data['conf_matrix_with_bin_i'].append(data_copy['conf_matrix_with_bin'])
        else:
            if 'conf_matrix_i' not in data:
                data['conf_matrix_i'] = []
            data['conf_matrix_i'].append(data_copy['conf_matrix'])

        with torch.no_grad():
            img0_coord_dist, img1_coord_dist, anchors = \
                self.get_coord_dist(data_copy, feat0, feat1)  # [N,L,AN,2], [N,AN,2,2]

        if 'anchors_i' not in data:  # 用于可视化
            data['anchors_i'] = []
        data['anchors_i'].append(anchors)

        img0_coord_dist_flatten = rearrange(img0_coord_dist, 'n l an c -> n l (an c)')
        img1_coord_dist_flatten = rearrange(img1_coord_dist, 'n s an c -> n s (an c)')

        img0_coord_dist_norm_flatten = F.normalize(img0_coord_dist_flatten, p=1, dim=2)  # [N,L,AN*2]
        img1_coord_dist_norm_flatten = F.normalize(img1_coord_dist_flatten, p=1, dim=2)  # [N,S,AN*2]

        img0_coord_dist_norm = rearrange(img0_coord_dist_norm_flatten, 'n l (an c)-> n l an c', c=2)
        img1_coord_dist_norm = rearrange(img1_coord_dist_norm_flatten, 'n l (an c)-> n l an c', c=2)

        # 加入距离信息，因为 norm 之后左右图的同一点到相同 anchor 点的距离相同

        img0_dist_norm = torch.sqrt(torch.sum(torch.square(img0_coord_dist_norm), dim=3))
        img1_dist_norm = torch.sqrt(torch.sum(torch.square(img1_coord_dist_norm), dim=3))

        geo_feat0 = torch.cat([img0_coord_dist_norm_flatten, img0_dist_norm], dim=-1)  # [N,L,AN*3]
        geo_feat1 = torch.cat([img1_coord_dist_norm_flatten, img1_dist_norm], dim=-1)  # [N,L,AN*3]

        geo_feat0 = self.geo_feat_linear(geo_feat0)
        geo_feat1 = self.geo_feat_linear(geo_feat1)

        # 计算信息熵，表观特征匹配好则权重低，否则权重高
        # 采用神经网络从信息熵回归，更鲁棒，人工设定不行
        # 包括 信息熵，最大值，标准差这三个信息
        # 用一个跳接
        # TODO: 最好再回归一个单应变换矩阵
        # TODO: 尝试去掉 pos encoding

        weight0, weight1 = self.weight_layer(conf_matrix)

        # vis_feat0 = feat0 * weight0[:, :, 0:1]
        # vis_feat1 = feat1 * weight1[:, :, 0:1]

        geo_feat0 *= weight0
        geo_feat1 *= weight1

        feat0_in = torch.cat([feat0, geo_feat0], dim=-1)
        feat1_in = torch.cat([feat1, geo_feat1], dim=-1)

        feat0 = self.merge_linear(feat0_in)
        feat1 = self.merge_linear(feat1_in)

        return feat0, feat1
