import random
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat

from ..utils.coarse_matching import CoarseMatching


class AnchorExtractor(nn.Module):
    def __init__(self, config):
        super(AnchorExtractor, self).__init__()
        self.anchor_num = config['anchor_num']
        self.matcher = CoarseMatching(config['matcher'])
        self.use_gt_matches_in_training = config['use_gt_matches_in_training']
        self.use_nms = config['use_nms']
        self.nms_pooling_ks = config['nms_pooling_ks']
        self.feat_channels = config['feat_channels']
        self.conv = nn.Conv1d(in_channels=self.feat_channels + 2 * self.anchor_num, out_channels=self.feat_channels,
                              kernel_size=(1,), stride=(1,))
        if self.use_nms:
            self.nms_pooling = nn.Sequential(
                nn.ConstantPad2d(padding=(0, 1, 0, 1), value=0),
                nn.MaxPool2d(kernel_size=self.nms_pooling_ks, stride=1),
            )

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
        j_ids_matrix = torch.empty(size=(bs, hw0_c[0], hw0_c[1]), dtype=j_ids.dtype, device=j_ids.device)
        j_ids_matrix_flatten = rearrange(j_ids_matrix, 'b h w -> b (h w)')
        j_ids_matrix_flatten[b_ids, i_ids] = j_ids
        conf0 = rearrange(conf0_flatten, 'b (h w) -> b h w', h=hw0_c[0])
        conf0_nms = self.nms_pooling(conf0)
        is_matches = (conf0 > 0) & (conf0 == conf0_nms)
        is_matches_flatten = rearrange(is_matches, 'b h w -> b (h w)')
        b_ids_new, i_ids_new = torch.where(is_matches_flatten)
        j_ids_new = j_ids_matrix_flatten[b_ids_new, i_ids_new]
        matches = self.get_matches(b_ids_new, i_ids_new, j_ids_new, hw0_c, hw1_c, bs)
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

    def forward(self, data: dict,
                feat0: torch.Tensor, feat1: torch.Tensor,
                mask0: torch.Tensor = None, mask1: torch.Tensor = None):

        assert feat0 is not feat1

        data_copy = data.copy()

        self.matcher(feat0, feat1, data_copy, mask0, mask1, only_find_matches=True)

        conf_matrix = data_copy['conf_matrix']  # 用于生成监督信号

        # 用于生成监督信号
        if self.matcher.match_type == 'sinkhorn':
            if 'conf_matrix_with_bin_i' not in data:
                data['conf_matrix_with_bin_i'] = []
            data['conf_matrix_with_bin_i'].append(data_copy['conf_matrix_with_bin'])
        else:
            if 'conf_matrix_i' not in data:
                data['conf_matrix_i'] = []
            data['conf_matrix_i'].append(data_copy['conf_matrix'])

        with torch.no_grad():

            b_ids = data_copy['b_ids']
            i_ids = data_copy['i_ids']
            j_ids = data_copy['j_ids']
            mconf = data_copy['mconf']

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

        if 'anchors_i' not in data:  # 用于可视化
            data['anchors_i'] = []
        data['anchors_i'].append(anchors)

        return anchors, conf_matrix
