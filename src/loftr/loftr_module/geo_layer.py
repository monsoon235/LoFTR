import random
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat

from .loftr_encoder import LoFTREncoderLayer
from ..utils.coarse_matching import CoarseMatching


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


# 计算信息熵，表观特征匹配好则权重低，否则权重高
# 采用神经网络从信息熵回归，更鲁棒，人工设定不行
# 包括 信息熵，最大值，标准差这三个信息
# 用一个跳接
# TODO: 最好再回归一个单应变换矩阵
# TODO: 尝试去掉 pos encoding
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


class GeoLayer(nn.Module):

    def __init__(self, config: dict):
        super(GeoLayer, self).__init__()
        self.use_warp = config['use_warp']
        self.use_gt_matches_in_training = config['use_gt_matches_in_training']
        self.anchor_num = config['anchor_num']
        self.use_weight = config['use_weight']
        self.feat_channels = config['feat_channels']
        self.use_attention = config['use_attention']
        self.use_loss = config['use_loss']
        self.attention_n_head = config['attention_n_head']

        self.geo_feat_linear = nn.Linear(in_features=3 * self.anchor_num, out_features=2 * self.anchor_num)
        if self.use_weight:
            self.weight_layer = WeightLayer()
        if self.use_attention:
            self.geo_self_attention = LoFTREncoderLayer(d_model=2 * self.anchor_num, nhead=self.attention_n_head)
            self.geo_cross_attention = LoFTREncoderLayer(d_model=2 * self.anchor_num, nhead=self.attention_n_head)
        if self.use_loss:
            self.matcher = CoarseMatching(config['loss_matcher'])
        self.merge_linear = nn.Linear(in_features=self.feat_channels + 2 * self.anchor_num,
                                      out_features=self.feat_channels)
        self.norm = nn.LayerNorm(self.feat_channels)

    def forward(self, data: dict, conf_matrix, anchors: torch.Tensor,
                feat0: torch.Tensor, feat1: torch.Tensor,
                mask0: torch.Tensor = None, mask1: torch.Tensor = None):

        hw0_c = data['hw0_c']
        hw1_c = data['hw1_c']

        _device = feat0.device
        _dtype = feat0.dtype
        bs = feat0.size(0)

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

        img0_coord_dist_flatten = rearrange(img0_coord_dist, 'n l an c -> n l (an c)')
        img1_coord_dist_flatten = rearrange(img1_coord_dist, 'n s an c -> n s (an c)')

        img0_coord_dist_norm_flatten = F.normalize(img0_coord_dist_flatten, p=1, dim=2)  # [N,L,AN*2]
        img1_coord_dist_norm_flatten = F.normalize(img1_coord_dist_flatten, p=1, dim=2)  # [N,S,AN*2]

        img0_coord_dist_norm = rearrange(img0_coord_dist_norm_flatten, 'n l (an c)-> n l an c', c=2)
        img1_coord_dist_norm = rearrange(img1_coord_dist_norm_flatten, 'n l (an c)-> n l an c', c=2)

        # 加入距离信息，因为 norm 之后左右图的同一点到相同 anchor 点的距离相同

        img0_dist_norm = torch.sqrt(torch.sum(torch.square(img0_coord_dist_norm), dim=3))
        img1_dist_norm = torch.sqrt(torch.sum(torch.square(img1_coord_dist_norm), dim=3))

        feat_geo0 = torch.cat([img0_coord_dist_norm_flatten, img0_dist_norm], dim=-1)  # [N,L,AN*3]
        feat_geo1 = torch.cat([img1_coord_dist_norm_flatten, img1_dist_norm], dim=-1)  # [N,L,AN*3]

        feat_geo0 = self.geo_feat_linear(feat_geo0)
        feat_geo1 = self.geo_feat_linear(feat_geo1)

        if self.use_attention:
            feat_geo0 = self.geo_self_attention(feat_geo0, feat_geo0, mask0, mask0)
            feat_geo1 = self.geo_self_attention(feat_geo1, feat_geo1, mask1, mask1)

            feat_geo0, feat_geo1 = \
                self.geo_cross_attention(feat_geo0, feat_geo1, mask0, mask1), \
                self.geo_cross_attention(feat_geo1, feat_geo0, mask1, mask0)

        if self.use_weight:
            weight0, weight1 = self.weight_layer(conf_matrix)
            feat_geo0 *= weight0
            feat_geo1 *= weight1

        if self.use_loss:
            data_copy = data.copy()
            self.matcher(feat_geo0, feat_geo1, data_copy, mask0, mask1, only_find_matches=True)

            if self.matcher.match_type == 'sinkhorn':
                if 'conf_matrix_geo_with_bin_i' not in data:
                    data['conf_matrix_geo_with_bin_i'] = []
                data['conf_matrix_geo_with_bin_i'].append(data_copy['conf_matrix_with_bin'])
            else:
                if 'conf_matrix_geo_i' not in data:
                    data['conf_matrix_geo_i'] = []
                data['conf_matrix_geo_i'].append(data_copy['conf_matrix'])

        feat0_in = torch.cat([feat0, feat_geo0], dim=-1)
        feat1_in = torch.cat([feat1, feat_geo1], dim=-1)

        feat0 = self.merge_linear(feat0_in)
        feat1 = self.merge_linear(feat1_in)

        feat0 = self.norm(feat0)
        feat1 = self.norm(feat1)

        return feat0, feat1
