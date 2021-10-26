import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from kornia.geometry.subpix import dsnt
from kornia.utils.grid import create_meshgrid

from einops.einops import rearrange, repeat
from einops.layers.torch import Rearrange


class FineMatching(nn.Module):
    """FineMatching with s2d paradigm"""

    def __init__(self, config):
        super().__init__()

        self.use_group_sim = 'num_group' in config and config['num_group'] > 0

        if self.use_group_sim:
            self.num_group = config['num_group']
            self.num_feat_c = config['num_feat_c']
            self.weight_use_right = config['weight_use_right']
            assert self.num_feat_c % self.num_group == 0
            self.index_nn = nn.Sequential(
                Rearrange('n l c -> n c l'),
                nn.Conv1d(in_channels=self.num_feat_c, out_channels=32, kernel_size=(1,), stride=(1,)),
                nn.BatchNorm1d(32),
                nn.LeakyReLU(inplace=False),
                nn.Conv1d(in_channels=32, out_channels=1, kernel_size=(1,), stride=(1,)),
                nn.BatchNorm1d(1),
                nn.Tanh(),
                Rearrange('n c l -> n l c'),
            )

    def forward(self, feat_f0, feat_f1, data):
        """
        Args:
            feat0 (torch.Tensor): [M, WW, C]
            feat1 (torch.Tensor): [M, WW, C]
            data (dict)
        Update:
            data (dict):{
                'expec_f' (torch.Tensor): [M, 3],
                'mkpts0_f' (torch.Tensor): [M, 2],
                'mkpts1_f' (torch.Tensor): [M, 2]}
        """
        M, WW, C = feat_f0.shape
        W = int(math.sqrt(WW))
        scale = data['hw0_i'][0] / data['hw0_f'][0]
        self.M, self.W, self.WW, self.C, self.scale = M, W, WW, C, scale

        # corner case: if no coarse matches found
        if M == 0:
            assert self.training == False, "M is always >0, when training, see coarse_matching.py"
            # logger.warning('No matches found in coarse-level.')
            data.update({
                'expec_f': torch.empty(0, 3, device=feat_f0.device),
                'mkpts0_f': data['mkpts0_c'],
                'mkpts1_f': data['mkpts1_c'],
            })
            return

        feat_f0_picked = feat_f0[:, WW // 2, :]

        if self.use_group_sim:
            feat_f0_picked_grouped = feat_f0_picked.view(M, self.num_group, C // self.num_group)  # [M, G, D]
            feat_f1_grouped = feat_f1.view(M, WW, self.num_group, C // self.num_group)  # [M, WW, G, D]
            sim_matrix_grouped = torch.einsum('mgd,mwgd->mwg', feat_f0_picked_grouped, feat_f1_grouped)  # [M, WW, G]
            feat_f0_picked_in = rearrange(feat_f0_picked, 'm c -> m 1 c')
            feat_f0_group_index = self.index_nn(feat_f0_picked_in)  # [M, 1, 1]
            if self.weight_use_right:
                raise NotImplementedError
            else:
                feat_group_index = feat_f0_group_index  # [M, 1, 1]
                sim_in = rearrange(sim_matrix_grouped, 'm w g -> m w 1 g')
                l_index = torch.arange(0, 1, dtype=feat_group_index.dtype, device=feat_group_index.device)  # [1]
                l_index = repeat(l_index, '1 -> m 1 1', m=M)
                index_in = torch.stack([l_index, feat_group_index], dim=-1)  # [M, 1, 1, 2]
                sim_matrix = F.grid_sample(sim_in, index_in, align_corners=False)  # [M, WW, 1, 1]
                sim_matrix = rearrange(sim_matrix, 'm w 1 1 -> m w')
        else:
            sim_matrix = torch.einsum('mc,mrc->mr', feat_f0_picked, feat_f1)

        softmax_temp = 1. / C ** .5
        heatmap = torch.softmax(softmax_temp * sim_matrix, dim=1).view(-1, W, W)

        # compute coordinates from heatmap
        coords_normalized = dsnt.spatial_expectation2d(heatmap[None], True)[0]  # [M, 2]
        grid_normalized = create_meshgrid(W, W, True, heatmap.device).reshape(1, -1, 2)  # [1, WW, 2]

        # compute std over <x, y>
        var = torch.sum(grid_normalized ** 2 * heatmap.view(-1, WW, 1), dim=1) - coords_normalized ** 2  # [M, 2]
        std = torch.sum(torch.sqrt(torch.clamp(var, min=1e-10)), -1)  # [M]  clamp needed for numerical stability

        # for fine-level supervision
        data.update({'expec_f': torch.cat([coords_normalized, std.unsqueeze(1)], -1)})

        # compute absolute kpt coords
        self.get_fine_match(coords_normalized, data)

    @torch.no_grad()
    def get_fine_match(self, coords_normed, data):
        W, WW, C, scale = self.W, self.WW, self.C, self.scale

        # mkpts0_f and mkpts1_f
        mkpts0_f = data['mkpts0_c']
        scale1 = scale * data['scale1'][data['b_ids']] if 'scale0' in data else scale
        mkpts1_f = data['mkpts1_c'] + (coords_normed * (W // 2) * scale1)[:len(data['mconf'])]

        data.update({
            "mkpts0_f": mkpts0_f,
            "mkpts1_f": mkpts1_f
        })
