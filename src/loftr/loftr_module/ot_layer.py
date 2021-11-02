import math

import torch
import torch.nn as nn
import torch.nn.functional as F

INF = 1e9


class OTLayer(nn.Module):

    def __init__(self, config):
        super(OTLayer, self).__init__()
        self.type = config['type']
        if self.type == 'dual_softmax':
            self.temperature = config['dsmax_temperature']
        elif self.type == 'sinkhorn':
            raise NotImplementedError
            try:
                from .superglue import log_optimal_transport
            except ImportError:
                raise ImportError("download superglue.py first!")
            self.log_optimal_transport = log_optimal_transport
            self.bin_score = nn.Parameter(torch.tensor(config['skh_init_bin_score'], requires_grad=True))
            self.skh_iters = config['skh_iters']
            self.skh_prefilter = config['skh_prefilter']
        else:
            raise NotImplementedError()

    def forward(self, feat0: torch.Tensor, feat1: torch.Tensor, mask0: torch.Tensor, mask1: torch.Tensor):

        N, L, H, D = feat0.shape
        _, K, _, _ = feat1.shape

        # N, L, S, C = feat0.size(0), feat0.size(1), feat1.size(1), feat0.size(2)

        # normalize
        feat0, feat1 = map(lambda feat: feat / feat.shape[-1] ** .5, [feat0, feat1])

        sim_matrix_out = torch.einsum("nlhd,nkhd->nlkh", feat0, feat1)

        if self.type == 'dual_softmax':
            sim_matrix = sim_matrix_out / self.temperature
            if mask0 is not None:
                sim_matrix.masked_fill_(
                    ~(mask0[..., None, None]).bool(),
                    -INF)
            if mask1 is not None:
                sim_matrix.masked_fill_(
                    ~(mask1[:, None, None]).bool(),
                    -INF)

            conf_matrix = F.softmax(sim_matrix, 1) * F.softmax(sim_matrix, 2)

        elif self.type == 'sinkhorn':
            raise NotImplementedError
            # sinkhorn, dustbin included
            sim_matrix = sim_matrix_out
            if mask0 is not None:
                sim_matrix[:, :L, :K].masked_fill_(
                    ~(mask0[..., None, None]).bool(),
                    -INF)
            if mask1 is not None:
                sim_matrix[:, :L, :K].masked_fill_(
                    ~(mask1[:, None, None]).bool(),
                    -INF)

            # build uniform prior & use sinkhorn
            log_assign_matrix = self.log_optimal_transport(sim_matrix, self.bin_score, self.skh_iters)
            assign_matrix = log_assign_matrix.exp()
            conf_matrix = assign_matrix[:, :-1, :-1]

        else:
            raise KeyError

        return sim_matrix_out, conf_matrix
