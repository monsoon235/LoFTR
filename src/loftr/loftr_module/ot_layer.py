import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat

INF = 1e9


def lns_sinkhorn(u: torch.Tensor, v: torch.Tensor, cost: torch.Tensor,
                 reg=0.05, num_iter_max=100, stop_thr=5e-5) -> torch.Tensor:
    # u [B, L]
    # v [b, K+1]
    # cost [B, L, K+1]
    K = torch.exp(-cost / reg)
    r = torch.ones_like(u)
    c = torch.ones_like(v)
    for i in range(num_iter_max):
        r0 = r
        r = u / torch.matmul(K, c.unsqueeze(-1)).squeeze(-1)
        c = v / torch.matmul(K.permute(0, 2, 1), r.unsqueeze(-1)).squeeze(-1)
        err = (r - r0).abs().mean()
        if err.item() < stop_thr:
            break
    T = torch.matmul(r.unsqueeze(-1), c.unsqueeze(-2)) * K
    return T


def lns_get_sim_ot(feat: torch.Tensor, prototype: torch.Tensor, feat_mask: torch.Tensor) -> torch.Tensor:
    # feat [B, L, C]
    # prototype [B, K, C]
    # feat_mask [B, L]
    B, L, C = feat.shape
    _, K, _ = prototype.shape
    att = torch.einsum('nlc,nkc->nlk', feat, prototype)  # [B, L, K]
    att /= (feat.norm(p=2, dim=-1) + 0.5)[:, :, None]
    att /= (prototype.norm(p=2, dim=-1) + 0.5)[:, None, :]
    trash = torch.zeros(size=(B, L, 1), device=att.device)
    trash = torch.masked_fill(trash, ~feat_mask[:, :, None], 2)
    cost_matrix = torch.cat([1 - att, trash], dim=-1)  # [B, L, K+1]
    col = torch.full(size=(B, L), fill_value=1 / L, device=att.device)
    bg_num = torch.sum(~feat_mask, dim=-1, keepdim=True)  # [B, 1]
    fg_num = L - bg_num
    row = torch.empty(size=(B, K + 1), device=att.device)
    row[:, :K] = bg_num / K
    row[:, K:] = fg_num
    row /= L
    T = lns_sinkhorn(col, row, cost_matrix, reg=0.05, num_iter_max=100, stop_thr=1e-1)
    T *= L
    weight = T[:, :, :-1]
    att_weight = weight * att

    return att_weight


def lns_ot(feat: torch.Tensor, prototype: torch.Tensor,
           feat_mask: torch.Tensor):
    sim = lns_get_sim_ot(feat, prototype, feat_mask)
    sim = torch.relu(sim)
    prototype_rec = torch.einsum('nlc,nlk->nkc', feat, sim)
    prototype_rec /= (torch.sum(sim, dim=1) + 1e-5)[:, :, None]
    return sim, prototype_rec


def lns_ot_entrypoint(feat: torch.Tensor, prototype: torch.Tensor,
                      feat_mask: torch.Tensor):
    sim, prototype_new = lns_ot(feat, prototype, feat_mask)
    prototype1 = prototype + prototype_new
    sim, prototype_new = lns_ot(feat, prototype1, feat_mask)
    prototype2 = prototype1 + prototype_new
    sim, prototype_new = lns_ot(feat, prototype2, feat_mask)
    return sim, prototype2


class OTLayer(nn.Module):

    def __init__(self, config):
        super(OTLayer, self).__init__()
        self.type = config['type']
        if self.type == 'dual_softmax':
            self.temperature = config['dsmax_temperature']
        elif self.type == 'sinkhorn':
            try:
                from .superglue import log_optimal_transport
            except ImportError:
                raise ImportError("download superglue.py first!")
            self.log_optimal_transport = log_optimal_transport
            # self.bin_score = nn.Parameter(torch.tensor(config['skh_init_bin_score'], requires_grad=True))
            self.skh_iters = config['skh_iters']
            self.skh_prefilter = config['skh_prefilter']
        else:
            raise NotImplementedError()

    def forward(self, feat: torch.Tensor, prototype: torch.Tensor, feat_mask: torch.Tensor):

        N, L, H, D = feat.shape
        _, K, _, _ = prototype.shape

        if self.type == 'dual_softmax':

            # normalize
            feat /= feat.shape[-1] ** .5
            prototype /= prototype.shape[-1] ** .5
            sim_matrix_origin = torch.einsum("nlhd,nkhd->nlkh", feat, prototype)
            sim_matrix = sim_matrix_origin / self.temperature
            if feat_mask is not None:
                sim_matrix.masked_fill_(
                    ~(feat_mask[..., None, None]).bool(),
                    -INF)

            sim = F.softmax(sim_matrix, 1) * F.softmax(sim_matrix, 2)
            sim *= sim_matrix_origin
            return sim

        elif self.type == 'sinkhorn':

            # lns version
            feat_in = rearrange(feat, 'n l h d -> (n h) l d')
            prototype_in = rearrange(prototype, 'n k h d -> (n h) k d')
            feat_mask_in = repeat(feat_mask, 'n l -> (n h) l', h=H)
            sim_out, prototype_out = lns_ot_entrypoint(feat_in, prototype_in, feat_mask_in)
            sim = rearrange(sim_out, '(n h) l k -> n l k h', h=H)
            return sim

        else:
            raise KeyError
