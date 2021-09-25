import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from .linear_attention import LinearAttention, FullAttention


class LoFTREncoderLayer(nn.Module):
    def __init__(self,
                 d_model,
                 nhead,
                 attention='linear'):
        super(LoFTREncoderLayer, self).__init__()

        self.dim = d_model // nhead
        self.nhead = nhead

        # multi-head attention
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.attention = LinearAttention() if attention == 'linear' else FullAttention()
        self.merge = nn.Linear(d_model, d_model, bias=False)

        # feed-forward network
        self.mlp = nn.Sequential(
            nn.Linear(d_model * 2, d_model * 2, bias=False),
            nn.ReLU(True),
            nn.Linear(d_model * 2, d_model, bias=False),
        )

        # norm and dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, source, x_mask=None, source_mask=None):
        """
        Args:
            x (torch.Tensor): [N, L, C]
            source (torch.Tensor): [N, S, C]
            x_mask (torch.Tensor): [N, L] (optional)
            source_mask (torch.Tensor): [N, S] (optional)
        """
        bs = x.size(0)
        query, key, value = x, source, source

        # multi-head attention
        query = self.q_proj(query).view(bs, -1, self.nhead, self.dim)  # [N, L, (H, D)]
        key = self.k_proj(key).view(bs, -1, self.nhead, self.dim)  # [N, S, (H, D)]
        value = self.v_proj(value).view(bs, -1, self.nhead, self.dim)
        message = self.attention(query, key, value, q_mask=x_mask, kv_mask=source_mask)  # [N, L, (H, D)]
        message = self.merge(message.view(bs, -1, self.nhead * self.dim))  # [N, L, C]
        message = self.norm1(message)

        # feed-forward network
        message = self.mlp(torch.cat([x, message], dim=2))
        message = self.norm2(message)

        return x + message


def get_conf_matrix(feat_c0: torch.Tensor, feat_c1: torch.Tensor, mask_c0=None, mask_c1=None) -> torch.Tensor:
    N, L, S, C = feat_c0.size(0), feat_c0.size(1), feat_c1.size(1), feat_c0.size(2)

    # normalize
    feat_c0, feat_c1 = map(lambda feat: feat / feat.shape[-1] ** .5,
                           [feat_c0, feat_c1])

    temperature = 0.1
    sim_matrix = torch.einsum("nlc,nsc->nls", feat_c0, feat_c1) / temperature

    if mask_c0 is not None:
        INF = 1e9
        sim_matrix.masked_fill_(
            ~(mask_c0[..., None] * mask_c1[:, None]).bool(),
            -INF)

    conf_matrix = F.softmax(sim_matrix, 1) * F.softmax(sim_matrix, 2)

    return conf_matrix


class LocalFeatureTransformer(nn.Module):
    """A Local Feature Transformer (LoFTR) module."""

    def __init__(self, config):
        super(LocalFeatureTransformer, self).__init__()

        self.config = config
        self.d_model = config['d_model']
        self.nhead = config['nhead']
        self.top_k = config.get('top_k', 0)
        self.top_k_thr = config.get('top_k_thr', 0.2)
        self.layer_names = config['layer_names']
        encoder_layer = LoFTREncoderLayer(config['d_model'], config['nhead'], config['attention'])
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(len(self.layer_names))])
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, feat0, feat1, mask0=None, mask1=None):
        """
        Args:
            feat0 (torch.Tensor): [N, L, C]
            feat1 (torch.Tensor): [N, S, C]
            mask0 (torch.Tensor): [N, L] (optional)
            mask1 (torch.Tensor): [N, S] (optional)
        """

        assert self.d_model == feat0.size(2), "the feature number of src and transformer must be equal"

        if self.top_k > 0:

            N = feat0.size(0)
            L = feat0.size(1)
            S = feat1.size(1)
            C = feat0.size(2)
            K = self.top_k

            # 选取 top k 个 点做 attention
            conf_matrix_00 = get_conf_matrix(feat0, feat0, mask0, mask0)  # [N, L, L]
            conf_matrix_11 = get_conf_matrix(feat1, feat1, mask1, mask1)  # [N, S, S]
            conf_matrix_01 = get_conf_matrix(feat0, feat1, mask0, mask1)  # [N, L, S]
            conf_matrix_10 = conf_matrix_01.transpose(1, 2)  # [N,S,L]

            # 只对大于阈值的点做 attention

            # def select2(feat: torch.Tensor, conf_matrix: torch.Tensor) -> torch.Tensor:
            #     n, l, c = feat.size()
            #     _, s, _ = conf_matrix.size()
            #     k = K
            #     # 选择存在大于阈值的匹配的点
            #     conf_mask: torch.Tensor = conf_matrix.max(dim=2) >= self.top_k_thr  # [N, S]
            #     # 计算不同 N 下的 S‘
            #     S_thr = conf_mask.sum(dim=1)
            #     S_end = torch.cumsum(S_thr, dim=0)
            #     S_beg = S_end - S_thr[0]
            #     feat_selected = torch.empty(size=[n, torch.sum(S_thr), k, c], dtype=feat.dtype, device=feat.device,
            #                                 requires_grad=feat.requires_grad)  # [N, sum(S'), K, C]
            #     for b in range(n):
            #         ss = S_end[b] - S_beg[b]  # S'
            #         conf_matrix_real = conf_matrix[b][:, conf_mask[b]]  # [ss, L]
            #         _, index = torch.topk(conf_matrix_real, k, dim=1)  # [ss, K]
            #         feat_selected[b, S_beg[b]:S_end[b]] = torch.gather(
            #             input=feat[b].as_strided(size=[ss, l, c], stride=[0, c, 1]),
            #             dim=2,
            #             index=index.as_strided(size=[ss, k, c], stride=[k, 1, 0])
            #         )

            # 使用 as_stride 加速
            def select(feat: torch.Tensor, conf_matrix: torch.Tensor) -> torch.Tensor:
                _, index = torch.topk(conf_matrix, K, dim=2)
                n, l, c = feat.size()
                _, s, k = index.size()
                return torch.gather(
                    input=feat.as_strided(size=[n, s, l, c], stride=[l * c, 0, c, 1]),
                    dim=2,
                    index=index.as_strided(size=[n, s, k, c], stride=[s * k, k, 1, 0])
                )  # [n, s, k, c]

            feat0_flatten = feat0.view(N * L, 1, C)
            feat1_flatten = feat1.view(N * S, 1, C)
            mask0_flatten = mask0.view(N * L, 1)
            mask1_flatten = mask1.view(N * S, 1)
            feat0_select_00 = select(feat0, conf_matrix_00).view(N * L, K, C)
            feat0_select_10 = select(feat0, conf_matrix_10).view(N * S, K, C)
            feat1_select_01 = select(feat1, conf_matrix_01).view(N * L, K, C)
            feat1_select_11 = select(feat1, conf_matrix_11).view(N * S, K, C)

            # [N*L, 1, C] 与 [N*L, K, C] 做 attention
            # [N*S, 1, C] 与 [N*S, K, C] 做 attention

            for layer, name in zip(self.layers, self.layer_names):

                if name == 'self':
                    feat0_flatten = layer(feat0_flatten, feat0_select_00, mask0_flatten, None)
                    feat1_flatten = layer(feat1_flatten, feat1_select_11, mask1_flatten, None)
                elif name == 'cross':
                    feat0_flatten = layer(feat0_flatten, feat1_select_01, mask0_flatten, None)
                    feat1_flatten = layer(feat1_flatten, feat0_select_10, mask1_flatten, None)
                else:
                    raise KeyError

            feat0 = feat0_flatten.view(N, L, C)
            feat1 = feat1_flatten.view(N, S, C)

        else:

            for layer, name in zip(self.layers, self.layer_names):
                if name == 'self':
                    feat0 = layer(feat0, feat0, mask0, mask0)
                    feat1 = layer(feat1, feat1, mask1, mask1)
                elif name == 'cross':
                    feat0 = layer(feat0, feat1, mask0, mask1)
                    feat1 = layer(feat1, feat0, mask1, mask0)
                else:
                    raise KeyError

        return feat0, feat1
