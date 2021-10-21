import copy
import math

import torch
import torch.nn as nn
from .linear_attention import LinearAttention, FullAttention


class ModifiedDETRDecoder(nn.Module):
    def __init__(self, d_model, n_head, attention='linear'):
        super(ModifiedDETRDecoder, self).__init__()
        self.n_head = n_head
        self.n_dim = d_model // n_head
        self.query_self_attention = LinearAttention() if attention == 'linear' else FullAttention()
        self.query_q_proj = nn.Linear(d_model, d_model, bias=False)
        self.query_k_proj = nn.Linear(d_model, d_model, bias=False)
        self.query_v_proj = nn.Linear(d_model, d_model, bias=False)
        self.query_merge = nn.Linear(d_model, d_model, bias=False)
        self.norm1 = nn.LayerNorm(d_model)
        self.prototype_extractor_attention = LinearAttention() if attention == 'linear' else FullAttention()
        self.prototype_q_proj = nn.Linear(d_model, d_model, bias=False)
        self.prototype_k_proj = nn.Linear(d_model, d_model, bias=False)
        self.prototype_v_proj = nn.Linear(d_model, d_model, bias=False)
        self.prototype_merge = nn.Linear(d_model, d_model, bias=False)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model, bias=False),
            nn.ReLU(True),
            nn.Linear(d_model, d_model, bias=False),
        )
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, query: torch.Tensor, feat: torch.Tensor, query_mask: torch.Tensor = None,
                feat_mask: torch.Tensor = None) -> torch.Tensor:
        bs = query.size(0)
        query_q = self.query_q_proj(query).view(bs, -1, self.n_head, self.n_dim)
        query_k = self.query_k_proj(query).view(bs, -1, self.n_head, self.n_dim)
        query_v = self.query_v_proj(query).view(bs, -1, self.n_head, self.n_dim)
        message = self.query_self_attention(query_q, query_k, query_v, query_mask, query_mask)
        message = message.view(bs, -1, self.n_head * self.n_dim)
        message = self.query_merge(message)

        message = self.norm1(query + message)

        message_q = self.prototype_q_proj(message).view(bs, -1, self.n_head, self.n_dim)
        feat_k = self.prototype_k_proj(feat).view(bs, -1, self.n_head, self.n_dim)
        feat_v = self.prototype_v_proj(feat).view(bs, -1, self.n_head, self.n_dim)
        prototype = self.prototype_extractor_attention(message_q, feat_k, feat_v, query_mask, feat_mask)
        prototype = prototype.view(bs, -1, self.n_head * self.n_dim)
        prototype = self.prototype_merge(prototype)

        prototype = self.norm2(message + prototype)
        prototype_ffn = self.ffn(prototype)
        prototype = self.norm3(prototype + prototype_ffn)

        return prototype


class PrototypeExtractor(nn.Module):

    def __init__(self, d_model: int, num_prototype: int, type: str, n_head: int = None, attention: str = 'linear'):
        super(PrototypeExtractor, self).__init__()
        self.d_model = d_model
        self.num_prototype = num_prototype
        # type 可选 linear 和 detr
        self.type = type
        if self.type == 'linear':
            self.linear = nn.Linear(d_model, num_prototype, bias=False)
        elif self.type == 'detr':
            self.query = nn.Embedding(num_prototype, d_model)
            self.decoder = ModifiedDETRDecoder(d_model, n_head, attention=attention)
        else:
            raise KeyError

    def forward(self, feat: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if self.type == 'linear':
            heatmap = self.linear(feat)  # [N, L, C] -> [N, L, K]
            heatmap = torch.masked_fill(heatmap, ~mask[:, :, None], -math.inf)
            # 全是 -inf 会导致 nan，但是这里没有全是 -inf 的情况
            heatmap = torch.softmax(heatmap, dim=1)
            prototype = torch.einsum('nlc,nlk->nkc', feat, heatmap)
            return prototype
        elif self.type == 'detr':
            query_in = self.query.weight.as_strided(
                size=(feat.size(0),) + self.query.weight.size(),
                stride=(0,) + self.query.weight.stride(),
            )
            return self.decoder(query_in, feat, query_mask=None, feat_mask=mask)
        else:
            raise KeyError


class LoFTREncoderLayer(nn.Module):
    def __init__(self,
                 d_model,
                 nhead,
                 attention='linear',
                 num_prototype: int = 0,
                 prototype_extractor_type: str = 'detr'):
        super(LoFTREncoderLayer, self).__init__()

        self.dim = d_model // nhead
        self.nhead = nhead

        self.use_prototype = num_prototype > 0

        # multi-head attention
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        if self.use_prototype:
            self.num_prototype = num_prototype
            self.prototype_extractor = PrototypeExtractor(d_model, num_prototype,
                                                          type=prototype_extractor_type,
                                                          n_head=nhead, attention=attention)
            self.qp_proj = nn.Linear(d_model, d_model, bias=False)
            self.kp_proj = nn.Linear(d_model, d_model, bias=False)
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
        if self.use_prototype:

            query = self.q_proj(query)
            key = self.k_proj(key)

            prototype = self.prototype_extractor(x, x_mask)
            prototype_p = self.qp_proj(prototype)
            prototype_k = self.kp_proj(prototype)

            assert self.num_prototype % self.nhead == 0
            query = torch.einsum('nlc,nkc->nlk', query, prototype_p)
            key = torch.einsum('nsc,nkc->nsk', key, prototype_k)

            # TODO: OT or DS
            # 暂时先使用 DS
            query = torch.masked_fill(query, ~x_mask[:, :, None], value=-math.inf)
            query = torch.softmax(query, dim=1) * torch.nan_to_num(torch.softmax(query, dim=2), nan=0)
            # if torch.any(torch.isnan(query)):
            #     print('break')
            key = torch.masked_fill(key, ~source_mask[:, :, None], value=-math.inf)
            key = torch.softmax(key, dim=1) * torch.nan_to_num(torch.softmax(key, dim=2), nan=0)
            # if torch.any(torch.isnan(key)):
            #     print('break')

            query = query.view(bs, -1, self.nhead, self.num_prototype // self.nhead)
            key = key.view(bs, -1, self.nhead, self.num_prototype // self.nhead)
        else:
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


class LocalFeatureTransformer(nn.Module):
    """A Local Feature Transformer (LoFTR) module."""

    def __init__(self, config):
        super(LocalFeatureTransformer, self).__init__()

        self.config = config
        self.d_model = config['d_model']
        self.nhead = config['nhead']
        self.layer_names = config['layer_names']
        encoder_layer = LoFTREncoderLayer(config['d_model'], config['nhead'], config['attention'],
                                          config.get('num_prototype', 0),
                                          config.get('prototype_extractor_type', 'detr'))
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

        for layer, name in zip(self.layers, self.layer_names):
            if name == 'self':
                feat0 = layer(feat0, feat0, mask0, mask0)
                feat1 = layer(feat1, feat1, mask1, mask1)
            elif name == 'cross':
                feat0, feat1 = layer(feat0, feat1, mask0, mask1), layer(feat1, feat0, mask1, mask0)
            else:
                raise KeyError

        return feat0, feat1
