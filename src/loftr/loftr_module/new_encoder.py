import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat

from .ot_layer import OTLayer
from .linear_attention import LinearAttention, FullAttention
from .prototype import PrototypeTransformer


class NewEncoder(nn.Module):

    def __init__(self, config, type, is_first=False):
        super(NewEncoder, self).__init__()

        self.type = type

        self.d_model = config['d_model']
        self.nhead = config['nhead']
        self.dim = self.d_model // self.nhead
        self.use_prototype = config['use_prototype']
        self.num_prototype = config['num_prototype']
        self.prototype_query_type = config['prototype_query_type']
        self.use_trans_matrix = config['use_trans_matrix']
        self.attention = config['attention']
        self.is_first = is_first

        self.q_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.k_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.v_proj = nn.Linear(self.d_model, self.d_model, bias=False)

        if self.use_prototype:
            self.prototype_extractor = PrototypeTransformer(config['prototype_extractor'], is_first=is_first)
            self.prototype_q_proj = nn.Linear(self.d_model, self.d_model, bias=False)
            self.prototype_k_proj = nn.Linear(self.d_model, self.d_model, bias=False)
            self.ot_layer = OTLayer(config['ot_layer'])

        # multi-head attention
        self.attention = LinearAttention() if self.attention == 'linear' else FullAttention()
        self.merge = nn.Linear(self.d_model, self.d_model, bias=False)

        # feed-forward network
        self.mlp = nn.Sequential(
            nn.Linear(self.d_model * 2, self.d_model * 2, bias=False),
            nn.ReLU(True),
            nn.Linear(self.d_model * 2, self.d_model, bias=False),
        )

        # norm and dropout
        self.norm1 = nn.LayerNorm(self.d_model)
        self.norm2 = nn.LayerNorm(self.d_model)

    def forward(self, data: dict,
                feat0: torch.Tensor, feat1: torch.Tensor,
                mask0: torch.Tensor, mask1: torch.Tensor,
                prototype0_query: torch.Tensor, prototype1_query: torch.Tensor,
                feat0_no_pe: torch.Tensor, feat1_no_pe: torch.Tensor):

        assert feat0 is not feat1  # 防止把两个相同的 feat 输入

        hw0_c = data['hw0_c']
        hw1_c = data['hw1_c']

        if self.prototype_query_type == 'anchor':
            prototype0_res = self.prototype_extractor(prototype0_query, feat0_no_pe, mask0, h=hw0_c[0], w=hw0_c[1],
                                                      use_query_pe=True, use_feat_pe=True, value=feat0)  # [N, AN, C]
            prototype1_res = self.prototype_extractor(prototype1_query, feat1_no_pe, mask1, h=hw1_c[0], w=hw1_c[1],
                                                      use_query_pe=True, use_feat_pe=True, value=feat1)
        elif self.prototype_query_type == 'query':
            prototype0_res = self.prototype_extractor(prototype0_query, feat0_no_pe, mask0, h=hw0_c[0], w=hw0_c[1],
                                                      use_query_pe=False, use_feat_pe=False, value=feat0)  # [N, AN, C]
            prototype1_res = self.prototype_extractor(prototype1_query, feat1_no_pe, mask1, h=hw1_c[0], w=hw1_c[1],
                                                      use_query_pe=False, use_feat_pe=False, value=feat1)
        else:
            raise KeyError

        if self.is_first:
            prototype0 = prototype0_res
            prototype1 = prototype1_res
        elif len(self.prototype_extractor.blocks) > 0:
            prototype0 = prototype0_query + prototype0_res
            prototype1 = prototype1_query + prototype1_res
        else:
            prototype0 = prototype0_query
            prototype1 = prototype1_query

        if 'feat0' not in data:
            data['feat0'] = []
        data['feat0'].append(feat0)
        if 'feat1' not in data:
            data['feat1'] = []
        data['feat1'].append(feat1)
        if 'prototype0' not in data:
            data['prototype0'] = []
        data['prototype0'].append(prototype0)
        if 'prototype1' not in data:
            data['prototype1'] = []
        data['prototype1'].append(prototype1)

        if self.type == 'self':
            feat0_out = self.real_forward(data, feat0, feat0, prototype0, prototype0, mask0, mask0)
            feat1_out = self.real_forward(data, feat1, feat1, prototype1, prototype1, mask1, mask1)
        elif self.type == 'cross':
            if self.use_trans_matrix:
                raise NotImplementedError
                # FIXME
                trans_matrix = torch.einsum('nkc,njc->nkj', prototype0, prototype1) / prototype0.size(2)
                prototype_0_to_1 = torch.einsum('nkc,nkj->njc', prototype0, trans_matrix)
                prototype_1_to_0 = torch.einsum('njc,nkj->nkc', prototype1, trans_matrix)
                feat0_out = self.real_forward(data, feat0, feat1, prototype0, prototype_1_to_0, mask0, mask1)
                feat1_out = self.real_forward(data, feat1, feat0, prototype1, prototype_0_to_1, mask1, mask0)
            else:
                feat0_out = self.real_forward(data, feat0, feat1, prototype0, prototype0, mask0, mask1)
                feat1_out = self.real_forward(data, feat1, feat0, prototype1, prototype1, mask1, mask0)
        else:
            raise KeyError

        return feat0_out, feat1_out, prototype0, prototype1

    def real_forward(self, data: dict,
                     x: torch.Tensor, source: torch.Tensor,
                     x_prototype: torch.Tensor, source_prototype: torch.Tensor,
                     x_mask: torch.Tensor, source_mask: torch.Tensor):

        bs = x.size(0)
        query, key, value = x, source, source

        # multi-head attention
        query = self.q_proj(query).view(bs, -1, self.nhead, self.dim)  # [N, L, (H, D)]
        key = self.k_proj(key).view(bs, -1, self.nhead, self.dim)  # [N, S, (H, D)]
        value = self.v_proj(value).view(bs, -1, self.nhead, self.dim)

        if self.use_prototype:
            prototype_query = self.prototype_q_proj(x_prototype).view(bs, -1, self.nhead, self.dim)  # [N, K, (H, D)]
            prototype_key = self.prototype_k_proj(source_prototype).view(bs, -1, self.nhead, self.dim)  # [N, K, (H, D)]

            query_sim = self.ot_layer(query, prototype_query, x_mask)
            key_sim = self.ot_layer(key, prototype_key, source_mask)

            query = rearrange(query_sim, 'n l k h -> n l h k')
            key = rearrange(key_sim, 'n s k h -> n s h k')

        message = self.attention(query, key, value, q_mask=x_mask, kv_mask=source_mask)  # [N, L, (H, D)]
        message = self.merge(message.view(bs, -1, self.nhead * self.dim))  # [N, L, C]
        message = self.norm1(message)

        # feed-forward network
        message = self.mlp(torch.cat([x, message], dim=2))
        message = self.norm2(message)

        return x + message
