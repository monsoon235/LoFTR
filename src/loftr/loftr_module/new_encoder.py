import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat

from .ot_layer import OTLayer
from .linear_attention import LinearAttention, FullAttention
from .prototype import PrototypeTransformer
from .anchor import AnchorExtractor
from .geo_layer import GeoLayer

from ..utils.position_encoding import PositionEncodingSine


class NewEncoder(nn.Module):

    def __init__(self, config, type):
        super(NewEncoder, self).__init__()

        self.type = type

        self.d_model = config['d_model']
        self.nhead = config['nhead']
        self.dim = self.d_model // self.nhead
        self.num_prototype = config['num_prototype']
        self.attention = config['attention']

        self.use_geo_feat = config['use_geo_feat']
        self.anchor_extractor = AnchorExtractor(config['anchor_extractor'])
        self.pos_encoding = PositionEncodingSine(self.d_model, temp_bug_fix=config['temp_bug_fix'])
        if self.use_geo_feat:
            self.geo_layer = GeoLayer(config['geo_layer'])
        # else:
        #     self.prototype_query = nn.Parameter(
        #         torch.empty(size=(self.num_prototype, self.d_model), dtype=torch.float32), requires_grad=True)
        self.q_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.k_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.v_proj = nn.Linear(self.d_model, self.d_model, bias=False)

        if self.num_prototype > 0:
            self.prototype_extractor = PrototypeTransformer(config['prototype_extractor'])
            self.prototype_q_proj = nn.Linear(self.d_model, self.d_model, bias=False)
            self.prototype_k_proj = nn.Linear(self.d_model, self.d_model, bias=False)
            self.ot_layer = OTLayer(config['ot_layer'])

        # multi-head attention
        if self.num_prototype > 0:
            self.attention = LinearAttention(with_feature_map=False) if self.attention == 'linear' else FullAttention()
        else:
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

    def get_anchor_query(self, data: dict, feat0: torch.Tensor, feat1: torch.Tensor, anchors: torch.Tensor):
        # hw0_c = data['hw0_c']
        # hw1_c = data['hw1_c']
        # feat0_nchw = rearrange(feat0, 'n (h w) c -> n c h w', h=hw0_c[0])
        # feat1_nchw = rearrange(feat1, 'n (h w) c -> n c h w', h=hw1_c[0])
        # feat0_nchw = self.pos_encoding.forward_anchors(feat0_nchw, anchors[:, :, 0, :])
        # feat1_nchw = self.pos_encoding.forward_anchors(feat1_nchw, anchors[:, :, 1, :])
        feat0_pe = self.pos_encoding.forward_anchors_only_pe(anchors[:, :, 0, :])
        feat1_pe = self.pos_encoding.forward_anchors_only_pe(anchors[:, :, 1, :])
        feat0_pe = rearrange(feat0_pe, 'n c an -> n an c')
        feat1_pe = rearrange(feat1_pe, 'n c an -> n an c')
        return feat0_pe, feat1_pe

    def forward(self, data: dict, feat0: torch.Tensor, feat1: torch.Tensor,
                mask0: torch.Tensor = None, mask1: torch.Tensor = None):

        assert feat0 is not feat1  # 防止把两个相同的 feat 输入

        hw0_c = data['hw0_c']
        hw1_c = data['hw1_c']

        anchors, conf_matrix = self.anchor_extractor(data, feat0, feat1, mask0, mask1)

        if self.use_geo_feat:
            feat0, feat1 = self.geo_layer(data, conf_matrix, anchors, feat0, feat1, mask0, mask1)  # 把结构化特征 concate 进去

        prototype0_query, prototype1_query = self.get_anchor_query(data, feat0, feat1, anchors)  # [N, AN, C]
        prototype0 = self.prototype_extractor(prototype0_query, feat0, mask0, h=hw0_c[0], w=hw0_c[1])  # [N, AN, C]
        prototype1 = self.prototype_extractor(prototype1_query, feat1, mask1, h=hw1_c[0], w=hw1_c[1])

        if self.type == 'self':
            feat0_out = self.real_forward(feat0, feat0, prototype0, prototype0, mask0, mask0)
            feat1_out = self.real_forward(feat1, feat1, prototype1, prototype1, mask1, mask1)
        elif self.type == 'cross':
            feat0_out = self.real_forward(feat0, feat1, prototype0, prototype1, mask0, mask1)
            feat1_out = self.real_forward(feat1, feat0, prototype1, prototype0, mask1, mask0)
        else:
            raise KeyError

        return feat0_out, feat1_out

    def real_forward(self, x: torch.Tensor, source: torch.Tensor, x_prototype: torch.Tensor,
                     source_prototype: torch.Tensor, x_mask: torch.Tensor = None, source_mask: torch.Tensor = None):
        bs = x.size(0)
        query, key, value = x, source, source

        # multi-head attention
        query = self.q_proj(query).view(bs, -1, self.nhead, self.dim)  # [N, L, (H, D)]
        key = self.k_proj(key).view(bs, -1, self.nhead, self.dim)  # [N, S, (H, D)]
        value = self.v_proj(value).view(bs, -1, self.nhead, self.dim)

        if self.num_prototype > 0:

            if x_prototype is not source_prototype:
                prototype_trans = torch.einsum('nkc,njc->nkj', x_prototype, source_prototype) / x_prototype.size(2)
                source_prototype = torch.einsum('njc,nkj->nkc', source_prototype, prototype_trans)

            prototype_query = self.prototype_q_proj(x_prototype).view(bs, -1, self.nhead, self.dim)  # [N, K, (H, D)]
            prototype_key = self.prototype_k_proj(source_prototype).view(bs, -1, self.nhead, self.dim)  # [N, K, (H, D)]

            # query_sim = torch.einsum('nlhd,nkhd->nlhk', query, prototype_query)
            # key_sim = torch.einsum('nlhd,nkhd->nlhk', key, prototype_key)

            # FIXME: OT
            query_sim, query_sim_ot = self.ot_layer(query, prototype_query, x_mask, None)
            key_sim, key_sim_ot = self.ot_layer(key, prototype_key, source_mask, None)

            query_sim = query_sim * query_sim_ot
            key_sim = key_sim * key_sim_ot

            query = rearrange(query_sim, 'n l k h -> n l h k')
            key = rearrange(key_sim, 'n s k h -> n s h k')

        message = self.attention(query, key, value, q_mask=x_mask, kv_mask=source_mask)  # [N, L, (H, D)]
        message = self.merge(message.view(bs, -1, self.nhead * self.dim))  # [N, L, C]
        message = self.norm1(message)

        # feed-forward network
        message = self.mlp(torch.cat([x, message], dim=2))
        message = self.norm2(message)

        return x + message
