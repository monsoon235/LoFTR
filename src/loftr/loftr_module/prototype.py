import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat

from .linear_attention import LinearAttention, FullAttention
from ..utils.position_encoding import PositionEncodingSine


class DETRBlock(nn.Module):

    def __init__(self, config):
        super(DETRBlock, self).__init__()

        d_model = config['d_model']
        self.n_head = config['n_head']
        self.n_dim = d_model // self.n_head
        attention = config['attention']

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

    def forward(self, query: torch.Tensor, query_pe: torch.Tensor, feat: torch.Tensor, feat_pe: torch.Tensor,
                feat_mask: torch.Tensor = None) -> torch.Tensor:
        bs = query.size(0)

        if query_pe is not None:
            query_in = query + query_pe
        else:
            query_in = query

        query_q = self.query_q_proj(query_in).view(bs, -1, self.n_head, self.n_dim)
        query_k = self.query_k_proj(query_in).view(bs, -1, self.n_head, self.n_dim)
        query_v = self.query_v_proj(query).view(bs, -1, self.n_head, self.n_dim)
        message = self.query_self_attention(query_q, query_k, query_v)
        message = message.view(bs, -1, self.n_head * self.n_dim)
        message = self.query_merge(message)

        message = self.norm1(query + message)

        if query_pe is not None:
            message_in = message + query_pe
        else:
            message_in = message
        if feat_pe is not None:
            feat_in = feat + feat_pe
        else:
            feat_in = feat

        message_q = self.prototype_q_proj(message_in).view(bs, -1, self.n_head, self.n_dim)
        feat_k = self.prototype_k_proj(feat_in).view(bs, -1, self.n_head, self.n_dim)
        feat_v = self.prototype_v_proj(feat).view(bs, -1, self.n_head, self.n_dim)
        prototype = self.prototype_extractor_attention(message_q, feat_k, feat_v, None, feat_mask)
        prototype = prototype.view(bs, -1, self.n_head * self.n_dim)
        prototype = self.prototype_merge(prototype)

        prototype = self.norm2(message + prototype)
        prototype_ffn = self.ffn(prototype)
        prototype = self.norm3(prototype + prototype_ffn)

        return prototype


class PrototypeTransformer(nn.Module):

    def __init__(self, config, is_first=False):
        super(PrototypeTransformer, self).__init__()
        num_block = config['num_block_first'] if is_first else config['num_block']
        self.blocks = nn.ModuleList([DETRBlock(config['block']) for _ in range(num_block)])
        self.pos_encoding = PositionEncodingSine(d_model=config['block']['d_model'], temp_bug_fix=True)

    def forward(self, query: torch.Tensor, feat: torch.Tensor, feat_mask: torch.Tensor, h: int, w: int,
                use_query_pe: bool, use_feat_pe: bool) -> torch.Tensor:
        query_in = query
        if use_query_pe:
            query_pe = query
        else:
            query_pe = None
        if use_feat_pe and len(self.blocks) > 0:
            feat_pe = self.pos_encoding.get_hw_flatten(feat.size(0), h, w)
        else:
            feat_pe = None
        for block in self.blocks:
            query_in = block.forward(query=query_in, query_pe=query_pe, feat=feat, feat_pe=feat_pe, feat_mask=feat_mask)
        return query_in
