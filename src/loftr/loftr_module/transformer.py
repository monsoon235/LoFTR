import copy
from typing import List

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
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


class LocalFeatureTransformer(nn.Module):
    """A Local Feature Transformer (LoFTR) module."""

    def __init__(self, config):
        super(LocalFeatureTransformer, self).__init__()

        self.config = config
        self.d_model = config['d_model']
        self.nhead = config['nhead']
        self.layer_names = config['layer_names']
        encoder_layer = LoFTREncoderLayer(config['d_model'], config['nhead'], config['attention'])
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(len(self.layer_names))])

        self.use_prototype = config.get('use_prototype', False)

        if self.use_prototype:
            self.n_prototype = config['n_prototype']
            p = torch.empty([self.n_prototype, self.d_model], dtype=torch.float)
            self.prototype = Parameter(p, requires_grad=True)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, feat0, feat1, mask0=None, mask1=None, feat0_wo_pe=None, feat1_wo_pe=None):
        """
        Args:
            feat0 (torch.Tensor): [N, L, C]
            feat1 (torch.Tensor): [N, S, C]
            mask0 (torch.Tensor): [N, L] (optional)
            mask1 (torch.Tensor): [N, S] (optional)
        """

        assert self.d_model == feat0.size(2), "the feature number of src and transformer must be equal"

        if self.use_prototype:

            bs = feat0.size(0)
            ks = self.n_prototype

            feat0_p = torch.einsum('nlc,pc->nlp', feat0_wo_pe, self.prototype)
            feat1_p = torch.einsum('nsc,pc->nsp', feat1_wo_pe, self.prototype)
            class0 = torch.argmax(feat0_p, dim=2)  # [N, L]
            class1 = torch.argmax(feat1_p, dim=2)  # [N, S]

            feat0_by_b_by_k: List[List[torch.Tensor]] = [
                [feat0[b][(mask0[b] != 0) & (class0[b] == k)] for k in range(ks)]
                for b in range(bs)
            ]
            feat1_by_b_by_k: List[List[torch.Tensor]] = [
                [feat1[b][(mask1[b] != 0) & (class1[b] == k)] for k in range(ks)]
                for b in range(bs)
            ]

            for layer, name in zip(self.layers, self.layer_names):
                for b in range(bs):
                    for k in range(ks):
                        if name == 'self-self':
                            feat0_by_b_by_k[b][k] = layer(
                                feat0_by_b_by_k[b][k].unsqueeze(0), feat0_by_b_by_k[b][k].unsqueeze(0),
                                None, None,
                            ).squeeze(0)
                            feat1_by_b_by_k[b][k] = layer(
                                feat1_by_b_by_k[b][k].unsqueeze(0), feat1_by_b_by_k[b][k].unsqueeze(0),
                                None, None,
                            ).squeeze(0)
                        elif name == 'cross-self':
                            feat0_by_b_by_k[b][k] = layer(
                                feat0_by_b_by_k[b][k].unsqueeze(0), feat1_by_b_by_k[b][k].unsqueeze(0),
                                None, None,
                            ).squeeze(0)
                            feat1_by_b_by_k[b][k] = layer(
                                feat1_by_b_by_k[b][k].unsqueeze(0), feat0_by_b_by_k[b][k].unsqueeze(0),
                                None, None,
                            ).squeeze(0)
                        else:
                            raise KeyError

            for b in range(bs):
                for k in range(ks):
                    feat0[b][(mask0[b] != 0) & (class0[b] == k)] = feat0_by_b_by_k[b][k]
                    feat1[b][(mask1[b] != 0) & (class1[b] == k)] = feat1_by_b_by_k[b][k]

            return feat0, feat1, class0, class1, feat0_p, feat1_p, self.prototype

            ##############

            # feat0_p = torch.einsum('nlc,pc->nlp', feat0_wo_pe, self.prototype)
            # feat1_p = torch.einsum('nsc,pc->nsp', feat1_wo_pe, self.prototype)
            # class0 = torch.argmax(feat0_p, dim=2)  # [N, L]
            # class1 = torch.argmax(feat1_p, dim=2)  # [N, S]
            #
            # for layer, name in zip(self.layers, self.layer_names):
            #
            #     # TODO: 必须每个 batch, kind 单独处理，并行度受限
            #     for b in range(feat0.size(0)):
            #
            #         for k in range(self.n_prototype):
            #
            #             if name == 'self-self':
            #                 feat0[b][class0[b] == k] = layer(
            #                     feat0[b][class0[b] == k].unsqueeze(0), feat0[b][class0[b] == k].unsqueeze(0),
            #                     mask0[b][class0[b] == k].unsqueeze(0), mask0[b][class0[b] == k].unsqueeze(0),
            #                 ).squeeze(0)
            #                 feat1[b][class1[b] == k] = layer(
            #                     feat1[b][class1[b] == k].unsqueeze(0), feat1[b][class1[b] == k].unsqueeze(0),
            #                     mask1[b][class1[b] == k].unsqueeze(0), mask1[b][class1[b] == k].unsqueeze(0),
            #                 ).squeeze(0)
            #             elif name == 'cross-self':
            #                 feat0[b][class0[b] == k] = layer(
            #                     feat0[b][class0[b] == k].unsqueeze(0), feat1[b][class1[b] == k].unsqueeze(0),
            #                     mask0[b][class0[b] == k].unsqueeze(0), mask1[b][class1[b] == k].unsqueeze(0),
            #                 ).squeeze(0)
            #                 feat1[b][class1[b] == k] = layer(
            #                     feat1[b][class1[b] == k].unsqueeze(0), feat0[b][class0[b] == k].unsqueeze(0),
            #                     mask1[b][class1[b] == k].unsqueeze(0), mask0[b][class0[b] == k].unsqueeze(0),
            #                 ).squeeze(0)
            #             elif name == 'prototype':
            #                 prototype_wo_k = torch.cat([self.prototype[:k, :], self.prototype[k + 1:, :]], dim=0)
            #                 feat0[b][class0[b] == k] = layer(
            #                     feat0[b][class0[b] == k].squeeze(0), prototype_wo_k.unsqueeze(0),
            #                     mask0[b][class0[b] == k].squeeze(0), None,
            #                 ).squeeze(0)
            #                 feat1[b][class1[b] == k] = layer(
            #                     feat1[b][class1[b] == k].unsqueeze(0), prototype_wo_k.unsqueeze(0),
            #                     mask1[b][class0[b] == k].unsqueeze(0), None,
            #                 ).squeeze(0)
            #             else:
            #                 raise KeyError
            #
            # return feat0, feat1, class0, class1, feat0_p, feat1_p, self.prototype

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

            return feat0, feat1, None, None, None, None, None
