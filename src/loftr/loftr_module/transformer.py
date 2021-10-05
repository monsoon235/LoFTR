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
        self.attention_class = attention
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

    def forward(self, x, source, x_mask=None, source_mask=None, x_class=None, source_class=None):
        """
        Args:
            x (torch.Tensor): [N, L, C]
            source (torch.Tensor): [N, S, C]
            x_mask (torch.Tensor): [N, L] (optional)
            source_mask (torch.Tensor): [N, S] (optional)
            x_class: [N, L]
            source_class: [N, S]
        """

        assert self.attention_class == 'full' or (x_class is None and source_class is None)

        bs = x.size(0)
        query, key, value = x, source, source

        # multi-head attention
        query = self.q_proj(query).view(bs, -1, self.nhead, self.dim)  # [N, L, (H, D)]
        key = self.k_proj(key).view(bs, -1, self.nhead, self.dim)  # [N, S, (H, D)]
        value = self.v_proj(value).view(bs, -1, self.nhead, self.dim)
        if self.attention_class == 'full':
            message = self.attention(query, key, value, q_mask=x_mask, kv_mask=source_mask, q_class=x_class,
                                     kv_class=source_class)  # [N, L, (H, D)]
        else:
            message = self.attention(query, key, value, q_mask=x_mask, kv_mask=source_mask)  # [N, L, (H, D)]
        message = self.merge(message.view(bs, -1, self.nhead * self.dim))  # [N, L, C]
        message = self.norm1(message)

        # feed-forward network
        message = self.mlp(torch.cat([x, message], dim=2))
        message = self.norm2(message)

        return x + message


class Sampler(nn.Module):

    def __init__(self, in_channels: int, num_class: int, num_sample: int):
        super(Sampler, self).__init__()
        self.num_sample = num_sample
        self.num_class = num_class
        self.conv_by_k = nn.ModuleList([
            nn.Conv1d(in_channels=in_channels, out_channels=num_sample, kernel_size=1, stride=1, bias=False)
            for _ in range(num_class)
        ])
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, feat: torch.Tensor, class_map: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        :param feat: [N, L, C]
        :param class_map: [N, L]
        :param mask: [N, L]
        :return: [N, num_sample * num_class, C]
        """
        feat_in = torch.transpose(feat, 1, 2)  # [N, C, L]
        bs = feat.size(0)
        ks = self.num_class
        feat_sampled_by_b = []
        for b in range(bs):
            feat_sampled_by_b_by_k = []
            for k in range(ks):
                feat_selected_b_k = feat_in[b:b + 1, :, class_map[b] == k]  # [1, C, L‘]
                if feat_selected_b_k.size(2) == 0:
                    # 如果某类不存在 token，用 0 填充
                    feat_selected_b_k = torch.zeros(
                        size=(1, feat_in.size(1), 1), dtype=feat.dtype, device=feat.device)
                class_conf = self.conv_by_k[k](feat_selected_b_k)  # [1, num_sample, L‘]
                if mask is not None:
                    mask_selected_b_k = mask[b:b + 1, class_map[b] == k]  # [1, L']
                    class_conf.masked_fill(~mask_selected_b_k[:, None, :], float('-inf'))  # 被 mask 的元素不参与采样
                class_conf = torch.softmax(class_conf, dim=2)
                feat_weighted = feat_selected_b_k[:, :, None, :] * class_conf[:, None, :,
                                                                   :]  # [1, C, num_sample, L']
                feat_sampled_b_k = feat_weighted.sum(dim=3)  # [1, C, num_sample]
                feat_sampled_by_b_by_k.append(feat_sampled_b_k)
            feat_sampled_by_b.append(torch.cat(feat_sampled_by_b_by_k, dim=2))
        feat_sampled = torch.cat(feat_sampled_by_b, dim=0)  # [N, C, num_sample * num_class]
        feat_sampled = torch.transpose(feat_sampled, 1, 2)  # [N, num_sample * num_class, C]
        return feat_sampled


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

        self.num_sample = config.get('num_sample', -1)
        if self.num_sample > 0:
            self.samplers = nn.ModuleList(
                [Sampler(in_channels=self.d_model, num_class=self.n_prototype, num_sample=self.num_sample) for _ in
                 range(len(self.layer_names))])

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

            for i, (layer, name) in enumerate(zip(self.layers, self.layer_names)):
                # 每层 attention 前都进行一次采样
                feat0_sampled = self.samplers[i](feat0, class0, mask0)  # [N, num_class * num_sample, C]
                feat1_sampled = self.samplers[i](feat1, class1, mask1)  # [N, num_class * num_sample, C]
                kv_class = torch.arange(0, ks, device=class0.device).view(-1, 1).repeat(1, self.num_sample) \
                    .view(-1)  # [num_class * num_sample]
                kv_class = kv_class.as_strided(size=(bs, ks * self.num_sample), stride=(0, 1))
                if name == 'self-self':
                    feat0 = layer(feat0, feat0_sampled, mask0, None, class0, kv_class)
                    feat1 = layer(feat1, feat1_sampled, mask1, None, class1, kv_class)
                elif name == 'cross-self':
                    feat0 = layer(feat0, feat1_sampled, mask1, None, class0, kv_class)
                    feat1 = layer(feat1, feat0_sampled, mask0, None, class1, kv_class)
                else:
                    raise KeyError

            return feat0, feat1, class0, class1, feat0_p, feat1_p, self.prototype

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
