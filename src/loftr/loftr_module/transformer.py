import copy
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

    def forward(self, feat0, feat1, mask0=None, mask1=None):
        """
        Args:
            feat0 (torch.Tensor): [N, L, C]
            feat1 (torch.Tensor): [N, S, C]
            mask0 (torch.Tensor): [N, L] (optional)
            mask1 (torch.Tensor): [N, S] (optional)
        """

        assert self.d_model == feat0.size(2), "the feature number of src and transformer must be equal"

        if self.use_prototype:

            # 禁止往 feat0 和 feat1 传播梯度
            feat0_p = torch.einsum('nlc,pc->nlp', feat0.detach(), self.prototype)
            feat1_p = torch.einsum('nsc,pc->nsp', feat1.detach(), self.prototype)
            class0 = torch.argmax(feat0_p, dim=2)  # [N, L]
            class1 = torch.argmax(feat1_p, dim=2)  # [N, S]

            # TODO: 必须每个 batch, kind 单独处理，并行度受限
            feat0_out = torch.empty_like(feat0)
            feat1_out = torch.empty_like(feat1)

            for b in range(feat0.size(0)):
                feat0_b = feat0[b]  # [L, C]
                feat1_b = feat1[b]  # [S, C]
                mask0_b = mask0[b]  # [L]
                mask1_b = mask1[b]  # [S]
                class0_b = class0[b]  # [L]
                class1_b = class1[b]  # [S]

                feat0_out_b_by_k = [None] * self.n_prototype  # [L', C] * P
                feat1_out_b_by_k = [None] * self.n_prototype  # [S', C] * P

                for k in range(self.n_prototype):
                    feat0_b_k = feat0_b[class0_b == k].unsqueeze(0)  # [1, L', C]
                    feat1_b_k = feat1_b[class1_b == k].unsqueeze(0)  # [1, S', C]
                    mask0_b_k = mask0_b[class0_b == k].unsqueeze(0)
                    mask1_b_k = mask1_b[class1_b == k].unsqueeze(0)
                    feat0_b_not_k = feat0_b[class0_b != k].unsqueeze(0)  # [1, L'', C]
                    feat1_b_not_k = feat1_b[class1_b != k].unsqueeze(0)  # [1, S'', C]
                    mask0_b_not_k = mask0_b[class0_b != k].unsqueeze(0)
                    mask1_b_not_k = mask1_b[class1_b != k].unsqueeze(0)
                    for layer, name in zip(self.layers, self.layer_names):
                        if name == 'self-self':
                            feat0_out_b_by_k[k] = layer(feat0_b_k, feat0_b_k, mask0_b_k, mask0_b_k)
                            feat1_out_b_by_k[k] = layer(feat1_b_k, feat1_b_k, mask1_b_k, mask1_b_k)
                        elif name == 'self-cross':
                            feat0_out_b_by_k[k] = layer(feat0_b_k, feat0_b_not_k, mask0_b_k, mask0_b_not_k)
                            feat1_out_b_by_k[k] = layer(feat1_b_k, feat1_b_not_k, mask1_b_k, mask1_b_not_k)
                        elif name == 'cross-self':
                            feat0_out_b_by_k[k] = layer(feat0_b_k, feat1_b_k, mask0_b_k, mask1_b_k)
                            feat1_out_b_by_k[k] = layer(feat1_b_k, feat0_b_k, mask1_b_k, mask0_b_k)
                        elif name == 'cross-cross':
                            feat0_out_b_by_k[k] = layer(feat0_b_k, feat1_b_not_k, mask0_b_k, mask1_b_not_k)
                            feat1_out_b_by_k[k] = layer(feat1_b_k, feat0_b_not_k, mask1_b_k, mask0_b_not_k)
                        else:
                            raise KeyError

                feat0_out[b] = torch.cat(feat0_out_b_by_k, dim=1)
                feat1_out[b] = torch.cat(feat1_out_b_by_k, dim=1)

            feat0, feat1 = feat0_out, feat1_out

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
