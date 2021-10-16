import copy
import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.einops import rearrange

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
        self.use_v4 = config.get('use_v4', False)
        if self.use_v4:
            self.class_num = config['class_num']
            self.window_size = config['window_size']
            self.select_window_num = config['select_window_num']
            self.other_window_sample_num = config['other_window_sample_num']
            self.classify_conv = nn.Conv1d(in_channels=self.d_model, out_channels=self.class_num, kernel_size=(1,),
                                           stride=(1,), bias=False)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def pad_split(self, tensor: torch.Tensor, h: int, w: int, ws: int) -> torch.Tensor:
        n, _, c = tensor.shape
        tensor_hw = rearrange(tensor, 'n (h w) c -> n c h w', h=h, w=w)
        if h % ws != 0 or w % ws != 0:
            pad = ((w % ws) // 2, (w % ws) - (w % ws) // 2,
                   (h % ws) // 2, (h % ws) - (h % ws) // 2)
            tensor_hw = F.pad(tensor, pad=pad, mode='constant', value=0)
        size = (n, c, tensor_hw.shape[2] // ws, tensor_hw.shape[3] // ws, ws, ws)
        stride = (tensor_hw.stride(0), tensor_hw.stride(1),
                  ws * tensor_hw.stride(2), ws * tensor_hw.stride(3),
                  tensor_hw.stride(2), tensor_hw.stride(3))
        tensor_hw_split = tensor_hw.as_strided(size=size, stride=stride)
        tensor_hw_split = rearrange(tensor_hw_split, 'n c hh ww ws1 ws2 -> n (hh ww) (ws1 ws2) c')
        return tensor_hw_split

    def preprocess(self, feat: torch.Tensor, mask: torch.Tensor, h: int, w: int) \
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        n, l, c = feat.shape
        ws = self.window_size
        cn = self.class_num

        feat_in = rearrange(feat, 'n hw c -> n c hw')
        feat_class_sim = self.classify_conv(feat_in)  # [n, cn, hw]
        feat_class_sim = rearrange(feat_class_sim, 'n cn hw -> n hw cn')

        feat_split = self.pad_split(feat, h=h, w=w, ws=ws)
        mask_split = self.pad_split(mask[:, :, None], h=h, w=w, ws=ws).squeeze(-1)
        feat_split_class_sim = self.pad_split(feat_class_sim, h=h, w=w, ws=ws)

        # 全是 -inf 时 softmax 输出 nan
        # 解决方法是定义一个 mask_win_rep [n, wn]
        # 如果某个区域内所有像素都被 mask，则它也被 mask
        mask_win_rep = torch.any(mask_split, dim=2)  # [n, wn]

        # feat_split_class_sim = torch.masked_fill(feat_split_class_sim, ~mask_split[:, :, :, None],
        #                                          value=-math.inf)  # 排除被 mask 的元素
        # feat_split_class_sim = torch.masked_fill(feat_split_class_sim, ~mask_win_rep[:, :, None, None],
        #                                          value=0)  # 避免计算时出现 nan
        feat_class_heatmap = torch.softmax(feat_split_class_sim, dim=3)  # [n, wn, wsws, cn]
        feat_class_avg = torch.mean(feat_class_heatmap, dim=2)  # [n, wn, cn]
        feat_class = torch.argmax(feat_class_heatmap, dim=3)  # [n, wn, wsws]

        prototype = self.classify_conv.weight.squeeze(2)  # [cn, c]

        feat_win_rep = torch.tensordot(feat_class_avg, prototype, dims=[[2], [0]])  # [n, wn, c]

        return feat_split, feat_win_rep, mask_split, mask_win_rep, feat_class

    def select(self, feat_split: torch.Tensor, feat_win_rep: torch.Tensor,
               mask_split: torch.Tensor, mask_win_rep: torch.Tensor, sim: torch.Tensor) \
            -> Tuple[torch.Tensor, torch.Tensor]:

        n, wn0, wsws, c = feat_split.shape
        _, wn1, _ = sim.shape
        k = self.select_window_num

        _, selected_wins_indices = torch.topk(sim, k=k, dim=2)  # [n, wn1, k]

        not_selected_wins_indices = torch.empty(n, wn1, wn0 - k, dtype=selected_wins_indices.dtype)
        for n_i in range(n):
            for wn_i in range(wn1):
                is_not_selected = torch.ones(wn0, dtype=torch.bool)
                is_not_selected[selected_wins_indices[n_i, wn_i]] = False
                not_selected_wins_indices[n_i, wn_i] = torch.where(is_not_selected)[0]
        not_selected_wins_indices = not_selected_wins_indices.to(device=selected_wins_indices.device)

        feat_selected = torch.gather(
            input=feat_split.as_strided(  # [n, wn0, wsws, c] -> [n, wn1, wn0, wsws, c]
                size=(n, wn1, wn0, wsws, c),
                stride=feat_split.stride()[:1] + (0,) + feat_split.stride()[1:]
            ),
            dim=2,
            index=selected_wins_indices.as_strided(  # [n, wn1, k] -> [n, wn1, k, wsws, c]
                size=(n, wn1, k, wsws, c),
                stride=selected_wins_indices.stride()[:3] + (0, 0)
            )
        )
        mask_selected = torch.gather(
            input=mask_split.as_strided(  # [n, wn0, wsws] -> [n, wn1, wn0, wsws]
                size=(n, wn1, wn0, wsws),
                stride=mask_split.stride()[:1] + (0,) + mask_split.stride()[1:]
            ),
            dim=2,
            index=selected_wins_indices.as_strided(  # [n, wn1, k] -> [n, wn1, k, wsws]
                size=(n, wn1, k, wsws),
                stride=selected_wins_indices.stride()[:3] + (0,)
            )
        )
        feat_not_selected = torch.gather(
            input=feat_win_rep.as_strided(  # [n, wn0, c] -> [n, wn1, wn0, c]
                size=(n, wn1, wn0, c),
                stride=feat_win_rep.stride()[:1] + (0,) + feat_win_rep.stride()[1:]
            ),
            dim=2,
            index=not_selected_wins_indices.as_strided(  # [n, wn1, wn0-k] -> [n, wn1, wn0-k, c]
                size=(n, wn1, wn0 - k, c),
                stride=not_selected_wins_indices.stride()[:3] + (0,)
            )
        )
        mask_not_selected = torch.gather(
            input=mask_win_rep.as_strided(  # [n, wn0] -> [n, wn1, wn0]
                size=(n, wn1, wn0),
                stride=mask_win_rep.stride()[:1] + (0,) + mask_win_rep.stride()[1:]
            ),
            dim=2,
            index=not_selected_wins_indices,  # [n, wn1, wn0-k]
        )

        feat_selected_flatten = rearrange(feat_selected, 'n wn k wsws c -> (n wn) (k wsws) c')
        mask_selected_flatten = rearrange(mask_selected, 'n wn k wsws -> (n wn) (k wsws)')
        feat_not_selected_flatten = rearrange(feat_not_selected, 'n wn wnk c -> (n wn) wnk c')
        mask_not_selected_flatten = rearrange(mask_not_selected, 'n wn wnk -> (n wn) wnk')

        feat_target_flatten = torch.cat([feat_selected_flatten, feat_not_selected_flatten], dim=1)
        mask_target_flatten = torch.cat([mask_selected_flatten, mask_not_selected_flatten], dim=1)

        return feat_target_flatten, mask_target_flatten

    def fold_clip_flatten(self, feat_split_flatten: torch.Tensor, h: int, w: int, n: int) -> torch.Tensor:
        ws = self.window_size
        feat_split = rearrange(feat_split_flatten, '(n wn) wsws c -> n (c wsws) wn', n=n)
        h_o = ws * (h // ws) + (h % ws)
        w_o = ws * (w // ws) + (w % ws)
        feat = F.fold(feat_split, output_size=(h_o, w_o), kernel_size=ws, stride=ws)  # [n, c, h_o, w_o]
        if h % ws != 0:
            feat = feat[:, :, (h % ws) // 2: -((h % ws) - (h % ws) // 2), :]  # [n, c, h, w]
        if w % ws != 0:
            feat = feat[:, :, :, (w % ws) // 2:-((w % ws) - (w % ws) // 2)]  # [n, c, h, w]

        return rearrange(feat, 'n c h w -> n (h w) c')

    def forward(self, feat0, feat1, data, mask0=None, mask1=None):
        """
        Args:
            feat0 (torch.Tensor): [N, L, C]
            feat1 (torch.Tensor): [N, S, C]
            data (dict)
            mask0 (torch.Tensor): [N, L] (optional)
            mask1 (torch.Tensor): [N, S] (optional)
        """

        if self.use_v4:

            N, L, C = feat0.shape
            _, S, _ = feat1.shape

            H0, W0 = data['hw0_c']
            H1, W1 = data['hw1_c']

            WS = self.window_size

            assert self.d_model == feat0.size(2), "the feature number of src and transformer must be equal"

            for layer, name in zip(self.layers, self.layer_names):

                feat0_split, feat0_win_rep, mask0_split, mask0_win_rep, feat0_class = \
                    self.preprocess(feat0, mask0, H0, W0)
                feat1_split, feat1_win_rep, mask1_split, mask1_win_rep, feat1_class = \
                    self.preprocess(feat1, mask1, H1, W1)

                feat0_split_flatten = rearrange(feat0_split, 'n wn wsws c -> (n wn) wsws c')
                feat1_split_flatten = rearrange(feat1_split, 'n wn wsws c -> (n wn) wsws c')
                mask0_split_flatten = rearrange(mask0_split, 'n wn wsws -> (n wn) wsws')
                mask1_split_flatten = rearrange(mask1_split, 'n wn wsws -> (n wn) wsws')

                if name == 'self':

                    sim00 = torch.cosine_similarity(feat0_win_rep[:, :, None, :], feat0_win_rep[:, None, :, :],
                                                    dim=3)  # [n wn wn]
                    sim11 = torch.cosine_similarity(feat1_win_rep[:, :, None, :], feat1_win_rep[:, None, :, :],
                                                    dim=3)  # [n wn wn]
                    torch.masked_fill(sim00, ~(mask0_win_rep[:, :, None] * mask0_win_rep[:, None, :]), value=-1)
                    torch.masked_fill(sim11, ~(mask1_win_rep[:, :, None] * mask1_win_rep[:, None, :]), value=-1)

                    feat0_target_flatten, mask0_target_flatten = \
                        self.select(feat0_split, feat0_win_rep, mask0_split, mask0_win_rep, sim00)

                    feat1_target_flatten, mask1_target_flatten = \
                        self.select(feat1_split, feat1_win_rep, mask1_split, mask1_win_rep, sim11)

                    feat0_split_flatten = layer(feat0_split_flatten, feat0_target_flatten,  # [n*wn, ws*ws, c]
                                                mask0_split_flatten, mask0_target_flatten)
                    feat1_split_flatten = layer(feat1_split_flatten, feat1_target_flatten,  # [n*wn, ws*ws, c]
                                                mask1_split_flatten, mask1_target_flatten)

                    feat0 = self.fold_clip_flatten(feat0_split_flatten, H0, W0, N)
                    feat1 = self.fold_clip_flatten(feat1_split_flatten, H1, W1, N)

                elif name == 'cross':

                    sim01 = torch.cosine_similarity(feat0_win_rep[:, :, None, :], feat0_win_rep[:, None, :, :],
                                                    dim=3)  # [n wn wn]
                    torch.masked_fill(sim01, ~(mask0_win_rep[:, :, None] * mask0_win_rep[:, None, :]), value=-1)

                    sim10 = sim01.transpose(1, 2)

                    feat0_target_flatten, mask0_target_flatten = \
                        self.select(feat0_split, feat0_win_rep, mask0_split, mask0_win_rep, sim10)

                    feat1_target_flatten, mask1_target_flatten = \
                        self.select(feat1_split, feat1_win_rep, mask1_split, mask1_win_rep, sim01)

                    feat0_split_flatten = layer(feat0_split_flatten, feat0_target_flatten,  # [n*wn, ws*ws, c]
                                                mask1_split_flatten, mask1_target_flatten)
                    feat1_split_flatten = layer(feat1_split_flatten, feat1_target_flatten,  # [n*wn, ws*ws, c]
                                                mask0_split_flatten, mask0_target_flatten)

                    feat0 = self.fold_clip_flatten(feat0_split_flatten, H0, W0, N)
                    feat1 = self.fold_clip_flatten(feat1_split_flatten, H1, W1, N)

                else:
                    raise KeyError
        else:

            assert self.d_model == feat0.size(2), "the feature number of src and transformer must be equal"

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
