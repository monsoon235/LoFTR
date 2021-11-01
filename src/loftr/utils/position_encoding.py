import math
import torch
from torch import nn

from einops import rearrange, repeat


class PositionEncodingSine(nn.Module):
    """
    This is a sinusoidal position encoding that generalized to 2-dimensional images
    """

    def __init__(self, d_model, max_shape=(256, 256), temp_bug_fix=True):
        """
        Args:
            max_shape (tuple): for 1/8 featmap, the max length of 256 corresponds to 2048 pixels
            temp_bug_fix (bool): As noted in this [issue](https://github.com/zju3dv/LoFTR/issues/41),
                the original implementation of LoFTR includes a bug in the pos-enc impl, which has little impact
                on the final performance. For now, we keep both impls for backward compatability.
                We will remove the buggy impl after re-training all variants of our released models.
        """
        super().__init__()

        pe = torch.zeros((d_model, *max_shape))
        y_position = torch.ones(max_shape).cumsum(0).float().unsqueeze(0)
        x_position = torch.ones(max_shape).cumsum(1).float().unsqueeze(0)
        if temp_bug_fix:
            div_term = torch.exp(torch.arange(0, d_model // 2, 2).float() * (-math.log(10000.0) / (d_model // 2)))
        else:  # a buggy implementation (for backward compatability only)
            div_term = torch.exp(torch.arange(0, d_model // 2, 2).float() * (-math.log(10000.0) / d_model // 2))
        div_term = div_term[:, None, None]  # [C//4, 1, 1]
        pe[0::4, :, :] = torch.sin(x_position * div_term)
        pe[1::4, :, :] = torch.cos(x_position * div_term)
        pe[2::4, :, :] = torch.sin(y_position * div_term)
        pe[3::4, :, :] = torch.cos(y_position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0), persistent=False)  # [1, C, H, W]

    def forward(self, x):
        """
        Args:
            x: [N, C, H, W]
        """
        return x + self.pe[:, :, :x.size(2), :x.size(3)]

    def forward_anchors(self, feat_nchw: torch.Tensor, anchors: torch.Tensor):
        n, c, h, w = feat_nchw.shape

        result = []
        for b in range(n):
            feat_b_select = feat_nchw[b, :, anchors[b, :, 0].long(), anchors[b, :, 1].long()]  # [C, AN]
            feat_b_select += self.pe[0, :, anchors[b, :, 0].long(), anchors[b, :, 1].long()]
            result.append(feat_b_select)
        result = torch.stack(result, dim=0)

        return result

    def forward_anchors_only_pe(self, anchors: torch.Tensor):
        n = anchors.size(0)
        result = []
        for b in range(n):
            select = self.pe[0, :, anchors[b, :, 0].long(), anchors[b, :, 1].long()]
            result.append(select)
        result = torch.stack(result, dim=0)

        return result

    def get_hw_flatten(self, n, h, w):
        result = self.pe[0, :, :h, :w]
        result = repeat(result, 'c h w -> n c h w', n=n)
        result = rearrange(result, 'n c h w -> n (h w) c')
        return result
