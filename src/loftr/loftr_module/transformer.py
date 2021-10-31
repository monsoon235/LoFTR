import copy
import torch
import torch.nn as nn

from .loftr_encoder import LoFTREncoderLayer
from .new_encoder import NewEncoder


class LocalFeatureTransformer(nn.Module):
    """A Local Feature Transformer (LoFTR) module."""

    def __init__(self, config):
        super(LocalFeatureTransformer, self).__init__()

        self.config = config
        self.d_model = config['d_model']
        self.nhead = config['nhead']
        self.layer_names = config['layer_names']

        layers = []

        for name in self.layer_names:
            if name == 'self' or name == 'cross':
                layers.append(LoFTREncoderLayer(config['d_model'], config['nhead'], config['attention']))
            elif name.startswith('new'):
                layers.append(NewEncoder(config, type='cross'))
            else:
                raise KeyError

        self.layers = nn.ModuleList(layers)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, data, feat0, feat1, mask0=None, mask1=None):
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
                feat0 = layer(feat0, feat1, mask0, mask1)
                feat1 = layer(feat1, feat0, mask1, mask0)
            elif name == 'new' or name == 'new-self' or name == 'new-cross':
                feat0, feat1 = layer(data, feat0, feat1, mask0, mask1)
            else:
                raise KeyError

        return feat0, feat1
