import copy
import torch
import torch.nn as nn
from .linear_attention import LinearAttention, FullAttention
from .loftr_encoder import LoFTREncoderLayer
from .new_encoder import NewEncoder
from .prototype import PrototypeTransformer

from einops import repeat


class LocalFeatureTransformer(nn.Module):
    """A Local Feature Transformer (LoFTR) module."""

    def __init__(self, config):
        super(LocalFeatureTransformer, self).__init__()

        self.config = config
        self.d_model = config['d_model']
        self.nhead = config['nhead']
        self.layer_names = config['layer_names']
        # encoder_layer = LoFTREncoderLayer(config['d_model'], config['nhead'], config['attention'])
        # self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(len(self.layer_names))])

        layers = []

        for name in self.layer_names:
            if name in ['self', 'cross']:
                layers.append(LoFTREncoderLayer(config['d_model'], config['nhead'], config['attention']))
            else:
                raise KeyError

        self.layers = nn.ModuleList(layers)

        self.use_prototype = config.get('use_prototype', None)
        self.num_prototype = config.get('num_prototype', None)

        if self.use_prototype:
            self.prototype_extractor = PrototypeTransformer(config['prototype_extractor'])
            self.prototype_query = nn.Parameter(
                torch.empty(size=(self.num_prototype, self.d_model), dtype=torch.float32),
                requires_grad=True)
            self.semantic_feat_merger = nn.Sequential(
                nn.Linear(in_features=self.d_model + self.num_prototype, out_features=self.d_model, bias=False),
                nn.LayerNorm(self.d_model),
            )

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, data, feat0, feat1, mask0, mask1, feat0_no_pe, feat1_no_pe):
        """
        Args:
            feat0 (torch.Tensor): [N, L, C]
            feat1 (torch.Tensor): [N, S, C]
            mask0 (torch.Tensor): [N, L] (optional)
            mask1 (torch.Tensor): [N, S] (optional)
        """

        assert self.d_model == feat0.size(2), "the feature number of src and transformer must be equal"

        bs = feat0.size(0)
        c = feat0.size(-1)
        hw0_c = data['hw0_c']
        hw1_c = data['hw1_c']

        if self.use_prototype:
            query_in = repeat(self.prototype_query, 'k c -> b k c', b=bs)

            prototype0 = self.prototype_extractor(query_in, feat0_no_pe, mask0, hw0_c[0], hw0_c[1], use_query_pe=False,
                                                  use_feat_pe=False, value=feat0)
            prototype1 = self.prototype_extractor(query_in, feat1_no_pe, mask1, hw1_c[0], hw1_c[1], use_query_pe=False,
                                                  use_feat_pe=False, value=feat1)

            sim0 = torch.cosine_similarity(feat0[:, :, None, :], prototype0[:, None, :, :], dim=-1)
            sim1 = torch.cosine_similarity(feat1[:, :, None, :], prototype1[:, None, :, :], dim=-1)

            feat0_cat = torch.cat([feat0, sim0], dim=-1)
            feat1_cat = torch.cat([feat1, sim1], dim=-1)

            feat0 = self.semantic_feat_merger(feat0_cat)
            feat1 = self.semantic_feat_merger(feat1_cat)

        for layer, name in zip(self.layers, self.layer_names):
            if name == 'self':
                feat0 = layer(feat0, feat0, mask0, mask0)
                feat1 = layer(feat1, feat1, mask1, mask1)
            elif name == 'cross':
                feat0, feat1 = layer(feat0, feat1, mask0, mask1), layer(feat1, feat0, mask1, mask0)
            else:
                raise KeyError

        return feat0, feat1
