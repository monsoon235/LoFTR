import copy
import torch
import torch.nn as nn

from einops import rearrange, repeat

from .loftr_encoder import LoFTREncoderLayer
from .new_encoder import NewEncoder
from .anchor import AnchorExtractor
from .geo_layer import GeoLayer
from ..utils.position_encoding import PositionEncodingSine


class LocalFeatureTransformer(nn.Module):
    """A Local Feature Transformer (LoFTR) module."""

    def __init__(self, config):
        super(LocalFeatureTransformer, self).__init__()

        self.config = config
        self.d_model = config['d_model']
        self.nhead = config['nhead']
        self.layer_names = config['layer_names']

        layers = []
        is_first = True
        for name in self.layer_names:
            if name in ['self', 'cross']:
                layers.append(LoFTREncoderLayer(config['d_model'], config['nhead'], config['attention']))
            elif name == 'new-self':
                layers.append(NewEncoder(config, type='self', is_first=is_first))
                is_first = False
            elif name == 'new-cross':
                layers.append(NewEncoder(config, type='cross', is_first=is_first))
                is_first = False
            else:
                raise KeyError
        self.layers = nn.ModuleList(layers)

        self.use_prototype = config.get('use_prototype', False)
        self.num_prototype = config.get('num_prototype', 0)
        self.prototype_query_type = config.get('prototype_query_type', None)  # query / anchor
        self.use_geo_feat = config.get('use_geo_feat', False)

        if self.use_geo_feat or (self.use_prototype and self.prototype_query_type == 'anchor'):
            self.anchor_extractor = AnchorExtractor(config['anchor_extractor'])
            self.pos_encoding = PositionEncodingSine(self.d_model, temp_bug_fix=config['temp_bug_fix'])

        if self.use_geo_feat:
            self.geo_layer = GeoLayer(config['geo_layer'])

        if self.use_prototype and self.prototype_query_type == 'query':
            self.prototype_query = nn.Parameter(
                torch.empty(size=(self.num_prototype, self.d_model), dtype=torch.float32), requires_grad=True)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def get_anchor_query(self, data: dict, feat0: torch.Tensor, feat1: torch.Tensor, anchors: torch.Tensor):
        hw0_c = data['hw0_c']
        hw1_c = data['hw1_c']
        feat0_nchw = rearrange(feat0, 'n (h w) c -> n c h w', h=hw0_c[0])
        feat1_nchw = rearrange(feat1, 'n (h w) c -> n c h w', h=hw1_c[0])
        feat0_nchw = self.pos_encoding.forward_anchors(feat0_nchw, anchors[:, :, 0, :])
        feat1_nchw = self.pos_encoding.forward_anchors(feat1_nchw, anchors[:, :, 1, :])
        feat0_pe = rearrange(feat0_nchw, 'n c an -> n an c')
        feat1_pe = rearrange(feat1_nchw, 'n c an -> n an c')
        return feat0_pe, feat1_pe
        # feat0_pe = self.pos_encoding.forward_anchors_only_pe(anchors[:, :, 0, :])
        # feat1_pe = self.pos_encoding.forward_anchors_only_pe(anchors[:, :, 1, :])
        # feat0_pe = rearrange(feat0_pe, 'n c an -> n an c')
        # feat1_pe = rearrange(feat1_pe, 'n c an -> n an c')
        # return feat0_pe, feat1_pe

    def forward(self, data, feat0, feat1, mask0=None, mask1=None):
        """
        Args:
            feat0 (torch.Tensor): [N, L, C]
            feat1 (torch.Tensor): [N, S, C]
            mask0 (torch.Tensor): [N, L] (optional)
            mask1 (torch.Tensor): [N, S] (optional)
        """

        assert self.d_model == feat0.size(2), "the feature number of src and transformer must be equal"

        if self.use_geo_feat or (self.use_prototype and self.prototype_query_type == 'anchor'):
            anchors, conf_matrix = self.anchor_extractor(data, feat0, feat1, mask0, mask1)

        if self.use_geo_feat:
            feat0, feat1 = self.geo_layer(data, conf_matrix, anchors, feat0, feat1, mask0, mask1)

        if self.use_prototype:
            if self.prototype_query_type == 'query':
                prototype_query = repeat(self.prototype_query, 'k c -> n k c', n=feat0.size(0))
                prototype0_query = prototype_query
                prototype1_query = prototype_query
            elif self.prototype_query_type == 'anchor':
                prototype0_query, prototype1_query = self.get_anchor_query(data, feat0, feat1, anchors)
            else:
                raise KeyError

        for layer, name in zip(self.layers, self.layer_names):
            if name == 'self':
                feat0 = layer(feat0, feat0, mask0, mask0)
                feat1 = layer(feat1, feat1, mask1, mask1)
            elif name == 'cross':
                feat0, feat1 = layer(feat0, feat1, mask0, mask1), layer(feat1, feat0, mask1, mask0)
            elif name.startswith('new'):
                feat0, feat1, prototype0_query, prototype1_query = \
                    layer(data, feat0, feat1, mask0, mask1, prototype0_query, prototype1_query)
            else:
                raise KeyError

        return feat0, feat1
