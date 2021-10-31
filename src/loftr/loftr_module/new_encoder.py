import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat

from .prototype import PrototypeTransformer
from .loftr_encoder import LoFTREncoderLayer


class NewEncoder(nn.Module):

    def __init__(self, config, type: str):
        super(NewEncoder, self).__init__()

        self.type = type

        self.num_prototype = config['num_prototype']
        self.d_model = config['d_model']
        self.nhead = config['nhead']

        self.prototype_extractor = PrototypeTransformer(config['prototype_extractor'])
        self.prototype_query = nn.Parameter(torch.empty(size=(self.num_prototype, self.d_model), dtype=torch.float32),
                                            requires_grad=True)

        self.semantic_feat_merger = nn.Sequential(
            nn.Linear(in_features=self.d_model + self.num_prototype, out_features=self.d_model),
            nn.LayerNorm(self.d_model),
        )

        self.encoder_layer = LoFTREncoderLayer(d_model=self.d_model, nhead=self.nhead, attention=config['attention'])

    def forward(self, data: dict, feat0: torch.Tensor, feat1: torch.Tensor, mask0: torch.Tensor, mask1: torch.Tensor):
        assert feat0 is not feat1

        bs = feat0.size(0)
        hw0_c = data['hw0_c']
        hw1_c = data['hw1_c']

        query_in = repeat(self.prototype_query, 'k c -> b k c', b=bs)

        prototype0 = self.prototype_extractor(query_in, feat0, mask0, hw0_c[0], hw0_c[1])
        prototype1 = self.prototype_extractor(query_in, feat1, mask1, hw1_c[0], hw1_c[1])

        sim0 = torch.einsum('nlc,nkc->nlk', feat0, prototype0)
        sim1 = torch.einsum('nsc,njc->nsj', feat1, prototype1)

        feat0_cat = torch.cat([feat0, sim0], dim=-1)
        feat1_cat = torch.cat([feat1, sim1], dim=-1)

        feat0_in = self.semantic_feat_merger(feat0_cat)
        feat1_in = self.semantic_feat_merger(feat1_cat)

        if self.type == 'self':

            feat0 = self.encoder_layer(feat0_in, feat0_in, mask0, mask0)
            feat1 = self.encoder_layer(feat1_in, feat1_in, mask1, mask1)

        elif self.type == 'cross':

            # 需要一个变换矩阵
            sim_trans_matrix = torch.einsum('nkc,njc->nkj', prototype0, prototype1)

            sim0_trans = torch.einsum('nlk,nkj->nlj', sim0, sim_trans_matrix)
            sim1_trans = torch.einsum('nsj,nkj->nsk', sim1, sim_trans_matrix)

            feat0_source = self.semantic_feat_merger(torch.cat([feat0, sim0_trans], dim=-1))
            feat1_source = self.semantic_feat_merger(torch.cat([feat1, sim1_trans], dim=-1))

            feat0 = self.encoder_layer(feat0_in, feat1_source, mask0, mask1)
            feat1 = self.encoder_layer(feat1_in, feat0_source, mask1, mask0)

        else:
            raise KeyError

        return feat0, feat1
