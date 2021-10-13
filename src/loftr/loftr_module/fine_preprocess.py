import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.einops import rearrange, repeat


class FinePreprocess(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.cat_c_feat = config['fine_concat_coarse_feat']
        self.W = self.config['fine_window_size']
        self.fine_use_window_sample = self.config['fine_use_window_sample']
        self.fine_sample_window_size = self.config['fine_sample_window_size']
        self.fine_sample_num = self.config['fine_sample_num']

        d_model_c = self.config['coarse']['d_model']
        d_model_f = self.config['fine']['d_model']
        self.d_model_f = d_model_f
        if self.cat_c_feat:
            self.down_proj = nn.Linear(d_model_c, d_model_f, bias=True)
            self.merge_feat = nn.Linear(2 * d_model_f, d_model_f, bias=True)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.kaiming_normal_(p, mode="fan_out", nonlinearity="relu")

    def forward(self, feat_f0, feat_f1, feat_c0, feat_c1, data):

        if self.fine_use_window_sample:

            W = self.fine_sample_window_size

            stride = data['hw0_f'][0] // data['hw0_c'][0]
            if data['b_ids'].shape[0] == 0:
                feat0 = torch.empty(0, self.fine_sample_num, self.d_model_f, device=feat_f0.device)
                feat1 = torch.empty(0, self.fine_sample_num, self.d_model_f, device=feat_f1.device)
                return feat0, feat1

            # 1. unfold(crop) all local windows
            feat_f0_unfold = F.unfold(feat_f0, kernel_size=(W, W), stride=stride, padding=W // 2)
            feat_f0_unfold = rearrange(feat_f0_unfold, 'n (c ww) l -> n l ww c', ww=W ** 2)
            feat_f1_unfold = F.unfold(feat_f1, kernel_size=(W, W), stride=stride, padding=W // 2)
            feat_f1_unfold = rearrange(feat_f1_unfold, 'n (c ww) l -> n l ww c', ww=W ** 2)

            # 2. select only the predicted matches
            feat_f0_unfold = feat_f0_unfold[data['b_ids'], data['i_ids']]  # [n, ww, cf]
            feat_f1_unfold = feat_f1_unfold[data['b_ids'], data['j_ids']]

            # 3. calc similarity and sample
            feat_f0_center = feat_f0_unfold[:, W * W // 2, :]  # [n, cf]
            feat_f0_unfold_no_center = torch.cat(
                [feat_f0_unfold[:, :W * W // 2, :], feat_f0_unfold[:, W * W // 2 + 1:, :]], dim=1)
            sim0 = torch.cosine_similarity(feat_f0_center.unsqueeze(1), feat_f0_unfold_no_center, dim=2)  # [n, ww-1]
            sim1 = torch.cosine_similarity(feat_f0_center.unsqueeze(1), feat_f1_unfold, dim=2)  # [n, ww]

            _, indices0 = torch.topk(sim0, k=self.fine_sample_num - 1, dim=1)  # [n, k-1]  排除掉自身
            _, indices1 = torch.topk(sim1, k=self.fine_sample_num, dim=1)  # [n, k]

            feat_f0_center_index = torch.tensor(W * W // 2, device=indices0.device, dtype=indices0.dtype) \
                .as_strided(size=(indices0.shape[0], 1), stride=(0, 1))
            indices0 = torch.cat([feat_f0_center_index, indices0], dim=1)  # 把中心位置的坐标放进来

            indices0_extended = indices0.as_strided(size=(indices0.shape[0], indices0.shape[1], self.d_model_f),
                                                    stride=(indices0.shape[1], 1, 0))
            indices1_extended = indices1.as_strided(size=(indices1.shape[0], indices1.shape[1], self.d_model_f),
                                                    stride=(indices1.shape[1], 1, 0))

            selected0 = torch.gather(feat_f0_unfold, dim=1, index=indices0_extended)  # [n, k, cf]
            selected1 = torch.gather(feat_f1_unfold, dim=1, index=indices1_extended)  # [n, k, cf]

            # 在小窗口中的坐标
            # pos0 = torch.stack([indices0 // W, indices0 % W], dim=2)  # [n, k, 2]
            pos1 = torch.stack([indices1 // W, indices1 % W], dim=2)  # [n, k, 2]

            # option: use coarse-level loftr feature as context: concat and linear
            if self.cat_c_feat:
                feat_c0_unfold = feat_c0[data['b_ids'], data['i_ids']]  # [n, c]
                feat_c1_unfold = feat_c1[data['b_ids'], data['j_ids']]  # [n, c]

                feat_c0_unfold = self.down_proj(feat_c0_unfold)  # [n, cf]
                feat_c1_unfold = self.down_proj(feat_c1_unfold)  # [n, cf]

                selected_c0 = feat_c0_unfold.as_strided(
                    size=(feat_c0_unfold.shape[0], self.fine_sample_num, feat_c0_unfold.shape[1]),
                    stride=(feat_c0_unfold.shape[1], 0, 1))  # [n, k, cf]
                selected_c1 = feat_c1_unfold.as_strided(
                    size=(feat_c1_unfold.shape[0], self.fine_sample_num, feat_c1_unfold.shape[1]),
                    stride=(feat_c1_unfold.shape[1], 0, 1))  # [n, k, cf]

                selected0 = self.merge_feat(torch.cat([selected0, selected_c0], dim=2))  # [n, k, cf]
                selected1 = self.merge_feat(torch.cat([selected1, selected_c1], dim=2))  # [n, k, cf]

            data.update({
                # 'pos0': pos0,
                'pos1': pos1,
            })

            return selected0, selected1

        else:

            W = self.W
            stride = data['hw0_f'][0] // data['hw0_c'][0]

            data.update({'W': W})
            if data['b_ids'].shape[0] == 0:
                feat0 = torch.empty(0, self.W ** 2, self.d_model_f, device=feat_f0.device)
                feat1 = torch.empty(0, self.W ** 2, self.d_model_f, device=feat_f0.device)
                return feat0, feat1

            # 1. unfold(crop) all local windows
            feat_f0_unfold = F.unfold(feat_f0, kernel_size=(W, W), stride=stride, padding=W // 2)
            feat_f0_unfold = rearrange(feat_f0_unfold, 'n (c ww) l -> n l ww c', ww=W ** 2)
            feat_f1_unfold = F.unfold(feat_f1, kernel_size=(W, W), stride=stride, padding=W // 2)
            feat_f1_unfold = rearrange(feat_f1_unfold, 'n (c ww) l -> n l ww c', ww=W ** 2)

            # 2. select only the predicted matches
            feat_f0_unfold = feat_f0_unfold[data['b_ids'], data['i_ids']]  # [n, ww, cf]
            feat_f1_unfold = feat_f1_unfold[data['b_ids'], data['j_ids']]

            # option: use coarse-level loftr feature as context: concat and linear
            if self.cat_c_feat:
                feat_c_win = self.down_proj(torch.cat([feat_c0[data['b_ids'], data['i_ids']],
                                                       feat_c1[data['b_ids'], data['j_ids']]], 0))  # [2n, c]
                feat_cf_win = self.merge_feat(torch.cat([
                    torch.cat([feat_f0_unfold, feat_f1_unfold], 0),  # [2n, ww, cf]
                    repeat(feat_c_win, 'n c -> n ww c', ww=W ** 2),  # [2n, ww, cf]
                ], -1))
                feat_f0_unfold, feat_f1_unfold = torch.chunk(feat_cf_win, 2, dim=0)

            return feat_f0_unfold, feat_f1_unfold
