
import torch, torch.nn as nn, torch.nn.functional as F
from fashion.models import HMModel


SEQ_TOPK_K = 6


class HMModel1(HMModel):
    def __init__(self, data_config, args=None):
        super().__init__(data_config, args=args)
        self.seq_topk_k = self.args.get('seq_topk_k', SEQ_TOPK_K)
        assert self.seq_topk_k <= data_config['seq_len']

        in_channels = 2*self.seq_topk_k + 1
        self.top = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=1), nn.LeakyReLU(),
            nn.Conv1d(32, 8, kernel_size=1), nn.LeakyReLU(),
            nn.Conv1d(8, 1, kernel_size=1))

    @staticmethod
    def add_argparse_args(parser):
        HMModel.add_argparse_args(parser)
        _add = parser.add_argument
        _add('--seq_topk_k', type=int, default=SEQ_TOPK_K)

    def forward(self, inputs):
        article_hist, week_hist = inputs[0], inputs[1]

        x = self.article_emb(article_hist)    # (N, seq_len, emb_size)
        x = F.normalize(x, dim=2)

        # (N, seq_len, num_class)
        x = x@F.normalize(self.article_emb.weight).T

        # (N, seq_topk_k, num_class)
        x, indices = torch.topk(x, k=self.seq_topk_k, dim=1)
        x = x.clamp(1e-3, 0.999)
        x = -torch.log(1/x - 1)

        topk_week = (
            week_hist
            .unsqueeze(1)
            .repeat(1, self.seq_topk_k, 1)
            .gather(dim=2, index=indices)
        )   # (N, seq_topk_k, num_class)

        x = torch.cat([
            x,
            topk_week,
            self.article_likelihood[None, None, :].repeat(x.shape[0], 1, 1)
            ], 
            axis=1)    # (N, 2 * seq_topk_k + 1, num_class)

        x = self.top(x).squeeze(1)    # (N, num_class)
        return x
