
import torch, torch.nn as nn, torch.nn.functional as F



class HMModel2(nn.Module):
    def __init__(self, data_config, args=None):
        super().__init__()
        self.args = vars(args) if args is not None else {}

        self.linear = nn.Linear(in_features=data_config['seq_len'],
                                out_features=data_config['num_article_ids'])

    @staticmethod
    def add_argparse_args(parser):
        pass

    def forward(self, inputs):
        article_hist, week_hist = inputs[0], inputs[1]

        return self.linear(week_hist)

