
import torch, torch.nn as nn, torch.nn.functional as F


class HMModel(nn.Module):
    def __init__(self, article_shape):
        super(HMModel, self).__init__()

        self.article_emb = nn.Embedding(
            article_shape[0], embedding_dim=article_shape[1])

        self.article_likelihood = nn.Parameter(
            torch.zeros(article_shape[0]), requires_grad=True)
        self.top = nn.Sequential(nn.Conv1d(3, 32, kernel_size=1), nn.LeakyReLU(),
                                 nn.Conv1d(
                                     32, 8, kernel_size=1), nn.LeakyReLU(),
                                 nn.Conv1d(8, 1, kernel_size=1))

    def forward(self, inputs):
        article_hist, week_hist = inputs[0], inputs[1]
        x = self.article_emb(article_hist)
        x = F.normalize(x, dim=2)

        x = x@F.normalize(self.article_emb.weight).T

        x, indices = x.max(axis=1)
        x = x.clamp(1e-3, 0.999)
        x = -torch.log(1/x - 1)

        max_week = week_hist.unsqueeze(2).repeat(
            1, 1, x.shape[-1]).gather(1, indices.unsqueeze(1).repeat(1, week_hist.shape[1], 1))
        max_week = max_week.mean(axis=1).unsqueeze(1)

        x = torch.cat([x.unsqueeze(1), max_week,
                       self.article_likelihood[None, None, :].repeat(x.shape[0], 1, 1)], axis=1)

        x = self.top(x).squeeze(1)
        return x
