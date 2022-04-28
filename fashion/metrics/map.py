
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np


def calc_map(topk_preds, target_array, k=12):
    metric = []
    tp, fp = 0, 0

    for pred in topk_preds:
        if target_array[pred]:
            tp += 1
            metric.append(tp/(tp + fp))
        else:
            fp += 1

    return np.sum(metric) / min(k, target_array.sum())


class MAP(nn.Module):
    def __init__(self, k=12):
        super().__init__()
        self.k = k

    @torch.no_grad()
    def forward(self, logits, targets):
        _, indices = torch.topk(logits, dim=1, k=self.k)

        indices = indices.detach().cpu().numpy()
        targets = targets.detach().cpu().numpy()

        maps = []
        for i in range(len(indices)):
            maps.append(calc_map(indices[i], targets[i]))

        return np.sum(maps)
