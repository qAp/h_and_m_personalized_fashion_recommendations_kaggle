
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
