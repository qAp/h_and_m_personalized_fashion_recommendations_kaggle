

def dice_loss(y_pred, y_true):
    y_pred = y_pred.sigmoid()
    intersect = (y_true*y_pred).sum(axis=1)

    return 1 - (intersect/(intersect + y_true.sum(axis=1) + y_pred.sum(axis=1))).mean()
