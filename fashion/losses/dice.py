

def dice_loss(y_pred, y_true):
    y_pred = y_pred.sigmoid()
    intersect = (y_true*y_pred).sum(axis=1)

    return 1 - (intersect/(intersect + y_true.sum(axis=1) + y_pred.sum(axis=1))).mean()


def dice_loss_1(y_pred, y_true):
    y_pred = y_pred.sigmoid()

    x1 = (y_true * y_pred).sum(dim=1)
    x2 = y_pred.sum(dim=1).clamp(min=1e-12)
    return (1 - x1/x2).mean()
