
import torch
import pytorch_lightning as pl
from fashion.losses.dice import dice_loss
from fashion.metrics.map import MAP


LR = 1e-4

class BaseLitModel(pl.LightningModule):

    def __init__(self, model, args=None):
        super().__init__()
        self.args = vars(args) if args is not None else {}
        self.model = model

        self.dice_loss = dice_loss
        self.bce_loss = torch.nn.BCEWithLogitsLoss()
        self.mean_average_precision = MAP(k=12)

        self.lr = self.args.get('lr', LR)

    def add_argparse_args(parser):
        _add = parser.add_argument
        _add('--lr', type=float, default=LR)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return {'optimizer': optimizer}

    def _shared_step(self, batch):
        article_hist, week_hist, target = batch
        inputs = (article_hist, week_hist)
        logits = self.model(inputs)
        return logits, target

    def training_step(self, batch, batch_idx):
        logits, target = self._shared_step(batch)
        loss = self.dice_loss(logits, target) + self.bce_loss(logits, target)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        logits, target = self._shared_step(batch)
        loss = self.dice_loss(logits, target) + self.bce_loss(logits, target)

        topked = torch.topk(logits, dim=1, k=12)
        score = self.mean_average_precision(topked.indices, target)
        self.log('train_loss', loss)
        self.log('valid_loss', score)

