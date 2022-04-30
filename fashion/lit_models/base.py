
import torch
import pytorch_lightning as pl
from torch.optim.lr_scheduler import MultiStepLR
from fashion.losses.dice import dice_loss, dice_loss_1
from fashion.metrics.map import MAP


LR = 1e-4
LR_SCHEDULER = None
MULTISTEP_GAMMA = 0.5
MULTISTEP_MILESTONES = [5, 10, 15, 20, 25, 30, 35, 40]

class BaseLitModel(pl.LightningModule):

    def __init__(self, model, args=None):
        super().__init__()
        self.args = vars(args) if args is not None else {}
        self.model = model

        self.dice_loss = dice_loss_1
        self.bce_loss = torch.nn.BCEWithLogitsLoss()
        self.mean_average_precision = MAP(k=12)

        self.lr = self.args.get('lr', LR)
        self.lr_scheduler = self.args.get('lr_scheduler', LR_SCHEDULER)
        self.multistep_gamma = self.args.get(
            'multistep_gamma', MULTISTEP_GAMMA)
        self.multistep_milestones = self.args.get(
            'multistep_milestones', MULTISTEP_MILESTONES)

    def add_argparse_args(parser):
        _add = parser.add_argument
        _add('--lr', type=float, default=LR)
        _add('--lr_scheduler', type=str, default=LR_SCHEDULER)
        _add('--multistep_gamma', type=float, default=MULTISTEP_GAMMA)
        _add('--multistep_milestones', type=int, nargs='+', 
             default=MULTISTEP_MILESTONES)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        if self.lr_scheduler is None:
            return {'optimizer': optimizer}

        elif self.lr_scheduler == 'MultiStepLR':
            lr_scheduler = MultiStepLR(optimizer, 
                                       gamma=self.multistep_gamma, 
                                       milestones=self.multistep_milestones)
            return {'optimizer': optimizer, 
                    'lr_scheduler': {'scheduler': lr_scheduler}
            }
        else:
            raise NotImplementedError
            
            

    def _shared_step(self, batch):
        article_hist, week_hist, target = batch
        inputs = (article_hist, week_hist)
        logits = self.model(inputs)
        return logits, target

    def training_step(self, batch, batch_idx):
        logits, target = self._shared_step(batch)
        dice_loss = self.dice_loss(logits, target)
        bce_loss = self.bce_loss(logits, target)
        loss = dice_loss + bce_loss
        self.log('train_dice_loss', dice_loss)
        self.log('train_bce_loss', bce_loss)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        logits, target = self._shared_step(batch)
        dice_loss = self.dice_loss(logits, target)
        bce_loss = self.bce_loss(logits, target)
        loss = dice_loss + bce_loss
        score = self.mean_average_precision(logits, target)
        
        self.log('valid_dice_loss', dice_loss)
        self.log('valid_bce_loss', bce_loss)
        self.log('valid_loss', loss)
        self.log('valid_score', score)

    @torch.no_grad()
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        logits, _ = self._shared_step(batch)
        topked = torch.topk(logits, dim=1, k=12)
        return topked.indices

