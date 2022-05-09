import os
import sys
import wandb
import argparse
import pandas as pd
import torch
import albumentations as albu
import pytorch_lightning as pl

from fashion.config import *
from fashion.utils import import_class
from training.run_experiment1 import setup_parser


def main():
    parser = setup_parser()

    args = parser.parse_args()

    data_class = import_class(f'fashion.data.{args.data_class}')
    model_class = import_class(f'fashion.models.{args.model_class}')
    lit_model_class = import_class(
        f'fashion.lit_models.{args.lit_model_class}')

    data = data_class(args)
    data.prepare_data()
    data.setup()

    model = model_class(data_config=data.config(), args=args)

    if args.load_from_checkpoint is not None:
        lit_model = lit_model_class.load_from_checkpoint(
            checkpoint_path=args.load_from_checkpoint,
            model=model,
            args=args)
    else:
        lit_model = lit_model_class(model, args=args)

    trainer = pl.Trainer.from_argparse_args(args)

    indices = trainer.predict(model=lit_model, 
                              dataloaders=data.test_dataloader()
                              )
    indices = torch.cat(indices).numpy()

    preds = []
    for ind in indices:
        preds.append(
            ' '.join(list(data.le_article.inverse_transform(ind)))
        )

    ss_df = pd.read_csv(f'{COMP_DIR}/sample_submission.csv')
    ss_df['prediction'] = preds
    ss_df.to_csv('/kaggle/working/submission.csv', index=False)



if __name__ == '__main__':
    main()
