
import os
import sys
import wandb
import argparse
import albumentations as albu
import pytorch_lightning as pl

from fashion.utils import import_class


def setup_parser():
    parser = argparse.ArgumentParser(add_help=False)
    pl.Trainer.add_argparse_args(parser)

    _add = parser.add_argument
    _add('--data_class', type=str, default='HM')
    _add('--model_class', type=str, default='HMModel')
    _add('--lit_model_class', type=str, default='BaseLitModel')
    _add('--dir_out', type=str, default='/kaggle/working/training/logs')
    _add('--load_from_checkpoint', type=str, default=None)
    _add('--wandb', action='store_true', default=False)

    args, _ = parser.parse_known_args()

    data_class = import_class(f'fashion.data.{args.data_class}')
    data_group = parser.add_argument_group('Data Args')
    data_class.add_argparse_args(data_group)

    model_class = import_class(f'fashion.models.{args.model_class}')
    model_group = parser.add_argument_group('Model Args')
    model_class.add_argparse_args(model_group)

    lit_model_class = import_class(
        f'fashion.lit_models.{args.lit_model_class}')
    lit_model_group = parser.add_argument_group('LitModel Args')
    lit_model_class.add_argparse_args(lit_model_group)

    parser.add_argument('--help', '-h', action='help')
    return parser


def main():
    parser = setup_parser()

    args = parser.parse_args()
    print(args)

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

    logger = pl.loggers.TensorBoardLogger(args.dir_out)
    if args.wandb:
        logger = pl.loggers.WandbLogger(project='fashion')
        logger.watch(model)
        logger.log_hyperparams(vars(args))

    model_checkpoint_callback = pl.callbacks.ModelCheckpoint(
        filename=('epoch{epoch:03d}'
                  '-valid_loss{valid_loss:.3f}' 
                  '-valid_score{valid_score:.3f}'),
        auto_insert_metric_name=False,
        monitor='valid_loss',
        mode='min',
        save_last=True)

    lr_monitor_callback = pl.callbacks.LearningRateMonitor()

    model_summary_callback = pl.callbacks.ModelSummary(max_depth=10)

    callbacks = [model_checkpoint_callback,
                 lr_monitor_callback,
                 model_summary_callback]

    trainer = pl.Trainer.from_argparse_args(args,
                                            weights_save_path=args.dir_out,
                                            weights_summary='full',
                                            logger=logger,
                                            callbacks=callbacks)

    trainer.tune(model=lit_model, datamodule=data)

    trainer.fit(model=lit_model, datamodule=data)

    best_model_path = model_checkpoint_callback.best_model_path
    if best_model_path:
        print('Best model saved at:', best_model_path)
        if args.wandb:
            wandb.save(best_model_path)
            print('Best model uploaded to W&B.')


if __name__ == '__main__':
    main()
