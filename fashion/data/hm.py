
import os
import argparse
import pathlib
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from fashion.config import *


class HMDataset(Dataset):
    def __init__(self, df, seq_len=16, num_article_ids=100, week_hist_max=5,
                 is_test=False):
        self.df = df.reset_index(drop=True)
        self.seq_len = seq_len
        self.num_article_ids = num_article_ids
        self.week_hist_max = week_hist_max
        self.is_test = is_test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]

        if self.is_test:
            target = torch.zeros(2).float()
        else:
            target = torch.zeros(self.num_article_ids).float()
            for t in row.target:
                target[t] = 1.

        article_hist = torch.zeros(self.seq_len).long()
        week_hist = torch.ones(self.seq_len).float()

        if isinstance(row.article_id, list):
            if len(row.article_id) >= self.seq_len:
                article_hist = torch.LongTensor(row.article_id[-self.seq_len:])
                week_hist = (
                    (torch.LongTensor(
                        row.week_history[-self.seq_len:]) - row.week)
                    / self.week_hist_max / 2
                )
            else:
                article_hist[-len(row.article_id):] = torch.LongTensor(row.article_id)  
                week_hist[-len(row.article_id):] = (
                    (torch.LongTensor(row.week_history) - row.week)
                    / self.week_hist_max / 2
                )

        return article_hist, week_hist, target


META_DATA_DIR = '/kaggle/working/hm_meta_data'
WEEK_HIST_MAX = 5
VAL_WEEKS = [0]
TRAIN_WEEKS = [1, 2, 3, 4]

class HM(pl.LightningDataModule):
    def __init__(self, args=None):
        self.args = vars(args) if args is not None else {}
        
        self.meta_data_dir = self.args.get('meta_data_dir', META_DATA_DIR)
        self.week_hist_max = self.args.get('week_hist_max', WEEK_HIST_MAX)
        self.val_weeks = self.args.get('val_weeks', VAL_WEEKS)
        self.train_weeks = self.args.get('train_weeks', TRAIN_WEEKS)

    @staticmethod
    def add_argparse_args(parser):
        _add = parser.add_argument
        _add('--meta_data_dir', type=str, default=META_DATA_DIR)
        _add('--week_hist_max', type=int, default=WEEK_HIST_MAX)
        _add('--val_weeks', type=int, nargs='+', default=VAL_WEEKS)
        _add('--train_weeks', type=int, nargs='+', default=TRAIN_WEEKS)

    def prepare_data(self):
        meta_data_path = f'{self.meta_data_dir}/train.parquet'
        label_encoder_path = f'{self.meta_data_dir}/label_encoder'

        meta_data_exists = os.path.exists(meta_data_path)
        label_encoder_exists = os.path.exists(label_encoder_path)
        if meta_data_exists and label_encoder_exists:
            print('Found existing meta data.  Data prepared.')
            return

        print('Loading transactions_train.csv')
        df = pd.read_csv(f'{COMP_DIR}/transactions_train.csv', 
                         dtype={'article_id': str})
        df['t_dat'] = pd.to_datetime(df['t_dat'])

        print('Keeping active articles only')
        active_articles = df.groupby('article_id')['t_dat'].max().reset_index()
        active_articles = active_articles[active_articles['t_dat'] 
                                        >= '2019-09-01'].reset_index()
        df = df[df['article_id'].isin(
            active_articles['article_id'])].reset_index(drop=True)

        df['week'] = (df['t_dat'].max() - df['t_dat']).dt.days // 7

        print('Encoding articles IDs')
        # article_ids = np.concatenate([['placeholder'], df['article_id'].unique()])
        article_ids = np.concatenate(
            [["placeholder"], np.unique(df["article_id"].values)])
        le_article = LabelEncoder()
        le_article.fit(article_ids)
        df['article_id'] = le_article.transform(df['article_id'])

        print('Writing meta data and label encoder to disk')
        pathlib.Path(self.meta_data_dir).mkdir(exist_ok=True, parents=True)
        df.to_parquet(meta_data_path)
        joblib.dump(le_article, label_encoder_path)

    def create_dataset(self, df, week):
        is_hist_week = (
            (week + self.week_hist_max >= df['week']) & (df['week'] > week)
        )
        is_target_week = (df['week'] == week)

        hist_df = (
            df[is_hist_week]
            .groupby('customer_id')
            .transform({'article_id': list, 'week': list})
            .reset_index()
            .rename(columns={'week': 'week_history'})
        )

        target_df = (
            df[is_target_week]
            .groupby('customer_id')
            .transform({'article_id': list, 'week': list})
            .reset_index()
            .rename(columns={'article_id': 'target'})
        )
        return target_df.merge(hist_df, on='customer_id', how='left')

    def setup(self):
        meta_data_path = f'{self.meta_data_dir}/train.parquet'
        label_encoder_path = f'{self.meta_data_dir}/label_encoder'

        df = pd.read_parquet(meta_data_path)
        le_article = joblib.load(label_encoder_path)

        if self.val_weeks:
            val_df = pd.concat([
                self.create_dataset(df, w) for w in self.val_weeks
                ]).reset_index(drop=True)

            self.valid_ds = HMDataset(val_df,
                                    self.seq_len,
                                    num_article_ids=len(le_article.classes_),
                                    week_hist_max=self.week_hist_max)

        train_df = pd.concat([
            self.create_dataset(df, w) for w in self.train_weeks
            ]).reset_index(drop=True)

        self.train_ds = HMDataset(train_df, 
                                  self.seq_len, 
                                  num_article_ids=len(le_article.classes_),
                                  week_hist_max=self.week_hist_max)

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.num_workes,
            pin_memory=self.on_gpu
        )

    def val_dataloader(self):
        if self.val_weeks:
            return DataLoader(
                self.valid_ds,
                batch_size=self.batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=self.num_workers,
                pin_memory=self.on_gpu
            )


def prepare_data():
    parser = argparse.ArgumentParser()

    HM.add_argparse_args(parser)
    args = parser.parse_args([])

    data = HM(args)
    data.prepare_data()
