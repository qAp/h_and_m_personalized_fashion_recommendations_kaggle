
import os
import argparse
import pathlib
from tqdm.auto import tqdm
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from fashion.config import *



def create_dataset(df, week=0, week_hist_max=5):
    is_hist_week = (
        (week + week_hist_max >= df['week']) & (df['week'] > week)
    )
    is_target_week = (df['week'] == week)

    hist_df = (
        df[is_hist_week]
        .groupby('customer_id')
        .agg({'article_id': list, 'week': list})
        .reset_index()
        .rename(columns={'week': 'week_history'})
    )

    target_df = (
        df[is_target_week]
        .groupby('customer_id')
        .agg({'article_id': list})
        .reset_index()
        .rename(columns={'article_id': 'target'})
    )
    target_df['week'] = week
    return target_df.merge(hist_df, on='customer_id', how='left')


def create_test_dataset(df, week_hist_max=5):
    test_df = pd.read_csv(f'{COMP_DIR}/sample_submission.csv')
    test_df.drop('prediction', axis=1, inplace=True)

    week = -1
    test_df['week'] = week

    is_hist_week = (
        (week + week_hist_max >= df['week']) & (df['week'] > week)
    )
    hist_df = (
        df[is_hist_week]
        .groupby('customer_id')
        .agg({'article_id': list, 'week': list})
        .reset_index()
        .rename(columns={'week': 'week_history'})
    )
    return test_df.merge(hist_df, on='customer_id', how='left')


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
SEQ_LEN = 16
BATCH_SIZE = 256
NUM_WORKERS = os.cpu_count()


class HM(pl.LightningDataModule):
    def __init__(self, args=None):
        self.args = vars(args) if args is not None else {}
        
        self.meta_data_dir = self.args.get('meta_data_dir', META_DATA_DIR)
        self.week_hist_max = self.args.get('week_hist_max', WEEK_HIST_MAX)
        self.val_weeks = self.args.get('val_weeks', VAL_WEEKS)
        self.train_weeks = self.args.get('train_weeks', TRAIN_WEEKS)
        self.seq_len = self.args.get('seq_len', SEQ_LEN)

        self.batch_size = self.args.get('batch_size', BATCH_SIZE)
        self.num_workers = self.args.get('num_workers', NUM_WORKERS)
        self.on_gpu = isinstance(self.args.get('gpus', None), (int, str))

    @staticmethod
    def add_argparse_args(parser):
        _add = parser.add_argument
        _add('--meta_data_dir', type=str, default=META_DATA_DIR)
        _add('--week_hist_max', type=int, default=WEEK_HIST_MAX)
        _add('--val_weeks', type=int, nargs='+', default=VAL_WEEKS)
        _add('--train_weeks', type=int, nargs='+', default=TRAIN_WEEKS)
        _add('--seq_len', type=int, default=SEQ_LEN)
        _add('--batch_size', type=int, default=BATCH_SIZE)
        _add('--num_workers', type=int, default=NUM_WORKERS)

    def prepare_data(self):
        meta_data_path = f'{self.meta_data_dir}/train.parquet'
        label_encoder_path = f'{self.meta_data_dir}/label_encoder'

        hm_df_paths = [
            f'hm_df_week{w}_hist_max{self.week_hist_max}.parquet'
            for w in self.val_weeks + self.train_weeks]
        hm_df_paths = [f'{self.meta_data_dir}/{p}' for p in hm_df_paths]

        all_exists = all(
            os.path.exists(p)
            for p in [meta_data_path, label_encoder_path] + hm_df_paths)

        if all_exists:
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
        article_ids = np.concatenate([
            [10*'0'], df["article_id"].unique()
            ])
        le_article = LabelEncoder()
        le_article.fit(article_ids)
        df['article_id'] = le_article.transform(df['article_id'])

        print('Writing meta data and label encoder to disk')
        pathlib.Path(self.meta_data_dir).mkdir(exist_ok=True, parents=True)
        df.to_parquet(meta_data_path)
        joblib.dump(le_article, label_encoder_path)

        print('HM dataframes...')
        for w, p in tqdm(zip(self.val_weeks + self.train_weeks, hm_df_paths),
                         total=len(hm_df_paths)):
            hm_df = create_dataset(df, w, self.week_hist_max)
            hm_df.to_parquet(p)

    def setup(self):
        meta_data_path = f'{self.meta_data_dir}/train.parquet'
        label_encoder_path = f'{self.meta_data_dir}/label_encoder'

        print(f'Loading meta data at {meta_data_path}...')
        df = pd.read_parquet(meta_data_path)
        le_article = joblib.load(label_encoder_path)
        self.le_article = le_article

        print('Creating validation set...')
        if self.val_weeks:
            val_df = pd.concat([
                create_dataset(df, w, self.week_hist_max) 
                for w in self.val_weeks
                ]).reset_index(drop=True)

            self.valid_ds = HMDataset(val_df,
                                    self.seq_len,
                                    num_article_ids=len(le_article.classes_),
                                    week_hist_max=self.week_hist_max)

        print('Creating training set...')
        train_df = pd.concat([
            create_dataset(df, w, self.week_hist_max) 
            for w in self.train_weeks
            ]).reset_index(drop=True)

        self.train_ds = HMDataset(train_df, 
                                  self.seq_len, 
                                  num_article_ids=len(le_article.classes_),
                                  week_hist_max=self.week_hist_max)

        print('Creating test set...')
        test_df = create_test_dataset(df, self.week_hist_max)
        self.test_ds = HMDataset(test_df,
                                 self.seq_len,
                                 num_article_ids=len(le_article.classes_),
                                 week_hist_max=self.week_hist_max,
                                 is_test=True)

    def config(self):
        return {'num_article_ids': len(self.le_article.classes_),
                'seq_len': self.seq_len}

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.num_workers,
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

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers,
            pin_memory=self.on_gpu
        )


def prepare_data():
    parser = argparse.ArgumentParser()

    HM.add_argparse_args(parser)
    args = parser.parse_args()

    data = HM(args)
    data.prepare_data()


def describe_history_length(df, week=0, week_hist_max=5):
    out_df = create_dataset(df, week, week_hist_max)

    history_length = out_df['week_history'].map(
        lambda x: len(x) if isinstance(x, list) else 0)

    print(f'week = {week}')
    print(f'week_hist_max = {week_hist_max}')
    print('Distribution of history lengths')

    print(history_length.describe())

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.hist(history_length.values, bins=50)
    ax.set_xlabel('History length')
    ax.set_ylabel('Count')


def describe_article_id(df, weeks=[0], week_hist_max=5):

    out_df = pd.concat(
        [create_dataset(df, w, week_hist_max) for w in weeks],
        axis=0
    ).reset_index(drop=True)

    history_list = []
    target_list = []
    for _, r in tqdm(out_df.iterrows(), total=len(out_df)):
        history_ids = r['article_id'] if isinstance(
            r['article_id'], list) else []
        target_ids = r['target'] if isinstance(r['target'], list) else []

        for hid in history_ids:
            for tid in target_ids:
                history_list.append(hid)
                target_list.append(tid)

    pairs = pd.DataFrame({'history': history_list, 'target': target_list})

    print('len(weeks)=', len(weeks), 'min(weeks)=',
          min(weeks), 'max(weeks)=', max(weeks))
    print('week_hist_max', week_hist_max)
    print('Number of unique ids in histories', pairs['history'].nunique())
    print('Number of unique ids in targets', pairs['target'].nunique())

    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(12, 8))
    axs = axs.flatten()
    ax = axs[0]
    ax.hist(pairs['history'].values, bins=20, color='blue', label='history')
    ax.hist(pairs['target'].values, bins=20,
            color='orange', label='target', alpha=0.7)
    ax.set_xlabel('article_id')
    ax.set_ylabel('Count')
    ax.legend()

    hist_cover = pairs.groupby('target')['history'].nunique().reset_index()
    ax = axs[1]
    ax.hist(hist_cover['history'].values, bins=100)
    ax.set_xlabel('Number of unique history ids')
    ax.set_ylabel('Number of target ids')


def describe_article_id_1(df, weeks=[0], week_hist_max=5):

    out_df = pd.concat(
        [create_dataset(df, w, week_hist_max) for w in weeks],
        axis=0
    ).reset_index(drop=True)

    history_cnt = {}
    target_cnt = {}
    history_cover = {}
    for _, r in tqdm(out_df.iterrows(), total=len(out_df)):
        history_ids = r['article_id'] if isinstance(
            r['article_id'], list) else []
        target_ids = r['target'] if isinstance(r['target'], list) else []

        for tid in target_ids:

            if tid not in history_cover:
                history_cover[tid] = set()
            else:
                history_cover[tid].update(set(history_ids))

            for hid in history_ids:

                if hid not in history_cnt:
                    history_cnt[hid] = 0
                else:
                    history_cnt[hid] += 1

                if tid not in target_cnt:
                    target_cnt[tid] = 0
                else:
                    target_cnt[tid] += 1

    print('len(weeks)=', len(weeks), 'min(weeks)=',
          min(weeks), 'max(weeks)=', max(weeks))
    print('week_hist_max', week_hist_max)

    print('Number of unique ids in histories', len(history_cnt.keys()))
    print('Number of unique ids in targets', len(target_cnt.keys()))
