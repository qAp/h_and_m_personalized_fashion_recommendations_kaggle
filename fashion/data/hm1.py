
import os
import pathlib
import gc
import joblib
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import torch
import pytorch_lightning as pl
from fashion.config import *
from fashion.data.hm import (HMDataset, HM,
                             create_dataset, 
                             load_hm_df, shrink_hm_df)


class HM1Dataset(HMDataset):

    def __getitem__(self, index):
        self.seq_len
        self.num_article_ids

        if self.is_test:
            target = torch.zeros(2).float()
        else:
            target = torch.randint(low=0, high=1 + 1, 
                                   size=(self.num_article_ids,)
                                   ).float()

        article_hist = torch.randint(low=0, high=self.num_article_ids,
                                     size=(self.seq_len,)
                                     ).long()
        week_hist = torch.rand(self.seq_len).float()

        return article_hist, week_hist, target


META_DATA_DIR = '/kaggle/working/hm_meta_data'
WEEK_HIST_MAX = 5
VAL_WEEKS = [0]
TRAIN_WEEKS = [1, 2, 3, 4]
SEQ_LEN = 16
BATCH_SIZE = 256
NUM_WORKERS = os.cpu_count()


class HM1(HM):
    def __init__(self, args=None):
        super().__init__(args)

    @staticmethod
    def add_argparse_args(parser):
        HM.add_argparse_args(parser)

    def setup(self):
        meta_data_path = f'{self.meta_data_dir}/train.parquet'
        label_encoder_path = f'{self.meta_data_dir}/label_encoder'

        print(f'Loading meta data at {self.meta_data_dir}...')
        le_article = joblib.load(label_encoder_path)
        self.le_article = le_article

        print('Creating validation set...')
        if self.val_weeks:
            hm_df = pd.concat([
                load_hm_df(
                    f'{self.meta_data_dir}/hm_df_week{w}_hist_max{self.week_hist_max}.parquet')
                for w in self.val_weeks
            ]).reset_index(drop=True)
            hm_df = shrink_hm_df(hm_df)

            self.valid_ds = HM1Dataset(hm_df,
                                      self.seq_len,
                                      num_article_ids=len(le_article.classes_),
                                      week_hist_max=self.week_hist_max)

        print('Creating training set...')
        hm_df = pd.concat([
            load_hm_df(
                f'{self.meta_data_dir}/hm_df_week{w}_hist_max{self.week_hist_max}.parquet')
            for w in self.train_weeks
        ]).reset_index(drop=True)
        hm_df = shrink_hm_df(hm_df)

        self.train_ds = HM1Dataset(hm_df,
                                  self.seq_len,
                                  num_article_ids=len(le_article.classes_),
                                  week_hist_max=self.week_hist_max)

        del hm_df
        gc.collect()

    def config(self):
        return {'num_article_ids': len(self.le_article.classes_),
                'seq_len': self.seq_len}
