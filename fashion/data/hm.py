
import os
import argparse
import pathlib
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pytorch_lightning as pl
from fashion.config import *


META_DATA_DIR = '/kaggle/working/hm_meta_data'


class HM(pl.LightningDataModule):
    def __init__(self, args=None):
        self.args = vars(args) if args is not None else {}
        
        self.meta_data_dir = self.args.get('meta_data_dir', META_DATA_DIR)

    @staticmethod
    def add_argparse_args(parser):
        _add = parser.add_argument
        _add('--meta_data_dir', type=str, default=META_DATA_DIR)

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

    def setup(self):
        pass

    def train_dataloader(self):
        pass

    def val_dataloader(self):
        pass


def prepare_data():
    parser = argparse.ArgumentParser()

    HM.add_argparse_args(parser)
    args = parser.parse_args([])

    data = HM(args)
    data.prepare_data()
