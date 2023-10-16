import os
import sys
import torch
import pandas as pd
import torch.nn as nn
import multiprocessing
import pytorch_lightning as pl

from torch.utils.data import TensorDataset, DataLoader
from transformers import BartTokenizer


class AnswerExtractionDataModule(pl.LightningDataModule):

    def __init__(self, 
                 dataset_name, 
                 pre_trained_model_name='indobenchmark/indobart-v2', 
                 max_length=512, 
                 batch_size=32, 
                 recreate=False
                 ) -> None:

        super(AnswerExtractionDataModule, self).__init__()

        self.seed = 42
        self.pre_trained_model_name = pre_trained_model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.recreate = recreate

        self.train_dataset_path = f"Datasets/Processed/{dataset_name}/train.csv"
        self.dev_dataset_path = f"Datasets/Processed/{dataset_name}/dev.csv"

        self.train_tokenized_pickle = f"Datasets/Tensor/{dataset_name}/train.pkl"
        self.dev_tokenized_pickle = f"Datasets/Tensor/{dataset_name}/dev.pkl"

        self.load_data()
    

    def load_data(self):
        if os.path.exists(self.train_tokenized_pickle) and os.path.exists(self.dev_tokenized_pickle) and not self.recreate:
            print('\n[ Loading Dataset ]')
            



    def setup(self, stage=None):
        train_data, valid_data, test_data = self.load_data()
        if stage == "fit":
            self.train_data = train_data
            self.valid_data = valid_data
        elif stage == "test":
            self.test_data = test_data


    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=multiprocessing.cpu_count()
        )


    def val_dataloader(self):
        return DataLoader(
            dataset=self.valid_data,
            batch_size=self.batch_size,
            num_workers=multiprocessing.cpu_count()
        )


    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_data,
            batch_size=self.batch_size,
            num_workers=multiprocessing.cpu_count()
        )
