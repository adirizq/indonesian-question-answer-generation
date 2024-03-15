import sys
import multiprocessing
import pytorch_lightning as pl

from torch.utils.data import DataLoader
from Utils.dataset import PipelineDataset, MultiTaskDataset



class PipelineQAGDataModule(pl.LightningDataModule):
    def __init__(self, model_type, task_type, dataset_csv_paths, dataset_tensor_paths, tokenizer, batch_size, recreate):

        super(PipelineQAGDataModule, self).__init__()

        self.train_csv_path, self.validation_csv_path, self.test_csv_path = dataset_csv_paths
        self.train_tensor_path, self.validation_tensor_path, self.test_tensor_path = dataset_tensor_paths

        self.model_type = model_type
        self.task_type = task_type
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.recreate = recreate


    def setup(self, stage=None):
        if stage == "fit":
            self.train_data = PipelineDataset(self.task_type, self.train_csv_path, self.train_tensor_path, self.tokenizer, self.recreate)
            self.valid_data = PipelineDataset(self.task_type, self.validation_csv_path, self.validation_tensor_path, self.tokenizer, self.recreate)
        elif stage == "test":
            self.test_data = PipelineDataset(self.task_type, self.test_csv_path, self.test_tensor_path, self.tokenizer, self.recreate)


    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True, num_workers=multiprocessing.cpu_count())


    def val_dataloader(self):
        return DataLoader(self.valid_data, batch_size=self.batch_size, shuffle=False, num_workers=multiprocessing.cpu_count())


    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False, num_workers=multiprocessing.cpu_count())
    

class MultiTaskQAGDataModule(pl.LightningDataModule):
    def __init__(self, model_type, dataset_csv_paths, dataset_tensor_paths, tokenizer, batch_size, recreate):

        super(MultiTaskQAGDataModule, self).__init__()

        self.train_csv_path, self.validation_csv_path, self.test_csv_path = dataset_csv_paths
        self.train_tensor_path, self.validation_tensor_path, self.test_tensor_path = dataset_tensor_paths

        self.model_type = model_type
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.recreate = recreate


    def setup(self, stage=None):
        if stage == "fit":
            self.train_data = MultiTaskDataset(self.train_csv_path, self.train_tensor_path, self.tokenizer, self.recreate)
            self.valid_data = MultiTaskDataset(self.validation_csv_path, self.validation_tensor_path, self.tokenizer, self.recreate)
        elif stage == "ae_test":
            self.test_data = MultiTaskDataset(self.test_csv_path, self.test_tensor_path, self.tokenizer, self.recreate, test_type='ae')
        elif stage == "qg_test":
            self.test_data = MultiTaskDataset(self.test_csv_path, self.test_tensor_path, self.tokenizer, self.recreate, test_type='qg')


    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True, num_workers=multiprocessing.cpu_count())


    def val_dataloader(self):
        return DataLoader(self.valid_data, batch_size=self.batch_size, shuffle=False, num_workers=multiprocessing.cpu_count())


    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False, num_workers=multiprocessing.cpu_count())