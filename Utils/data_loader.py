import os
import sys
import torch
import pandas as pd
import torch.nn as nn
import multiprocessing
import pytorch_lightning as pl

from tqdm import tqdm
from transformers import BertTokenizer
from indobenchmark import IndoNLGTokenizer
from torch.utils.data import TensorDataset, DataLoader


class AnswerExtractionDataModule(pl.LightningDataModule):

    def __init__(self, 
                 dataset_name, 
                 input_type,
                 output_type,
                 pre_trained_model_name='indobenchmark/indobart-v2', 
                 max_length=512, 
                 batch_size=32, 
                 recreate=False,
                 ) -> None:

        super(AnswerExtractionDataModule, self).__init__()

        self.seed = 42
        self.dataset_name = dataset_name
        self.input_type = input_type
        self.output_type = output_type
        self.pre_trained_model_name = pre_trained_model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.recreate = recreate
        
        self.train_dataset_path = f"Datasets/Processed/{dataset_name}/prepared_train.csv"
        self.valid_dataset_path = f"Datasets/Processed/{dataset_name}/prepared_dev.csv"
        self.test_dataset_path = f"Datasets/Processed/{dataset_name}/prepared_test.csv"

        self.train_tensor_dataset_path = f"Datasets/Tensor/{dataset_name}_{input_type}_to_{output_type}/train.pt"
        self.valid_tensor_dataset_path = f"Datasets/Tensor/{dataset_name}_{input_type}_to_{output_type}/dev.pt"
        self.test_tensor_dataset_path = f"Datasets/Tensor/{dataset_name}_{input_type}_to_{output_type}/test.pt"

        self.tokenizer = IndoNLGTokenizer.from_pretrained(self.pre_trained_model_name, additional_special_tokens=['<hl>', '<mask>'])
        
        # Must add model.resize_token_embeddings(len(tokenizer)) after adding new special tokens
    

    def get_tokenizer(self):
        return self.tokenizer


    def load_data(self):
        if os.path.exists(self.train_tensor_dataset_path) and os.path.exists(self.valid_tensor_dataset_path) and os.path.exists(self.test_tensor_dataset_path) and not self.recreate:
            
            print('\n[ Loading Dataset ]')
            train_data = torch.load(self.train_tensor_dataset_path)
            valid_data = torch.load(self.valid_tensor_dataset_path)
            test_data = torch.load(self.test_tensor_dataset_path)
            print('[ Load Completed ]')

            return train_data, valid_data, test_data
        
        else:

            print('\n[ Processing Dataset ]')
            
            data_csv = {
                'train': pd.read_csv(self.train_dataset_path),
                'dev': pd.read_csv(self.valid_dataset_path),
                'test': pd.read_csv(self.test_dataset_path),
            }

            tokenized_data = {
                'train': [],
                'dev': [],
                'test': [],
            }

            for key, data in data_csv.items():

                input_ids, attention_mask, target_ids = [], [], []

                for (context, context_key_sentence, context_answer, question, answer) in tqdm(data.values.tolist(), desc=f'Tokenizing {key} data'):

                    input_text = locals()[self.input_type]
                    output_text = locals()[self.output_type]

                    encoded_input = self.tokenizer(f'<s>{input_text}</s>', 
                                                                  add_special_tokens=True, 
                                                                  max_length=self.max_length,
                                                                  padding="max_length",
                                                                  truncation=True)
                    
                    encoded_output =  self.tokenizer(f'<s>{output_text}</s>',
                                                     add_special_tokens=True, 
                                                     max_length=self.max_length,
                                                     padding="max_length",
                                                     truncation=True)

                    input_ids.append(encoded_input['input_ids'])
                    attention_mask.append(encoded_input['attention_mask'])
                    target_ids.append(encoded_output['input_ids'])
                
                tensordataset_save_dir = f"Datasets/Tensor/{self.dataset_name}_{self.input_type}_to_{self.output_type}"

                if not os.path.exists(tensordataset_save_dir):
                    os.makedirs(tensordataset_save_dir)

                tokenized_data[key] = TensorDataset(torch.tensor(input_ids), torch.tensor(attention_mask), torch.tensor(target_ids))
                torch.save(tokenized_data[key], f'{tensordataset_save_dir}/{key}.pt')

            print('[ Process Completed ]')
            
            return tokenized_data['train'], tokenized_data['dev'], tokenized_data['test']


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
