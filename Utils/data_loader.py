import os
import sys
import torch
import pandas as pd
import torch.nn as nn
import multiprocessing
import pytorch_lightning as pl

from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader, Dataset
from Utils.dataset import QAGDataset


class QAGDataModule(pl.LightningDataModule):


    def __init__(self,
                 dataset_name,
                 tokenizer,
                 model,
                 max_length=512,
                 batch_size=8,
                 recreate=False,
                 ) -> None:
    
            super(QAGDataModule, self).__init__()
    
            self.seed = 42
            self.dataset_name = dataset_name
            self.tokenizer = tokenizer
            self.model = model
            self.max_length = max_length
            self.batch_size = batch_size
            self.recreate = recreate
    
            self.train_dataset_path = f"Datasets/Processed/{dataset_name}/train.csv"
            self.valid_dataset_path = f"Datasets/Processed/{dataset_name}/validation.csv"
            self.test_dataset_path = f"Datasets/Processed/{dataset_name}/test.csv"
    
            self.train_tensor_dataset_path = f"Datasets/Tensor/QAG_{dataset_name}/train.pt"
            self.valid_tensor_dataset_path = f"Datasets/Tensor/QAG_{dataset_name}/validation.pt"
            self.test_tensor_dataset_path = f"Datasets/Tensor/QAG_{dataset_name}/test.pt"
    

    def load_data(self):
        if os.path.exists(self.train_tensor_dataset_path) and os.path.exists(self.valid_tensor_dataset_path) and os.path.exists(self.test_tensor_dataset_path) and not self.recreate:
            
            print('\n[ Loading Dataset ]\n')
            train_data = torch.load(self.train_tensor_dataset_path)
            valid_data = torch.load(self.valid_tensor_dataset_path)
            test_data = torch.load(self.test_tensor_dataset_path)
            print('\n[ Load Completed ]\n')

            return train_data, valid_data, test_data
        
        else:

            print('\n[ Processing Dataset ]\n')
            
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

                encoded_context_input_ids, encoded_context_attention_mask = [], []
                encoded_answer_input_ids = []
                encoded_question_input_ids = []

                for (context, answer, question) in tqdm(data.values.tolist(), desc=f'Tokenizing {key} data'):

                    if self.model == 'IndoBART':
                        context = f'{locals()["context"]}{self.tokenizer.eos_token}'
                    

                    encoded_context = self.tokenizer(context, 
                                                     add_special_tokens=True, 
                                                     max_length=self.max_length,
                                                     padding="max_length",
                                                     truncation=True)

                    encoded_answer = self.tokenizer(answer, 
                                                    add_special_tokens=True, 
                                                    max_length=self.max_length,
                                                    padding="max_length",
                                                    truncation=True)
                    
                    encoded_question = self.tokenizer(question.split('<sep>'),
                                                      add_special_tokens=True, 
                                                      max_length=self.max_length,
                                                      padding="max_length",
                                                      truncation=True)

                    encoded_context_input_ids.append(encoded_context['input_ids'])
                    encoded_context_attention_mask.append(encoded_context['attention_mask'])
                    encoded_answer_input_ids.append(encoded_answer['input_ids'])
                    encoded_question_input_ids.append(torch.tensor(encoded_question['input_ids']))
                
                tensordataset_save_dir = f"Datasets/Tensor/QAG_{self.dataset_name}"
                os.makedirs(tensordataset_save_dir, exist_ok=True)

                tokenized_data[key] = QAGDataset(
                    torch.tensor(encoded_context_input_ids), 
                    torch.tensor(encoded_context_attention_mask),
                    torch.tensor(encoded_answer_input_ids),
                    encoded_question_input_ids
                )
                
                torch.save(tokenized_data[key], f'{tensordataset_save_dir}/{key}.pt')

            print('\n[ Process Completed ]\n')
            
            return tokenized_data['train'], tokenized_data['dev'], tokenized_data['test']


    def setup(self, stage=None):
        train_data, valid_data, test_data = self.load_data()
        if stage == "fit":
            self.train_data = train_data
            self.valid_data = valid_data
        elif stage == "test":
            self.test_data = test_data


    def collate_fn(self, batch):
        context_input_ids, context_attention_mask, answer_input_ids, question_input_ids = zip(*batch)
        
        context_input_ids = torch.utils.data._utils.collate.default_collate(context_input_ids)
        context_attention_mask = torch.utils.data._utils.collate.default_collate(context_attention_mask)
        answer_input_ids = torch.utils.data._utils.collate.default_collate(answer_input_ids)
        
        question_input_ids = list(question_input_ids)
        
        return context_input_ids, context_attention_mask, answer_input_ids, question_input_ids


    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=multiprocessing.cpu_count(),
            collate_fn=self.collate_fn
        )


    def val_dataloader(self):
        return DataLoader(
            dataset=self.valid_data,
            batch_size=self.batch_size,
            num_workers=multiprocessing.cpu_count(),
            collate_fn=self.collate_fn
        )


    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_data,
            batch_size=self.batch_size,
            num_workers=multiprocessing.cpu_count(),
            collate_fn=self.collate_fn
        )


class QAGMultiTaskDataModule(pl.LightningDataModule):

    def __init__(self, 
                 dataset_name, 
                 tokenizer,
                 model,
                 max_length=512, 
                 batch_size=8, 
                 recreate=False,
                 test=False,
                 ) -> None:

        super(QAGMultiTaskDataModule, self).__init__()

        self.seed = 42
        self.dataset_name = dataset_name
        self.tokenizer = tokenizer
        self.model = model
        self.max_length = 128 if test else max_length
        self.batch_size = batch_size
        self.recreate = recreate
        self.test = test
        
        self.train_dataset_path = f"Datasets/MultiTask/{dataset_name}/train.csv"
        self.valid_dataset_path = f"Datasets/MultiTask/{dataset_name}/validation.csv"
        self.test_dataset_path = f"Datasets/MultiTask/{dataset_name}/test.csv"

        self.save_train_tensor_dataset_dir_path = f"Datasets/Tensor/MultiTask/{dataset_name}"
        self.train_tensor_dataset_path = f"{self.save_train_tensor_dataset_dir_path}/train.pt"
        self.valid_tensor_dataset_path = f"{self.save_train_tensor_dataset_dir_path}/validation.pt"
        self.ae_test_tensor_dataset_path = f"{self.save_train_tensor_dataset_dir_path}/ae_test.pt"
        self.qg_test_tensor_dataset_path = f"{self.save_train_tensor_dataset_dir_path}/qg_test.pt"


    def load_data(self):
        if os.path.exists(self.train_tensor_dataset_path) and os.path.exists(self.valid_tensor_dataset_path) and os.path.exists(self.ae_test_tensor_dataset_path) and os.path.exists(self.qg_test_tensor_dataset_path) and not self.recreate:
            
            print('\n[ Loading Dataset ]\n')
            train_data = torch.load(self.train_tensor_dataset_path)
            valid_data = torch.load(self.valid_tensor_dataset_path)
            ae_test_data = torch.load(self.ae_test_tensor_dataset_path)
            qg_test_data = torch.load(self.qg_test_tensor_dataset_path)
            print('\n[ Load Completed ]\n')

            return train_data, valid_data, ae_test_data, qg_test_data
        
        else:

            print('\n[ Processing Dataset ]\n')
            
            data_csv = {
                'train': pd.read_csv(self.train_dataset_path),
                'validation': pd.read_csv(self.valid_dataset_path),
                'test': pd.read_csv(self.test_dataset_path),
            }

            if self.test:
                data_csv = {
                'train': pd.read_csv(self.train_dataset_path)[:123],
                'validation': pd.read_csv(self.valid_dataset_path)[:123],
                'test': pd.read_csv(self.test_dataset_path)[:26],
            }

            tokenized_data = {
                'train': [],
                'validation': [],
                'ae_test': [],
                'qg_test': [],
            }

            for key, data in data_csv.items():

                if key != 'test':

                    input_ids, attention_mask, target_ids = [], [], []

                    for (task, input_text, output_text) in tqdm(data.values.tolist(), desc=f'Tokenizing {key} data'):
                        
                        if self.model == 'IndoBART':
                            input_text = f'{locals()["input_text"]}{self.tokenizer.eos_token}'


                        encoded_input = self.tokenizer(input_text, 
                                                    add_special_tokens=True, 
                                                    max_length=self.max_length,
                                                    padding="max_length",
                                                    truncation=True)
                        
                        encoded_output =  self.tokenizer(output_text,
                                                        add_special_tokens=True, 
                                                        max_length=self.max_length,
                                                        padding="max_length",
                                                        truncation=True)
                        
                        input_ids.append(encoded_input['input_ids'])
                        attention_mask.append(encoded_input['attention_mask'])
                        target_ids.append(encoded_output['input_ids'])

                    tokenized_data[key] = TensorDataset(torch.tensor(input_ids), torch.tensor(attention_mask), torch.tensor(target_ids))

                    os.makedirs(self.save_train_tensor_dataset_dir_path, exist_ok=True)
                    torch.save(tokenized_data[key], f'{self.save_train_tensor_dataset_dir_path}/{key}.pt')
                
                elif key == 'test':

                    ae_input_ids, ae_attention_mask, ae_target_ids = [], [], []
                    qg_input_ids, qg_attention_mask, qg_target_ids = [], [], []

                    for (task, input_text, output_text) in tqdm(data.values.tolist(), desc=f'Tokenizing {key} data'):
                        
                        if self.model == 'IndoBART':
                            input_text = f'{locals()["input_text"]}{self.tokenizer.eos_token}'


                        encoded_input = self.tokenizer(input_text, 
                                                    add_special_tokens=True, 
                                                    max_length=self.max_length,
                                                    padding="max_length",
                                                    truncation=True)
                        
                        encoded_output =  self.tokenizer(output_text,
                                                        add_special_tokens=True, 
                                                        max_length=self.max_length,
                                                        padding="max_length",
                                                        truncation=True)
                        
                        if locals()["task"] == 'AE':
                            ae_input_ids.append(encoded_input['input_ids'])
                            ae_attention_mask.append(encoded_input['attention_mask'])
                            ae_target_ids.append(encoded_output['input_ids'])
                        elif locals()["task"] == 'QG':
                            qg_input_ids.append(encoded_input['input_ids'])
                            qg_attention_mask.append(encoded_input['attention_mask'])
                            qg_target_ids.append(encoded_output['input_ids'])


                    tokenized_data['ae_test'] = TensorDataset(torch.tensor(ae_input_ids), torch.tensor(ae_attention_mask), torch.tensor(ae_target_ids))
                    tokenized_data['qg_test'] = TensorDataset(torch.tensor(qg_input_ids), torch.tensor(qg_attention_mask), torch.tensor(qg_target_ids))

                    os.makedirs(self.save_train_tensor_dataset_dir_path, exist_ok=True)
                    torch.save(tokenized_data['ae_test'], f'{self.save_train_tensor_dataset_dir_path}/ae_test.pt')
                    torch.save(tokenized_data['qg_test'], f'{self.save_train_tensor_dataset_dir_path}/qg_test.pt')
                    
                
            print('\n[ Process Completed ]\n')
            
            return tokenized_data['train'], tokenized_data['validation'], tokenized_data['ae_test'], tokenized_data['qg_test']


    def setup(self, stage=None):
        train_data, valid_data, ae_test_data, qg_test_data = self.load_data()
        if stage == "fit":
            self.train_data = train_data
            self.valid_data = valid_data
        elif stage == "ae_test":
            self.test_data = ae_test_data
        elif stage == "qg_test":
            self.test_data = qg_test_data


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



class AnswerExtractionDataModule(pl.LightningDataModule):

    def __init__(self, 
                 dataset_name, 
                 tokenizer,
                 model,
                 input_type,
                 output_type,
                 max_length=512, 
                 batch_size=8, 
                 recreate=False,
                 test=False,
                 ) -> None:

        super(AnswerExtractionDataModule, self).__init__()

        self.seed = 42
        self.dataset_name = dataset_name
        self.tokenizer = tokenizer
        self.model = model
        self.input_type = input_type
        self.output_type = output_type
        self.max_length = 128 if test else max_length
        self.batch_size = batch_size
        self.recreate = recreate
        self.test = test
        
        self.train_dataset_path = f"Datasets/Processed/{dataset_name}/prepared_train.csv"
        self.valid_dataset_path = f"Datasets/Processed/{dataset_name}/prepared_dev.csv"
        self.test_dataset_path = f"Datasets/Processed/{dataset_name}/prepared_test.csv"

        self.train_tensor_dataset_path = f"Datasets/Tensor/{dataset_name}/train.pt"
        self.valid_tensor_dataset_path = f"Datasets/Tensor/{dataset_name}_{input_type}_to_{output_type}/dev.pt"
        self.test_tensor_dataset_path = f"Datasets/Tensor/{dataset_name}_{input_type}_to_{output_type}/test.pt"


    def load_data(self):
        if os.path.exists(self.train_tensor_dataset_path) and os.path.exists(self.valid_tensor_dataset_path) and os.path.exists(self.test_tensor_dataset_path) and not self.recreate:
            
            print('\n[ Loading Dataset ]\n')
            train_data = torch.load(self.train_tensor_dataset_path)
            valid_data = torch.load(self.valid_tensor_dataset_path)
            test_data = torch.load(self.test_tensor_dataset_path)
            print('\n[ Load Completed ]\n')

            return train_data, valid_data, test_data
        
        else:

            print('\n[ Processing Dataset ]\n')
            
            data_csv = {
                'train': pd.read_csv(self.train_dataset_path),
                'dev': pd.read_csv(self.valid_dataset_path),
                'test': pd.read_csv(self.test_dataset_path),
            }

            if self.test:
                data_csv = {
                'train': pd.read_csv(self.train_dataset_path)[:123],
                'dev': pd.read_csv(self.valid_dataset_path)[:123],
                'test': pd.read_csv(self.test_dataset_path)[:123],
            }

            tokenized_data = {
                'train': [],
                'dev': [],
                'test': [],
            }

            for key, data in data_csv.items():

                input_ids, attention_mask, target_ids = [], [], []

                for (context, context_key_sentence, context_answer, question, answer) in tqdm(data.values.tolist(), desc=f'Tokenizing {key} data'):

                    if self.model == 'IndoBART':
                        input_text = f'{self.tokenizer.bos_token}{locals()[self.input_type]}{self.tokenizer.eos_token}'
                        output_text = f'{locals()[self.output_type]}'
                    elif self.model == 'Flan-T5':
                        input_text = locals()[self.input_type]
                        output_text = locals()[self.output_type]

                    encoded_input = self.tokenizer(input_text, 
                                                   add_special_tokens=True, 
                                                   max_length=self.max_length,
                                                   padding="max_length",
                                                   truncation=True)
                    
                    encoded_output =  self.tokenizer(output_text,
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

            print('\n[ Process Completed ]\n')
            
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
