import os
import sys
import torch
import pandas as pd

from Utils.utils import PipeLineTaskType
from torch.utils.data import Dataset


class PipelineDataset(Dataset):
    def __init__(self, task_type, csv_path, save_path, tokenizer, recreate):
        
        if os.path.exists(save_path) and not recreate:
            print(f'[INFO] Loading data from {save_path}')
            self.data = torch.load(save_path)

        else:
            print(f'[INFO] Tokenizing data from {csv_path}')
            self.csv_path = csv_path
            self.save_path = save_path
            self.recreate = recreate
            self.task_type = task_type
            self.tokenizer = tokenizer
            self.data = self.load_data()


    def __getitem__(self, index):
        return self.data[0][index], self.data[1][index], self.data[2][index]


    def __len__(self):
        return self.data.shape[1]


    def load_data(self):
        if self.task_type == PipeLineTaskType.ANSWER_EXTRACTION:
            x_column = 'sentence_highlighted_context'
            y_column = 'answer_highlighted_context'
        
        if self.task_type == PipeLineTaskType.QUESTION_GENERATION:
            x_column = 'answer_highlighted_context'
            y_column = 'question'

        df = pd.read_csv(self.csv_path)[[x_column, y_column]]

        encoded_x = self.tokenizer.tokenize(df[x_column].tolist())
        encoded_y = self.tokenizer.tokenize(df[y_column].tolist())

        data = torch.stack((torch.tensor(encoded_x.input_ids), 
                            torch.tensor(encoded_x.attention_mask), 
                            torch.tensor(encoded_y.input_ids)))

        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        torch.save(data, self.save_path)
        
        return data