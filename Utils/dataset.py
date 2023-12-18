import os
import sys
import torch
import pandas as pd

from torch.utils.data import Dataset


class AnswerExtractionDataset(Dataset):
    
    def __init__(self, csv_path, save_path, tokenizer, max_length = 512, recreate = False):
        self.csv_path = csv_path
        self.save_path = save_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.recreate = recreate

        if os.path.exists(self.save_path) and not self.recreate:
            self.data = torch.load(self.save_path)
        else:
            self.data = self.load_data()
        
        self.x_input_ids = self.data[0]
        self.x_attention_mask = self.data[1]
        self.y_input_ids = self.data[2]


    def __getitem__(self, index):
        return self.x_input_ids[index], self.x_attention_mask[index], self.y_input_ids[index]


    def __len__(self):
        return len(self.data)


    def load_data(self):
        # Read csv and convert to context, answer1<sep>answer2<sep>...
        data_df = pd.read_csv(self.csv_path, dtype=str)[['context', 'answer_text']]
        data_df = data_df.groupby('context')['answer_text'].apply('<sep>'.join).reset_index()

        # Add eos token to end of context and answer_text
        data_df['context'] = data_df['context'].apply(lambda x: f'{x}{self.tokenizer.eos_token}')
        data_df['answer_text'] = data_df['answer_text'].apply(lambda x: f'{x}{self.tokenizer.eos_token}')

        # Tokenize context
        x = self.tokenizer(data_df['context'].tolist(), 
                       add_special_tokens=True,
                       max_length=self.max_length,
                       padding="max_length",
                       truncation=True,
                       return_tensors='pt')

        # Tokenize answer_text
        y = self.tokenizer(data_df['answer_text'].tolist(), 
                       add_special_tokens=True,
                       max_length=self.max_length,
                       padding="max_length",
                       truncation=True,
                       return_tensors='pt')
        
        # Combine x and y into one tensor, and save it
        data = torch.stack((x.input_ids, x.attention_mask, y.input_ids))
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        torch.save(data, self.save_path)
        
        return data
    

