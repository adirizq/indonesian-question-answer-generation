import sys
import torch
import torch.nn as nn


class QAGPipelineModel(nn.Module):
    
    def __init__(self, pretrained_model, tokenizer, model_task):
        super(QAGPipelineModel, self).__init__()

        self.model = pretrained_model
        self.tokenizer = tokenizer
        self.model_task = model_task

        self.model_task_inf = {
            'ae': 'Answer Extraction',
            'qg': 'Question Generator'
        }
    

    def forward(self, input_ids, attention_mask, labels):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    

    def save_pretrained(self, save_path):
        self.model.save_pretrained(save_path)


    def decode(self, text):
        decoded = self.tokenizer.decode(text).replace('<pad>', '').replace('<s>', '').replace('</s>', '').replace('<unk>', '').strip()
        if self.model_task == 'ae':
            decoded = [d.strip() for d in decoded.split('<sep>')]
        return decoded

