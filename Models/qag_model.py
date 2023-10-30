import sys
import torch
import evaluate
import pytorch_lightning as pl

from torch import nn
from textwrap import dedent
from torch.nn import functional as F
from indobenchmark import IndoNLGTokenizer
from transformers import BartForConditionalGeneration
from torch.optim.lr_scheduler import StepLR


class QAGModel(pl.LightningModule):
    
    def __init__(self, model, tokenizer, lr_scheduler, input_type, output_type, model_task, learning_rate=1e-5) -> None:
        super(QAGModel, self).__init__()

        self.save_hyperparameters(ignore=['model', 'tokenizer'])

        self.model = model
        self.tokenizer = tokenizer
        self.lr_scheduler = lr_scheduler
        self.input_type = input_type
        self.output_type = output_type
        self.model_task = model_task
        self.lr = learning_rate

        self.bleu = evaluate.load('bleu')
        self.meteor = evaluate.load('meteor')
        self.rouge = evaluate.load('rouge')

        self.test_step_outputs = {
          'input_ids': [],
          'outputs': [],
          'labels': [],
        }

        self.valid_step_outputs ={
          'input_ids': [],
          'outputs': [],
          'labels': [],
        }

        self.model_task_inf = {
            'ae': 'Answer Extraction',
            'qg': 'Question Generator'
        }

    
    def save_pretrained(self, save_path):
        self.model.save_pretrained(save_path)

    
    def decode(self, text):
        decoded = self.tokenizer.decode(text).replace('<pad>', '').replace('<s>', '').replace('</s>', '')
        decoded = decoded.split('<hl>')[1] if '<hl>' in decoded else decoded
        return decoded.strip()
    

    def exact_match_evaluation(self, predictions, references):
        assert len(predictions) == len(references), "The number of predictions and references should be the same"
        
        exact_matches = 0
        for pred, refs in zip(predictions, references):
            if pred in refs:
                exact_matches += 1
        
        return exact_matches / len(predictions)

    
    def forward(self, input_ids, attention_mask, labels):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr,  betas=(0.9, 0.999), eps=1e-08)

        if self.lr_scheduler:
            scheduler = StepLR(optimizer, step_size=1, gamma=0.9)
            return {"optimizer": optimizer, "lr_scheduler": scheduler}

        return optimizer
    

    def training_step(self, train_batch, batch_idx):
        input_ids, attention_mask, labels = train_batch

        out = self(input_ids, attention_mask, labels)

        loss = out.loss
        logits = out.logits

        self.log_dict({'train_loss': loss}, prog_bar=True, on_epoch=True)

        return loss


    def validation_step(self, valid_batch, batch_idx):
        input_ids, attention_mask, labels = valid_batch

        out = self(input_ids, attention_mask, labels)
        
        loss = out.loss
        logits = out.logits

        self.log_dict({'val_loss': loss}, prog_bar=True, on_epoch=True)

        return loss
    

    def test_step(self, test_batch, batch_idx):
        input_ids, attention_mask, labels = test_batch

        out = self.model.generate(input_ids)

        for idx in range(len(input_ids)):
            self.test_step_outputs['input_ids'].append(self.decode(input_ids[idx]))
            self.test_step_outputs['outputs'].append(self.decode(out[idx]))
            self.test_step_outputs['labels'].append(self.decode(labels[idx]))

        return 0
    

    def on_test_epoch_end(self):
        label_dict = {}
        processed_labels = [] 

        for d_input, d_label in zip(self.test_step_outputs['input_ids'], self.test_step_outputs['labels']):
          if not d_label in label_dict:
            label_dict[d_input] = []
          label_dict[d_input].append(d_label)

        for d_input in self.test_step_outputs['input_ids']:
          processed_labels.append(label_dict[d_input])
        
        score_exact_match = self.exact_match_evaluation(predictions=self.test_step_outputs['outputs'], references=processed_labels)
        score_bleu = self.bleu.compute(predictions=self.test_step_outputs['outputs'], references=processed_labels)["bleu"]
        score_meteor = self.meteor.compute(predictions=self.test_step_outputs['outputs'], references=processed_labels)["meteor"]
        score_rouge = self.rouge.compute(predictions=self.test_step_outputs['outputs'], references=processed_labels)
        score_rouge1 = score_rouge['rouge1']
        score_rouge2 = score_rouge['rouge2']
        score_rougeL = score_rouge['rougeL']
        score_rougeLsum = score_rouge['rougeLsum']

        print(dedent(f'''
        -----------------------------------------------
                {self.model_task_inf[self.model_task]} Test Result        
        -----------------------------------------------
        Name                | Value       
        -----------------------------------------------
        Input Type          | {self.input_type}
        Output Type         | {self.output_type}
        Exact Match         | {score_exact_match}
        Bleu                | {score_bleu}
        Meteor              | {score_meteor}
        Rouge1              | {score_rouge1}
        Rouge2              | {score_rouge2}
        RougeL              | {score_rougeL}
        RougeLsum           | {score_rougeLsum}
        -----------------------------------------------

        '''))

        print(dedent(f'''
        -----------------------------------------------
              {self.model_task_inf[self.model_task]} Prediction Result        
        -----------------------------------------------
        '''))
        for d_pred, d_label in zip(self.test_step_outputs['outputs'], processed_labels):
            print(f'Predictions:\n{d_pred}')
            print(f'Labels:\n{d_label}\n') 
        

        self.log_dict({'test_exact_match': score_exact_match,
                       'test_bleu': score_bleu,
                       'test_meteor': score_meteor,
                       'test_rouge1': score_rouge1,
                       'test_rouge2': score_rouge2,
                       'test_rougeL': score_rougeL,
                       'test_rougeLsum': score_rougeLsum,
                       }, on_epoch=True)

