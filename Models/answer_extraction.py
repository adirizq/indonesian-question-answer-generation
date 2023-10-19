import sys
import evaluate
import torch
import pytorch_lightning as pl

from torch import nn
from torch.nn import functional as F
from indobenchmark import IndoNLGTokenizer
from transformers import BartForConditionalGeneration


class BartAnswerExtraction(pl.LightningModule):
    
    def __init__(self, tokenizer, learning_rate=1e-5, max_length=512) -> None:
        super(BartAnswerExtraction, self).__init__()

        self.tokenizer = tokenizer
        self.lr = learning_rate
        self.model = BartForConditionalGeneration.from_pretrained('indobenchmark/indobart-v2')
        self.model.resize_token_embeddings(len(self.tokenizer) + 1)
        self.model.config.max_length = max_length

        self.bleu = evaluate.load('bleu')
        self.meteor = evaluate.load('meteor')
        self.rouge = evaluate.load('rouge')

        self.test_step_outpus = {
          'input_ids': [],
          'outputs': [],
          'labels': [],
        }

    
    def decode(self, text):
        decoded = self.tokenizer.decode(text).replace('<pad>', '').replace('<s>', '').replace('</s>', '')
        decoded = decoded.split('<hl>')[1] if '<hl>' in decoded else decoded
        return decoded

    
    def forward(self, input_ids, attention_mask, labels):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
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

        self.test_step_outpus['input_ids'].append(input_ids)
        self.test_step_outpus['outputs'].append(out)
        self.test_step_outpus['labels'].append(labels)

        return 0
    

    def on_test_epoch_end(self):
        input_ids = torch.cat(self.test_step_outpus['input_ids'], dim=0)
        outputs = torch.cat(self.test_step_outpus['outputs'], dim=0)
        labels = torch.cat(self.test_step_outpus['labels'], dim=0)

        decoded_inputs = []
        decoded_outputs = []
        decoded_labels = []

        decoded_inputs_labels  = {}

        for idx in range(len(input_ids)):
            decoded_input = self.decode(input_ids[idx])
            decoded_output = self.decode(outputs[idx])

            decoded_inputs.append(decoded_input)
            decoded_outputs.append(decoded_output)

            if not decoded_input in decoded_inputs_labels:
                decoded_inputs_labels[decoded_input] = []
            
            decoded_inputs_labels[decoded_input].append(self.decode(labels[idx]))

        for decoded_input in decoded_inputs:
            decoded_labels.append(decoded_inputs_labels[decoded_input])
        

        score_bleu = self.bleu.compute(predictions=decoded_inputs, references=decoded_labels)["bleu"]
        score_meteor = self.meteor.compute(predictions=decoded_inputs, references=decoded_labels)["meteor"]
        score_rouge = self.rouge.compute(predictions=decoded_inputs, references=decoded_labels)
        score_rouge1 = score_rouge['rouge1']
        score_rouge2 = score_rouge['rouge2']
        score_rougeL = score_rouge['rougeL']
        score_rougeLsum = score_rouge['rougeLsum']

        print('\n[ Test Results ]\n')
        print(f'Bleu: {score_bleu}')
        print(f'Meteor: {score_meteor}')
        print(f'Rouge1: {score_rouge1}')
        print(f'Rouge2: {score_rouge2}')
        print(f'RougeL: {score_rougeL}')
        print(f'RougeLsum: {score_rougeLsum}')
