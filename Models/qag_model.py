import re
import sys
import torch
import evaluate
import torch.nn as nn
import pytorch_lightning as pl

from typing import Any
from textwrap import dedent
from torch.optim.lr_scheduler import StepLR
from Utils.utils import Tokenizer, Evaluator


class QAGModel(pl.LightningModule):

    def __init__(self, ae_model, qg_model, batch, tokenizer, learning_rate, lr_scheduler, cuda=False):
        super(QAGModel, self).__init__()

        self.ae_model = ae_model
        self.qg_model = qg_model
        self.batch = batch
        self.tokenizer = tokenizer
        self.lr_scheduler = lr_scheduler
        self.lr = learning_rate
        self.cuda = 'cuda' if cuda else 'cpu'

        self.pad_token_id = tokenizer.pad_token_id
        self.separator_token_id = tokenizer.convert_tokens_to_ids('<sep>')
        self.tensor_separator_token_id = torch.tensor([tokenizer.convert_tokens_to_ids('<sep>')]).to(self.cuda)

        self.ae_criterion = nn.CrossEntropyLoss(ignore_index=self.pad_token_id)
        self.qg_criterion = nn.CrossEntropyLoss(ignore_index=self.pad_token_id)
        

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr,  betas=(0.9, 0.999), eps=1e-08)

        if self.lr_scheduler:
            scheduler = StepLR(optimizer, step_size=1, gamma=0.9)
            return {"optimizer": optimizer, "lr_scheduler": scheduler}

        return optimizer
    

    def split_tensor(self, tensor, value):
        indices = torch.where(tensor == value)[0]

        start = 0
        split_tensors = []

        for idx in indices:
            split_tensors.append(tensor[start:idx])
            start = idx + 1

        split_tensors.append(tensor[start:])
        split_tensors = [t[t != 1] for t in split_tensors if t.nelement() > 0]

        return split_tensors


    def forward(self, context_input_ids, context_attention_mask, answer_input_ids, question_input_ids):
        self.ae_model.train()
        ae_output = self.ae_model(input_ids=context_input_ids, attention_mask=context_attention_mask, labels=answer_input_ids)

        self.ae_model.eval()
        predicted_answers = self.ae_model.generate(context_input_ids)
        self.ae_model.train()

        with torch.no_grad():
            qg_inputs_ids = []
            qg_attention_mask = []
            qg_labels = []

            batch_split_answers = [ self.split_tensor(predicted_answer, self.separator_token_id) for predicted_answer in predicted_answers ]

            for i, split_answers in enumerate(batch_split_answers):
                for j, answer in enumerate(split_answers):
                    if j < len(question_input_ids[i]):
                        input_ids = torch.cat([answer, self.tensor_separator_token_id, context_input_ids[i]])[:len(context_input_ids[i])]
                        qg_inputs_ids.append(input_ids)
                        qg_attention_mask.append(((input_ids != 1).clone().detach()).int())
                        qg_labels.append(question_input_ids[i][j])
            
            qg_inputs_ids = torch.stack(qg_inputs_ids).to(self.cuda).split(self.batch)
            qg_attention_mask = torch.stack(qg_attention_mask).to(self.cuda).split(self.batch)
            qg_labels = torch.stack(qg_labels).to(self.cuda).split(self.batch)

        qg_ouputs = []
        predicted_questions_answer = []

        for input_ids, attention_mask, labels in zip(qg_inputs_ids, qg_attention_mask, qg_labels):
            self.qg_model.train()
            qg_output = self.qg_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)  
            qg_ouputs.append(qg_output)

            self.qg_model.eval()
            predicted_questions = self.qg_model.generate(input_ids=input_ids)
            for i, predicted_question in enumerate(predicted_questions):
                predicted_answer = self.split_tensor(input_ids[i], self.separator_token_id)[0]
                predicted_questions_answer.append([predicted_question, predicted_answer])
            self.qg_model.train()

        
        return ae_output, qg_ouputs, predicted_questions_answer, qg_labels


    def training_step(self, train_batch, batch_idx):
        context_input_ids, context_attention_mask, answer_input_ids, question_input_ids = train_batch

        ae_output, qg_ouputs, predicted_questions_answer, qg_labels = self(context_input_ids, context_attention_mask, answer_input_ids, question_input_ids)

        ae_loss = self.ae_criterion(ae_output.logits.view(-1, ae_output.logits.size(-1)), answer_input_ids.view(-1))

        qg_outs_logits = []
        for qg_outs in qg_ouputs:
            qg_outs_logits.append(qg_outs.logits)

        qg_outs_logits = torch.cat(qg_outs_logits)
        qg_loss = self.qg_criterion(qg_outs_logits.view(-1, qg_outs_logits.size(-1)), torch.cat(qg_labels).view(-1))
        
        avg_loss = (ae_loss + qg_loss) / 2

        print('AE Loss:', ae_loss.item())
        print('QG Loss:', qg_loss.item())

        self.log_dict({'train_loss': avg_loss}, prog_bar=True, on_epoch=True)

        print('\n[ TRAINING STEP ][ TARGET ]\n')
        for a_ids, q_ids in zip(answer_input_ids, question_input_ids):
            for q in q_ids:
                print(self.tokenizer.decode(q, skip_special_tokens=True), '\n')
            print(self.tokenizer.decode(a_ids, skip_special_tokens=True), '\n')

        print('\n[ TRAINING STEP ][ PREDICTED ]\n')
        for qa in predicted_questions_answer:
            print('Question:\n', self.tokenizer.decode(qa[0], skip_special_tokens=True), '\n')
            print('Answer:\n', self.tokenizer.decode(qa[1], skip_special_tokens=True), '\n')

        return avg_loss


    def validation_step(self, valid_batch, batch_idx):
        context_input_ids, context_attention_mask, answer_input_ids, question_input_ids = valid_batch

        ae_output, qg_ouputs, predicted_questions_answer, qg_labels = self(context_input_ids, context_attention_mask, answer_input_ids, question_input_ids)

        ae_loss = self.ae_criterion(ae_output.logits.view(-1, ae_output.logits.size(-1)), answer_input_ids.view(-1))

        qg_outs_logits = []
        for qg_outs in qg_ouputs:
            qg_outs_logits.append(qg_outs.logits)

        qg_outs_logits = torch.cat(qg_outs_logits)
        qg_loss = self.qg_criterion(qg_outs_logits.view(-1, qg_outs_logits.size(-1)), torch.cat(qg_labels).view(-1))
        
        avg_loss = (ae_loss + qg_loss) / 2

        print('AE Loss:', ae_loss.item())
        print('QG Loss:', qg_loss.item())

        self.log_dict({'val_loss': avg_loss}, prog_bar=True, on_epoch=True)

        print('\n[ EVALUATION ][ TARGET ]\n')
        for a_ids, q_ids in zip(answer_input_ids, question_input_ids):
            for q in q_ids:
                print(self.tokenizer.decode(q, skip_special_tokens=True), '\n')
            print(self.tokenizer.decode(a_ids, skip_special_tokens=True), '\n')

        print('\n[ EVALUATION ][ PREDICTED ]\n')
        for qa in predicted_questions_answer:
            print('Question:\n', self.tokenizer.decode(qa[0], skip_special_tokens=True), '\n')
            print('Answer:\n', self.tokenizer.decode(qa[1], skip_special_tokens=True), '\n')

        return avg_loss


class QAGMultiTaskModel(pl.LightningModule):
    def __init__(self, pretrained_model, model_type, tokenizer: Tokenizer, lr_scheduler, learning_rate=1e-5):
        super(QAGMultiTaskModel, self).__init__()

        self.model = pretrained_model
        self.model_type = model_type
        self.tokenizer = tokenizer
        self.lr_scheduler = lr_scheduler
        self.lr = learning_rate

        self.bleu = evaluate.load('bleu')
        self.meteor = evaluate.load('meteor')
        self.rouge = evaluate.load('rouge')

        self.test_type = None

        self.test_step_outputs = {
          'input_ids': [],
          'outputs': [],
          'labels': [],
        }

    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr,  betas=(0.9, 0.999), eps=1e-08)

        if self.lr_scheduler:
            scheduler = StepLR(optimizer, step_size=1, gamma=0.9)
            return {"optimizer": optimizer, "lr_scheduler": scheduler}

        return optimizer
    

    def forward(self, input_ids, attention_mask, labels):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)


    def training_step(self, train_batch, batch_idx):
        input_ids, attention_mask, labels = train_batch

        out = self(input_ids, attention_mask, labels)
        self.log_dict({'train_loss': out.loss}, prog_bar=True, on_epoch=True)

        return out.loss
    

    def validation_step(self, valid_batch, batch_idx):
        input_ids, attention_mask, labels = valid_batch

        out = self(input_ids, attention_mask, labels)
        self.log_dict({'val_loss': out.loss}, prog_bar=True, on_epoch=True)

        return out.loss
    

    def test_step(self, test_batch, batch_idx):
        input_ids, attention_mask, labels = test_batch

        out = self.model.generate(input_ids)

        for idx in range(len(input_ids)):
            self.test_step_outputs['input_ids'].append(self.tokenizer.decode(input_ids[idx]))
            self.test_step_outputs['outputs'].append(self.tokenizer.decode_for_answer_or_question(out[idx]))
            self.test_step_outputs['labels'].append(self.tokenizer.decode_for_answer_or_question(labels[idx]))

        return 0


    def on_test_epoch_end(self):

        evaluator = Evaluator()
        
        score_exact_match, score_bleu, score_meteor, score_rouge1, score_rouge2, score_rougeL, score_rougeLsum = evaluator.evaluate(self.test_type, self.test_step_outputs, f'./Predictions/Multitask/{self.model_type}.csv')

        self.log_dict({f"{self.test_type.value}_test_exact_match": score_exact_match,
                       f"{self.test_type.value}_test_bleu": score_bleu,
                       f"{self.test_type.value}_test_meteor": score_meteor,
                       f"{self.test_type.value}_test_rouge1": score_rouge1,
                       f"{self.test_type.value}_test_rouge2": score_rouge2,
                       f"{self.test_type.value}_test_rougeL": score_rougeL,
                       f"{self.test_type.value}_test_rougeLsum": score_rougeLsum
                       }, on_epoch=True)


class QAGPipelineModel(pl.LightningModule):
    
    def __init__(self, pretrained_model, model_type, task_type, tokenizer: Tokenizer, lr_scheduler, learning_rate=1e-5):
        super(QAGPipelineModel, self).__init__()

        self.model = pretrained_model
        self.model_type = model_type
        self.task_type = task_type
        self.tokenizer = tokenizer
        self.lr_scheduler = lr_scheduler
        self.lr = learning_rate

        self.bleu = evaluate.load('bleu')
        self.meteor = evaluate.load('meteor')
        self.rouge = evaluate.load('rouge')

        self.test_step_outputs = {
          'input_ids': [],
          'outputs': [],
          'labels': [],
        }


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr,  betas=(0.9, 0.999), eps=1e-08)

        if self.lr_scheduler:
            scheduler = StepLR(optimizer, step_size=1, gamma=0.9)
            return {"optimizer": optimizer, "lr_scheduler": scheduler}

        return optimizer
    

    def forward(self, input_ids, attention_mask, labels):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    

    def training_step(self, train_batch, batch_idx):
        input_ids, attention_mask, labels = train_batch

        out = self(input_ids, attention_mask, labels)
        self.log_dict({'train_loss': out.loss}, prog_bar=True, on_epoch=True)

        return out.loss


    def validation_step(self, valid_batch, batch_idx):
        input_ids, attention_mask, labels = valid_batch

        out = self(input_ids, attention_mask, labels)
        self.log_dict({'val_loss': out.loss}, prog_bar=True, on_epoch=True)

        return out.loss
    

    def test_step(self, test_batch, batch_idx):
        input_ids, attention_mask, labels = test_batch

        out = self.model.generate(input_ids)

        for idx in range(len(input_ids)):
            self.test_step_outputs['input_ids'].append(self.tokenizer.decode(input_ids[idx]))
            self.test_step_outputs['outputs'].append(self.tokenizer.decode_for_answer_or_question(out[idx]))
            self.test_step_outputs['labels'].append(self.tokenizer.decode_for_answer_or_question(labels[idx]))

        return 0


    def on_test_epoch_end(self):

        evaluator = Evaluator()
        
        score_exact_match, score_bleu, score_meteor, score_rouge1, score_rouge2, score_rougeL, score_rougeLsum = evaluator.evaluate(self.task_type, self.test_step_outputs, f'./Predictions/Pipeline/{self.task_type.value}/{self.model_type}.csv')

        self.log_dict({f"{self.task_type.value}_test_exact_match": score_exact_match,
                       f"{self.task_type.value}_test_bleu": score_bleu,
                       f"{self.task_type.value}_test_meteor": score_meteor,
                       f"{self.task_type.value}_test_rouge1": score_rouge1,
                       f"{self.task_type.value}_test_rouge2": score_rouge2,
                       f"{self.task_type.value}_test_rougeL": score_rougeL,
                       f"{self.task_type.value}_test_rougeLsum": score_rougeLsum
                       }, on_epoch=True)


class QAGEnd2EndModel(pl.LightningModule):
    
    def __init__(self, pretrained_model, model_type, task_type, tokenizer: Tokenizer, lr_scheduler, learning_rate=1e-5):
        super(QAGPipelineModel, self).__init__()

        self.model = pretrained_model
        self.model_type = model_type
        self.task_type = task_type
        self.tokenizer = tokenizer
        self.lr_scheduler = lr_scheduler
        self.lr = learning_rate

        self.bleu = evaluate.load('bleu')
        self.meteor = evaluate.load('meteor')
        self.rouge = evaluate.load('rouge')

        self.test_step_outputs = {
          'input_ids': [],
          'outputs': [],
          'labels': [],
        }


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr,  betas=(0.9, 0.999), eps=1e-08)

        if self.lr_scheduler:
            scheduler = StepLR(optimizer, step_size=1, gamma=0.9)
            return {"optimizer": optimizer, "lr_scheduler": scheduler}

        return optimizer
    

    def forward(self, input_ids, attention_mask, labels):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    

    def training_step(self, train_batch, batch_idx):
        input_ids, attention_mask, labels = train_batch

        out = self(input_ids, attention_mask, labels)
        self.log_dict({'train_loss': out.loss}, prog_bar=True, on_epoch=True)

        return out.loss


    def validation_step(self, valid_batch, batch_idx):
        input_ids, attention_mask, labels = valid_batch

        out = self(input_ids, attention_mask, labels)
        self.log_dict({'val_loss': out.loss}, prog_bar=True, on_epoch=True)

        return out.loss
    

    def test_step(self, test_batch, batch_idx):
        input_ids, attention_mask, labels = test_batch

        out = self.model.generate(input_ids)

        for idx in range(len(input_ids)):
            self.test_step_outputs['input_ids'].append(self.tokenizer.decode(input_ids[idx]))
            self.test_step_outputs['outputs'].append(self.tokenizer.decode_for_answer_or_question(out[idx]))
            self.test_step_outputs['labels'].append(self.tokenizer.decode_for_answer_or_question(labels[idx]))

        return 0


    def on_test_epoch_end(self):

        evaluator = Evaluator()
        
        score_exact_match, score_bleu, score_meteor, score_rouge1, score_rouge2, score_rougeL, score_rougeLsum = evaluator.evaluate(self.task_type, self.test_step_outputs, f'./Predictions/End2End/{self.task_type.value}/{self.model_type}.csv')

        self.log_dict({f"{self.task_type.value}_test_exact_match": score_exact_match,
                       f"{self.task_type.value}_test_bleu": score_bleu,
                       f"{self.task_type.value}_test_meteor": score_meteor,
                       f"{self.task_type.value}_test_rouge1": score_rouge1,
                       f"{self.task_type.value}_test_rouge2": score_rouge2,
                       f"{self.task_type.value}_test_rougeL": score_rougeL,
                       f"{self.task_type.value}_test_rougeLsum": score_rougeLsum
                       }, on_epoch=True)