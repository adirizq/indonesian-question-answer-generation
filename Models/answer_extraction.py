import sys
import torch
import pytorch_lightning as pl

from torch import nn
from torch.nn import functional as F
from indobenchmark import IndoNLGTokenizer
from transformers import BartForConditionalGeneration


class BartAnswerExtraction(pl.LightningModule):
    
    def __init__(self, tokenizer, learning_rate=1e-5) -> None:
        super(BartAnswerExtraction, self).__init__()

        self.tokenizer = tokenizer
        self.lr = learning_rate
        self.model = BartForConditionalGeneration.from_pretrained('indobenchmark/indobart-v2')
        self.model.resize_token_embeddings(len(self.tokenizer) + 1)

    
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

        return loss


    def validation_step(self, valid_batch, batch_idx):
        input_ids, attention_mask, labels = valid_batch

        out = self(input_ids, attention_mask, labels)
        
        loss = out.loss
        logits = out.logits

        return loss
    

    def test_step(self, test_batch, batch_idx):
        input_ids, attention_mask, labels = test_batch

        out = self.model.generate(input_ids)

        return 0