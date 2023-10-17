import os
import sys
import argparse
import torch
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from textwrap import dedent

from Utils.data_loader import AnswerExtractionDataModule
from Models.answer_extraction import BartAnswerExtraction


if __name__ == "__main__":

    seed_everything(seed=42, workers=True)

    data_module = AnswerExtractionDataModule(dataset_name="TyDiQA", input_type="context", output_type='context_answer', batch_size=1)
    model = BartAnswerExtraction(tokenizer=data_module.get_tokenizer())

    trainer = Trainer(
        accelerator='gpu',
        max_epochs=1,
        log_every_n_steps=5,
        deterministic=True  # To ensure reproducible results
    )

    trainer.fit(model, datamodule=data_module)