import os
import sys
import argparse
import torch
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from textwrap import dedent

from Utils.data_loader import AnswerExtractionDataModule
from Models.answer_extraction import BartAnswerExtraction


if __name__ == "__main__":

    seed_everything(seed=42, workers=True)

    parser = argparse.ArgumentParser(description='Trainer Parser')
    parser.add_argument('--accelerator', choices=['gpu', 'cpu'], default='gpu', help='Use gpu for training')
    parser.add_argument('-rd', '--recreate_dataset', choices=[0, 1], default=0, help='Use gpu for training')

    args = parser.parse_args()
    config = vars(args)
    
    recreate = True if config['accelerator'] == 1 else False
    data_module = AnswerExtractionDataModule(dataset_name="TyDiQA", input_type="context", output_type='context_answer', batch_size=1, recreate=recreate)
    model = BartAnswerExtraction(tokenizer=data_module.get_tokenizer())

    trainer = Trainer(
        accelerator=config['accelerator'],
        max_epochs=5,
        log_every_n_steps=5,
        deterministic=True 
    )

    trainer.fit(model, datamodule=data_module)