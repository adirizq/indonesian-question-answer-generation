import os
import sys
import argparse
import torch
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar, EarlyStopping
from pytorch_lightning.loggers import CSVLogger
from textwrap import dedent

from Utils.data_loader import AnswerExtractionDataModule
from Models.answer_extraction import BartAnswerExtraction


if __name__ == "__main__":

    seed_everything(seed=42, workers=True)

    parser = argparse.ArgumentParser(description='Trainer Parser')
    parser.add_argument('-ac', '--accelerator', choices=['gpu', 'cpu'], default='gpu', help='Use gpu for training')
    parser.add_argument('-rd', '--recreate_dataset', default=False, help='Recreate dataset', action=argparse.BooleanOptionalAction)
    parser.add_argument('--test', default=False, action=argparse.BooleanOptionalAction)

    args = parser.parse_args()
    config = vars(args)

    if config['test']:
        print('\n[ Test Mode ]\n')
    
    data_module = AnswerExtractionDataModule(dataset_name="TyDiQA", input_type="context", output_type='context_answer', batch_size=1, recreate=config['recreate_dataset'], test=config['test'])
    model = BartAnswerExtraction(tokenizer=data_module.get_tokenizer())

    # Initialize callbacks and progressbar
    csv_logger = CSVLogger('csv_logs', name=f'logs')
    checkpoint_callback = ModelCheckpoint(dirpath=f'./checkpoints', monitor='val_loss', mode='min')
    early_stop_callback = EarlyStopping(monitor='val_loss', min_delta=0.00, check_on_train_epoch_end=1, patience=3, mode='min')
    tqdm_progress_bar = TQDMProgressBar()

    trainer = Trainer(
        accelerator=config['accelerator'],
        default_root_dir=f'./checkpoints',
        callbacks=[checkpoint_callback, early_stop_callback, tqdm_progress_bar],
        logger=[csv_logger],
        max_epochs= 1 if config['test'] else 5,
        log_every_n_steps=5,
        deterministic=True 
    )

    trainer.fit(model, datamodule=data_module)
    trainer.test(datamodule=data_module, ckpt_path='best')
