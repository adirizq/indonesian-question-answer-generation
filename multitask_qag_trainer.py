import os
import sys
import argparse
import torch
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

from pathlib import Path
from textwrap import dedent
from indobenchmark import IndoNLGTokenizer
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning import Trainer, seed_everything
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar, EarlyStopping

from Utils.data_loader import QAGMultiTaskDataModule
from Models.qag_model import QAGMultiTask


def initialize_pretrained_tokenizer(name):
    if 'IndoBART'.lower() in name.lower():
        tokenizer = IndoNLGTokenizer.from_pretrained(name)
    else:
        tokenizer = AutoTokenizer.from_pretrained(name, use_fast=False)
    
    tokenizer_len = len(tokenizer) + 1
    new_add_special_tokens = tokenizer.additional_special_tokens + ['<hl>']
    tokenizer.add_special_tokens({'additional_special_tokens': new_add_special_tokens})
    
    return tokenizer, tokenizer_len


def initialize_pretrained_model(name, tokenizer_len, max_length):
    model = AutoModelForSeq2SeqLM.from_pretrained(name)
    model.resize_token_embeddings(tokenizer_len)
    model.config.max_length = max_length
    model.config.early_stopping = True
    model.config.num_beams = 4

    return model



if __name__ == '__main__':

    seed_everything(seed=42, workers=True)

    parser = argparse.ArgumentParser(description='Trainer Parser')
    parser.add_argument('-p', '--pretrained_model', choices=['IndoBART', 'Flan-T5'], required=True, help='Pretrained model for training')

    args = parser.parse_args()
    config = vars(args)

    pretrained_model_type = config['pretrained_model']

    model_inf = {
        'IndoBART': {'type': 'BART', 'tokenizer': 'indobenchmark/indobart-v2', 'pre_trained': 'indobenchmark/indobart-v2', 'lr_scheduler': True},
        'Flan-T5': {'type': 'Flan-T5', 'tokenizer': 'google/flan-t5-base', 'pre_trained': 'google/flan-t5-base', 'lr_scheduler': False}
    }

    dataset = 'TyDiQA'

    tokenizer, tokenizer_len = initialize_pretrained_tokenizer(model_inf[pretrained_model_type]['tokenizer'])
    pretrained_model = initialize_pretrained_model(model_inf[pretrained_model_type]['pre_trained'], tokenizer_len, 512)

    data_module = QAGMultiTaskDataModule(dataset_name=dataset, 
                                         tokenizer=tokenizer, 
                                         model=pretrained_model_type,
                                         batch_size=8,
                                         max_length=512,
                                         recreate=True,
                                         test=False)
    
    model = QAGMultiTask(pretrained_model=pretrained_model,
                         tokenizer=tokenizer, 
                         lr_scheduler=model_inf[pretrained_model_type]['lr_scheduler'],
                         learning_rate=1e-5)
    
    Path(f'./CSV Logs/QAGMultiTask').mkdir(parents=True, exist_ok=True)
    Path(f'./Checkpoints/QAGMultiTask').mkdir(parents=True, exist_ok=True)

    csv_logger = CSVLogger(f'CSV Logs', name=f'QAGMultiTask/{pretrained_model_type}_{dataset}')
    checkpoint_callback = ModelCheckpoint(dirpath=f'./Checkpoints/QAGMultiTask/{pretrained_model_type}_{dataset}', filename='{epoch}-{val_loss:.2f}', monitor='val_loss', mode='min')
    early_stop_callback = EarlyStopping(monitor='val_loss', min_delta=0.000, check_on_train_epoch_end=1, patience=5, mode='min')
    tqdm_progress_bar = TQDMProgressBar()

    trainer = Trainer(
        accelerator='gpu',
        default_root_dir=f'./Checkpoints/QAGMultiTask/{pretrained_model_type}_{dataset}',
        callbacks=[checkpoint_callback, early_stop_callback, tqdm_progress_bar],
        logger=[csv_logger],
        max_epochs=20,
        log_every_n_steps=5,
        deterministic=True 
    )


    print('\n[ Start Training ]\n')
    trainer.fit(model, datamodule=data_module)


    print('\n[ Start AE Test ]\n')
    data_module.setup('ae_test')
    model.test_type = 'ae'
    trainer.test(model, datamodule=data_module)


    print('\n[ Start QG Test ]\n')
    data_module.setup('qg_test')
    model.test_type = 'qg'
    trainer.test(model, datamodule=data_module)

    
