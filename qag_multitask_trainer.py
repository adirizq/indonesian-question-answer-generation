import os
import sys
import argparse
import torch
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

from pathlib import Path
from textwrap import dedent
from transformers import AutoModelForSeq2SeqLM
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar, EarlyStopping

from Utils.data_loader import MultiTaskQAGDataModule
from Utils.utils import ModelType, MultiTaskTestType, Tokenizer
from Models.qag_model import QAGMultiTaskModel

torch.multiprocessing.set_sharing_strategy('file_system')

def initialize_pretrained_model(name, tokenizer_len, max_length):
    model = AutoModelForSeq2SeqLM.from_pretrained(name)
    model.resize_token_embeddings(tokenizer_len)
    model.config.max_length = max_length
    model.config.early_stopping = True
    model.config.num_beams = 4

    return model


if __name__ == "__main__":
    seed_everything(seed=42, workers=True)

    parser = argparse.ArgumentParser(description='Trainer Parser')
    parser.add_argument('-ac', '--accelerator', choices=['gpu', 'cpu'], default='gpu', help='Accelerator for training')
    parser.add_argument('-rd', '--recreate_dataset', default=False, help='Recreate dataset for data loader', action=argparse.BooleanOptionalAction)
    parser.add_argument('-m', '--model_type', choices=['IndoBART', 'Flan-T5'], required=True, help='Pretrained model for training')
    parser.add_argument('-e', '--max_epochs', type=int, default=50, help='Max epochs for training')
    parser.add_argument('-b', '--batch_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-5, help='Learning rate for training')
    parser.add_argument('-ml', '--max_length', type=int, default=512, help='Max length for input sequence')

    args = parser.parse_args()
    config = vars(args)

    accelerator = config['accelerator']
    recreate = config['recreate_dataset']
    model_type = ModelType(config['model_type'])
    max_epochs = config['max_epochs']
    batch_size = config['batch_size']
    learning_rate = config['learning_rate']
    max_length = config['max_length']

    model_info = {
        ModelType.INDOBART : {'type': 'BART', 'tokenizer': 'indobenchmark/indobart-v2', 'pre_trained': 'indobenchmark/indobart-v2', 'lr_scheduler': True},
        ModelType.FLAN_T5: {'type': 'Flan-T5', 'tokenizer': 'google/flan-t5-small', 'pre_trained': 'google/flan-t5-small', 'lr_scheduler': False}
    }

    print(dedent(f'''
    -----------------------------------------------
          MultiTask Train Information        
    -----------------------------------------------
     Name                | Value       
    -----------------------------------------------
     Model Type          | {model_info[model_type]['type']}
     Pretrained Model    | {model_type}
     Batch Size          | {batch_size}
     Learning Rate       | {learning_rate}
     LR Scheduler        | {model_info[model_type]['lr_scheduler']}
     Input Max Length    | {max_length}  
    -----------------------------------------------

    '''))


    tokenizer = Tokenizer(model_type=model_type, max_length=max_length)
    pretrained_model = initialize_pretrained_model(model_info[model_type]['pre_trained'], tokenizer.tokenizer_len(), max_length)
    model = QAGMultiTaskModel(pretrained_model, model_info[model_type]['type'], tokenizer, model_info[model_type]['lr_scheduler'], learning_rate)


    dataset_csv_paths = [
            'Datasets/Splitted/gemini_wiki_train.csv', 
            'Datasets/Splitted/gemini_wiki_validation.csv', 
            'Datasets/Splitted/gemini_wiki_test.csv'
            ]
    
    dataset_tensor_paths = [
            f'Datasets/Tensor/gemini_wiki_train_multitask_{model_type.value}.pt', 
            f'Datasets/Tensor/gemini_wiki_validation_multitask_{model_type.value}.pt', 
            f'Datasets/Tensor/gemini_wiki_ae_test_multitask_{model_type.value}.pt',
            f'Datasets/Tensor/gemini_wiki_qg_test_multitask_{model_type.value}.pt',
            ]

    data_module = MultiTaskQAGDataModule(
        model_type=model_type,
        dataset_csv_paths=dataset_csv_paths,
        dataset_tensor_paths=dataset_tensor_paths,
        tokenizer=tokenizer,
        batch_size=batch_size,
        recreate=recreate
    )


    Path(f'./CSV Logs/MultiTask').mkdir(parents=True, exist_ok=True)
    Path(f'./Checkpoints/MultiTask').mkdir(parents=True, exist_ok=True)

    csv_logger = CSVLogger(f'CSV Logs', name=f'MultiTask/{model_type.value}')
    checkpoint_callback = ModelCheckpoint(dirpath=f'./Checkpoints/MultiTask/{model_type.value}', filename='{epoch}-{val_loss:.2f}', monitor='val_loss', mode='min')
    early_stop_callback = EarlyStopping(monitor='val_loss', min_delta=0.000, check_on_train_epoch_end=1, patience=3, mode='min')
    tqdm_progress_bar = TQDMProgressBar()

    trainer = Trainer(
        accelerator=accelerator,
        default_root_dir=f'./Checkpoints/MultiTask/{model_type.value}',
        callbacks=[checkpoint_callback, early_stop_callback, tqdm_progress_bar],
        logger=[csv_logger],
        max_epochs=max_epochs,
        deterministic=True 
    )


    print('\n[ Start Training ]\n')
    trainer.fit(model, datamodule=data_module)


    # print('\n[ Start AE Test ]\n')
    # data_module.setup('ae_test')
    # model.test_type = MultiTaskTestType('ae')
    # trainer.test(model, datamodule=data_module, ckpt_path='best')


    # print('\n[ Start QG Test ]\n')
    # data_module.setup('qg_test')
    # model.test_type = MultiTaskTestType('qg')
    # trainer.test(model, datamodule=data_module, ckpt_path='best')


    # print('\n[ Save Trained Model ]\n')

    # Path(f'./Trained Model/MultiTask/{model_type.value}').mkdir(parents=True, exist_ok=True)
