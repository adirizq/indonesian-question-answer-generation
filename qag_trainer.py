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

from Utils.data_loader import AnswerExtractionDataModule
from Models.qag_model import QAGModel


def initialize_pretrained_tokenizer(name):
    if 'IndoBART'.lower() in name.lower():
        tokenizer = IndoNLGTokenizer.from_pretrained(name)
    else:
        tokenizer = AutoTokenizer.from_pretrained(name)
    
    tokenizer_len = len(tokenizer) + 1
    new_add_special_tokens = tokenizer.additional_special_tokens + ['<hl>']
    tokenizer.add_special_tokens({'additional_special_tokens': new_add_special_tokens})
    
    return tokenizer, tokenizer_len


def initialize_pretrained_model(name, tokenizer_len, max_length):
    model = AutoModelForSeq2SeqLM(name)
    model.resize_token_embeddings(tokenizer_len)
    model.config.max_length = max_length
    model.config.early_stopping = True
    model.config.num_beams = 4

    return model


if __name__ == "__main__":

    seed_everything(seed=42, workers=True)

    parser = argparse.ArgumentParser(description='Trainer Parser')
    parser.add_argument('-t', '--test', default=False, help='Enable test mode', action=argparse.BooleanOptionalAction)
    parser.add_argument('-ac', '--accelerator', choices=['gpu', 'cpu'], default='gpu', help='Accelerator for training')
    parser.add_argument('-rd', '--recreate_dataset', default=False, help='Recreate dataset for data loader', action=argparse.BooleanOptionalAction)
    parser.add_argument('-d', '--dataset', choices=['TyDiQA', 'SQuAD'], required=True, help='Dataset for training')
    parser.add_argument('-m', '--model_task', choices=['ae', 'qg'], required=True, help='Model task for training')
    parser.add_argument('-p', '--pretrained_model', choices=['IndoBART', 'Flan-T5'], required=True, help='Pretrained model for training')
    parser.add_argument('-i', '--input_type', choices=['context_key_sentence', 'context', 'context_answer'], required=True, help='Input type for training')
    parser.add_argument('-o', '--output_type', choices=['context_answer', 'answer', 'question'], required=True, help='Output type training')
    parser.add_argument('-e', '--max_epochs', type=int, default=50, help='Max epochs for training')

    args = parser.parse_args()
    config = vars(args)

    accelerator = config['accelerator']
    recreate_dataset = config['recreate_dataset']
    is_test = config['test']
    max_epochs = 1 if is_test else config['max_epochs']
    model_task = config['model_task']
    input_type = config['input_type']
    output_type = config['output_type']
    pretrained_model_type = config['pretrained_model']
    dataset = config['dataset']
    batch_size = 8
    learning_rate = 1e-5
    max_length = 128 if is_test else 512

    model_task_inf = {
        'ae': 'Answer Extraction',
        'qg': 'Question Generator'
    }

    model_inf = {
        'IndoBART': {'type': 'BART', 'tokenizer': 'indobenchmark/indobart-v2', 'pre_trained': 'indobenchmark/indobart-v2', 'lr_scheduler': True}
    }

    if is_test:
        print(dedent(f'''

    -----------------------------------------------
                      TEST MODE        
    -----------------------------------------------
    '''))
    

    print(dedent(f'''
    -----------------------------------------------
          {model_task_inf[model_task]} Train Information        
    -----------------------------------------------
     Name                | Value       
    -----------------------------------------------
     Model Type          | {model_inf[pretrained_model_type]['type']}
     Pretrained Model    | {pretrained_model_type}
     Dataset             | {dataset}
     Input Type          | {input_type}
     Output Type         | {output_type}
     Batch Size          | {batch_size}
     Learning Rate       | {learning_rate}
     LR Scheduler        | {model_inf[pretrained_model_type]['lr_scheduler']}
     Input Max Length    | {max_length}  
    -----------------------------------------------

    '''))


    print('\n[ Initializing Pretrained Tokenizer ]\n')
    tokenizer, tokenizer_len = initialize_pretrained_tokenizer(model_inf[pretrained_model_type]['pre_trained'])
    print('\n[ Initialize Completed ]\n')

    print('\n[ Initializing Pretrained Model ]\n')
    pretrained_model = initialize_pretrained_model(model_inf[pretrained_model_type]['pre_trained'], tokenizer_len, max_length)
    print('\n[ Initialize Completed ]\n')


    print('\n[ Initializing Data Module & Model ]\n')

    data_module = AnswerExtractionDataModule(dataset_name=dataset, 
                                             input_type=input_type,
                                             tokenizer=tokenizer, 
                                             output_type=output_type,
                                             batch_size=batch_size,
                                             max_length=max_length,
                                             recreate=recreate_dataset, 
                                             test=is_test
                                             )
    
    model = QAGModel(model=pretrained_model,
                     tokenizer=data_module.get_tokenizer(), 
                     lr_scheduler=model_inf[pretrained_model_type]['lr_scheduler'],
                     learning_rate=learning_rate,
                     input_type=input_type,
                     output_type=output_type,
                     model_task=model_task
                     )
    
    print('\n[ Initialize Completed ]\n')


    Path(f'./CSV Logs/{model_task}').mkdir(parents=True, exist_ok=True)
    Path(f'./Checkpoints/{model_task}').mkdir(parents=True, exist_ok=True)

    csv_logger = CSVLogger(f'CSV Logs', name=f'{model_task}/{pretrained_model_type}_{dataset}_{input_type}_to_{output_type}')
    checkpoint_callback = ModelCheckpoint(dirpath=f'./Checkpoints/{model_task}/{pretrained_model_type}_{dataset}_{input_type}_to_{output_type}', monitor='val_loss', mode='min')
    early_stop_callback = EarlyStopping(monitor='val_loss', min_delta=0.000, check_on_train_epoch_end=1, patience=5, mode='min')
    tqdm_progress_bar = TQDMProgressBar()

    trainer = Trainer(
        accelerator=accelerator,
        default_root_dir=f'./Checkpoints/{model_task}/{pretrained_model_type}_{dataset}_{input_type}_to_{output_type}',
        callbacks=[checkpoint_callback, early_stop_callback, tqdm_progress_bar],
        logger=[csv_logger],
        max_epochs=max_epochs,
        log_every_n_steps=5,
        deterministic=True 
    )


    print('\n[ Start Training ]\n')

    trainer.fit(model, datamodule=data_module)
    trainer.test(datamodule=data_module, ckpt_path='best')


    print('\n[ Saving Trained Model ]\n')

    Path(f'./Pretrained/{model_task}/{pretrained_model_type}_{dataset}_{input_type}_to_{output_type}/hf').mkdir(parents=True, exist_ok=True)
    Path(f'./Pretrained/{model_task}/{pretrained_model_type}_{dataset}_{input_type}_to_{output_type}/pt').mkdir(parents=True, exist_ok=True)

    model.save_pretrained(f'Pretrained/{model_task}/{pretrained_model_type}_{dataset}_{input_type}_to_{output_type}/hf')
    torch.save(model.state_dict(), f'Pretrained/{model_task}/{pretrained_model_type}_{dataset}_{input_type}_to_{output_type}/pt/pretrained.pth')
