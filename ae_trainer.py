import os
import sys
import argparse
import torch
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

from pathlib import Path
from textwrap import dedent
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar, EarlyStopping

from Utils.data_loader import AnswerExtractionDataModule
from Models.answer_extraction import BartAnswerExtraction


if __name__ == "__main__":

    seed_everything(seed=42, workers=True)

    parser = argparse.ArgumentParser(description='Trainer Parser')
    parser.add_argument('-ac', '--accelerator', choices=['gpu', 'cpu'], default='gpu', help='Accelerator for training')
    parser.add_argument('-rd', '--recreate_dataset', default=False, help='Recreate dataset for data loader', action=argparse.BooleanOptionalAction)
    parser.add_argument('-t', '--test', default=False, help='Test mode', action=argparse.BooleanOptionalAction)
    parser.add_argument('-e', '--max_epochs', type=int, default=30, help='Max epochs for training')
    parser.add_argument('-i', '--input_type', choices=['context_key_sentence', 'context'], required=True, help='Input type for training')
    parser.add_argument('-o', '--output_type', choices=['context_answer', 'answer'], required=True, help='Output type training')

    args = parser.parse_args()
    config = vars(args)

    accelerator = config['accelerator']
    recreate_dataset = config['recreate_dataset']
    is_test = config['test']
    max_epochs = 1 if is_test else config['max_epochs']
    input_type = config['input_type']
    output_type = config['output_type']
    model_type = 'BART'
    pretrained_model_type = 'IndoBART'
    dataset = 'TyDiQA'
    batch_size = 10
    learning_rate = 1e-5
    max_length = 128 if is_test else 512

    if is_test:
        print(dedent(f'''

    -----------------------------------------------
                      TEST MODE        
    -----------------------------------------------
    '''))
    

    print(dedent(f'''
    -----------------------------------------------
          Answer Extraction Train Information        
    -----------------------------------------------
     Name                | Value       
    -----------------------------------------------
     Model Type          | {model_type}
     Pretrained Model    | {pretrained_model_type}
     Dataset             | {dataset}
     Input Type          | {input_type}
     Output Type         | {output_type}
     Batch Size          | {batch_size}
     Learning Rate       | {learning_rate}
     Input Max Length    | {max_length}  
    -----------------------------------------------

    '''))
    
    data_module = AnswerExtractionDataModule(dataset_name=dataset, 
                                             input_type=input_type, 
                                             output_type=output_type,
                                             batch_size=batch_size,
                                             max_length=max_length,
                                             recreate=recreate_dataset, 
                                             test=is_test
                                             )
    
    model = BartAnswerExtraction(tokenizer=data_module.get_tokenizer(), 
                                 max_length=max_length,
                                 learning_rate=learning_rate,
                                 input_type=input_type,
                                 output_type=output_type,
                                 )

    csv_logger = CSVLogger(f'csv_logs', name=f'ae_{pretrained_model_type}_{dataset}_{input_type}_to_{output_type}')
    checkpoint_callback = ModelCheckpoint(dirpath=f'./checkpoints/ae_{pretrained_model_type}_{dataset}_{input_type}_to_{output_type}', monitor='val_loss', mode='min')
    early_stop_callback = EarlyStopping(monitor='val_loss', min_delta=0.000, check_on_train_epoch_end=1, patience=3, mode='min')
    tqdm_progress_bar = TQDMProgressBar()

    trainer = Trainer(
        accelerator=accelerator,
        default_root_dir=f'./checkpoints/ae_{pretrained_model_type}_{dataset}_{input_type}_to_{output_type}',
        callbacks=[checkpoint_callback, early_stop_callback, tqdm_progress_bar],
        logger=[csv_logger],
        max_epochs=max_epochs,
        log_every_n_steps=5,
        deterministic=True 
    )

    trainer.fit(model, datamodule=data_module)
    trainer.test(datamodule=data_module, ckpt_path='best')

    Path(f'./pretrained/hf_ae_{pretrained_model_type}_{dataset}_{input_type}_to_{output_type}').mkdir(parents=True, exist_ok=True)
    Path(f'./pretrained/pt_ae_{pretrained_model_type}_{dataset}_{input_type}_to_{output_type}').mkdir(parents=True, exist_ok=True)

    model.save_pretrained(f'./pretrained/hf_ae_{pretrained_model_type}_{dataset}_{input_type}_to_{output_type}')
    torch.save(model.state_dict(), f'./pretrained/pt_ae_{pretrained_model_type}_{dataset}_{input_type}_to_{output_type}/pretrained.pth')
