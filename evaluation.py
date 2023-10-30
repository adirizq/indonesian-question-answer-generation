import os
import sys
import torch
import argparse
import evaluate
import pandas as pd

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from indobenchmark import IndoNLGTokenizer
from textwrap import dedent

from Models.qag_model import QAGModel

def encode(tokenizer, text, max_length):
    encoded = tokenizer(f'<s>{text}</s>', 
                        add_special_tokens=True, 
                        max_length=max_length,
                        padding="max_length",
                        truncation=True)
    return encoded['input_ids']


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Evaluation Parser')
    parser.add_argument('-t', '--type', choices=['Pipeline', 'Multitask', 'End2end'], required=True, help='Evaluation type')

    args = parser.parse_args()
    config = vars(args)

    bleu = evaluate.load('bleu')
    meteor = evaluate.load('meteor')
    rouge = evaluate.load('rouge')

    evaluation_data = pd.read_csv('Datasets/Processed/TyDiQA/prepared_test.csv')
    print(evaluation_data)
    sys.exit()

    if config['type'] == 'Pipeline':

        models = {
            'ae': [],
            'qg': []
        }
        
        root = 'Checkpoints'

        for task_type in os.listdir(root):
            if task_type in ['ae', 'qg']:
                path = os.path.join(root, task_type)
                for checkpoint in os.listdir(path):
                    model_info = checkpoint.split('_')
                    model = model_info[0]
                    dataset = model_info[1]
                    input_type = '_'.join(model_info[2:(model_info.index('to'))])
                    output_type = '_'.join(model_info[(model_info.index('to'))+1:])
                    ckpt_path = os.path.join(os.path.join(path, checkpoint), os.listdir(os.path.join(path, checkpoint))[0])
                    
                    models[task_type].append({'model': model, 'dataset': dataset, 'input_type': input_type, 'output_type': output_type, 'ckpt_path': ckpt_path})
    
        for ae in models['ae']:
            for qg in models['qg']:
                print(dedent(f'''
                -------------------------------------------------------------------------
                                            Pipeline Evaluation        
                -------------------------------------------------------------------------
                Name                | Value       
                -------------------------------------------------------------------------
                AE Model            | {ae['model']}
                AE Input Output     | {ae['input_type']}_to_{ae['output_type']}
                AE Train Dataset    | {ae['dataset']}
                -------------------------------------------------------------------------
                QG Model            | {qg['model']} 
                QG Input Output     | {qg['input_type']}_to_{ae['output_type']}
                QG Train Dataset    | {qg['dataset']}
                -------------------------------------------------------------------------
                '''))
                print('')
                # print(f'{qe["ckpt_path"]} : {qg["ckpt_path"]}')
                ae_model = QAGModel.load_from_checkpoint(ae['ckpt_path'])
                qg_model = QAGModel.load_from_checkpoint(qg['ckpt_path'])
            