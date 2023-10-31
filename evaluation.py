import os
import sys
import torch
import argparse
import evaluate
import pandas as pd

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from indobenchmark import IndoNLGTokenizer
from textwrap import dedent
from tqdm import tqdm

from Models.qag_model import QAGModel



def initialize_pretrained_tokenizer(name):
    if 'IndoBART'.lower() in name.lower():
        tokenizer = IndoNLGTokenizer.from_pretrained(name)
    else:
        tokenizer = AutoTokenizer.from_pretrained(name)
    
    new_add_special_tokens = tokenizer.additional_special_tokens + ['<hl>']
    tokenizer.add_special_tokens({'additional_special_tokens': new_add_special_tokens})
    
    return tokenizer


def exact_match_evaluation(predictions, references):
        assert len(predictions) == len(references), "The number of predictions and references should be the same"
        
        exact_matches = 0
        for pred, refs in zip(predictions, references):
            if pred in refs:
                exact_matches += 1
        
        return exact_matches / len(predictions)


def encode(tokenizer, text, max_length):
    encoded = tokenizer(f'<s>{text}</s>', 
                        add_special_tokens=True, 
                        max_length=max_length,
                        padding="max_length",
                        truncation=True)
    return encoded['input_ids']


def decode_clean(tokenizer, text):
        decoded = tokenizer.decode(text).replace('<pad>', '').replace('<s>', '').replace('</s>', '')
        decoded = decoded.split('<hl>')[1] if '<hl>' in decoded else decoded
        return decoded.strip()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Evaluation Parser')
    parser.add_argument('-t', '--type', choices=['Pipeline', 'Multitask', 'End2end'], required=True, help='Evaluation type')

    args = parser.parse_args()
    config = vars(args)

    bleu = evaluate.load('bleu')
    meteor = evaluate.load('meteor')
    rouge = evaluate.load('rouge')

    evaluation_data = pd.read_csv('Datasets/Processed/TyDiQA/prepared_test.csv')
    evaluation_data['qa'] = evaluation_data.apply(lambda x: f'question: {x["question"]} <sep> answer: {x["answer"]}', axis=1)

    labels_dict = {}

    for index, data in evaluation_data.iterrows():
        if data['context'] in labels_dict:
            labels_dict[data['context']].append(data['qa'])
        else:
            labels_dict[data['context']] = [data['qa']]
    
    evaluation_data['labels'] = evaluation_data['context'].apply(lambda x: labels_dict[x])

    model_inf = {
        'IndoBART': {'type': 'BART', 'tokenizer': 'indobenchmark/indobart-v2'}
    }

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
                    model_path = os.path.join(os.path.join(path, checkpoint), 'hf')
                    
                    models[task_type].append({'model': model, 'dataset': dataset, 'input_type': input_type, 'output_type': output_type, 'model_path': model_path})

                    print(model_path)
    
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

                ae_model = AutoModelForSeq2SeqLM.from_pretrained(ae['model_path']).cuda()
                qg_model = AutoModelForSeq2SeqLM.from_pretrained(qg['model_path']).cuda()

                ae_tokenizer = initialize_pretrained_tokenizer(model_inf[ae['model']]['tokenizer'])
                qg_tokenizer = initialize_pretrained_tokenizer(model_inf[qg['model']]['tokenizer'])

                predicted = []

                for index, data in tqdm(evaluation_data.iterrows(), total=evaluation_data.shape[0]):
                    ae_input_ids = encode(ae_tokenizer, data[ae['input_type']], ae_model.config.max_length)
                    ae_pred = ae_model.generate(torch.tensor([ae_input_ids]).cuda())[0]
                    ae_pred_clean = decode_clean(ae_tokenizer, ae_pred)

                    qg_input_ids = encode(qg_tokenizer, ae_tokenizer.decode(ae_pred).strip(), qg_model.config.max_length)
                    qg_pred = decode_clean(qg_tokenizer, qg_model.generate(torch.tensor([qg_input_ids]).cuda())[0])

                    predicted.append(f'question: {qg_pred}, answer: {ae_pred_clean}')
                
                score_exact_match = exact_match_evaluation(predictions=predicted, references=evaluation_data['labels'].to_list())
                score_bleu = bleu.compute(predictions=predicted, references=evaluation_data['labels'].to_list())["bleu"]
                score_meteor = meteor.compute(predictions=predicted, references=evaluation_data['labels'].to_list())["meteor"]
                score_rouge = rouge.compute(predictions=predicted, references=evaluation_data['labels'].to_list())
                score_rouge1 = score_rouge['rouge1']
                score_rouge2 = score_rouge['rouge2']
                score_rougeL = score_rouge['rougeL']
                score_rougeLsum = score_rouge['rougeLsum']

                print(dedent(f'''
                -------------------------------------------------------------------------
                                      Pipeline Evaluation Result     
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
                Exact Match         | {score_exact_match}
                Bleu                | {score_bleu}
                Meteor              | {score_meteor}
                Rouge1              | {score_rouge1}
                Rouge2              | {score_rouge2}
                RougeL              | {score_rougeL}
                RougeLsum           | {score_rougeLsum}
                -------------------------------------------------------------------------

                '''))