import os
import sys
import json
import stanza
import shutil
import zipfile
import pandas as pd

from tqdm import tqdm

# Data preparation function
# Read json file
# Modify: remove qas without answer, refactor json structure
# Save modified json
def prepare_data(data_path, save_path):

    modified_data = []

    with open(data_path, 'r') as f:
        data = json.load(f)
    
    for p_key, p_value in tqdm(data['paragraphs'].items(), desc=f'Preparing {data_path}'):
        
        article_dict = {
            'title': '',
            'paragraphs': [],
        }

        article_dict['title'] = data['title'][p_key]

        for paragraph in p_value:

            paragraph_dict = {
                'context': '',
                'qas': []
            }

            paragraph_dict['context'] = paragraph['context']

            for qas in paragraph['qas']:

                qas_dict = {'question': qas['question']}
                
                if 'is_impossible' in qas:
                    if not qas['is_impossible']:
                        if int(qas['indonesian_answers'][0]['answer_start']) != -1:
                            qas_dict.update(qas['indonesian_answers'][0])
                            paragraph_dict['qas'].append(qas_dict)
                else:
                    if int(qas['indonesian_answers'][0]['answer_start']) != -1:
                        qas_dict.update(qas['indonesian_answers'][0])
                        paragraph_dict['qas'].append(qas_dict)

            if len(paragraph_dict['qas']) > 0:
                article_dict['paragraphs'].append(paragraph_dict)

        modified_data.append(article_dict)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, "w") as outfile:
        json.dump(modified_data, outfile)

    

if __name__ == "__main__":

    # Extract qag_dataset.zip
    with zipfile.ZipFile('Datasets/qag_dataset.zip', 'r') as zip_ref:
        zip_ref.extractall('Datasets/Temp')

    # Prepare SQuAD2.0 and TyDiQA
    prepare_data('Datasets/Temp/train-v2.0-translated_fixed_enhanced.json', 'Datasets/Raw/SQuAD2.0/train.json')
    prepare_data('Datasets/Temp/dev-v2.0-translated_fixed_enhanced.json', 'Datasets/Raw/SQuAD2.0/dev.json')
    prepare_data('Datasets/Temp/tydiqa-goldp-v1.1-train-indonesian_prepared_enhanced.json', 'Datasets/Raw/TyDiQA/train.json')
    prepare_data('Datasets/Temp/tydiqa-goldp-v1.1-dev-indonesian_prepared_enhanced.json', 'Datasets/Raw/TyDiQA/dev.json')
    
    # Delete temp extracted data
    shutil.rmtree('Datasets/Temp')