import os
import sys
import json
import pandas as pd

from tqdm import tqdm

def save_to_csv(path, save_path):
    # open json
    with open(path, 'r') as f:
        data = json.load(f)

    # flattening json
    flattened_data = []
    for d in tqdm(data['data']):
        title = d['title']
        for p in d['paragraphs']:
            context = p['context']
            for qa in p['qas']:
                question = qa['question']
                qa_id = qa['id'] 
                is_impossible = qa['is_impossible'] if 'is_impossible' in qa else False
                for answer in qa['answers']:
                    answer_text = answer['text']
                    answer_start = answer['answer_start']
                    flattened_data.append({'id': qa_id, 'title': title, 'context': context, 'question': question, 'answer_text': answer_text, 'answer_start': answer_start, 'is_impossible': is_impossible})

    # convert to dataframe
    df = pd.DataFrame.from_dict(flattened_data, orient='columns')

    # get only indonesian data and drop duplicates
    df = df.drop_duplicates(subset=['context', 'question', 'answer_text'])

    # save to csv
    df.to_csv(save_path, index=False)


if __name__ == "__main__":
    save_paths = ['Datasets/Csv/TyDiQA', 'Datasets/Csv/SQuAD']

    for path in save_paths:
        if not os.path.exists(path):
            os.makedirs(path)

    save_to_csv('Datasets/Raw/SQuAD/train.json', 'Datasets/Csv/SQuAD/train.csv')
    save_to_csv('Datasets/Raw/SQuAD/dev.json', 'Datasets/Csv/SQuAD/dev.csv')
    
    save_to_csv('Datasets/Raw/TyDiQA/train.json', 'Datasets/Csv/TyDiQA/train.csv')
    save_to_csv('Datasets/Raw/TyDiQA/dev.json', 'Datasets/Csv/TyDiQA/dev.csv')