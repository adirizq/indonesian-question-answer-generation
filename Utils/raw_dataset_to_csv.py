import os
import sys
import json
import random
import pandas as pd

from tqdm import tqdm


def split_train(data):
    randomized_data = random.sample(data, len(data))

    index = int(len(randomized_data) * 0.8)

    train_split = randomized_data[:index]
    test_split = randomized_data[index:]

    return train_split, test_split


def open_data(path, filter_indonesian = False):
    with open(path, 'r') as f:
        data = json.load(f)

    if filter_indonesian:
        filtered_data = []

        for d in tqdm(data['data']):
            for p in d['paragraphs']:
                for qa in p['qas']:
                    if 'indonesian' in qa['id']:
                        filtered_data.append(d)

        return filtered_data

    return data['data']


def save_to_csv(data, save_path):
    flattened_data = []
    for d in tqdm(data):
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

    squad_train_json = open_data('Datasets/Raw/SQuAD/train.json')
    squad_dev_json = open_data('Datasets/Raw/SQuAD/dev.json') 
    squad_train_json, squad_test_json = split_train(squad_train_json)

    save_to_csv(squad_train_json, 'Datasets/Csv/SQuAD/train.csv')
    save_to_csv(squad_dev_json, 'Datasets/Csv/SQuAD/dev.csv')
    save_to_csv(squad_test_json, 'Datasets/Csv/SQuAD/test.csv')

    tydiqa_train_json = open_data('Datasets/Raw/TyDiQA/train.json', True)
    tydiqa_dev_json = open_data('Datasets/Raw/TyDiQA/dev.json', True) 
    tydiqa_train_json, tydiqa_test_json = split_train(tydiqa_train_json)

    save_to_csv(tydiqa_train_json, 'Datasets/Csv/TyDiQA/train.csv')
    save_to_csv(tydiqa_dev_json, 'Datasets/Csv/TyDiQA/dev.csv')
    save_to_csv(tydiqa_test_json, 'Datasets/Csv/TyDiQA/test.csv')