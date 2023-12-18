import os
import sys
import json
import random
import pandas as pd

from tqdm import tqdm


def split_train(data):
    randomized_data = random.sample(data, len(data))

    index = int(len(randomized_data) * 0.9)

    train_split = randomized_data[:index]
    test_split = randomized_data[index:]

    return train_split, test_split


def open_data(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data


def save_to_csv(data, save_path):
    flattened_data = []
    for d in tqdm(data):
        title = d['title']
        for p in d['paragraphs']:
            context = p['context']
            for qas in p['qas']:
                question = qas['question']
                answer_text = qas['text']
                answer_start = qas['answer_start']
                flattened_data.append({'title': title, 'context': context, 'question': question, 'answer_text': answer_text, 'answer_start': answer_start})

    # convert to dataframe
    df = pd.DataFrame.from_dict(flattened_data, orient='columns')

    # drop duplicates
    df = df.drop_duplicates(subset=['context', 'question', 'answer_text'])

    # save to csv
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)


if __name__ == "__main__":
    random.seed(42)
    
    squad_train_json = open_data('Datasets/Raw/SQuAD2.0/train.json')
    squad_dev_json = open_data('Datasets/Raw/SQuAD2.0/dev.json') 
    squad_train_json, squad_val_json = split_train(squad_train_json)

    save_to_csv(squad_train_json, 'Datasets/CSV/SQuAD2.0/train.csv')
    save_to_csv(squad_val_json, 'Datasets/CSV/SQuAD2.0/validation.csv')
    save_to_csv(squad_dev_json, 'Datasets/CSV/SQuAD2.0/test.csv')

    tydiqa_train_json = open_data('Datasets/Raw/TyDiQA/train.json')
    tydiqa_dev_json = open_data('Datasets/Raw/TyDiQA/dev.json') 
    tydiqa_train_json, tydiqa_val_json = split_train(tydiqa_train_json)

    save_to_csv(tydiqa_train_json, 'Datasets/CSV/TyDiQA/train.csv')
    save_to_csv(tydiqa_val_json, 'Datasets/CSV/TyDiQA/validation.csv')
    save_to_csv(tydiqa_dev_json, 'Datasets/CSV/TyDiQA/test.csv')