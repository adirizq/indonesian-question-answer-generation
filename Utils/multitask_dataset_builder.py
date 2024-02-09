import os
import sys
import pandas as pd

from tqdm import tqdm


def build_multitask_dataset(data_path, save_path):
    data_dict = {
        'task': [],
        'input': [],
        'target': []
    }

    data_df = pd.read_csv(data_path)

    for index, row in tqdm(data_df.iterrows(), total=len(data_df)):
        # Answer Extraction Task
        data_dict['task'].append('AE')
        data_dict['input'].append('[AE]: ' + row['context_highlighted_sentence'])
        data_dict['target'].append(row['context_highlighted_answer'])

        # Question Generation Task
        data_dict['task'].append('QG')
        data_dict['input'].append('[QG]: ' + row['context_highlighted_answer'])
        data_dict['target'].append(row['question'])

    multitask_df = pd.DataFrame(data_dict)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    multitask_df.to_csv(save_path, index=False)


if __name__ == "__main__":

    # Prepare SQuAD2.0 
    build_multitask_dataset('Datasets/AE_QG/SQuAD2.0/train.csv', 'Datasets/MultiTask/SQuAD2.0/train.csv')
    build_multitask_dataset('Datasets/AE_QG/SQuAD2.0/validation.csv', 'Datasets/MultiTask/SQuAD2.0/validation.csv')
    build_multitask_dataset('Datasets/AE_QG/SQuAD2.0/test.csv', 'Datasets/MultiTask/SQuAD2.0/test.csv')

    # Prepare TyDiQA
    build_multitask_dataset('Datasets/AE_QG/TyDiQA/train.csv', 'Datasets/MultiTask/TyDiQA/train.csv')
    build_multitask_dataset('Datasets/AE_QG/TyDiQA/validation.csv', 'Datasets/MultiTask/TyDiQA/validation.csv')
    build_multitask_dataset('Datasets/AE_QG/TyDiQA/test.csv', 'Datasets/MultiTask/TyDiQA/test.csv')
    
