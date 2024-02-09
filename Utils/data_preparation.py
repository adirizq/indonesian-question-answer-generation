import os
import pandas as pd


def prepare_data(data_path, save_path):

    data_df = pd.read_csv(data_path)
    context_answer_data_df = data_df.groupby('context')['answer_text'].apply('<sep>'.join).reset_index()
    context_question_data_df = data_df.groupby('context')['question'].apply('<sep>'.join).reset_index()

    qag_data_df = pd.merge(context_answer_data_df, context_question_data_df, on='context')
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    qag_data_df.to_csv(save_path, index=False)


if __name__ == "__main__":

    # Prepare SQuAD2.0 
    prepare_data('Datasets/CSV/SQuAD2.0/train.csv', 'Datasets/Processed/SQuAD2.0/train.csv')
    prepare_data('Datasets/CSV/SQuAD2.0/validation.csv', 'Datasets/Processed/SQuAD2.0/validation.csv')
    prepare_data('Datasets/CSV/SQuAD2.0/test.csv', 'Datasets/Processed/SQuAD2.0/test.csv')

    # Prepare TyDiQA
    prepare_data('Datasets/CSV/TyDiQA/train.csv', 'Datasets/Processed/TyDiQA/train.csv')
    prepare_data('Datasets/CSV/TyDiQA/validation.csv', 'Datasets/Processed/TyDiQA/validation.csv')
    prepare_data('Datasets/CSV/TyDiQA/test.csv', 'Datasets/Processed/TyDiQA/test.csv')