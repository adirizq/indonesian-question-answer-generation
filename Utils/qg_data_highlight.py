import os
import sys
import pandas as pd


def highlight_answer_in_context(context, answer, start_index):
    end_index = start_index + len(answer)
    highlighted_context = context[:start_index] + '<hl>' + context[start_index:end_index] + '<hl>' + context[end_index:]
    return highlighted_context.strip()


def highlight_answer(data_path, save_path):

    data_df = pd.read_csv(data_path)
    data_df['context_highlighted_answer'] = data_df.apply(lambda x: highlight_answer_in_context(x['context'], x['answer_text'], x['answer_start']), axis=1)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    data_df.to_csv(save_path, index=False)


if __name__ == "__main__":

    # Prepare SQuAD2.0 
    highlight_answer('Datasets/CSV/SQuAD2.0/train.csv', 'Datasets/AE_QG/SQuAD2.0/train.csv')
    highlight_answer('Datasets/CSV/SQuAD2.0/validation.csv', 'Datasets/AE_QG/SQuAD2.0/validation.csv')
    highlight_answer('Datasets/CSV/SQuAD2.0/test.csv', 'Datasets/AE_QG/SQuAD2.0/test.csv')

    # Prepare TyDiQA
    highlight_answer('Datasets/CSV/TyDiQA/train.csv', 'Datasets/AE_QG/TyDiQA/train.csv')
    highlight_answer('Datasets/CSV/TyDiQA/validation.csv', 'Datasets/AE_QG/TyDiQA/validation.csv')
    highlight_answer('Datasets/CSV/TyDiQA/test.csv', 'Datasets/AE_QG/TyDiQA/test.csv')