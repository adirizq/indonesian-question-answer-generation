import os
import sys
import stanza
import pandas as pd

from tqdm import tqdm


def data_preparation(data_path, save_path):
    data = pd.read_csv(data_path)

    new_data = []

    for index, row in tqdm(data.iterrows(), total=data.shape[0]):
        context = row['context']
        answer_start = row['answer_start']
        answer_text = row['answer_text']

        context_answer = f"{context[:answer_start]}<hl>{answer_text}<hl>{context[(answer_start + len(str(answer_text))):]}"
        context_key_sentence = [sentence.text for sentence in nlp(context_answer).sentences]
        context_key_sentence = " ".join([ f'<hl>{s.replace("<hl>", "")}<hl>' if "<hl>" in s else s for s in context_key_sentence])

        new_data.append({'context': row['context'], 'context_key_sentence': context_key_sentence, 'context_answer': context_answer, 'question': row['question'], 'answer': row['answer_text']})
        
    new_data_df = pd.DataFrame.from_dict(new_data)
    new_data_df.to_csv(save_path, index=False)


if __name__ == "__main__":

    stanza.download('id')
    nlp = stanza.Pipeline('id', processors='tokenize')

    save_paths = ['Datasets/Processed/TyDiQA', 'Datasets/Processed/SQuAD']

    for path in save_paths:
        if not os.path.exists(path):
            os.makedirs(path)

    data_preparation('Datasets/Csv/SQuAD/train.csv', 'Datasets/Processed/SQuAD/prepared_train.csv')
    data_preparation('Datasets/Csv/SQuAD/dev.csv', 'Datasets/Processed/SQuAD/prepared_dev.csv')
    data_preparation('Datasets/Csv/SQuAD/test.csv', 'Datasets/Processed/SQuAD/prepared_test.csv')

    data_preparation('Datasets/Csv/TyDiQA/train.csv', 'Datasets/Processed/TyDiQA/prepared_train.csv')
    data_preparation('Datasets/Csv/TyDiQA/dev.csv', 'Datasets/Processed/TyDiQA/prepared_dev.csv')
    data_preparation('Datasets/Csv/TyDiQA/test.csv', 'Datasets/Processed/TyDiQA/prepared_test.csv')