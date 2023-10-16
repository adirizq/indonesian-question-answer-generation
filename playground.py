import sys
import pandas as pd


if __name__ == "__main__":
    data_squad = pd.concat([pd.read_csv('Datasets/Csv/SQuAD/train.csv'), pd.read_csv('Datasets/Csv/SQuAD/dev.csv')]) 
    # data_squad['context_senteces_length'] = data_squad['context'].apply(lambda x: len(x.split('. ')))

    new_data = []

    for index, row in data_squad.iterrows():
        context = row['context']
        answer_start = row['answer_start']
        answer_text = row['answer_text']

        context_answer = f"{context[:answer_start]}<hl>{answer_text}<hl>{context[(answer_start + len(answer_text)):]}"
        context_key_sentence = context_answer.split('. ')
        context_key_sentence = " ".join([ f'<hl>{s.replace("<hl>", "")}.<hl>' if "<hl>" in s else f"{s}." for s in context_key_sentence])

        new_data.append({'context': row['context'], 'context_key_sentence': context_key_sentence, 'context_answer': context_answer, 'question': row['question'], 'answer': row['answer_text']})
        
    df = pd.DataFrame.from_dict(new_data)
    print(df)