import os
import sys
import pandas as pd

from py4j.java_gateway import JavaGateway
from tqdm import tqdm


def highlight_sentence_in_context(isd, row):
    formatted_splitted_sentences = []
    splitted_sentences = isd.splitSentence(row['context_highlighted_answer'])
    for sentence in splitted_sentences:
        sentence = sentence.strip()
        sentence = sentence.replace(' .', '.')
        if '<hl>' in sentence:
            sentence = sentence.replace('<hl>', '')
            sentence = f'<hl>{sentence}<hl>'
        formatted_splitted_sentences.append(sentence)
        
    return ' '.join(formatted_splitted_sentences)


def highlight_sentence(data_path, save_path, isd):

    tqdm.pandas()
    data_df = pd.read_csv(data_path)
    data_df['context_highlighted_sentence'] = data_df.progress_apply(lambda x: highlight_sentence_in_context(isd, x), axis=1)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    data_df.to_csv(save_path, index=False)


if __name__ == "__main__":

    java = JavaGateway.launch_gateway(classpath="Utils/InaNLP.jar")
    isd = java.jvm.IndonesianNLP.IndonesianSentenceDetector()

    # Prepare SQuAD2.0 
    highlight_sentence('Datasets/AE_QG/SQuAD2.0/train.csv', 'Datasets/AE_QG/SQuAD2.0/train.csv', isd)
    highlight_sentence('Datasets/AE_QG/SQuAD2.0/validation.csv', 'Datasets/AE_QG/SQuAD2.0/validation.csv', isd)
    highlight_sentence('Datasets/AE_QG/SQuAD2.0/test.csv', 'Datasets/AE_QG/SQuAD2.0/test.csv', isd)

    # Prepare TyDiQA
    highlight_sentence('Datasets/AE_QG/TyDiQA/train.csv', 'Datasets/AE_QG/TyDiQA/train.csv', isd)
    highlight_sentence('Datasets/AE_QG/TyDiQA/validation.csv', 'Datasets/AE_QG/TyDiQA/validation.csv', isd)
    highlight_sentence('Datasets/AE_QG/TyDiQA/test.csv', 'Datasets/AE_QG/TyDiQA/test.csv', isd)