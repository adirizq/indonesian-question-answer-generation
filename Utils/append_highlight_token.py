import os
import sys
import pandas as pd

from py4j.java_gateway import JavaGateway
from tqdm import tqdm


def highlight_answer(row):
    answer_end = row.answer_start + len(row.answer)
    answer_highlighted_context = row.context[:row.answer_start] + '<hl>' + row.context[row.answer_start:answer_end] + '<hl>' + row.context[answer_end:]
    return answer_highlighted_context


def highlight_sentence(row, isd):
    formatted_splitted_sentences = []
    splitted_sentences = isd.splitSentence(row['answer_highlighted_context'])
    for sentence in splitted_sentences:
        sentence = sentence.strip()
        sentence = sentence.replace(' .', '.')
        if '<hl>' in sentence:
            sentence = sentence.replace('<hl>', '')
            sentence = f'<hl>{sentence}<hl>'
        formatted_splitted_sentences.append(sentence)
        
    return ' '.join(formatted_splitted_sentences)


if __name__ == "__main__":

    tqdm.pandas()
    java = JavaGateway.launch_gateway(classpath="Utils/InaNLP.jar")
    isd = java.jvm.IndonesianNLP.IndonesianSentenceDetector()

    # Highlight answer in context
    data_df = pd.read_csv('Datasets/Deduplicated/gemini.csv')
    data_df['answer_highlighted_context'] = data_df.progress_apply(highlight_answer, axis=1)

    # Highlight sentence in context
    data_df['context_highlighted_sentence'] = data_df.progress_apply(lambda x: highlight_sentence(x, isd), axis=1)
    
    os.makedirs('Datasets/Highlighted', exist_ok=True)
    data_df.to_csv('Datasets/Highlighted/gemini.csv', index=False)
