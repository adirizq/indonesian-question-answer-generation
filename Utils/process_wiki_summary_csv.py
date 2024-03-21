import os
import json
import pandas as pd

from py4j.java_gateway import JavaGateway
from tqdm import tqdm


if __name__ == '__main__':

    java = JavaGateway.launch_gateway(classpath="Utils\InaNLP.jar")
    isd = java.jvm.IndonesianNLP.IndonesianSentenceDetector()
    
    data_df = pd.read_csv('Datasets/wiki_summary.csv')

    data_dict = {
        'data': []
    }

    for idx, row in tqdm(data_df.iterrows(), total=data_df.shape[0]):

        temp_title = row['title']
        temp_paragraphs = []

        paragraphs = row['summary'].split('\n')

        for paragraph in paragraphs:
            splitted_sentences = isd.splitSentence(paragraph)
            if len(splitted_sentences) > 1:
                temp_paragraphs.append(paragraph)

        if len(temp_paragraphs) > 0:
            data_dict['data'].append({
                'title': temp_title,
                'paragraphs': [{'context': p} for p in temp_paragraphs]
            })

    chunks = [data_dict['data'][i:i+5000] for i in range(0, len(data_dict['data']), 5000)]

    for i, chunk in enumerate(chunks):
        filename = f'Datasets/Wiki Summary/wiki_summary_chunk_{i+1}.json'
        with open(filename, 'w') as f:
            json.dump({'data': chunk}, f)
