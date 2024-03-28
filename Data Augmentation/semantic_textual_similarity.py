import os
import json
import torch
import pandas as pd
import argparse

from sentence_transformers import SentenceTransformer, util


def flatten(json_data):
    flat_data = []
    for item in json_data:
        context = item['context']
        for qa in item['qa']:
            flat_data.append({
                'context': context,
                'question': qa['question'],
                'answer': qa['answer']
            })
    return flat_data



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Similarity Parser')
    parser.add_argument('-fp', '--folder_path', type=str, required=True, help='Folder path of the dataset')
    parser.add_argument('-of', '--output_filename', type=str, required=True, help='Output filename of the calculated similarity scores')

    args = parser.parse_args()
    config = vars(args)

    folder_path = config['folder_path']
    output_filename = config['output_filename']
    accelerator = 'cuda' if torch.cuda.is_available() else 'cpu'


    model = SentenceTransformer('distiluse-base-multilingual-cased-v2', device=accelerator)


    print('[INFO] Combining data')

    # Load and combine all JSON files
    combined_data = []
    directory = folder_path
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            with open(os.path.join(directory, filename), 'r') as f:
                data = json.load(f)
                combined_data.extend(data)


    print('[INFO] Processing raw data')

    flat_data = flatten(combined_data)
    df = pd.DataFrame(flat_data)

    contexts = df['context'].values
    questions = df['question'].values
    answers = df['answer'].values
    question_answers = [f"{q} {a}" for q, a in zip(df['question'].values, df['answer'].values)]


    print('[INFO] Encoding data')

    contexts_embeddings = model.encode(contexts, convert_to_tensor=True, show_progress_bar=True, device=accelerator)
    questions_embeddings = model.encode(question_answers, convert_to_tensor=True, show_progress_bar=True, device=accelerator)
    question_answers_embeddings = model.encode(question_answers, convert_to_tensor=True, show_progress_bar=True, device=accelerator)


    print('[INFO] Cosine Similarity Calculation')

    cosine_scores_context_question = util.pytorch_cos_sim(contexts_embeddings, questions_embeddings)
    cosine_scores_context_question_answer = util.pytorch_cos_sim(contexts_embeddings, question_answers_embeddings)


    print('[INFO] Finished')

    data = {
        'context': [],
        'question': [],
        'answer': [],
        'cosine_score_context_question': [],
        'cosine_score_context_question_answer': []
    }

    for i in range(len(contexts)):
        data['context'].append(contexts[i])
        data['question'].append(questions[i])
        data['answer'].append(answers[i])
        data['cosine_score_context_question'].append(cosine_scores_context_question[i][i].item())
        data['cosine_score_context_question_answer'].append(cosine_scores_context_question_answer[i][i].item())


    # Save to CSV
    df = pd.DataFrame(data)
    df.to_csv(output_filename, index=False)


    # Print average
    print('Average cosine similarity between context and question:', sum(data['cosine_score_context_question']) / len(data['cosine_score_context_question']))
    print('Average cosine similarity between context and question+answer:', sum(data['cosine_score_context_question_answer']) / len(data['cosine_score_context_question_answer']))
