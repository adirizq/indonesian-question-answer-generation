import os
import re
import json
import nltk
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from py4j.java_gateway import JavaGateway
from tqdm import tqdm


def flatten_json(json_data):
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


def find_answer_index(row, vectorizer):
    context = row.context
    question = row.question
    answer = row.answer

    # Find all occurrences of the answer in the context
    start_positions = [match.start() for match in re.finditer(re.escape(answer), context)]

    # If there is only one occurrence, return its position directly
    if len(start_positions) == 1:
        return start_positions[0]
    else:
        # For each occurrence, calculate the cosine similarity with the question
        similarities = []
        for start_pos in start_positions:
            end_pos = start_pos + len(answer)
            surrounding_context = context[max(0, start_pos - len(question)):min(len(context), end_pos + len(question))]
            tfidf_matrix = vectorizer.fit_transform([question, surrounding_context])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
            similarities.append(similarity[0][0])

        # Find the position with the highest similarity
        max_similarity_index = similarities.index(max(similarities))

        return start_positions[max_similarity_index]


if __name__ == "__main__":

    tqdm.pandas()
    vectorizer = TfidfVectorizer()

    # Load and combine all JSON files
    combined_data = []
    directory = './Datasets/Augmentation/Gemini' 
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            with open(os.path.join(directory, filename), 'r') as f:
                data = json.load(f)
                combined_data.extend(data)

    # Flatten the data
    flat_data = flatten_json(combined_data)

    # Convert to a DataFrame
    df = pd.DataFrame(flat_data)

    # Remove duplicates
    df = df.drop_duplicates()

    # Remove leading and trailing whitespaces
    df.context = df.context.str.strip()
    df.question = df.question.str.strip()
    df.answer = df.answer.str.strip()

    # Find answer position
    df['answer_start'] = df.progress_apply(lambda x: find_answer_index(x, vectorizer), axis=1)

    # Save data in CSV format
    df.to_csv('./Datasets/Deduplicated/gemini.csv', index=False)

    # Convert back to JSON format and save
    json_data = []
    for context, group in df.groupby('context'):
        qa_list = group[['question', 'answer', 'answer_start']].to_dict('records')
        json_data.append({
            'context': context,
            'qa': qa_list
        })

    with open('./Datasets/Deduplicated/gemini.json', 'w') as f:
        json.dump(json_data, f)

    print('total qa:', len(df))
    print('total context:', len(json_data))