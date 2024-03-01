import os
import re
import sys
import json
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from py4j.java_gateway import JavaGateway
from tqdm import tqdm


class Highlighter:
    def __init__(self):
        self.java = JavaGateway.launch_gateway(classpath="Utils/InaNLP.jar")
        self.isd = self.java.jvm.IndonesianNLP.IndonesianSentenceDetector()

    def highlight_answer(self, row):
        answer_end = row.answer_start + len(row.answer)
        answer_highlighted_context = row.context[:row.answer_start] + '<hl>' + row.context[row.answer_start:answer_end] + '<hl>' + row.context[answer_end:]
        return answer_highlighted_context

    def highlight_sentence(self, row):
        formatted_splitted_sentences = []
        splitted_sentences = self.isd.splitSentence(row['answer_highlighted_context'])
        for sentence in splitted_sentences:
            sentence = sentence.strip()
            sentence = sentence.replace(' .', '.')
            if '<hl>' in sentence:
                sentence = sentence.replace('<hl>', '')
                sentence = f'<hl>{sentence}<hl>'
            formatted_splitted_sentences.append(sentence)

        return ' '.join(formatted_splitted_sentences)


class AugmentedDataFormatter:
    def __init__(self) -> None:
        self.vectorizer = TfidfVectorizer()
        
    def remove_quotes(self, context):
        context = context.strip()
        if (context.startswith('"') and context.endswith('"')) or (context.startswith("'") and context.endswith("'")):
            return context[1:-1]
        return context


    def flatten(self, json_data):
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


    def find_answer_index(self, row):
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
                tfidf_matrix = self.vectorizer.fit_transform([question, surrounding_context])
                similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
                similarities.append(similarity[0][0])

            # Find the position with the highest similarity
            max_similarity_index = similarities.index(max(similarities))

            return start_positions[max_similarity_index]


    def process_raw_data(self, folder_path, output_filename):
        tqdm.pandas()

        os.makedirs('Datasets/Deduplicated', exist_ok=True)
        os.makedirs('Datasets/Highlighted', exist_ok=True)
        os.makedirs('Datasets/Splitted', exist_ok=True)

        output_deduplicated_csv_path = os.path.join(f'Datasets/Deduplicated/{output_filename}.csv')
        output_deduplicated_json_path = os.path.join(f'Datasets/Deduplicated/{output_filename}.json')

        output_highlighted_csv_path = os.path.join(f'Datasets/Highlighted/{output_filename}.csv')

        output_train_csv_path = os.path.join(f'Datasets/Splitted/{output_filename}_train.csv')
        output_test_csv_path = os.path.join(f'Datasets/Splitted/{output_filename}_test.csv')
        output_validation_csv_path = os.path.join(f'Datasets/Splitted/{output_filename}_validation.csv')


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

        # Flatten the data
        flat_data = self.flatten(combined_data)

        # Convert to a DataFrame
        df = pd.DataFrame(flat_data)

        # Remove duplicates
        df = df.drop_duplicates()

        # Remove leading and trailing whitespaces and quotes
        df.context = df.context.apply(self.remove_quotes)
        df.question = df.question.apply(self.remove_quotes)
        df.answer = df.answer.apply(self.remove_quotes)

        # Find answer position
        df['answer_start'] = df.progress_apply(self.find_answer_index, axis=1)

        # Save data in CSV format
        df.to_csv(output_deduplicated_csv_path, index=False)

        # Convert back to JSON format and save
        json_data = []
        for context, group in df.groupby('context'):
            qa_list = group[['question', 'answer', 'answer_start']].to_dict('records')
            json_data.append({
                'context': context,
                'qa': qa_list
            })

        with open(output_deduplicated_json_path, 'w') as f:
            json.dump(json_data, f)

        
        print('[INFO] Highlighting data')

        highlighter = Highlighter()

        # Highlight answer adn context in context
        df['answer_highlighted_context'] = df.progress_apply(highlighter.highlight_answer, axis=1)
        df['sentence_highlighted_context'] = df.progress_apply(highlighter.highlight_sentence, axis=1)
        
        # Save data in CSV format
        df.to_csv(output_highlighted_csv_path, index=False)


        print('[INFO] Splitting data')

        grouped = [group for _, group in df.groupby('context')]
        train_validation_df, test_df = train_test_split(grouped, test_size=0.2, random_state=42)
        train_df, validation_df = train_test_split(train_validation_df, test_size=0.1, random_state=42)

        train_df = pd.concat(train_df)
        validation_df = pd.concat(validation_df)
        test_df = pd.concat(test_df)

        train_df.to_csv(output_train_csv_path, index=False)
        test_df.to_csv(output_test_csv_path, index=False)
        validation_df.to_csv(output_validation_csv_path, index=False)

        print()
        print(f'[RESULT] Data saved in {output_deduplicated_csv_path}') 
        print(f'[RESULT] Data saved in {output_deduplicated_json_path}')
        print(f'[RESULT] Data saved in {output_highlighted_csv_path}')
        print(f'[RESULT] Data saved in {output_train_csv_path}')
        print(f'[RESULT] Data saved in {output_test_csv_path}')
        print(f'[RESULT] Data saved in {output_validation_csv_path}')

        print(f'[RESULT] Total QA: {len(df)}')
        print(f'[RESULT] Total Context: {len(json_data)}')

        print(f'[RESULT] Train data: {len(train_df)}')
        print(f'[RESULT] Validation data: {len(validation_df)}')
        print(f'[RESULT] Test data: {len(test_df)}')


        print()
        print('[INFO] Checking for overlapping contexts between sets')
        # Convert the 'context' columns of each set to a set of unique values
        train_contexts_set = set(train_df['context'].unique())
        val_contexts_set = set(validation_df['context'].unique())
        test_contexts_set = set(test_df['context'].unique())

        # Check any overlapping contexts between the sets
        train_in_val_or_test = train_contexts_set.intersection(val_contexts_set.union(test_contexts_set))
        val_in_train_or_test = val_contexts_set.intersection(train_contexts_set.union(test_contexts_set))
        test_in_train_or_val = test_contexts_set.intersection(train_contexts_set.union(val_contexts_set))

        # Print the results
        print()
        print(f"[WARNING] Contexts in train that are also in val or test: {len(train_in_val_or_test)}")
        print(f"[WARNING] Contexts in val that are also in train or test: {len(val_in_train_or_test)}")
        print(f"[WARNING] Contexts in test that are also in train or val: {len(test_in_train_or_val)}")


class TokenizerHelper:
    def __init__(self, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def indobart_tokenize(self, text):
        text = f'{text}{self.tokenizer.eos_token}'
        return self.tokenizer(text, max_length=self.max_length, add_special_tokens=True, truncation=True, padding='max_length')

    def flan_t5_tokenize(self, text):
        return self.tokenizer(text, max_length=self.max_length, add_special_tokens=True, truncation=True, padding='max_length')