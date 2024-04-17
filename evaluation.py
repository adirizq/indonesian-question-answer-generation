import os
import sys
import evaluate
import pandas as pd

from Utils.qa_aligned_f1_score import QAAlignedF1Score
from textwrap import dedent
from tqdm import tqdm


def exact_match_evaluation(predictions, references):
        assert len(predictions) == len(references), "The number of predictions and references should be the same"
        
        exact_matches = 0
        for pred, refs in zip(predictions, references):
            if pred in refs:
                exact_matches += 1
        
        return exact_matches / len(predictions)


if __name__ == "__main__":

    bleu = evaluate.load('bleu')
    meteor = evaluate.load('meteor')
    rouge = evaluate.load('rouge')
    qa_aligned_f1_mover = QAAlignedF1Score('moverscore')
    qa_aligned_f1_bert = QAAlignedF1Score('bertscore')


    labels = pd.read_csv('Datasets/Splitted/gemini_wiki_test.csv')
    labels['qa_format_labels'] = labels.apply(lambda x: f"pertanyaan: {x['question']}, jawaban: {x['answer']}", axis=1)

    prediction_data = {
        'task': [],
        'name': [],
        'predictions': []
    }

    for file in os.listdir('Predictions'):
        if file.endswith('.csv'):
            prediction_data['task'].append(file.split('_')[0].upper())
            prediction_data['name'].append('_'.join(file.split('_')[1:]).removesuffix('.csv'))
            prediction_data['predictions'].append(pd.read_csv(f'Predictions/{file}'))


    prediction_data_df = pd.DataFrame(prediction_data)


    evaluation_results = {
        'task': [],
        'name': [],
        'exact_match': [],
        'bleu': [],
        'meteor': [],
        'rouge1': [],
        'rouge2': [],
        'rougeL': [],
        'rougeLsum': [],
        'qa_aligned_f1_mover': [],
        'qa_aligned_f1_bert': []
    }


    for idx, (task, name, predictions) in prediction_data_df.iterrows():

        print(f"Evaluating {task} - {name}...")
        print(f"[{idx+1}/{len(prediction_data_df)}]")

        predictions['qa_format_preds'] = predictions['qa_format_preds'].apply(lambda x: x.strip())  
        labels['qa_format_labels'] = labels['qa_format_labels'].apply(lambda x: x.strip())

        combined = pd.DataFrame({
            'qa_format_preds': predictions['qa_format_preds'],
            'qa_format_labels': labels['qa_format_labels'],
            'context': labels['context']
        })

        grouped = combined.groupby('context')

        qa_aligned_f1_mover_score = 0
        qa_aligned_f1_bert_score = 0

        for _, group in tqdm(grouped, desc='Calculating QA Aligned F1 Score...'):
            group_qa_aligned_f1_mover_score = qa_aligned_f1_mover.get_score(hyps=group['qa_format_preds'].to_list(), refs=group['qa_format_labels'].to_list()) 
            group_qa_aligned_f1_bert_score = qa_aligned_f1_bert.get_score(hyps=group['qa_format_preds'].to_list(), refs=group['qa_format_labels'].to_list()) 

            qa_aligned_f1_mover_score += group_qa_aligned_f1_mover_score.mean()
            qa_aligned_f1_bert_score += group_qa_aligned_f1_bert_score.mean()
        
        qa_aligned_f1_mover_score /= len(grouped)
        qa_aligned_f1_bert_score /= len(grouped)
        score_exact_match = exact_match_evaluation(predictions=combined['qa_format_preds'], references=combined['qa_format_labels'].to_list())
        score_bleu = bleu.compute(predictions=combined['qa_format_preds'], references=combined['qa_format_labels'].to_list())["bleu"]
        score_meteor = meteor.compute(predictions=combined['qa_format_preds'], references=combined['qa_format_labels'].to_list())["meteor"]
        score_rouge = rouge.compute(predictions=combined['qa_format_preds'], references=combined['qa_format_labels'].to_list())
        score_rouge1 = score_rouge['rouge1']
        score_rouge2 = score_rouge['rouge2']
        score_rougeL = score_rouge['rougeL']
        score_rougeLsum = score_rouge['rougeLsum']

        print(dedent(f'''
        -------------------------------------------------------------------------
                                {task} Evaluation Result     
        -------------------------------------------------------------------------
                                {name}
        -------------------------------------------------------------------------
        Name                | Value       
        -------------------------------------------------------------------------
        Exact Match         | {score_exact_match}
        Bleu                | {score_bleu}
        Meteor              | {score_meteor}
        Rouge1              | {score_rouge1}
        Rouge2              | {score_rouge2}
        RougeL              | {score_rougeL}
        RougeLsum           | {score_rougeLsum}
        QA Aligned F1 Mover | {qa_aligned_f1_mover_score}
        QA Aligned F1 Bert  | {qa_aligned_f1_bert_score}
        -------------------------------------------------------------------------

        '''))

        evaluation_results['task'].append(task)
        evaluation_results['name'].append(name)
        evaluation_results['exact_match'].append(score_exact_match)
        evaluation_results['bleu'].append(score_bleu)
        evaluation_results['meteor'].append(score_meteor)
        evaluation_results['rouge1'].append(score_rouge1)
        evaluation_results['rouge2'].append(score_rouge2)
        evaluation_results['rougeL'].append(score_rougeL)
        evaluation_results['rougeLsum'].append(score_rougeLsum)
        evaluation_results['qa_aligned_f1_mover'].append(qa_aligned_f1_mover_score)
        evaluation_results['qa_aligned_f1_bert'].append(qa_aligned_f1_bert_score)


    os.makedirs('Evaluation', exist_ok=True)

    evaluation_results_df = pd.DataFrame(evaluation_results)
    evaluation_results_df.to_csv('evaluation_results.csv', index=False)