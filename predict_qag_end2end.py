import os
import re
import sys
import torch
import argparse
import pandas as pd
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


from pytorch_lightning import seed_everything
from transformers import AutoModelForSeq2SeqLM

from Utils.utils import ModelType, Tokenizer
from Utils.data_loader import End2EndQAGDataModule
from Models.qag_model import QAGMultiTaskModel
from tqdm import tqdm


def initialize_pretrained_model(name, tokenizer_len, max_length):
    model = AutoModelForSeq2SeqLM.from_pretrained(name)
    model.resize_token_embeddings(tokenizer_len)
    model.config.max_length = max_length
    model.config.early_stopping = True
    model.config.num_beams = 5

    return model


def extract_qa(s):
    question_regex_pattern = r"pertanyaan:(.*?) jawaban:"
    answer_regex_pattern = r"jawaban:(.*)"

    question = re.search(question_regex_pattern, s)
    answer = re.search(answer_regex_pattern, s)

    if question:
        question = question.group(1).strip()
    if answer:
        answer = answer.group(1).strip()

    return question, answer


if __name__ == "__main__":
    seed_everything(seed=42, workers=True)

    parser = argparse.ArgumentParser(description='Evaluation Parser')
    parser.add_argument('-m', '--model_type', choices=['IndoBART', 'Flan-T5'], required=True, help='End2End Model Type')
    parser.add_argument('-p', '--model_path', type=str, required=True, help='Multitask Model Path')
    parser.add_argument('-b', '--batch_size', type=int, default=8, help='Batch size for evaluation')
    parser.add_argument('-ml', '--max_length', type=int, default=512, help='Max length for input sequence')
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-5, help='Learning rate for initializing model')
    

    args = parser.parse_args()
    config = vars(args)

    model_type = ModelType(config['model_type'])
    model_path = config['model_path']
    batch_size = config['batch_size']
    max_length = config['max_length']
    learning_rate = config['learning_rate']

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model_info = {
        ModelType.INDOBART : {'type': 'BART', 'tokenizer': 'indobenchmark/indobart-v2', 'pre_trained': 'indobenchmark/indobart-v2', 'lr_scheduler': True},
        ModelType.FLAN_T5: {'type': 'Flan-T5', 'tokenizer': 'google/flan-t5-small', 'pre_trained': 'google/flan-t5-small', 'lr_scheduler': False}
    }

    tokenizer = Tokenizer(model_type, max_length=max_length)
    pretrained_model = initialize_pretrained_model(model_info[model_type]['pre_trained'], tokenizer.tokenizer_len(), max_length)


    model = QAGMultiTaskModel.load_from_checkpoint(
        checkpoint_path = model_path, 
        pretrained_model = pretrained_model, 
        model_type = model_info[model_type]['type'],  
        tokenizer = tokenizer, 
        lr_scheduler = model_info[model_type]['lr_scheduler'], 
        learning_rate = learning_rate
        )


    dataset_csv_paths = [
            'Datasets/Splitted/gemini_wiki_train.csv', 
            'Datasets/Splitted/gemini_wiki_validation.csv', 
            'Datasets/Splitted/gemini_wiki_test.csv'
            ]
    
    dataset_tensor_paths = [
            f'Datasets/Tensor/gemini_wiki_train_end2end_{model_type.value}.pt', 
            f'Datasets/Tensor/gemini_wiki_validation_end2end_{model_type.value}.pt', 
            f'Datasets/Tensor/gemini_wiki_test_end2end_{model_type.value}.pt'
            ]

    data_module = End2EndQAGDataModule(
        model_type=model_type,
        dataset_csv_paths=dataset_csv_paths,
        dataset_tensor_paths=dataset_tensor_paths,
        tokenizer=tokenizer,
        batch_size=batch_size,
        recreate=False
    )

    
    data_module.setup(stage='test')

    dataloader = data_module.test_dataloader()

    model.to(device)
    model.eval()

    prediction_results = {
        'ae_predictions': [],
        'ae_labels': [],
        'qg_predictions': [],
        'qg_labels': []
    }

    qa_predictions = []

    for batch in tqdm(dataloader, total=len(dataloader), desc='Generating Predictions'):
        input_ids, attention_mask, labels = batch

        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = model.model.generate(input_ids)

            for idx in range(len(outputs)):
                pred_question, pred_answer = extract_qa(tokenizer.decode_for_answer_or_question(outputs[idx]).strip())
                label_question, label_answer = extract_qa(tokenizer.decode_for_answer_or_question(labels[idx]).strip())

                prediction_results['ae_predictions'].append(pred_answer)
                prediction_results['ae_labels'].append(label_answer)
                prediction_results['qg_predictions'].append(pred_question)
                prediction_results['qg_labels'].append(label_question)

    
    os.makedirs('Predictions', exist_ok=True)

    predictions_df = pd.DataFrame(prediction_results)
    predictions_df['qa_format_preds'] = predictions_df.apply(lambda x: f"pertanyaan: {x['qg_predictions']}, jawaban: {x['ae_predictions']}", axis=1)
    predictions_df['qa_format_labels'] = predictions_df.apply(lambda x: f"pertanyaan: {x['qg_labels']}, jawaban: {x['ae_labels']}", axis=1)
    predictions_df.to_csv(f'Predictions/end2end_{model_type}.csv', index=False)