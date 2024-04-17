import os
import sys
import torch
import argparse
import pandas as pd
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


from pytorch_lightning import seed_everything
from transformers import AutoModelForSeq2SeqLM

from Utils.utils import ModelType, Tokenizer
from Utils.data_loader import MultiTaskQAGDataModule
from Models.qag_model import QAGMultiTaskModel
from tqdm import tqdm


def initialize_pretrained_model(name, tokenizer_len, max_length):
    model = AutoModelForSeq2SeqLM.from_pretrained(name)
    model.resize_token_embeddings(tokenizer_len)
    model.config.max_length = max_length
    model.config.early_stopping = True
    model.config.num_beams = 5

    return model


if __name__ == "__main__":
    seed_everything(seed=42, workers=True)

    parser = argparse.ArgumentParser(description='Evaluation Parser')
    parser.add_argument('-m', '--model_type', choices=['IndoBART', 'Flan-T5'], required=True, help='Multitask Model Type')
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
            f'Datasets/Tensor/gemini_wiki_train_multitask_{model_type.value}.pt', 
            f'Datasets/Tensor/gemini_wiki_validation_multitask_{model_type.value}.pt', 
            f'Datasets/Tensor/gemini_wiki_ae_test_multitask_{model_type.value}.pt',
            f'Datasets/Tensor/gemini_wiki_qg_test_multitask_{model_type.value}.pt',
            ]

    ae_data_module = MultiTaskQAGDataModule(
        model_type=model_type,
        dataset_csv_paths=dataset_csv_paths,
        dataset_tensor_paths=dataset_tensor_paths,
        tokenizer=tokenizer,
        batch_size=batch_size,
        recreate=False
    )

    qg_data_module = MultiTaskQAGDataModule(
        model_type=model_type,
        dataset_csv_paths=dataset_csv_paths,
        dataset_tensor_paths=dataset_tensor_paths,
        tokenizer=tokenizer,
        batch_size=batch_size,
        recreate=False
    )

    
    ae_data_module.setup(stage='ae_test')
    qg_data_module.setup(stage='qg_test')

    ae_dataloader = ae_data_module.test_dataloader()
    qg_dataloader = qg_data_module.test_dataloader()


    model.to(device)
    model.eval()

    prediction_results = {
        'inputs': [],
        'ae_predictions': [],
        'ae_labels': [],
        'qg_predictions': [],
        'qg_labels': []
    }

    for ae_batch, qg_batch in tqdm(zip(ae_dataloader, qg_dataloader), total=len(ae_dataloader), desc='Generating Predictions'):
        ae_raw_predictions = []

        ae_input_ids, ae_attention_mask, ae_labels = ae_batch
        _, _, qg_labels = qg_batch

        ae_input_ids = ae_input_ids.to(device)
        ae_attention_mask = ae_attention_mask.to(device)
        ae_labels = ae_labels.to(device)
        qg_labels = qg_labels.to(device)

        with torch.no_grad():
            ae_outputs = model.model.generate(ae_input_ids)

            for idx in range(len(ae_outputs)):
                prediction_results['inputs'].append(tokenizer.decode(ae_input_ids[idx]).strip())
                ae_raw_predictions.append(tokenizer.decode(ae_outputs[idx]).strip())
                prediction_results['ae_predictions'].append(tokenizer.decode_for_answer_or_question(ae_outputs[idx]).strip())
                prediction_results['ae_labels'].append(tokenizer.decode_for_answer_or_question(ae_labels[idx]).strip())

            encoded_qg_inputs = torch.tensor(tokenizer.tokenize(ae_raw_predictions).input_ids).to(device)
            qg_outputs = model.model.generate(encoded_qg_inputs)

            for idx in range(len(qg_outputs)):
                prediction_results['qg_predictions'].append(tokenizer.decode_for_answer_or_question(qg_outputs[idx]).strip())
                prediction_results['qg_labels'].append(tokenizer.decode_for_answer_or_question(qg_labels[idx]).strip())

    
    os.makedirs('Predictions', exist_ok=True)

    predictions_df = pd.DataFrame(prediction_results)
    predictions_df['qa_format_preds'] = predictions_df.apply(lambda x: f"pertanyaan: {x['qg_predictions']}, jawaban: {x['ae_predictions']}", axis=1)
    predictions_df['qa_format_labels'] = predictions_df.apply(lambda x: f"pertanyaan: {x['qg_labels']}, jawaban: {x['ae_labels']}", axis=1)
    predictions_df.to_csv(f'Predictions/multitask_{model_type.name}.csv', index=False)