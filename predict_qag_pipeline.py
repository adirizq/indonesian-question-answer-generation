import os
import sys
import torch
import argparse
import pandas as pd
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


from pytorch_lightning import seed_everything
from transformers import AutoModelForSeq2SeqLM

from Utils.utils import ModelType, PipeLineTaskType, Tokenizer
from Utils.data_loader import PipelineQAGDataModule
from Models.qag_model import QAGPipelineModel
from tqdm import tqdm


def initialize_pretrained_model(name, tokenizer_len, max_length):
    model = AutoModelForSeq2SeqLM.from_pretrained(name)
    model.resize_token_embeddings(tokenizer_len)
    model.config.max_length = max_length
    model.config.early_stopping = True
    model.config.num_beams = 4

    return model


if __name__ == "__main__":
    seed_everything(seed=42, workers=True)

    parser = argparse.ArgumentParser(description='Evaluation Parser')
    parser.add_argument('-ae_m', '--ae_model_type', choices=['IndoBART', 'Flan-T5'], required=True, help='Answer Extraction Model Type')
    parser.add_argument('-qg_m', '--qg_model_type', choices=['IndoBART', 'Flan-T5'], required=True, help='Question Generation Model Type')
    parser.add_argument('-ae', '--ae_model_path', type=str, required=True, help='Answer Extraction Model Path')
    parser.add_argument('-qg', '--qg_model_path', type=str, required=True, help='Question Generation Model Path')
    parser.add_argument('-b', '--batch_size', type=int, default=8, help='Batch size for evaluation')
    parser.add_argument('-ml', '--max_length', type=int, default=512, help='Max length for input sequence')
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-5, help='Learning rate for initializing model')
    

    args = parser.parse_args()
    config = vars(args)

    ae_model_type = ModelType(config['ae_model_type'])
    qg_model_type = ModelType(config['qg_model_type'])
    ae_task_type = PipeLineTaskType('ae')
    qg_task_type = PipeLineTaskType('qg')
    ae_model_path = config['ae_model_path']
    qg_model_path = config['qg_model_path']
    batch_size = config['batch_size']
    max_length = config['max_length']
    learning_rate = config['learning_rate']

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model_info = {
        ModelType.INDOBART : {'type': 'BART', 'tokenizer': 'indobenchmark/indobart-v2', 'pre_trained': 'indobenchmark/indobart-v2', 'lr_scheduler': True},
        ModelType.FLAN_T5: {'type': 'Flan-T5', 'tokenizer': 'google/flan-t5-small', 'pre_trained': 'google/flan-t5-small', 'lr_scheduler': False}
    }

    ae_tokenizer = Tokenizer(ae_model_type, max_length=max_length)
    qg_tokenizer = Tokenizer(qg_model_type, max_length=max_length)

    ae_pretrained_model = initialize_pretrained_model(model_info[ae_model_type]['pre_trained'], ae_tokenizer.tokenizer_len(), max_length)
    qg_pretrained_model = initialize_pretrained_model(model_info[qg_model_type]['pre_trained'], qg_tokenizer.tokenizer_len(), max_length)

    ae_model = QAGPipelineModel.load_from_checkpoint(
        checkpoint_path = ae_model_path, 
        pretrained_model = ae_pretrained_model, 
        model_type = model_info[ae_model_type]['type'],  
        task_type = ae_task_type, 
        tokenizer = ae_tokenizer, 
        lr_scheduler = model_info[ae_model_type]['lr_scheduler'], 
        learning_rate = learning_rate
        )
    qg_model = QAGPipelineModel.load_from_checkpoint(
        checkpoint_path = qg_model_path, 
        pretrained_model = qg_pretrained_model, 
        model_type = model_info[qg_model_type]['type'],  
        task_type = qg_task_type, 
        tokenizer = qg_tokenizer, 
        lr_scheduler = model_info[qg_model_type]['lr_scheduler'], 
        learning_rate = learning_rate
        )


    ae_qg_dataset_csv_paths = [
        'Datasets/Splitted/gemini_wiki_train.csv', 
        'Datasets/Splitted/gemini_wiki_validation.csv', 
        'Datasets/Splitted/gemini_wiki_test.csv'
        ]
    
    ae_dataset_tensor_paths = [
        f'Datasets/Tensor/gemini_wiki_train_{ae_task_type.value}_{ae_model_type.value}.pt', 
        f'Datasets/Tensor/gemini_wiki_validation_{ae_task_type.value}_{ae_model_type.value}.pt', 
        f'Datasets/Tensor/gemini_wiki_test_{ae_task_type.value}_{ae_model_type.value}.pt'
        ]

    ae_data_module = PipelineQAGDataModule(
        model_type=ae_model_type,
        task_type=ae_task_type,
        dataset_csv_paths=ae_qg_dataset_csv_paths,
        dataset_tensor_paths=ae_dataset_tensor_paths,
        tokenizer=ae_tokenizer,
        batch_size=batch_size,
        recreate=False
    )

    qg_dataset_tensor_paths = [
        f'Datasets/Tensor/gemini_wiki_train_{qg_task_type.value}_{qg_model_type.value}.pt', 
        f'Datasets/Tensor/gemini_wiki_validation_{qg_task_type.value}_{qg_model_type.value}.pt', 
        f'Datasets/Tensor/gemini_wiki_test_{qg_task_type.value}_{qg_model_type.value}.pt'
        ]
    
    qg_data_module = PipelineQAGDataModule(
        model_type=qg_model_type,
        task_type=qg_task_type,
        dataset_csv_paths=ae_qg_dataset_csv_paths,
        dataset_tensor_paths=qg_dataset_tensor_paths,
        tokenizer=qg_tokenizer,
        batch_size=batch_size,
        recreate=False
    )

    ae_data_module.setup(stage='test')
    qg_data_module.setup(stage='test')

    ae_dataloader = ae_data_module.test_dataloader()
    qg_dataloader = qg_data_module.test_dataloader()


    ae_model.to(device)
    qg_model.to(device)

    ae_model.eval()
    qg_model.eval()

    prediction_results = {
        'ae_predictions': [],
        'ae_labels': [],
        'qg_predictions': [],
        'qg_labels': []
    }

    for ae_batch, qg_batch in tqdm(zip(ae_dataloader, qg_dataloader), total=len(ae_dataloader)):
        ae_input_ids, ae_attention_mask, ae_labels = ae_batch
        _, _, qg_labels = qg_batch

        ae_input_ids = ae_input_ids.to(device)
        ae_attention_mask = ae_attention_mask.to(device)
        ae_labels = ae_labels.to(device)
        qg_labels = qg_labels.to(device)

        temp_ae_predictions_for_qg_input = []

        with torch.no_grad():
            ae_outputs = ae_model.model.generate(ae_input_ids)

            for idx in range(len(ae_outputs)):
                temp_ae_predictions_for_qg_input.append(ae_model.tokenizer.decode(ae_outputs[idx]).strip())
                prediction_results['ae_predictions'].append(ae_model.tokenizer.decode_for_answer_or_question(ae_outputs[idx]).strip())
                prediction_results['ae_labels'].append(ae_model.tokenizer.decode_for_answer_or_question(ae_labels[idx]).strip())
            
            encoded_qg_inputs = torch.tensor(qg_model.tokenizer.tokenize(temp_ae_predictions_for_qg_input).input_ids).to(device)
            qg_outputs = qg_model.model.generate(encoded_qg_inputs)

            for idx in range(len(qg_outputs)):
                prediction_results['qg_predictions'].append(qg_model.tokenizer.decode_for_answer_or_question(qg_outputs[idx]).strip())
                prediction_results['qg_labels'].append(qg_model.tokenizer.decode_for_answer_or_question(qg_labels[idx]).strip())

            break
    
    os.makedirs('Predictions', exist_ok=True)

    predictions_df = pd.DataFrame(prediction_results)
    predictions_df['qa_format'] = predictions_df.apply(lambda x: f"pertanyaan: {x['qg_predictions']}, jawaban: {x['ae_predictions']}", axis=1)
    predictions_df.to_csv('Predictions/pipeline.csv', index=False)