import os
import sys
import torch
import shutil
import argparse

from transformers import AutoModelForSeq2SeqLM
from optimum.onnxruntime import ORTModelForSeq2SeqLM

from Models.qag_model import QAGEnd2EndModel
from Utils.utils import ModelType, Tokenizer


def initialize_pretrained_model(name, tokenizer_len, max_length):
    model = AutoModelForSeq2SeqLM.from_pretrained(name)
    model.resize_token_embeddings(tokenizer_len)
    model.config.max_length = max_length
    model.config.early_stopping = True
    model.config.num_beams = 5

    return model


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='End2End Model Convert to ONNX')
    parser.add_argument('-m', '--model_type', choices=['IndoBART', 'Flan-T5'], required=True, help='Model Type')
    parser.add_argument('-i', '--input_model_path', type=str, required=True, help='Input Model Path')
    parser.add_argument('-ml', '--max_length', type=int, default=512, help='Max length for input sequence')
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-5, help='Learning rate for initializing model')


    args = parser.parse_args()
    config = vars(args)

    model_type = ModelType(config['model_type'])
    input_model_path = config['input_model_path']
    max_length = config['max_length']
    learning_rate = config['learning_rate']

    output_model_folder = f'ONNX Models/End2End/{model_type.name}'
    output_model_filename = f'{model_type.name}.onnx'

    os.makedirs(output_model_folder, exist_ok=True)

    model_info = {
        ModelType.INDOBART : {'type': 'BART', 'tokenizer': 'indobenchmark/indobart-v2', 'pre_trained': 'indobenchmark/indobart-v2', 'lr_scheduler': True},
        ModelType.FLAN_T5: {'type': 'Flan-T5', 'tokenizer': 'google/flan-t5-small', 'pre_trained': 'google/flan-t5-small', 'lr_scheduler': False}
    }

    print('[INFO] Loading model from', input_model_path)

    tokenizer = Tokenizer(model_type, max_length=max_length)
    pretrained_model = initialize_pretrained_model(model_info[model_type]['pre_trained'], tokenizer.tokenizer_len(), max_length).to('cpu')

    lightning_model = QAGEnd2EndModel.load_from_checkpoint(
        checkpoint_path = input_model_path, 
        pretrained_model = pretrained_model, 
        model_type = model_info[model_type]['type'],  
        tokenizer = tokenizer, 
        lr_scheduler = model_info[model_type]['lr_scheduler'], 
        learning_rate = learning_rate
        ).to('cpu')
    

    transformers_model = lightning_model.model.to('cpu')
    transformers_model.save_pretrained('temp_model')

    print('[INFO] Exporting model to', output_model_folder)

    onnx_model = ORTModelForSeq2SeqLM.from_pretrained('temp_model', from_transformers=True)
    onnx_model.save_pretrained(output_model_folder, filename=output_model_filename)
    tokenizer.tokenizer.save_pretrained(output_model_folder)

    print('[INFO] Model exported successfully')

    shutil.rmtree('temp_model')
