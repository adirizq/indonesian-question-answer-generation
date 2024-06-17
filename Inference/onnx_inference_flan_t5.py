import onnxruntime

from optimum.onnxruntime import ORTModelForSeq2SeqLM
from transformers import AutoTokenizer, pipeline


if __name__ == '__main__':
    model_path = 'ONNX Models/End2End/FLAN_T5'

    model = ORTModelForSeq2SeqLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    onnx_seq2seq = pipeline('text2text-generation', model=model, tokenizer=tokenizer)

    preds = onnx_seq2seq(inputs='<hl>Apoclausirion adalah genus kumbang tanduk panjang yang tergolong famili Cerambycidae.<hl> Genus ini juga merupakan bagian dari ordo Coleoptera, kelas Insecta, filum Arthropoda, dan kingdom Animalia.')

    print(preds)