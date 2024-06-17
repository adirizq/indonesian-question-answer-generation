import os
import logging
import tarfile
from typing import Any, Dict, List

from transformers import pipeline, AutoTokenizer
from optimum.onnxruntime import ORTModelForSeq2SeqLM

from google.cloud.aiplatform.prediction.predictor import Predictor
from google.cloud.aiplatform.utils import prediction_utils

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class HuggingFacePredictor(Predictor):
    def __init__(self) -> None:
        pass


    def load(self, artifacts_uri: str) -> None:

        prediction_utils.download_model_artifacts(artifacts_uri)

        model_path = "./model"

        os.makedirs(model_path, exist_ok=True)
        with tarfile.open("model.tar.gz", "r:gz") as tar:
            tar.extractall(path=model_path)

        model = ORTModelForSeq2SeqLM.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        self._pipeline = pipeline('text2text-generation', model=model, tokenizer=tokenizer, device_map="auto")


    def preprocess(self, prediction_input: Dict) -> List[str]:
        inputs = prediction_input['instances']
        return inputs


    def predict(self, instances):
        prediction_results = []

        for instance in instances:
            prediction_results.append(self._pipeline(instance)['generated_text'])

        return prediction_results
    

    def postprocess(self, prediction_results: Any) -> Any:
        return {"predictions": prediction_results}