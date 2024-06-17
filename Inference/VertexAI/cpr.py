import os
from google.cloud.aiplatform.prediction import LocalModel
from predictor import HuggingFacePredictor

local_model = LocalModel.build_cpr_model(
    ".",
    "asia-southeast1-docker.pkg.dev//indonesian-qag-end2end-flan-t5/indonesian-qag:end2end-flan-t5",
    predictor=HuggingFacePredictor,
    requirements_path="requirements.txt",
    base_image="python:3.10"
)

local_model.get_serving_container_spec()

with local_model.deploy_to_local_endpoint(
    artifact_uri="gs://model-indonesian-qag-end2end-flan-t5/model.tar.gz",
    credential_path="creds.json"
) as local_endpoint:
    predicts = local_endpoint.predict.predict(
        request_file = "input.json",
        headers={"Content-Type": "application/json"},
    )   
    print(predicts)

local_model.push_image()