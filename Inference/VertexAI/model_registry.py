from google.cloud import aiplatform

aiplatform.init(project="", location="asia-southeast1")

model = aiplatform.Model.upload(
    display_name="indonesian-qag-endend2-flan-t5",
    artifact_uri="gs://model-indonesian-qag-end2end-flan-t5",
    serving_container_image_uri="asia-southeast1-docker.pkg.dev//indonesian-qag-end2end-flan-t5/indonesian-qag:end2end-flan-t5",
    serving_container_environment_variables={
        "VERTEX_CPR_WEB_CONCURRENCY": 1,
    },
)