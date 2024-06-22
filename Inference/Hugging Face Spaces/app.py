from fastapi import FastAPI
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline


app = FastAPI()

model = AutoModelForSeq2SeqLM.from_pretrained('Model', local_files_only=True)
tokenizer = AutoTokenizer.from_pretrained('Tokenizer', local_files_only=True)


@app.get("/")
def index():
    return {"message": "Server Running"}


@app.get("/check-status")
def check_status():
    return {"message": "Server Running"}


@app.post("/predict")
def predict(text: str):
    text2text = pipeline('text2text-generation', model=model, tokenizer=tokenizer, device_map="auto")

    prediction = text2text(text)[0]['generated_text']

    try:
        question = prediction.split("pertanyaan:")[1].split(" jawaban:")[0]
        answer = prediction.split("jawaban:")[1]

        if question and answer:
            return {
                "question": question,
                "answer": answer
            }

        return "Model failed to predict question and answer", 500 

    except Exception as e:
        return str(e), 500


