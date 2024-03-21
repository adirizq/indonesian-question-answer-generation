import os
import json
import argparse
import google.generativeai as genai

from dotenv import load_dotenv
from prompts import Prompts
from tqdm import tqdm


load_dotenv()


def setup_gemini(temperature):

  genai.configure(api_key=os.environ.get('GEMINI_API_KEY'))

  generation_config = {
    "temperature": temperature,
    "top_p": 1,
    "top_k": 1,
  }

  safety_settings = [
    {
      "category": "HARM_CATEGORY_HARASSMENT",
      "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
      "category": "HARM_CATEGORY_HATE_SPEECH",
      "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
      "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
      "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
      "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
      "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
  ]

  gemini = genai.GenerativeModel(model_name="gemini-pro",
                                generation_config=generation_config,
                                safety_settings=safety_settings)
  
  return gemini


def generate_save_qa(context_file_name, file_name, gemini_creative, gemini_strict):
  prompts = Prompts()
  generated_qas = []
  total_generated_qas = 0
  
  with(open(context_file_name, 'r')) as f:
    data = json.load(f)['data']

  for article in tqdm(data):
    title = article['title']
    paragraphs = article['paragraphs']
    
    generated_qas_temp = []

    for paragraph in paragraphs:
      context = paragraph['context']
      qas_temp = {'context': context, 'qa': []}

      # Create question and answer generation prompt
      prompt_generate_qa = prompts.QUESTION_ANSWER_WIKI_PROMPT.replace("{CONTEXT}", context)

      # Generate question and answer
      try:
        response_qa = gemini_creative.generate_content(prompt_generate_qa)
        generated_qa = response_qa.text
      except:
        continue

      # Verify question and answer
      try:
        qas = json.loads(generated_qa.replace('```', '').replace('json', ''))

        for qa in qas:
          if 'answer' in qa and 'question' in qa:
            if qa['answer'] in context:

              prompt_verify_qa = prompts.VERIFY_QUESTION_ANSWER_WIKI_PROMPT.replace("{CONTEXT}", context)
              prompt_verify_qa = prompt_verify_qa.replace("{QUESTION}", qa['question'])
              prompt_verify_qa = prompt_verify_qa.replace("{ANSWER}", qa['answer'])

              try:
                response_verify_qa = gemini_strict.generate_content(prompt_verify_qa)
                if 'true' in response_verify_qa.text.lower():
                  qas_temp['qa'].append(qa)
              except:
                continue
      except:
        pass

      # Save if there is any verified qas
      if len(qas_temp['qa']) > 0:
        total_generated_qas += len(qas_temp['qa'])
        generated_qas_temp.append(qas_temp)
    
    if len(generated_qas_temp) > 0:
      generated_qas.append({'title': title, 'paragraphs': generated_qas_temp})
      json.dump(generated_qas, open(file_name, 'w'))


if __name__ == "__main__":
  
  # Setup
  gemini_creative = setup_gemini(0.9)
  gemini_strict = setup_gemini(0)
  prompts = Prompts()

  # Parser
  parser = argparse.ArgumentParser(description='QA Wiki Generation Parser')
  parser.add_argument('-i', '--input', type=str, required=True, help='Input JSON file')
  parser.add_argument('-o', '--output', type=str, required=True, help='Output JSON file')

  args = parser.parse_args()
  config = vars(args)

  input_file = config['input']
  output_file = config['output']

  # Generate and save QA
  generate_save_qa(input_file, output_file, gemini_creative, gemini_strict)