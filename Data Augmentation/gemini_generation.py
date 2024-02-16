import os
import json
import time
import random
import threading
import google.generativeai as genai

from prompts import Prompts


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

def generate_save_qa(file_name, gemini_creative, gemini_strict, prompts):
  generated_qas = []
  total_generated_qas = 0
  target = 20000

  for i in range(target):
    st = time.time()

    qa_temp = {'context': '', 'qa': []}

    # Create story generation prompt
    prompt_generate_story = prompts.STORY_PROMPT.replace("{STORY_GENRE}", " ".join(random.sample(prompts.STORY_GENRE, 3)))
    prompt_generate_story = prompt_generate_story.replace("{WORD_LENGTH}", str(random.sample(prompts.STORY_WORD_LENGTH, 1)[0]))

    # Generate story
    try:
      response_story = gemini_creative.generate_content(prompt_generate_story)
      generated_story = response_story.text
    except:
      continue
    qa_temp['context'] = generated_story

    # Create question and answer generation prompt
    prompt_generate_qa = prompts.QUESTION_ANSWER_PROMPT.replace("{STORY}", generated_story)

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
          if qa['answer'] in generated_story:

            prompt_verify_qa = prompts.VERIFY_QUESTION_ANSWER_PROMPT.replace("{STORY}", generated_story)
            prompt_verify_qa = prompt_verify_qa.replace("{QUESTION}", qa['question'])
            prompt_verify_qa = prompt_verify_qa.replace("{ANSWER}", qa['answer'])

            try:
              response_verify_qa = gemini_strict.generate_content(prompt_verify_qa)
              if 'true' in response_verify_qa.text.lower():
                qa_temp['qa'].append(qa)
            except:
              continue
    except:
      pass

    # Save if there is any verified qas
    if len(qa_temp['qa']) > 0:
      total_generated_qas += len(qa_temp['qa'])
      generated_qas.append(qa_temp)
      json.dump(generated_qas, open(file_name, 'w'))

    # Logging
    print(f'{file_name} - {i+1}/{target} - {time.time() - st:.2f}s - total_generated_qas: {total_generated_qas}')


if __name__ == "__main__":
  
  # Setup
  gemini_creative = setup_gemini(0.9)
  gemini_strict = setup_gemini(0)
  prompts = Prompts()

  # Run progress
  os.makedirs('Datasets/Augmentation/Gemini', exist_ok=True)
  generate_save_qa('Datasets/Augmentation/Gemini/qa.json', gemini_creative, gemini_strict, prompts)
