
import aisuite as ai

def main_inference(model_name,symptoms,cbc_report,xray):
  client = ai.Client()

  if model_name == "Gemma:2b":
    models = "ollama:gemma:2b"
  elif model_name == "Mistral:7b":
    models = "ollama:mistral"
  elif model_name == "Medllama2:7b":
    models = "ollama:medllama2"
  elif model_name == "Meditron:7b":
    models = "ollama:meditron"
  elif model_name == "GPT4":
    models = "openai:gpt-4"
  else :
    models = "ollama:gemma:2b"

  messages = [
      {"role": "system",
       "content": "You are a doctor, and your role is to diagnose a patient based on the symptoms provided. I will describe the symptoms, and you need to tell me the potential disease the patient might be suffering from, with a clear and detailed answer and also tell why you think that the patient might be suffering from that. Do not provide any other information or dialogue besides the diagnosis. Provide the diagnosis in full immediately after receiving the symptoms."},
      {"role": "user",
       "content": f"symptoms are {symptoms}, blood test report is {cbc_report} , and xray result {xray} . what disease he is suffering from ?"},
  ]


  response = client.chat.completions.create(
      model=models,
      messages=messages,
      temperature=0.1
  )
  return response.choices[0].message.content

