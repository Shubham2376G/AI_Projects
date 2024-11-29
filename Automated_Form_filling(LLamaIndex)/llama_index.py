from llama_index.llms import OpenAI
from llama_index.prompts import Prompt
from llama_index.core.llm_predictor import LLMPredictor
# Initialize the LLM
llama_llm = OpenAI(model="gpt-4", base_url="http://localhost:1234/v1", api_key="lm-studio")

# Define your query
user_prompt = "Explain the significance of SaaS."

# Create a prompt wrapper (optional but useful for structuring)
query_prompt = Prompt(template="{query}")

# Call the LLM
response = llama_llm.complete(prompt=query_prompt.format(query=user_prompt))

# Print the response
print(response)
