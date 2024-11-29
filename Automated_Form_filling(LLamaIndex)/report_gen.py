from llama_index.core.llms import ChatMessage
# from langchain.schema import ChatMessage
from langchain_community.chat_models import ChatOpenAI
# sllm = llm.as_structured_llm(output_cls=CompanyParameters)

lm_studio_llm = ChatOpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

# Create a ChatMessage
user_prompt = ChatMessage(role="user", content="Explain the significance of SaaS.")

# Call the LLM
response = lm_studio_llm([user_prompt])

# Output the response
print(response)