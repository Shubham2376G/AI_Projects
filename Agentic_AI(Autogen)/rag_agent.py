import autogen
import os
from autogen import AssistantAgent

from autogen.agentchat.contrib.retrieve_user_proxy_agent import RetrieveUserProxyAgent
import chromadb
gemma = {
    "config_list": [
        {
            "model": "lmstudio-community/gemma-2-2b-it-GGUF/gemma-2-2b-it-Q4_K_M.gguf",
            "base_url": "http://localhost:1234/v1",
            "api_key": "not-needed",
        },
    ],
}

assistant = AssistantAgent(
    name="assistant",
    system_message="You are a helpful assistant.",
    llm_config=gemma
)
ragproxyagent = RetrieveUserProxyAgent(
    name="ragproxyagent",
    human_input_mode="NEVER",
    retrieve_config={
        "task": "qa",
        "docs_path": "llm.pdf",
        "chunk_token_size": 2000,
        "vector_db": "chroma",
        "model": "lmstudio-community/gemma-2-2b-it-GGUF/gemma-2-2b-it-Q4_K_M.gguf",
        "embedding_model": "D:/QA_APP/LaMini-T5-738M",
        "get_or_create": True
    },
    code_execution_config=False
)

query = "tell features of LLMs in points"
ragproxyagent.initiate_chat(
    assistant, message=ragproxyagent.message_generator, problem=query
)