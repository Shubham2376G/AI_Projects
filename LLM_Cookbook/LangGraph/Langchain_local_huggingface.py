from langchain_huggingface.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_core.messages import AIMessage
from langchain.schema import (
    HumanMessage,
    SystemMessage,
)
from langchain_huggingface import ChatHuggingFace


# message=[
#     ("system","you are helpful assistant"),
#     ("human","India is a big country. it is")
# ]

message = [
    SystemMessage(content="You're a helpful assistant"),
    HumanMessage(
        content="What happens when an "
    ),
]

# message = [
#     {
#         "role": "system",
#         "content": "You are a friendly chatbot who always responds in the style of a pirate",
#     },
#     {"role": "user", "content": "How many helicopters can "},
#  ]


model_id = "C:/Users/lenovo/Desktop/Ramji_sir_FAQ/llama2"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=30, repetition_penalty=1.03, return_full_text=False) # must args return_full_text = False
llm = HuggingFacePipeline(pipeline=pipe)
chat_model = ChatHuggingFace(llm=llm)
print(chat_model.invoke(message))