from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface import ChatHuggingFace
from langchain_huggingface.llms import HuggingFacePipeline


#chat model initiate
model_id = "C:/Users/lenovo/Desktop/Ramji_sir_FAQ/llama2"    # Must support tool-call
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=30,repetition_penalty=1.03, return_full_text=False)
llm = HuggingFacePipeline(pipeline=pipe)
chat_model = ChatHuggingFace(llm=llm)

class Structured(BaseModel):
    question:str = Field(description="The Question field")
    answer: str = Field(description="The Answer field")


result=chat_model.with_structured_output(Structured).invoke("tell me a history question and answer")

