from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import MessagesState
from IPython.display import Image, display
from langgraph.graph import StateGraph, START, END
from langchain_huggingface.llms import HuggingFacePipeline
from sympy import content
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface import ChatHuggingFace
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict, Annotated
from operator import add

#chat model initiate
model_id = "C:/Users/lenovo/Desktop/Ramji_sir_FAQ/llama2"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=30,repetition_penalty=1.03, return_full_text=False)
llm = HuggingFacePipeline(pipeline=pipe)
chat_model = ChatHuggingFace(llm=llm)



#schema
class QA(TypedDict):
    ques:str
    content:Annotated[list,add]
    ans:str



def google(state):

    r1=chat_model.invoke([HumanMessage(content=state["ques"])])
    return {"content": [r1.content]}

def bing(state):
    r2=chat_model.invoke([HumanMessage(content=state["ques"])])
    return {"content": [r2.content]}

def assistant(state):

    net= " ".join(state["content"])
    sys=SystemMessage(content=f"summarize the given question {state["ques"]} ?, using the content {net}")

    return {"ans": chat_model.invoke([sys]+[HumanMessage(content="summarize it")])}


builder=StateGraph(QA)
# Initialize each node with node_secret
builder.add_node("search_web",google)
builder.add_node("search_wikipedia", bing)
builder.add_node("generate_answer", assistant)

# Flow
builder.add_edge(START, "search_wikipedia")
builder.add_edge(START, "search_web")
builder.add_edge("search_wikipedia", "generate_answer")
builder.add_edge("search_web", "generate_answer")
builder.add_edge("generate_answer", END)
graph = builder.compile()


result = graph.invoke({"ques": "What are Computers and laptops"})
print(result)
print(result["ans"].content)


