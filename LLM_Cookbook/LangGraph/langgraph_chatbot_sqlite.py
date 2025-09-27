from langchain_core.messages import HumanMessage
from langgraph.graph import MessagesState, add_messages, StateGraph, START, END
from langgraph.graph.message import RemoveMessage
from transformers import AutoModelForCausalLM , AutoTokenizer, pipeline
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver

db_path = "state_db/example.db"  # make the folder only
conn = sqlite3.connect(db_path, check_same_thread=False)
memory=SqliteSaver(conn)

checkpoint= "C:/Users/lenovo/Desktop/Ramji_sir_FAQ/llama2"
model=AutoModelForCausalLM.from_pretrained(checkpoint)
tokenizer=AutoTokenizer.from_pretrained(checkpoint)
pipe=pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=50,temperature=0.1, repetition_penalty=1.03, return_full_text=False)

llm=HuggingFacePipeline(pipeline=pipe)
chat_llm=ChatHuggingFace(llm=llm)


#nodes
def assistant(state: MessagesState):
    return {"messages": [chat_llm.invoke(state["messages"][-4:])]}

builder=StateGraph(MessagesState)
builder.add_node("assistant",assistant)
builder.add_edge(START,"assistant")
builder.add_edge("assistant",END)



graph=builder.compile(checkpointer=memory)

with open("graph_image.png", "wb") as f:
    f.write(graph.get_graph().draw_mermaid_png())

config = {"configurable": {"thread_id": "1"}}

message1=[HumanMessage(content="can you tell me about AI")]

messages = graph.invoke({"messages": message1},config)

for m in messages['messages']:
    m.pretty_print()