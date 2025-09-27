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


def advice(state: MessagesState):
    # return {"messages": state["messages"]}
    return {"messages": [HumanMessage(content="OK")]}

builder=StateGraph(MessagesState)
builder.add_node("assistant",assistant)
builder.add_node("advice",advice)
builder.add_edge(START,"assistant")
builder.add_edge("assistant","advice")
builder.add_edge("advice",END)



graph=builder.compile(checkpointer=memory,interrupt_before=["advice"])

with open("graph_image.png", "wb") as f:
    f.write(graph.get_graph().draw_mermaid_png())

config = {"configurable": {"thread_id": "1"}}

initial_input={"messages": HumanMessage(content="tell me about AI")}

for event in graph.stream(initial_input,config,stream_mode="values"):
    event["messages"][-1].pretty_print()

user_approval=input("accept this yes/no")

if user_approval.lower() == "yes":
    for event in graph.stream(None, config, stream_mode="values"):
        event["messages"][-1].pretty_print()

else:
    print("Operation cancelled by user.")

