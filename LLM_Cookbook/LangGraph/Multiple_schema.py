#Most of the time you will be using simple schema only, so dont worry


from typing import  TypedDict
from langgraph.graph import StateGraph, START, END


class Overall(TypedDict):
    ques:str
    content:str
    ans:str

class Input(TypedDict):
    ques:str

class Output(TypedDict):
    ans: str


#nodes
#No need to worry about auto add, Just a simple rule
#if the outgoing type has incoming key, then that key's value will also be set to the outgoing key too. Thats all, simple


def node1(state:Input) -> Overall:

    return {"content":"Ai is a artificial intelligence"}

def node2(state:Overall) -> Output:

    return {"ans":"yes you are right"}


builder=StateGraph(Overall,input=Input, output=Output)

builder.add_node("node1",node1)
builder.add_node("node2",node2)

builder.add_edge(START,"node1")
builder.add_edge("node1","node2")
builder.add_edge("node2",END)



graph=builder.compile()

final=graph.invoke({"ques":"What is AI"})

print(final)