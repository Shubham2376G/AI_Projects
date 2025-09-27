# Normally TypedDict bydefault overwrites the value, We dont want that, so here we are
#And also you cannot parallelize it cause you cannot overwrite 2 things at same time

from typing_extensions import TypedDict
from langgraph.graph import add_messages, StateGraph, START, END

#Reduceers
from operator import add
from typing import Annotated

class State(TypedDict):
    foo: Annotated[list[int],add] # this key takes values as List and instead of overwrite, It appends it


def node_1(state):
    print("---Node 1---")
    return {"foo": [state['foo'][-1] + 1]}    #key's value must be a list

def node_2(state):
    print("---Node 2---")
    return {"foo": [state['foo'][-1] + 1]}

def node_3(state):
    print("---Node 3---")
    return {"foo": [state['foo'][-1] + 1]}




# Build graph
builder = StateGraph(State)
builder.add_node("node_1", node_1)
builder.add_node("node_2", node_2)
builder.add_node("node_3", node_3)

# Logic
builder.add_edge(START, "node_1")
builder.add_edge("node_1", "node_2")
builder.add_edge("node_1", "node_3")
builder.add_edge("node_2", END)
builder.add_edge("node_3", END)

# Add
graph = builder.compile()


with open("graph_image.png", "wb") as f:
    f.write(graph.get_graph().draw_mermaid_png())


final=graph.invoke({"foo": [1] })  #key's value must be a list so {"foo": 1} will give error
print(final)


#bonus, This is exactly how StateMessage works

"""
class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
"""
