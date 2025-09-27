from typing_extensions import TypedDict
from langgraph.graph import add_messages, StateGraph, START, END


#I have just defined a type here, No biggie, Calm down
class Hidden(TypedDict):
    a:int
    b:int
    c:int
    d:str
    e:bool


def node1(state):   # state= {"c": 15}
    return {"b":2}   # it is basically setting/updating the value of this key of that object


def node2(state):
    return {"b":state["b"]+4}   # it is basically setting/updating the value of this key of that object

def node3(state):
    return {"a":1}      # it is basically setting/updating the value of this key of that object



builder=StateGraph(Hidden)    # It means your graph states must be of type "Hidden" as you defined
builder.add_node("node1",node1)
builder.add_node("node2",node2)
builder.add_node("node3",node3)
builder.add_edge(START,"node1")
builder.add_edge("node1","node2")
builder.add_edge("node2","node3")
builder.add_edge("node3",END)

graph=builder.compile()


mess=graph.invoke({"c":15})    # so basically I am passing state = {"c":15}, Yup its a valid TypedDict format, It has the specified keys and specified format of value
print(mess)

# Note that there is only 1 object, so all the effects will take place on THAT object only