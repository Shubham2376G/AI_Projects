from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import MessagesState
from IPython.display import Image, display
from langgraph.graph import StateGraph, START, END
from langchain_huggingface.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface import ChatHuggingFace
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition
from langgraph.checkpoint.memory import MemorySaver


#tools
def multiply(a: int, b: int) -> int:
    """Multiply a and b.

    Args:
        a: first int
        b: second int
    """
    return a * b
def add(a: int, b: int) -> int:
    """Adds a and b.

    Args:
        a: first int
        b: second int
    """
    return a + b
def divide(a: int, b: int) -> float:
    """Divide a and b.

    Args:
        a: first int
        b: second int
    """
    return a / b


#chat model initiate
model_id = "C:/Users/lenovo/Desktop/Ramji_sir_FAQ/llama2"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=30,repetition_penalty=1.03, return_full_text=False, temperature=0.1)
llm = HuggingFacePipeline(pipeline=pipe)
chat_model = ChatHuggingFace(llm=llm)

all_tools = [add, multiply, divide]
llm_with_tools = chat_model.bind_tools(all_tools)

# tool_call = llm_with_tools.invoke([HumanMessage(content=f"What is 23456 multiplied by 367", name="Lance")])

sys_msg= [SystemMessage(content="you are a helpful agent and talks in pirate language")]

# Node
def Assistant(state: MessagesState):
    return {"messages": [llm_with_tools.invoke( sys_msg + state["messages"])]}


builder = StateGraph(MessagesState)
builder.add_node("llm_Assistant", Assistant)  # return either chat or tool_call formatted chat
builder.add_node("tools", ToolNode(all_tools)) # performs the operation/task on the tool_call formatted chat
builder.add_edge(START, "llm_Assistant")
builder.add_conditional_edges(
    "llm_Assistant",
    # If the latest message (result) from assistant is a tool call -> tools_condition routes to tools
    # If the latest message (result) from assistant is a not a tool call -> tools_condition routes to END
    tools_condition,
)
builder.add_edge("tools", "llm_Assistant")
graph = builder.compile()

# Conditional_edges -> Literal['tools', '__end__'] – The next node to route to.


# View in local
with open("graph_image.png", "wb") as f:
    f.write(graph.get_graph().draw_mermaid_png())

#View in colab
# display(Image(graph.get_graph().draw_mermaid_png()))


#memory
memory = MemorySaver()
react_graph_memory = builder.compile(checkpointer=memory)

# Specify a thread
config = {"configurable": {"thread_id": "1"}} # you can change the thread ids to start a new chat or continue the old one, it store all history


messages = [HumanMessage(content="i want cakey")] # you can append more messages like ai , human here
messages = react_graph_memory.invoke({"messages": messages},config)
for m in messages['messages']:
    m.pretty_print()


