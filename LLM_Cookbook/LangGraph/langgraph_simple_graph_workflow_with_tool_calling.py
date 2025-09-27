from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import MessagesState
from IPython.display import Image, display
from langgraph.graph import StateGraph, START, END
from langchain_huggingface.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface import ChatHuggingFace
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition


def multiply(a: int, b: int) -> int:
    """Multiply a and b.

    Args:
        a: first int
        b: second int
    """
    return a * b


model_id = "C:/Users/lenovo/Desktop/Ramji_sir_FAQ/llama2"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=30,repetition_penalty=1.03, return_full_text=False, temperature=0.1)
llm = HuggingFacePipeline(pipeline=pipe)
chat_model = ChatHuggingFace(llm=llm)
llm_with_tools = chat_model.bind_tools([multiply])

# tool_call = llm_with_tools.invoke([HumanMessage(content=f"What is 23456 multiplied by 367", name="Lance")])

# Node
def tool_calling_llm(state: MessagesState):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


builder = StateGraph(MessagesState)
builder.add_node("tool_calling_llm", tool_calling_llm)  # return either chat or tool_call formatted chat
builder.add_node("tools", ToolNode([multiply])) # performs the operation/task on the tool_call formatted chat
builder.add_edge(START, "tool_calling_llm")
builder.add_conditional_edges(
    "tool_calling_llm",
    # If the latest message (result) from assistant is a tool call -> tools_condition routes to tools
    # If the latest message (result) from assistant is a not a tool call -> tools_condition routes to END
    tools_condition,
)
builder.add_edge("tools", END)
graph = builder.compile()

# Conditional_edges -> Literal['tools', '__end__'] – The next node to route to.


# View in local
with open("graph_image.png", "wb") as f:
    f.write(graph.get_graph().draw_mermaid_png())

#View in colab
# display(Image(graph.get_graph().draw_mermaid_png()))

messages = [HumanMessage(content="Hello world.")] # you can append more messages like ai , human here
messages = graph.invoke({"messages": messages})
for m in messages['messages']:
    m.pretty_print()