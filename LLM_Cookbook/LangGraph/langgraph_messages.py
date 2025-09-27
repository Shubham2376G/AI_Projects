from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, RemoveMessage
from langgraph.graph import MessagesState
from langgraph.graph.message import add_messages

initial_message=[HumanMessage(content="hello, how are you"),
                 AIMessage(content="hello, how can I help you")]

new_message=[HumanMessage(content="What is RAG")]


#Adding
final=add_messages(initial_message,new_message) # you can also use append , but we use this to add bulk message at once

initial_message.append(new_message[0]) # appending item must be a simple item not a list

# note that appending wont overwrite messages with same id

print(initial_message)

#overwriting (same ids are replaced)
initial_messages = [AIMessage(content="Hello! How can I assist you?", name="Model", id="1"),
                    HumanMessage(content="I'm looking for information on marine biology.", name="Lance", id="2")
                   ]
new_message = HumanMessage(content="I'm looking for information on whales, specifically", name="Lance", id="2")

final2=add_messages(initial_messages , new_message)

#Removal

messages = [AIMessage("Hi.", name="Bot", id="1")]
messages.append(HumanMessage("Hi.", name="Lance", id="2"))
messages.append(AIMessage("So you said you were researching ocean mammals?", name="Bot", id="3"))
messages.append(HumanMessage("Yes, I know about whales. But what others should I learn about?", name="Lance", id="4"))

# Isolate messages to delete
delete_messages = [RemoveMessage(id=m.id) for m in messages[2:]]

final3=add_messages(messages , delete_messages)
print(final3)