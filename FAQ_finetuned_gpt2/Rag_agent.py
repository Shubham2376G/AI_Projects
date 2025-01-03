import autogen
from autogen import AssistantAgent
from autogen.agentchat.contrib.retrieve_user_proxy_agent import RetrieveUserProxyAgent
import chromadb
from autogen import ConversableAgent
gemma = {
    "config_list": [
        {
            "model": "lmstudio-community/gemma-2-2b-it-GGUF/gemma-2-2b-it-Q4_K_M.gguf",
            "base_url": "http://localhost:1234/v1",
            "api_key": "not-needed",
        },
    ],
}



human_proxy=ConversableAgent(
    name="user",
    llm_config=False,
    human_input_mode="ALWAYS",
)

hyperlink_agent=ConversableAgent(
    name="hyperlink_Agent",
    system_message="""You are a tool designed to process user input and format it according to specific guidelines. Your task is to return the complete text provided by the user, while enclosing some words in square brackets and appending their corresponding unique code as a hyperlink. Follow these rules:

Use the given dictionary to identify words or phrases and their unique codes:
{sim only deals: a, prepay plan: a, esim: b, top up: c, lyca studio: d}

Enclose in square brackets [ ] only those words or phrases that closely match the keys in the dictionary, even if the words are not identical.

Example: For input "Visit the lyca studio website to download the app," return:
Visit the [lyca studio]('d') website to download the app.
Avoid overusing brackets; prioritize the most relevant matches and keep the formatting natural.

If no matches are found, return the text as is, without any brackets.

Remember, your goal is to make the output clear and user-friendly while adhering to the dictionary mappings.""",
    llm_config=gemma,
    human_input_mode="NEVER",
)



human_proxy.description = "human user"
hyperlink_agent.description = "text editor"


from autogen import GroupChat, GroupChatManager



group_chat=GroupChat(
    agents=[human_proxy, hyperlink_agent],
    messages=[],
    max_round=12,
    speaker_selection_method="round_robin",
    send_introductions=True,
    speaker_transitions_type="allowed"
)

group_chat_manager=GroupChatManager(
    groupchat=group_chat,
    llm_config=gemma
)

chat_result=hyperlink_agent.initiate_chat(
    group_chat_manager,
    message="Hello, Please mention your text",
    summary_method="reflection_with_llm"
)


