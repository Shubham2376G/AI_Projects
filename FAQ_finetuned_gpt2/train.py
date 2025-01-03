import os
import re
from autogen import AssistantAgent, UserProxyAgent


def train1(query):
    gemma = {
        "config_list": [
            {
                "model": "lmstudio-community/gemma-2-2b-it-GGUF/gemma-2-2b-it-Q4_K_M.gguf",
                "base_url": "http://localhost:1234/v1",
                "api_key": "not-needed",
            },
        ],
    }


    assistant = AssistantAgent("assistant", llm_config=gemma, system_message="""
    You are a tool designed to process user input and format it according to specific guidelines. Your task is to return the complete text provided by the user, while enclosing some words in square brackets and appending their corresponding unique code as a hyperlink. Follow these rules:
    
    Use the given dictionary to identify words or phrases and their unique codes:
    {sim only deals: a, esim prepay plan: b, esim: c, top up: d, lyca studio: e}
    
    Enclose in square brackets [ ] only those words or phrases that closely match the keys in the dictionary, even if the words are not identical.
    
    Example: For input "Visit the lyca studio website to download the app," return:
    Visit the [lyca studio]('d') website to download the app.
    Avoid overusing brackets; 1 2 per paragraph. prioritize the most relevant matches and keep the formatting natural.
    
    If no matches are found, return the text as is, without any brackets.
    
    Remember, your goal is to make the output clear and user-friendly while adhering to the dictionary mappings.""")

    user_proxy = UserProxyAgent("user_proxy", code_execution_config=False)

    # Start the chat
    mess=user_proxy.initiate_chat(
        assistant,
        message=f"format this: {query}",
        max_turns=1
    )

    database= {"a":"(https://www.lycamobile.co.uk/en/bundles/sim-only-deals/#30-day-plans)","b":"(https://www.lycamobile.co.uk/en/bundles/sim-only-deals/#30-day-plans)","c":"(https://www.lycamobile.co.uk/en/introducing-esim-in-uk-all-you-need-to-know/)","d":"(https://www.lycamobile.co.uk/en/quick-top-up/)","e":"https://www.lycamobile.co.uk)"}

    def hyperlink(text, database):
        result = []
        for i in range(len(text)):
            if text[i] == "(" and i + 1 < len(text):
                # Replace the next character using the database
                result.append(database[text[i + 1]])  # Replace or keep original
                i += 1  # Skip the replaced character

            elif text[i] == ")":
                result.pop()
            else:
                result.append(text[i])
        return "".join(result)


    return hyperlink(mess.chat_history[1]["content"],database)

