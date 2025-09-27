from autogen import ConversableAgent
import agentops
agentops.init("40b759df-e200-4eb6-a67c-390610caf38f")
agentops.start_session()
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

fitness_agent=ConversableAgent(
    name="fitness_Agent",
    system_message="you are a expert at understanding fitness needs, age-specific requirements, and gender-specific considerations, Skilled in developing customized exercise routines and fitness strategies. your goal is to analyze the fitness requirements for the user using his age , gender ,(disease if any mentioned) and suggest exercise routines and fitness strategies.Leave the diet plan to the nutritionist agent. Once the user is satisfied, you must say thank you to them",
    llm_config=gemma,
    human_input_mode="NEVER",
    is_termination_msg=lambda msg: "thank you" in msg["content"].lower()
)

nutritionist=ConversableAgent(
    name="nutritionist",
    system_message='you are knowledgeable in nutrition for different age groups and genders and provides tailored dietary advice based on specific nutritional needs.you goal is to Assess nutritional requirements for a perosn using his age, gender and disease if mentioned any, and provide dietary recommendations according of the fitness plan provided by the fitness_agent',
    llm_config=gemma,
    human_input_mode="NEVER"
)

human_proxy.description = "human user"
fitness_agent.description = "The best fitness coach"
nutritionist.description = "The best nutritionist"

from autogen import GroupChat, GroupChatManager

allowed_transitions = {
    human_proxy: [fitness_agent,nutritionist],
    fitness_agent: [human_proxy,nutritionist],
    nutritionist: [human_proxy],
}

group_chat=GroupChat(
    agents=[human_proxy,fitness_agent,nutritionist],
    messages=[],
    max_round=12,
    speaker_selection_method="auto",
    send_introductions=True,
    allowed_or_disallowed_speaker_transitions=allowed_transitions,
    speaker_transitions_type="allowed"
)

group_chat_manager=GroupChatManager(
    groupchat=group_chat,
    llm_config=gemma
)

chat_result=fitness_agent.initiate_chat(
    group_chat_manager,
    message="Hello, Please mention your age, gender and diesease if you are suffering from any",
    summary_method="reflection_with_llm"
)

agentops.end_session("Success")