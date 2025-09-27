pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=52,                    # Generate up to 52 tokens
    do_sample=False,                      # Deterministic generation (no randomness) (prefer temperature =0.1 instead to this)
    repetition_penalty=1.03,              # Slight penalty for repeated tokens
    return_full_text=False,                # Exclude the input prompt from the output
    temperature=0.1
)

"""
message = [
    {
        "role": "system",
        "content": "You are a friendly chatbot who always responds in the style of a pirate",
    },
    {"role": "user", "content": "Hello "},
 ]
 
 [{'generated_text': "Yer lookin' fer a friendly chat, eh? Well, matey, ye've come to the right ship! I be here to sw"}]
 
 """