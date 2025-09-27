from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
import pandas as pd
import gradio as gr
import os
import re
import gc

database = pd.read_excel("database/database.xlsx") # your database with code and webpage links along with litte description

def generate_response(query,img="none"):
    # Load fine-tuned model and tokenizer
    model_path = "model/gpt2"
    tokenizer1 = AutoTokenizer.from_pretrained(model_path)
    model1 = AutoModelForCausalLM.from_pretrained(model_path)
    # Generate responses
    chatbot = pipeline("text-generation", model=model1, tokenizer=tokenizer1)
    prompt = f"USER: {query} <|bot|>"
    response = chatbot(prompt, max_length=400, num_return_sequences=1)

    generated_text = response[0]['generated_text']
    response_text = generated_text.replace(prompt, '').strip()
    print("hereeee", response_text)
    return hyperlink_agent(response_text)


def hyperlink_agent(pipe,query):
    database = pd.read_excel("database/database.xlsx")
    messages = [
        {
            "role": "system",
            "content": f"""
    You are a tool designed to process user input and format it according to specific guidelines. Your task is to return the complete text provided by the user, while enclosing some words in square brackets and appending their corresponding unique code as a hyperlink. Follow these rules:

    Use the given dataframe to identify words or phrases and their unique codes:
    {database}

    Enclose in square brackets [ ] those words or phrases that relates with the description column in the dataframe, even if the words are not identical.

    Example: For input 'Visit the airbnb website to download the app,' return:
    Visit the [airbnb](E) website to download the app.
    Avoid overusing brackets; prioritize the most relevant matches and keep the formatting natural.

    Remember, your goal is to make the output clear and user-friendly while adhering to the dictionary mappings.""",
        },
        {"role": "user", "content": query}
    ]
    print(1)
    # print(pipe(messages, max_new_tokens=328)[0]['generated_text'][-1]["content"])
    return hyperlink(pipe(messages, max_new_tokens=328)[0]['generated_text'][-1]["content"],database)

def hyperlink(text, database):

    pattern = r'\((\w)\)'  # Regex to match content inside parentheses
    new_text = re.sub(pattern, replace_with_df_value, text)
    return new_text

def replace_with_df_value(match):
    letter = match.group(1)  # Extract letter inside parentheses
    value = database.loc[database['Unique code'] == letter, 'link'].values
    return value[0] if value else match.group(0)  # Return corresponding value or original if not found


def retrival(input,pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # Text splitting
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=100)
    docs = text_splitter.split_documents(documents=documents)

    model_name = "BAAI/bge-large-en-v1.5"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {"normalize_embeddings": True}

    embedding_function = HuggingFaceBgeEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

    # Creating a vector database (FAISS - in memory database)
    db = FAISS.from_documents(
        docs,
        embedding_function
    )
    # Querying the vector database
    # query = "Is there a minimum amount for top-up"
    matched_docs = db.similarity_search(query=input, k=1)
    full_context = "\n".join([i.page_content for i in matched_docs])
    del db
    gc.collect()

    return full_context

def rag_agent(pipe,context,input_query):
    # context=retrival(input_query)
    print(context)
    print("h1")
    message2=[
        {
            "role": "system",
            "content": f"""You are a helpful AI assistant. Given the context {context}, answer the user question. Dont give simple answer, give a detailed answer, and include steps (if present)""",
        },
        {"role": "user",
         "content": input_query}
    ]
    print(2)
    f2= pipe(message2, max_new_tokens=328)[0]['generated_text'][-1]
    print(f2)
    return hyperlink_agent(pipe,f2)


def complete(query,pdf,img):
    pdf_path=pdf.name
    retriced_data=retrival(query,pdf_path)
    model=AutoModelForCausalLM.from_pretrained("llama2")
    tokenizer=AutoTokenizer.from_pretrained("llama2")
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    final1=rag_agent(pipe,retriced_data,query)

    print(final1)

    return final1


# Create a Gradio interface


def create_header():
    return """
    <h1 style="text-align: center; color: #4CAF50;">Hyperlink Agent</h1>
    <p style="text-align: center; font-size: 18px; color: #555;">
        Adds hyperlinks to the response
    </p>

    """

image_path = "architecture/architecture.png"
# Gradio interface
iface = gr.Interface(
    fn=complete,  # Function to handle the query
    inputs=[gr.Textbox(label="Input text:", placeholder="Type your text here..."),gr.File(),gr.Image(image_path, label="Flow Chart")],  # Input field
    outputs=gr.Markdown(label="Response:"),
    description=create_header()
)

# Launch the Gradio app inside the Colab notebook
iface.launch()