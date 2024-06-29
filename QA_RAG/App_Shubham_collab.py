import streamlit as st
import tempfile
from langchain_community.document_loaders import YoutubeLoader, PyPDFLoader
from ingest_Shubham import ingest_docs
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForQuestionAnswering
from transformers import pipeline
import torch
import base64
import textwrap
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma, FAISS
from langchain_huggingface import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
checkpoint="LaMini-T5-738M"
# checkpoint="MBZUAI/LaMini-T5-738M"
# checkpoint="Intel/dynamic_tinybert"
offload_folder="offload_folder"
tokenizers=AutoTokenizer.from_pretrained(checkpoint)
base_model=AutoModelForSeq2SeqLM.from_pretrained(
    checkpoint,use_safetensors=True,
    device_map="cpu",
    torch_dtype=torch.float32,
    offload_folder=offload_folder
)
# base_model.to("cpu")
# base_model = base_model.to_empty(device="cpu")
# base_model.eval()
@st.cache_resource
def llm_pipeline():
    pipe=pipeline(
        "text2text-generation",
        model=base_model,
        tokenizer=tokenizers,
        max_new_tokens=500
    )
    local_llm=HuggingFacePipeline(pipeline=pipe,
                                  model_kwargs={"temperature": 0.1})
    return local_llm


@st.cache_resource
def qa_llm():
    llm=llm_pipeline()
    model_name = "all-MiniLM-l6-v2"
    device = "cpu"
    model_kwargs = {'device': device}
    encode_kwargs = {"normalize_embeddings": True}

    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs)
    # db=FAISS(embedding_function=embeddings)
    new_db = FAISS.load_local("faiss_index",embeddings= embeddings,allow_dangerous_deserialization=True)
    retriever=new_db.as_retriever(search_type="mmr", search_kwargs={"k":3})

    template = """
    You are an assistant for question-answering tasks.
    Use the provided context only to answer the following question:

    <context>
    {context}
    </context>

    Question: {input}
    """
    QA_CHAIN_PROMPT = ChatPromptTemplate.from_template(template)

    doc_chain = create_stuff_documents_chain(llm, QA_CHAIN_PROMPT)
    chain = create_retrieval_chain(retriever, doc_chain)
    # qa=RetrievalQA.from_chain_type(
    #     llm=llm,
    #     chain_type="stuff",
    #     retriever=retriever,
    #     return_source_documents=True,
    #     chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
    # )
    # return qa
    return chain

def process_answer(instruction):
    instruction=instruction
    qa=qa_llm()
    generated_text=qa.invoke({"input": instruction})
    answer=generated_text["answer"]

    return answer,generated_text

def get_youtube_info(link):
  # Implement logic to extract information from YouTube link using youtube-data-api
  # This example function returns a placeholder string
  loader = YoutubeLoader.from_youtube_url(link, add_video_info=True)
  documents = loader.load()
  ingest_docs(documents)

def process_pdf(pdf_data):
  # Implement logic to process PDF content using PyPDF2
  # This example function returns a placeholder string
  with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
    temp_file.write(pdf_data)
    loader = PyPDFLoader(temp_file.name)  # Use the temporary file path here
    # Use Langchain document object for further processing
    documents = loader.load()
  # return documents
  ingest_docs(documents)

def main():
  with st.expander("about the app"):
    st.markdown(
      """
      This is a app to answer your questions
      """
    )
  st.title("Choose your Input:")
  # Use radio buttons to allow users to select input type
  input_type = st.radio("Select Input Type:", ("YouTube Link", "Upload PDF"))

  if input_type == "YouTube Link":
    youtube_link = st.text_input("Enter YouTube Link")
    if youtube_link:
      try:
        # Call the function to get information from YouTube link
        get_youtube_info(youtube_link)
        # st.write(youtube_info)
      except Exception as e:
        st.error(f"Error processing YouTube link: {e}")

  elif input_type == "Upload PDF":
    pdf_file = st.file_uploader("Upload PDF file", type=('pdf'))
    if pdf_file:
      data = pdf_file.getvalue()
      try:
        # Call the function to process PDF content
        process_pdf(data)
        # st.write(processed_text)
      except Exception as e:
        st.error(f"Error processing PDF file: {e}")

  else:
    st.warning("Please select an input type.")


  question = st.text_area("enter your question")
  if st.button("serach"):
    st.info("your question:" + question)
    st.info("your answer")
    answer, metadata = process_answer(question)
    st.write(answer)
    st.write(metadata)


if __name__ == "__main__":
  main()



