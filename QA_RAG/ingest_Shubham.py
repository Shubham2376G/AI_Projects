import torch
from langchain_community.document_loaders import YoutubeLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma, FAISS
import os
from langchain.retrievers import ParentDocumentRetriever
persist_directory="database"

def ingest_docs(documents):
    text_splitter=RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ' ', ''],
        chunk_size=1000,
        chunk_overlap=150
    )
    texts=text_splitter.split_documents(documents)

    model_name = "all-MiniLM-l6-v2"
    device = "cpu"
    model_kwargs = {'device': device}
    encode_kwargs = {"normalize_embeddings": True}

    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs)
    db = FAISS.from_documents(texts, embeddings)
    db.save_local("faiss_index")
    # db= Chroma.from_documents(texts,embeddings,persist_directory=persist_directory)
    # db=None


