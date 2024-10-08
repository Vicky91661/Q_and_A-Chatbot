from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import streamlit as st
import os

from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings 
from langchain_community.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader

import os
from dotenv import load_env
load_env()

## Load the GROQ API Key
os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")
groq_api_key =  os.getenv("GROQ_API_KEY")

prompt = ChatPromptTemplate.from_messages(
   """
    Answer the questions based on the provided context only.
    Please provide the most accurate respone based on the question
    <context>
    {context}
    </context>
    Question:{input}
   """
)

llm = ChatGroq(groq_api_key = groq_api_key, model_name = "Llama-8b-8192")


def create_vector_embeddings():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = OllamaEmbeddings()
        st.session_state.loader = PyPDFDirectoryLoader("/research_paper")
        st.session_state.docs =st.session_state.loader.load()
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:50])
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents,st.session_state.embeddings)

st.title("RAG Doucmnet Q&A With Groq and Lama3")

user_prompt = st.text_input("enter your query from the reseach paper")

if st.button("Document Embedding"):
    create_vector_embeddings()
    st.write("Vector database is ready")

import time

if user_prompt:
    document_chain = create_stuff_documents_chain(llm,prompt)
    retriever = st.session_state.vector.as_retriever()
    retriever_chain = create_retrieval_chain(retriever,document_chain)

    start = time.process_time()
    resposne = retriever_chain.invoke({
        'input':user_prompt
    })

    print(f"Response time :{time.process_time()-start}")

    st.write(resposne['answer'])

    with st.expander("document similarity search"):
        for i,doc in enumerate(resposne['context']):
            st.write(doc.page_content)
            st.write('-----------------------')
            