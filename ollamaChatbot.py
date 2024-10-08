from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
import streamlit as st
import os

from dotenv import load_dotenv
load_dotenv()


## Langsmith Tracking
os.environ['LANGCHAIN_API_KEY'] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Q&A Chatbot with OPENAI"

## Prompt Template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system","You are a helpful assistant. Please response to the user queries"),
        ("user","Question:{question}")
    ]
)

def generate_response(question, engine, temperature, max_tokens):
   
    llm = Ollama(model = engine )
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser
    answer = chain.invoke({'question':question})
    return answer

## Title of the app
st.title("Enhanced Q&A Chatbot with Ollama")


## Drop down to select various Ollama AI Models
llm = st.disebar.selectbox("Select an Ollama AI Model",["mistral"])


## Adjust resposne parameter
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value = 1.0, value=0.7)
max_tokens = st.sidebar.slider("Max Tokens", min_value = 50, max_value = 300, value=150)

## Main interface for user input
st.write("GO ahead and ask any question")
user_input = st.text_input("You:")


if user_input:
    response = generate_response(user_input,llm,temperature,max_tokens)
    st.write(response)
else:
    st.write("Please provide the query")
