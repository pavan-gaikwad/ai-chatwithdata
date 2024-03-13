from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.vectorstores import Chroma
import streamlit as st
import os
import openai
import sys
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_openai import OpenAI
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_openai import AzureOpenAIEmbeddings
from langchain_openai import AzureChatOpenAI
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq

from dotenv import load_dotenv
load_dotenv()

openai.api_key = os.environ['OPENAI_API_KEY']

agent = None
temperature = 0
model = os.getenv('MODEL_NAME')
op_mode = os.getenv("MODE")
openai_api_version = os.getenv("AZURE_OPENAI_VERSION")
azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
azure_embeddings_deployment = os.getenv(
    "AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT_NAME")
chunk_size = int(os.getenv("CHUNK_SIZE"))
chunk_overlap = int(os.getenv("CHUNK_OVERLAP"))
if op_mode == "openai":
    persist_directory = 'docs/openai'
    embedding = OpenAIEmbeddings()
    llm = ChatOpenAI(model_name=model, temperature=0)
elif op_mode == "azure-openai":
    persist_directory = 'docs/openai'
    embedding = AzureOpenAIEmbeddings(
        azure_deployment=azure_embeddings_deployment,
        openai_api_version=openai_api_version,
    )
    llm = AzureChatOpenAI(
        openai_api_version=openai_api_version,
        azure_deployment=azure_deployment,
    )
elif op_mode == "ollama":
    persist_directory = 'docs/ollama'
    embedding = OllamaEmbeddings()
    llm = ChatOllama(model=model)
elif op_mode == "groq":
    persist_directory = 'docs/groq'
    embedding = OllamaEmbeddings()  # use ollama embeddings for now
    # todo: add option to configure separate embedding models
    llm = ChatGroq(temperature=0, model_name=model)

st.title("Simple Chat with LLM")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        stream = llm.stream(prompt)
        response = st.write_stream(stream)
    st.session_state.messages.append({"role": "assistant", "content": response})
