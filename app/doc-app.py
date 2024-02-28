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
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq


openai.api_key = os.environ['OPENAI_API_KEY']

agent = None
temperature = 0
model = os.getenv('MODEL_NAME')
op_mode = os.getenv("MODE")
openai_api_version = os.getenv("AZURE_OPENAI_VERSION")
azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
if op_mode == "openai":
    persist_directory = 'docs/openai'
    embedding = OpenAIEmbeddings()
    llm = ChatOpenAI(model_name=model, temperature=0)
elif op_mode == "azure-openai":
    persist_directory = 'docs/openai'
    embedding = AzureOpenAIEmbeddings(
        azure_deployment=azure_deployment,
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
    embedding = OllamaEmbeddings() # use ollama embeddings for now
    # todo: add option to configure separate embedding models
    llm = ChatGroq(temperature=0, model_name=model)


def init():

    if not os.path.exists('files'):
        os.makedirs('files')
    if not os.path.exists('docs'):
        os.makedirs('docs')
    if not os.path.exists('docs/openai'):
        os.makedirs('docs/openai')
    if not os.path.exists('docs/ollama'):
        os.makedirs('docs/ollama')
    if not os.path.exists('docs/groq'):
        os.makedirs('docs/groq')

    vectordb = Chroma(
        embedding_function=embedding,
        persist_directory=persist_directory
    )
    return vectordb


def process_files(files):
    for f in files:
        bytes_data = f.getvalue()
        uploaded_file_path = os.path.join("files", f.name)
        with open(uploaded_file_path, "wb") as f:
            f.write(bytes_data)

        file_extension = os.path.splitext(f.name)[1]

        if file_extension == '.csv':
            # Handle CSV files
            loader = CSVLoader(uploaded_file_path)
            data = loader.load()
            r_splitter = RecursiveCharacterTextSplitter(
                chunk_size=450,
                chunk_overlap=150
            )
            splits = r_splitter.split_documents(data)

        elif file_extension == '.docx':
            # Handle DOCX files
            loader = UnstructuredWordDocumentLoader(uploaded_file_path)
            data = loader.load()
            r_splitter = RecursiveCharacterTextSplitter(
                chunk_size=450,
                chunk_overlap=150
            )
            splits = r_splitter.split_documents(data)
        elif file_extension == '.md':
            # Handle Markdown files
            loader = UnstructuredMarkdownLoader(uploaded_file_path)
            data = loader.load()
            r_splitter = RecursiveCharacterTextSplitter(
                chunk_size=450,
                chunk_overlap=150
            )
            splits = r_splitter.split_documents(data)

        elif file_extension == '.pdf':
            # Handle PDF files
            loader = PyPDFLoader(uploaded_file_path)
            data = loader.load()
            r_splitter = RecursiveCharacterTextSplitter(
                chunk_size=450,
                chunk_overlap=150
            )
            splits = r_splitter.split_documents(data)

        elif file_extension == '.txt':
            # Handle TXT files
            loader = TextLoader(uploaded_file_path)
            data = loader.load()
            r_splitter = RecursiveCharacterTextSplitter(
                chunk_size=450,
                chunk_overlap=150
            )
            splits = r_splitter.split_documents(data)
        else:
            print(f'Unsupported file extension: {file_extension}')

        vectordb = Chroma.from_documents(
            documents=splits,
            embedding=embedding,
            persist_directory=persist_directory
        )


def get_source(docs):
    sources = []
    for doc in docs:
        source = ""
        if doc.metadata['source'].endswith('.pdf'):
            source += f"- Source: {doc.metadata['source']} : Page {doc.metadata['page']}"
        elif doc.metadata['source'].endswith('.csv'):
            source += f"- Source: {doc.metadata['source']} : Row {doc.metadata['row']}"
        else:
            source += f"- Source: {doc.metadata['source']}"
        sources.append(source)

    # convert to markdown
    source = ""
    for s in sources:
        source += f"- {s} \n"

    return source


def retrieve_docs(vectordb, q):
    metadata_field_info = [
        AttributeInfo(
            name="host",
            description="Server hostname",
            type="string or list[string]",
        ),
        AttributeInfo(
            name="alert",
            description="A monitoring alert for SRE",
            type="string",
        )
    ]
    document_content_description = "A query by SRE team"
    llm = OpenAI(temperature=0)
    retriever = SelfQueryRetriever.from_llm(
        llm, vectordb, document_content_description, metadata_field_info, verbose=True
    )
    docs = retriever.get_relevant_documents(q)
    return docs


def chatQA(vectordb, llm, question):

    # Build prompt
    template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Keep the answer as concise as possible. 
    {context}
    Question: {question}
    Helpful Answer:"""
    QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=vectordb.as_retriever(),
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
    )
    result = qa_chain({"query": question})
    return result


st.set_page_config(page_title="Chat with data")
st.markdown("""
    <style>
        .reportview-container {
            margin-top: -2em;
        }
        #MainMenu {visibility: hidden;}
        .stDeployButton {display:none;}
        footer {visibility: hidden;}
        #stDecoration {display:none;}
    </style>
""", unsafe_allow_html=True)
st.title('Chat with your data')
chat, u_data = st.tabs(["Chat", "Upload Data"])

files_selected = u_data.file_uploader("Upload your data.", type=[
    "csv", "pdf", "docx", "txt", "md"], accept_multiple_files=True)

if files_selected:
    if u_data.button("Process"):
        with st.spinner(text='Processing your documents, please wait...'):
            process_files(files_selected)
            st.info('Processing done.')


vectordb = init()


if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask something"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Show a spinner during a process
    with st.spinner(text='Thinking...'):
        result = chatQA(vectordb, llm, prompt)
        print(result)
        sources = get_source(result["source_documents"])

        st.session_state.messages.append(
            {"role": "assistant", "content": f"{result['result']} \n {sources}"})

    with st.chat_message("assistant"):
        sources = get_source(result["source_documents"])
        st.markdown(f"{result['result']} \n {sources}")
