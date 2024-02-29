import streamlit as st
from langchain_openai import ChatOpenAI
import os
from langchain_openai import ChatOpenAI
from langchain_openai import AzureChatOpenAI
from langchain_community.llms import Ollama
from langchain_community.agent_toolkits import create_sql_agent
from utils import load_csv_to_db


agent = None
temperature = 0
models = [model for model in os.getenv('MODEL_NAME').split(',') if model]
op_mode = os.getenv("MODE")
openai_api_version = os.getenv("AZURE_OPENAI_VERSION")
azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
db = None
st.set_page_config(page_title="Chat with csv")
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
st.title('Chat with your csv data')

chat, u_data = st.tabs(["Chat", "Upload Data"])

with u_data:
    uploaded_file = st.file_uploader(
        'File uploader', type=["csv"])
    if uploaded_file:
        st.info(f"Active file is set to: {uploaded_file.name}")
        db = load_csv_to_db(uploaded_file)
        st.info("File ready to start")

if db:
    if op_mode.lower() == "openai":
        # Can be used wherever a "file-like" object is accepted:
        agent = create_sql_agent(
            ChatOpenAI(temperature=temperature, model=models[0]),
            db=db,
            verbose=True
        )

    if op_mode.lower() == "azure-openai":
        agent = create_sql_agent(AzureChatOpenAI(
            openai_api_version=openai_api_version,
            azure_deployment=azure_deployment,
        ),
            db=db)

    if op_mode.lower() == "ollama":
        agent = create_sql_agent(Ollama(model="llama2"), db=db)

    st.info(f'Chatting with file: {uploaded_file.name}')
    file_ready = True

    if "csv_messages" not in st.session_state:
        st.session_state.csv_messages = []

    for message in st.session_state.csv_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("What is up?"):
        st.session_state.csv_messages.append(
            {"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Show a spinner during a process
        with st.spinner(text='Thinking...'):
            response = {}
            try:
                response = agent.invoke(prompt)
            except ValueError as e:
                print(e)
                if "Could not parse LLM output:" in str(e):
                    result = str(e).split("Could not parse LLM output:")
                    response['output'] = result[1]
                else:
                    response['output'] = "LLM could not understand or parse your question."

        st.session_state.csv_messages.append(
            {"role": "assistant", "content": response['output']})

        with st.chat_message("assistant"):
            st.markdown(response['output'])
