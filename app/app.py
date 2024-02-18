import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage
import openai
import os
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain_openai import ChatOpenAI, OpenAI
from langchain_openai import AzureChatOpenAI
from langchain_community.llms import Ollama
from langchain_community.agent_toolkits import create_sql_agent
import pandas as pd
from langchain_community.utilities import SQLDatabase
from sqlalchemy import create_engine
import uuid


agent = None
temperature = 0
models = [model for model in os.getenv('MODEL_NAME').split(',') if model]
op_mode = os.getenv("MODE")
openai_api_version = os.getenv("AZURE_OPENAI_VERSION")
azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")

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
file_ready = False

if not file_ready:
    uploaded_file = st.file_uploader('File uploader', type=["csv"])


def load_csv_to_db(file_path):
    df = pd.read_csv(file_path)

    file_name = os.path.splitext(file_path.name)[0]
    if os.path.exists(f"files/{file_name}.db"):
        os.remove(f"files/{file_name}.db")
    engine = create_engine(f"sqlite:///files/{file_name}.db")
    df.to_sql(file_name, engine, index=False)
    db = SQLDatabase(engine=engine)
    print(db.get_usable_table_names())
    return db


if uploaded_file is not None:

    db = load_csv_to_db(uploaded_file)

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

    st.info('File ready to chat.')
    file_ready = True

    if op_mode == "openai":
        model = st.selectbox('Select Model', models)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("What is up?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Show a spinner during a process
        with st.spinner(text='Thinking...'):
            response = agent.invoke(prompt)

        st.session_state.messages.append(
            {"role": "assistant", "content": response['output']})

        with st.chat_message("assistant"):
            st.markdown(response['output'])
