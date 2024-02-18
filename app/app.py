import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage
import openai
import os
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain_openai import ChatOpenAI, OpenAI

agent = None
temperature = 0
model = "gpt-3.5-turbo-0125"
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

if uploaded_file is not None:
    # Can be used wherever a "file-like" object is accepted:
    agent = create_csv_agent(
        ChatOpenAI(temperature=temperature, model=model),
        uploaded_file,
        verbose=True,
        agent_type=AgentType.OPENAI_FUNCTIONS,
    )

    st.info('File ready to chat.')
    file_ready = True
    model = st.selectbox('Select Model', ['gpt-4', 'gpt-3.5-turbo-0125'])

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
            response = agent.run(prompt)

        st.session_state.messages.append(
            {"role": "assistant", "content": response})

        with st.chat_message("assistant"):
            st.markdown(response)
