import streamlit as st

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

st.title("Chat with your data")
st.write("Designed for personal use, this open-source tool lets you query, analyze, and extract insights from your data effortlessly. Start exploring your data in a conversational way today!")
