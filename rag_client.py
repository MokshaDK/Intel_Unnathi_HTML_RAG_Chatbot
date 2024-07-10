import os
from dotenv import load_dotenv
import json
import requests

import streamlit as st

#loading information from .env
load_dotenv(verbose=True)
server_url  = os.environ['SERVER_URL']
server_port = os.environ['SERVER_PORT']

st.title('HTML-Based RAG Chatbot for AI, ML and DL Knowledge Retrieval')

if 'messages' not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

prompt = st.chat_input('Ask me anything realted to AI, ML or DL.')

if prompt:
    with st.chat_message('user'):
        st.markdown(prompt)
        st.session_state.messages.append({'role':'user', 'content':prompt})

    with st.chat_message('assistant'):
        payload = {"query":prompt}
        ans = requests.get(f'http://{server_url}:{server_port}/chatbot/1', params=payload)
        ans = json.loads(ans.text)
        st.markdown(ans['response'])
        st.session_state.messages.append({'role':'assistant', 'content':ans['response']})