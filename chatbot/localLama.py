'''

'''
# Import library
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain_ollama import ChatOllama
from dotenv import load_dotenv
from langsmith import traceable
import streamlit as st
import os

load_dotenv()

os.environ['LANGCHAIN_TRACING_V2'] = os.getenv('LANGCHAIN_TRACING_V2')
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')
os.environ['LANGCHAIN_ENDPOINT'] = os.getenv('LANGCHAIN_ENDPOINT')
os.environ['LANGCHAIN_PROJECT'] = os.getenv('LANGCHAIN_PROJECT')

st.set_page_config(page_title='Local LLM', page_icon='🤖')

st.title('PDF Extraction with llama')

upload_file = st.file_uploader('Upload your PDF file', type=['pdf'])


# Response

def get_response(query, chat_history):
    template = """
    Answer the following question based only on the provided context.
    Think step by step before providing a detailed answer.

    Chat history: {chat_history}
    User question: {user_question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    llm = ChatOllama(model='llama3.2:1b')

    chain = prompt | llm | StrOutputParser()

    return chain.invoke({
        'chat_history': chat_history,
        'user_question': query,
    })

# Conversation

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for massage in st.session_state.chat_history:
    if isinstance(massage, HumanMessage):
        with st.chat_message("Human"):
            st.markdown(massage.content)
    else:
        with st.chat_message("AI"):
            st.markdown(massage.content)

query = st.chat_input('Ask about your PDF file')
if query is not None and query != '':
    st.session_state.chat_history.append(HumanMessage(query))

    with st.chat_message("Human"):
        st.markdown(query)

    with st.chat_message("AI"):
        response = get_response(query, st.session_state.chat_history)
        st.markdown(response)

    st.session_state.chat_history.append(AIMessage(response))