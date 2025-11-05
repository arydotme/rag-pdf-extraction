""" """

import time

# Import library
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage, AIMessage
from langchain_ollama import ChatOllama
from dotenv import load_dotenv
import streamlit as st
import os
import tempfile

load_dotenv()

# os.environ['LANGCHAIN_TRACING_V2'] = os.getenv('LANGCHAIN_TRACING_V2')
# os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')
# os.environ['LANGCHAIN_ENDPOINT'] = os.getenv('LANGCHAIN_ENDPOINT')
# os.environ['LANGCHAIN_PROJECT'] = os.getenv('LANGCHAIN_PROJECT')

st.set_page_config(page_title="Local LLM", page_icon="🤖")

st.title("PDF Extraction with llama")

upload_file = st.file_uploader("Upload your PDF file", type=["pdf"])

if upload_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(upload_file.read())
        tmp_path = tmp_file.name

    st.success("File Uploaded!")

# Response


def get_response(input, chat_history):
    if upload_file is not None:
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=20
        )
        documents = text_splitter.split_documents(docs)

        embeddings = OllamaEmbeddings(model="nomic-embed-text:v1.5")

        db = Chroma.from_documents(documents, embedding=embeddings)

        llm = ChatOllama(model="llama3.2:latest")

        template = """
            Answer the following question based only on the provided context.
            Think step by step before providing a detailed answer.

            Chat history: {chat_history}
            Context: {context}
            User question: {input}
            """
        prompt = ChatPromptTemplate.from_template(template)

        document_chain = create_stuff_documents_chain(llm, prompt)

        retriever = db.as_retriever()

        retriever_chain = create_retrieval_chain(retriever, document_chain)

        for event in retriever_chain.stream(
            {
                "input": input,
                "chat_history": chat_history,
            }
        ):
            if "answer" in event:
                yield event["answer"]

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

query = st.chat_input("Ask about your PDF file")

if query is not None and query != "":
    st.session_state.chat_history.append(HumanMessage(query))

    with st.chat_message("Human"):
        st.markdown(query)

    with st.chat_message("AI"):
        response = get_response(query, st.session_state.chat_history)
        result = st.write_stream(response)

    st.session_state.chat_history.append(AIMessage(content=result))
