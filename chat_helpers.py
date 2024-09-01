import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import HuggingFaceHub

def initialize_environment():
    load_dotenv()

def extract_text_from_pdfs(pdf_files):
    combined_text = ""
    for pdf in pdf_files:
        reader = PdfReader(pdf)
        for page in reader.pages:
            combined_text += page.extract_text()
    return combined_text

def split_text_into_chunks(text):
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return splitter.split_text(text)

def create_vector_database(chunks):
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = FAISS.from_texts(texts=chunks, embedding=embedding_model)
    return vector_db

def setup_conversational_agent(vector_db):
    language_model = HuggingFaceHub(
        repo_id="google/flan-t5-large",
        model_kwargs={"temperature": 0.5, "max_length": 512}
    )

    conversation_memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversational_chain = ConversationalRetrievalChain.from_llm(
        llm=language_model,
        retriever=vector_db.as_retriever(),
        memory=conversation_memory
    )
    return conversational_chain

def process_user_input(user_input):
    response = st.session_state.conversation({'question': user_input})
    st.session_state.chat_history = response['chat_history']

    for idx, msg in enumerate(st.session_state.chat_history):
        if idx % 2 == 0:
            st.markdown(style_user_message(msg.content), unsafe_allow_html=True)
        else:
            st.markdown(style_bot_message(msg.content), unsafe_allow_html=True)

def style_user_message(message):
    return f"""
    <div style="background-color: #483D8B; border-radius: 5px; padding: 10px; margin: 10px 0; width: fit-content; max-width: 70%;">
        <strong>User:</strong> {message}
    </div>
    """

def style_bot_message(message):
    return f"""
    <div style="background-color: #00008B; border-radius: 5px; padding: 10px; margin: 10px 0; width: fit-content; max-width: 70%; margin-left: auto;">
        <strong>Bot:</strong> {message}
    </div>
    """
