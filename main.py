import streamlit as st
from chat_helpers import initialize_environment, extract_text_from_pdfs, split_text_into_chunks, create_vector_database, setup_conversational_agent, process_user_input


def run_app():
    initialize_environment()

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.title("Interactive PDF Chatbot with AI :books:")
    user_input = st.text_input("Enter your question regarding the documents:")
    if user_input:
        process_user_input(user_input)

    with st.sidebar:
        st.subheader("Upload your PDF documents")
        pdf_files = st.file_uploader(
            "Upload PDF files and click 'Start'", accept_multiple_files=True)
        if st.button("Start"):
            with st.spinner("Extracting content and preparing the chat..."):
                # Extract text from PDFs
                extracted_text = extract_text_from_pdfs(pdf_files)

                # Split text into manageable chunks
                text_chunks = split_text_into_chunks(extracted_text)
                
                # Create a vector-based search database
                vector_db = create_vector_database(text_chunks)
                
                # Set up the conversational AI with the vector store
                st.session_state.conversation = setup_conversational_agent(vector_db)

if __name__ == '__main__':
    run_app()
