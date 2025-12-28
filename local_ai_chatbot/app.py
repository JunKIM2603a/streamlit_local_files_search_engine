import streamlit as st
import os
import tempfile
from backend import load_documents, create_vector_store, get_rag_chain

# Streamlit Page Configuration
st.set_page_config(page_title="Local AI Chatbot", page_icon="🤖", layout="wide")

st.title("🤖 Local AI Chatbot (No GPU)")

# Session State Initialization
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

if "documents_loaded" not in st.session_state:
    st.session_state.documents_loaded = False

# Sidebar for Configuration
with st.sidebar:
    st.header("Data Sources")

    # Folder Selection
    folder_path = st.text_input("Local Folder Path", placeholder="/path/to/documents")

    # File Upload
    uploaded_files = st.file_uploader("Upload Files", type=["txt", "pdf"], accept_multiple_files=True)

    process_button = st.button("Process Documents")

    if process_button:
        all_docs = []
        status_text = st.empty()
        status_text.text("Processing documents...")

        # Process Local Folder
        if folder_path and os.path.exists(folder_path):
            docs = load_documents(folder_path)
            all_docs.extend(docs)
            st.success(f"Loaded {len(docs)} documents from folder.")
        elif folder_path:
            st.error("Invalid folder path.")

        # Process Uploaded Files
        if uploaded_files:
            temp_dir = tempfile.mkdtemp()
            for uploaded_file in uploaded_files:
                temp_file_path = os.path.join(temp_dir, uploaded_file.name)
                with open(temp_file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                docs = load_documents(temp_file_path)
                all_docs.extend(docs)
            st.success(f"Processed {len(uploaded_files)} uploaded files.")

        if all_docs:
            status_text.text("Creating Vector Store (this may take a while)...")
            try:
                # Create Vector Store
                vector_store = create_vector_store(all_docs)

                # Create RAG Chain
                status_text.text("Loading LLM and creating Chain (this make take a while)...")

                chain = get_rag_chain(vector_store)
                st.session_state.rag_chain = chain
                st.session_state.documents_loaded = True
                status_text.text("Ready!")
                st.success("System is ready! You can now ask questions.")
            except Exception as e:
                st.error(f"An error occurred: {e}")
        else:
            st.warning("No documents found to process.")

# Chat Interface
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about your documents..."):
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.chat_history.append({"role": "user", "content": prompt})

    # Generate response
    if st.session_state.rag_chain:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.rag_chain.invoke(prompt)
                    answer = response['result']
                    st.markdown(answer)

                    st.session_state.chat_history.append({"role": "assistant", "content": answer})
                except Exception as e:
                    st.error(f"Error generating response: {e}")
    else:
        with st.chat_message("assistant"):
            st.warning("Please upload or select documents and click 'Process Documents' first.")
