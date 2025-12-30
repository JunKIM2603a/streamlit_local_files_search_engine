import streamlit as st
import os
import tempfile
from backend import load_documents, create_vector_store, get_rag_chain, rewrite_query, rerank_documents

# Streamlit Page Configuration
st.set_page_config(page_title="Local AI Chatbot", page_icon="🤖", layout="wide")

st.title("🤖 Local AI Chatbot (No GPU)")
st.caption("하이브리드 검색 (BM25 70% + Vector 30%) + FlashRank 리랭킹")

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
            with tempfile.TemporaryDirectory() as temp_dir:
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
                # Create Vector Store with hybrid support
                vector_store, all_texts = create_vector_store(all_docs)

                # Create RAG Chain
                status_text.text("Loading LLM and creating Chain (this may take a while)...")

                chain = get_rag_chain(vector_store, all_texts)
                st.session_state.rag_chain = chain
                st.session_state.documents_loaded = True
                status_text.text("Ready!")
                st.success("System is ready! You can now ask questions.")
            except Exception as e:
                st.error(f"An error occurred: {e}")
                import traceback
                st.code(traceback.format_exc())
        else:
            st.warning("No documents found to process.")

    # 검색 설정 정보
    st.divider()
    st.subheader("🔧 검색 설정")
    st.info("""
    **하이브리드 검색 (RRF)**
    - BM25 가중치: 70%
    - Vector 가중치: 30%
    
    **리랭킹**
    - FlashRank (MiniLM)
    - CPU 최적화
    """)

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
                    # 1. Query Expansion
                    refined_query = rewrite_query(prompt)
                    
                    # 2. Retrieval & Generation
                    response = st.session_state.rag_chain.invoke(refined_query)
                    context_docs = response.get('context', [])
                    
                    # 3. FlashRank Reranking
                    reranked_docs, original_docs = rerank_documents(refined_query, context_docs, top_k=5)
                    
                    answer = response['answer']
                    st.markdown(answer)

                    # 검색 분석 확장기 (보고서 섹션 7.3 권장)
                    with st.expander("🔍 검색 분석 (Query Analysis)"):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("**쿼리 변환**")
                            st.write(f"- 원본: `{prompt}`")
                            st.write(f"- 변환: `{refined_query}`")
                        
                        with col2:
                            st.markdown("**검색 가중치**")
                            st.write("- BM25: **70%** (키워드)")
                            st.write("- Vector: **30%** (의미론)")
                        
                        st.divider()
                        
                        # 리랭킹 전/후 비교
                        st.markdown("**📊 리랭킹 비교 (FlashRank)**")
                        
                        col_before, col_after = st.columns(2)
                        
                        with col_before:
                            st.markdown("*하이브리드 검색 결과 (리랭킹 전)*")
                            for i, doc in enumerate(original_docs[:3]):
                                content = doc.page_content[:80] + "..." if len(doc.page_content) > 80 else doc.page_content
                                st.code(f"{i+1}. {content}", language=None)
                        
                        with col_after:
                            st.markdown("*FlashRank 리랭킹 후*")
                            for i, doc in enumerate(reranked_docs[:3]):
                                content = doc.page_content[:80] + "..." if len(doc.page_content) > 80 else doc.page_content
                                st.code(f"{i+1}. {content}", language=None)
                        
                        # 메타데이터 표시 (있는 경우)
                        if reranked_docs and reranked_docs[0].metadata.get('command'):
                            st.divider()
                            st.markdown("**📋 상위 결과 메타데이터**")
                            for i, doc in enumerate(reranked_docs[:2]):
                                meta = doc.metadata
                                if meta.get('command'):
                                    st.write(f"**{i+1}. 명령어:** `{meta.get('command')}`")
                                    if meta.get('source'):
                                        st.write(f"   - 소스: `{meta.get('source')}`")
                                    if meta.get('destination'):
                                        st.write(f"   - 목적지: `{meta.get('destination')}`")

                    st.session_state.chat_history.append({"role": "assistant", "content": answer})
                except Exception as e:
                    st.error(f"Error generating response: {e}")
                    import traceback
                    st.code(traceback.format_exc())
    else:
        with st.chat_message("assistant"):
            st.warning("Please upload or select documents and click 'Process Documents' first.")
