import os
import torch
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from huggingface_hub import login
from langchain.chains import RetrievalQA

# Constants
EMBEDDING_MODEL_NAME = "nlpai-lab/KURE-v1"
LLM_MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"

def authenticate_hf():
    """Authenticates with Hugging Face using token from environment."""
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        print("Warning: HF_TOKEN environment variable not set. Please set it to access gated models.")
        return None

    try:
        login(token=hf_token)
        print("Successfully logged in to Hugging Face.")
    except Exception as e:
        print(f"Failed to login to Hugging Face: {e}")
    return hf_token

def load_documents(source_path):
    """
    Loads documents from a directory or a single file.
    Args:
        source_path (str): Path to a directory or a file.
    Returns:
        list: List of Document objects.
    """
    documents = []

    if os.path.isfile(source_path):
        files = [source_path]
    elif os.path.isdir(source_path):
        files = [os.path.join(source_path, f) for f in os.listdir(source_path)
                 if f.endswith('.txt') or f.endswith('.pdf')]
    else:
        return []

    for file_path in files:
        try:
            if file_path.endswith('.pdf'):
                loader = PyPDFLoader(file_path)
                documents.extend(loader.load())
            elif file_path.endswith('.txt'):
                loader = TextLoader(file_path, encoding='utf-8')
                documents.extend(loader.load())
        except Exception as e:
            print(f"Error loading {file_path}: {e}")

    return documents

def create_vector_store(documents):
    """
    Creates a FAISS vector store from documents.
    """
    if not documents:
        return None

    # Split text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    texts = text_splitter.split_documents(documents)

    # Embeddings
    # Using cpu explicitly
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': True}

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

    # Create Vector Store
    vector_store = FAISS.from_documents(texts, embeddings)
    return vector_store

def get_llm():
    """
    Loads the Local LLM.
    """
    hf_token = authenticate_hf()

    print(f"Loading LLM: {LLM_MODEL_NAME} on CPU...")

    try:
        tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME, token=hf_token)

        # Load model on CPU.
        # Using float32 for CPU compatibility.
        model = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL_NAME,
            token=hf_token,
            device_map="cpu",
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )

        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512,
            temperature=0.1,
            top_p=0.95,
            repetition_penalty=1.15
            # device argument removed as it conflicts with device_map="cpu"
        )

        llm = HuggingFacePipeline(pipeline=pipe)
        return llm
    except Exception as e:
        print(f"Error loading LLM: {e}")
        raise e

def get_rag_chain(vector_store):
    """
    Creates a RetrievalQA chain.
    """
    llm = get_llm()

    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )

    return qa_chain
