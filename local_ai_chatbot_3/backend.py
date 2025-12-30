import os
from typing import Any, List, Optional, Tuple
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models.llms import LLM as BaseLLM
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from pydantic import Field
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama
from flashrank import Ranker, RerankRequest
import streamlit as st
from dotenv import load_dotenv
from pathlib import Path
import re

load_dotenv()

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Constants
# ---------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
os.makedirs(MODELS_DIR, exist_ok=True)


# ## **임베딩 (Embeddings) 및 벡터 스토어 (Vector Store)**
# ---------------------------------------------------------
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
EMBEDDING_MODEL_PATH = MODELS_DIR / "bge-m3"

# LLM Models - GGUF format
LLM_MODEL_NAME = "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF"
LLM_MODEL_PATH = MODELS_DIR / "Qwen2.5-Coder-7B-Instruct-GGUF"
LLM_MODEL_FILE = "qwen2.5-coder-7b-instruct-q4_k_m.gguf"

LLM_QUERY_MODEL_NAME = "Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF"
LLM_QUERY_MODEL_PATH = MODELS_DIR / "Qwen2.5-Coder-1.5B-Instruct-GGUF"
LLM_QUERY_MODEL_FILE = "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"


# =============================================================================
# Command Metadata Parser
# =============================================================================

def parse_command_metadata(line: str) -> dict:
    """
    명령어에서 메타데이터 추출 (cp, SetPrimary 등)
    보고서 섹션 3.2 권장사항 구현
    """
    metadata = {
        "command": None,
        "source": None,
        "destination": None,
        "args": None
    }
    
    # cp 명령어 패턴: cp "소스" "목적지"
    cp_pattern = r'^cp\s+"([^"]+)"\s*,\s*"([^"]+)"'
    match = re.match(cp_pattern, line.strip())
    if match:
        metadata["command"] = "cp"
        metadata["source"] = match.group(1)
        metadata["destination"] = match.group(2)
        metadata["args"] = f"{match.group(1)}, {match.group(2)}"
        return metadata
    
    # SetPrimary 명령어 패턴
    setprimary_pattern = r'^SetPrimary\s+(\S+)'
    match = re.match(setprimary_pattern, line.strip())
    if match:
        metadata["command"] = "SetPrimary"
        metadata["args"] = match.group(1)
        return metadata
    
    # netstat 명령어 패턴
    if line.strip().startswith("netstat"):
        metadata["command"] = "netstat"
        return metadata
    
    # Show* 명령어 패턴
    show_pattern = r'^(Show\w+)'
    match = re.match(show_pattern, line.strip())
    if match:
        metadata["command"] = match.group(1)
        return metadata
    
    return metadata


# =============================================================================
# Hybrid Retriever - Custom Implementation (inherits BaseRetriever)
# 보고서 섹션 4.3 권장사항: BM25 70%, Vector 30%
# =============================================================================
class HybridRetriever(BaseRetriever):
    """Hybrid retriever combining Vector and BM25 (30:70 ratio - BM25 우선)
    
    Inherits from LangChain's BaseRetriever for full Runnable compatibility.
    보고서 권장: 키워드 매칭이 중요하므로 BM25 비중을 높게 설정
    """
    
    vector_retriever: Any = Field(description="Vector store retriever")
    bm25_retriever: Any = Field(description="BM25 keyword retriever")
    vector_weight: float = Field(default=0.3, description="Weight for vector results (0-1), BM25 gets 1-vector_weight")
    k_constant: int = Field(default=60, description="RRF constant for ranking")
    
    class Config:
        arbitrary_types_allowed = True
    
    @property
    def bm25_weight(self) -> float:
        return 1.0 - self.vector_weight
    
    def _get_relevant_documents(
        self, 
        query: str, 
        *, 
        run_manager: Optional[CallbackManagerForRetrieverRun] = None
    ) -> List[Document]:
        """
        Get documents from both retrievers and combine using RRF
        보고서 섹션 4.3: Reciprocal Rank Fusion 구현
        """
        # Get results from both retrievers
        vector_docs = self.vector_retriever.invoke(query)
        bm25_docs = self.bm25_retriever.invoke(query)
        
        # RRF (Reciprocal Rank Fusion) 구현
        # Score(d) = Σ (weight / (k + rank(d, r)))
        rrf_scores = {}
        doc_map = {}
        
        # Vector 검색 결과 RRF 점수 계산
        for rank, doc in enumerate(vector_docs):
            content = doc.page_content
            score = self.vector_weight / (self.k_constant + rank + 1)
            rrf_scores[content] = rrf_scores.get(content, 0) + score
            doc_map[content] = doc
        
        # BM25 검색 결과 RRF 점수 계산
        for rank, doc in enumerate(bm25_docs):
            content = doc.page_content
            score = self.bm25_weight / (self.k_constant + rank + 1)
            rrf_scores[content] = rrf_scores.get(content, 0) + score
            doc_map[content] = doc
        
        # 점수순 정렬
        sorted_contents = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        
        # 상위 문서 반환
        return [doc_map[content] for content in sorted_contents[:10]]


# =============================================================================
# FlashRank Reranker - CPU 최적화 (보고서 섹션 5.2)
# =============================================================================

@st.cache_resource
def get_reranker():
    """
    FlashRank 리랭커 로드 (CPU 최적화)
    보고서 권장: ms-marco-MiniLM-L-12-v2 모델 사용
    """
    print("Loading FlashRank reranker...")
    ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir=str(MODELS_DIR))
    print("✓ FlashRank reranker loaded successfully")
    return ranker


def rerank_documents(query: str, docs: List[Document], top_k: int = 5) -> Tuple[List[Document], List[Document]]:
    """
    FlashRank를 사용한 문서 리랭킹
    보고서 섹션 5.1: 크로스 인코더로 정밀한 재순위화
    
    Returns:
        Tuple: (리랭킹된 문서들, 원본 문서들) - UI에서 비교 표시용
    """
    if not docs:
        return [], []
    
    original_docs = docs.copy()
    
    try:
        ranker = get_reranker()
        
        # FlashRank 형식으로 변환
        passages = [{"id": i, "text": doc.page_content} for i, doc in enumerate(docs)]
        
        # 리랭킹 수행
        rerank_request = RerankRequest(query=query, passages=passages)
        results = ranker.rerank(rerank_request)
        
        # 리랭킹된 순서로 문서 반환
        reranked_docs = [docs[result["id"]] for result in results[:top_k]]
        
        return reranked_docs, original_docs[:top_k]
        
    except Exception as e:
        print(f"Reranking failed: {e}, returning original order")
        return docs[:top_k], original_docs[:top_k]


# =============================================================================
# LangChain Wrapper for llama-cpp-python
# =============================================================================

class LlamaCppLLM(BaseLLM):
    """LangChain wrapper for llama-cpp-python"""
    llm: Any
    max_tokens: int = 512
    temperature: float = 0.1
    top_p: float = 0.95
    
    class Config:
        arbitrary_types_allowed = True
    
    def _call(
        self, 
        prompt: str, 
        stop: Optional[List[str]] = None,
        **kwargs
    ) -> str:
        """Generate text from prompt"""
        response = self.llm(
            prompt,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            stop=stop or [],
            echo=False
        )
        return response['choices'][0]['text']
    
    @property
    def _llm_type(self) -> str:
        return "llama_cpp"


# =============================================================================
# Document Loading
# =============================================================================

def load_documents(source_path):
    """
    Loads documents from a directory or a single file.
    Args:
        source_path (str): Path to a directory or a file.
    Returns:
        list: List of Document objects.
    """
    documents = []
    files = []

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


# =============================================================================
# Custom Chunking Strategy for Command/Script Data
# 보고서 섹션 3.1: 라인 단위 원자적 청킹
# =============================================================================

def create_command_chunks(documents):
    """
    라인 단위 원자적 청킹 + 메타데이터 추출
    보고서 섹션 3.1, 3.2 권장사항 구현
    
    - 각 라인을 독립된 Document로 처리
    - cp 명령어에서 source/destination 메타데이터 추출
    - 빈 라인과 의미없는 라인 필터링
    
    Args:
        documents: List of Document objects
    
    Returns:
        List of Document objects with proper chunking and metadata
    """
    chunks = []
    
    for doc in documents:
        lines = doc.page_content.split('\n')
        
        for line_num, line in enumerate(lines):
            stripped = line.strip()
            
            # 빈 라인 스킵
            if not stripped:
                continue
            
            # 메타데이터 추출
            cmd_metadata = parse_command_metadata(stripped)
            
            # 기본 메타데이터
            metadata = {
                **doc.metadata,
                'chunk_type': 'command',
                'line_number': line_num + 1,
                **cmd_metadata  # command, source, destination, args
            }
            
            chunks.append(Document(
                page_content=stripped,
                metadata=metadata
            ))
    
    print(f"Created {len(chunks)} atomic line chunks from {len(documents)} documents")
    return chunks


# =============================================================================
# Embeddings
# =============================================================================

# Download embedding model if not exists
if not EMBEDDING_MODEL_PATH.exists():
    print(f"임베딩 모델({EMBEDDING_MODEL_NAME})을 다운로드하여 로컬에 저장합니다...")
    try:
        model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        model.save(str(EMBEDDING_MODEL_PATH))
        print("임베딩 모델 저장 완료!")
    except Exception as e:
        print(f"임베딩 모델 다운로드 중 오류 발생: {e}")


@st.cache_resource
def get_embeddings(model_path):
    """Get HuggingFace embeddings model"""
    return HuggingFaceEmbeddings(
        model_name=str(model_path),
        model_kwargs={
            'device': 'cpu', 
            'trust_remote_code': True,
        },
        encode_kwargs={'normalize_embeddings': True}
    )


# =============================================================================
# Vector Store with In-Memory FAISS Index
# =============================================================================

def create_vector_store(documents):
    """
    Creates a FAISS vector store with in-memory index (IndexFlatL2) for speed.
    Returns both vector store and text list for hybrid retrieval.
    
    Args:
        documents: List of Document objects
    
    Returns:
        tuple: (vector_store, texts) - FAISS vector store and list of chunked documents
    """
    if not documents:
        return None, None

    # 라인 단위 원자적 청킹 (보고서 권장)
    texts = create_command_chunks(documents)
    
    if not texts:
        return None, None

    # Embeddings
    embeddings = get_embeddings(EMBEDDING_MODEL_PATH)
    
    # Create FAISS with in-memory index (IndexFlatL2)
    # This is fast for small to medium datasets with lots of RAM
    vector_store = FAISS.from_documents(texts, embeddings)
    
    print(f"Created in-memory FAISS index with {len(texts)} chunks")
    return vector_store, texts


# =============================================================================
# LLM Loading with llama-cpp-python
# =============================================================================

@st.cache_resource
def get_llm():
    """
    Loads the main LLM using llama-cpp-python (GGUF format).
    Uses CPU with optimized settings.
    """
    model_file = LLM_MODEL_PATH / LLM_MODEL_FILE
    
    if not model_file.exists():
        print(f"⚠️ GGUF 모델 파일을 찾을 수 없습니다: {model_file}")
        print(f"다음 경로에 GGUF 모델을 다운로드하세요: {LLM_MODEL_PATH}")
        print(f"HuggingFace에서 다운로드: https://huggingface.co/{LLM_MODEL_NAME}")
        raise FileNotFoundError(f"GGUF model file not found: {model_file}")
    
    print(f"Loading LLM from {model_file}...")
    
    try:
        llm = Llama(
            model_path=str(model_file),
            n_ctx=4096,          # Context window
            n_threads=8,         # CPU threads
            n_gpu_layers=0,      # CPU only
            verbose=False,
            n_batch=512,         # Batch size for prompt processing
        )
        
        # Wrap in LangChain compatible class
        llm_wrapper = LlamaCppLLM(
            llm=llm,
            max_tokens=512,
            temperature=0.1,
            top_p=0.95
        )
        
        print("✓ LLM loaded successfully")
        return llm_wrapper
        
    except Exception as e:
        print(f"Error loading LLM: {e}")
        raise e


@st.cache_resource
def get_query_llm():
    """
    Loads the Query Expansion LLM using llama-cpp-python (GGUF format).
    Smaller model for faster query rewriting.
    """
    model_file = LLM_QUERY_MODEL_PATH / LLM_QUERY_MODEL_FILE
    
    if not model_file.exists():
        print(f"⚠️ Query GGUF 모델 파일을 찾을 수 없습니다: {model_file}")
        print(f"다음 경로에 GGUF 모델을 다운로드하세요: {LLM_QUERY_MODEL_PATH}")
        print(f"HuggingFace에서 다운로드: https://huggingface.co/{LLM_QUERY_MODEL_NAME}")
        raise FileNotFoundError(f"GGUF model file not found: {model_file}")
    
    print(f"Loading Query LLM from {model_file}...")
    
    try:
        llm = Llama(
            model_path=str(model_file),
            n_ctx=2048,
            n_threads=4,
            n_gpu_layers=0,
            verbose=False,
            n_batch=256,
        )
        
        llm_wrapper = LlamaCppLLM(
            llm=llm,
            max_tokens=128,
            temperature=0.3,
            top_p=0.9
        )
        
        print("✓ Query LLM loaded successfully")
        return llm_wrapper
        
    except Exception as e:
        print(f"Error loading Query LLM: {e}")
        raise e


# =============================================================================
# RAG Chain with 30% Vector + 70% BM25 Hybrid Retrieval + FlashRank Reranking
# 보고서 섹션 4, 5 권장사항 구현
# =============================================================================

def get_rag_chain(vector_store, all_texts):
    """
    Creates RAG chain with hybrid retrieval (30% Vector + 70% BM25) + FlashRank Reranking.
    보고서 권장: BM25에 더 높은 가중치 부여
    
    Args:
        vector_store: FAISS vector store
        all_texts: List of all chunked documents for BM25
    
    Returns:
        RAG chain with hybrid retrieval and reranking
    """
    llm = get_llm()
    
    # Vector retriever (30% 가중치)
    vector_retriever = vector_store.as_retriever(
        search_kwargs={"k": 30}  # 후보군 확보
    )
    
    # BM25 retriever (70% 가중치) - for exact keyword matching
    bm25_retriever = BM25Retriever.from_documents(all_texts)
    bm25_retriever.k = 30  # 후보군 확보
    
    # Hybrid retriever with 30:70 ratio (Vector:BM25)
    # 보고서 권장: BM25 0.7, Vector 0.3
    hybrid_retriever = HybridRetriever(
        vector_retriever=vector_retriever,
        bm25_retriever=bm25_retriever,
        vector_weight=0.3  # BM25가 0.7
    )
    
    # Domain-specific prompt
    prompt = ChatPromptTemplate.from_template("""
    너는 하드웨어 제어 및 시스템 엔지니어링 전문가야. 
    제공된 [Context]에 기반하여 사용자의 질문에 답해줘.

    [답변 규칙]
    오직 제공된 [Context]에 있는 정보만 사용해. 모르는 내용은 추측하지 마.
    실행해야 할 명령어(cp, SetPrimary 등)는 코드 블럭(```)을 사용하여 명확히 표시해.
    주석(#)이 포함된 데이터의 경우 주석의 의미(예: 마스터/슬레이브 설정)를 함께 설명해.
    파일 경로(예: /nvme0:2/)를 생략하지 말고 전체 경로를 그대로 적어줘.
    만약 비슷한 버전이 여러 개 있다면, 버전 번호를 비교하여 사용자에게 확인시켜줘.

    [답변 형식]
    요약: 질문에 대한 한 줄 답변
    명령어: [명령어 코드 블럭]
    상세 설명: 각 파라미터나 경로에 대한 설명

    [Context] {context}
    User: {question}
    Answer:""",
    )
    
    # RAG chain
    rag_chain_from_docs = (
        RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
        | prompt
        | llm.bind(stop=["User:", "[Context]", "Answer:"])
        | StrOutputParser()
    )
    
    # With source documents
    rag_chain_with_source = RunnableParallel(
        {"context": hybrid_retriever, "question": RunnablePassthrough()}
    ).assign(answer=rag_chain_from_docs)
    
    return rag_chain_with_source


# =============================================================================
# Query Rewriting - 보고서 섹션 6.1 도메인 특화 쿼리 재작성
# =============================================================================

def rewrite_query(original_query):
    """
    Rewrites the user query using the Query LLM to improve retrieval.
    Extracts technical keywords for better search.
    
    보고서 섹션 6.1 권장사항:
    - 동의어 확장: "복사" → cp, "네트워크" → netstat
    - 식별자 보존: V09 등 정확히 유지
    - 약어 풀이: sdi → SDI.txt, sdi.out 등
    """
    try:
        llm = get_query_llm()
        
        # 보고서 권장 프롬프트 적용
        prompt = (
            "당신은 쉘 명령어 전문가입니다. 사용자의 질문을 검색 엔진에 최적화된 쿼리로 변환하세요.\n"
            "[규칙]\n"
            "1. 사용자가 언급한 파일명이나 숫자(V09, V10 등)는 정확히 유지하세요.\n"
            "2. '복사', '설정' 같은 자연어는 'cp', 'SetPrimary' 같은 실제 명령어로 변환하여 추가하세요.\n"
            "3. 모듈명(MSR, HTR, SDI, SDIT, ISP)을 반드시 포함하세요.\n"
            "4. 파일 확장자(.bit, .out, .txt)가 유추되면 포함하세요.\n"
            "5. 불필요한 조사나 서술어는 제외하고 콤마(,)로 구분된 키워드만 출력하세요.\n"
            "[입력 예시] \"MSR 필터 기능이 들어간 v09 비트스트림 복사하는 법 알려줘\"\n"
            "[출력 예시] cp, MSR, V09, ping_filter, .bit\n"
            f"질문: {original_query}\n"
            "키워드:"
        )

        # Generate response with stop token to prevent hallucinating more examples
        response = llm._call(prompt)
        
        # Post-processing: Take only the first line and clean up
        refined_query = response.strip()
        
        # Clean up common prefix patterns
        if "키워드:" in refined_query:
            refined_query = refined_query.split("키워드:")[-1].strip()
        
        if refined_query and refined_query != original_query:
            print(f"Original Query: {original_query} → Refined Query: {refined_query}")
            return refined_query
        else:
            print(f"Using original query: {original_query}")
            return original_query
            
    except Exception as e:
        print(f"Query rewriting failed: {e}")
        return original_query
