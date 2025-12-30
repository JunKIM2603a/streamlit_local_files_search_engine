# ---------------------------------------------------------
# Streamlit 시멘틱 검색엔진
# 시맨틱 검색 엔진 구축하기
# 이 노트북은 LangChain의 문서 로더, 임베딩, 그리고 벡터 스토어 추상화 개념에 익숙해지도록 설계되었습니다. 이러한 추상화는 (벡터) 데이터베이스 및 기타 소스에서 데이터를 검색하여 LLM 워크플로우에 통합하도록 지원합니다. 이는 **RAG (Retrieval-Augmented Generation)** 와 같은 모델 추론의 일부로 데이터를 검색해 활용하는 애플리케이션에 중요합니다.
# 이 노트북에서는 PDF 문서에 대한 검색 엔진을 구축합니다. 이를 통해 입력 쿼리와 유사한 PDF의 특정 구절을 검색할 수 있습니다.

# ### 개념
# 이 가이드는 텍스트 데이터 검색에 중점을 둡니다. 아래 개념을 다룰 것입니다:
# - 문서 및 문서 로더
# - 텍스트 분할기 (Text splitters)
# - 임베딩 (Embeddings)
# - 벡터 스토어 및 검색기 (Vector stores and retrievers)

# ### 문서 및 문서 로더 (Documents and Document Loaders)
# LangChain은 Document 추상화를 제공합니다. 이 객체는 텍스트 단위와 관련 메타데이터를 나타내기 위해 설계되었습니다.  

# Document의 주요 속성:  
# - page_content: 문서의 내용을 나타내는 문자열입니다.  
# - metadata: 문서의 출처, 다른 문서와의 관계 등 임의의 메타데이터를 포함하는 딕셔너리입니다.  
# - id: (선택적) 문서를 식별하는 문자열입니다.  

# metadata 속성은 문서의 출처, 문서 간 관계 및 기타 정보를 포함할 수 있습니다. 일반적으로 하나의 Document 객체는 더 큰 문서의 일부(청크)를 나타냅니다.

# 예제 문서 생성:
# ```python
# from langchain_core.documents import Document

# documents = [
#     Document(
#         page_content="개는 충성심과 친근함으로 잘 알려진 훌륭한 동반자입니다. ",
#         metadata={"source": "mammal-pets-doc"},
#     ),
#     Document(
#         page_content="고양이는 독립적인 반려동물로, 종종 자신만의 공간을 즐깁니다.",
#         metadata={"source": "mammal-pets-doc"},
#     ),
# ]
# ```
# ---------------------------------------------------------

import os
import asyncio

import streamlit as st
from streamlit_chat import message

# 문서 로더 (Document Loaders)
# 각 파일 형식에 맞는 로더를 사용하여 안정성을 높입니다.
from langchain_community.document_loaders import (
    DirectoryLoader,           # 디렉토리 내 여러 파일을 한 번에 로드하는 기본 로더
    PyPDFLoader,                # PDF 파일 전용 로더
    TextLoader,                 # 텍스트 파일 전용 로더
    UnstructuredExcelLoader,    # 엑셀 파일 로더 (unstructured 패키지 필요)
    UnstructuredPowerPointLoader # PPT 파일 로더 (unstructured 패키지 필요)
)
# 백업용/기타 파일용
from langchain_unstructured import UnstructuredLoader

# 텍스트 분할기
from langchain_text_splitters import RecursiveCharacterTextSplitter
# VectorScore
from langchain_core.vectorstores import InMemoryVectorStore
# 검색기
from typing import List
from langchain_core.documents import Document
from langchain_core.runnables import chain

# ---------------------------------------------------------
# ## **문서 로드하기**
# 다양한 포맷(.pdf, .txt, .xlsx, .pptx)의 파일을 `Document` 객체로 로드합니다.
# ---------------------------------------------------------
# 1. 확장자별로 사용할 로더 클래스를 매핑 (관리하기 편하도록 설정)
loader_mapping = {
    ".pdf": PyPDFLoader,
    ".xlsx": UnstructuredExcelLoader,
    ".pptx": UnstructuredPowerPointLoader,
    ".txt": TextLoader,
}

def load_all_documents(directory_path):
    """
    지정된 디렉토리 내의 모든 문서를 확장자별 로더를 사용하여 로드합니다.
    """
    all_docs = []
    
    # 2. 로더 맵에 정의된 각 확장자별로 순회하며 로드 수행
    for ext, loader_cls in loader_mapping.items():
        try:
            print(f"[{ext}] 파일 로드 시작...")
            
            # 3. 로더별 특수 설정 (kwargs) 준비
            loader_kwargs = {}
            if ext == ".txt":
                # 한글 윈도우 환경에서 작성된 TXT는 utf-8 또는 cp949 인코딩이 필요함
                loader_kwargs = {"encoding": "utf-8"}

            # 4. DirectoryLoader 설정
            # - glob: "**/*{ext}" 는 모든 하위 디렉토리에서 해당 확장자를 찾으라는 의미
            # - show_progress: 로드 과정을 진행 바로 표시
            # - use_multithreading: 여러 파일을 동시에 읽어 속도 향상
            loader = DirectoryLoader(
                directory_path,
                glob=f"**/*{ext}", 
                loader_cls=loader_cls,
                loader_kwargs=loader_kwargs,
                show_progress=True,
                use_multithreading=True 
            )
            
            # 실제 문서 로드 및 리스트 추가
            loaded_docs = loader.load()
            all_docs.extend(loaded_docs)
            print(f"[{ext}] 로드 완료: {len(loaded_docs)}개 문서")
            
        except Exception as e:
            # 5. TXT 파일의 경우 인코딩 에러(UnicodeDecodeError)가 자주 발생하므로 재시도 로직 추가
            if ext == ".txt":
                 print(f"utf-8 로드 실패. cp949 인코딩으로 재시도합니다... 에러: {e}")
                 try:
                     loader = DirectoryLoader(
                         directory_path, 
                         glob="**/*.txt", 
                         loader_cls=TextLoader, 
                         loader_kwargs={"encoding": "cp949"}
                     )
                     all_docs.extend(loader.load())
                 except Exception as retry_e:
                     print(f"cp949 재시도도 실패했습니다: {retry_e}")
            else:
                print(f"{ext} 파일 그룹 로드 중 오류 발생: {e}")

    return all_docs

# --- 실행 부분 ---

# 문서가 포함된 폴더 경로 (상대 경로 혹은 절대 경로)
doc_dir = "./doc"
docs = []
# 경로가 실제로 존재하는지 확인 후 로드 시작
if os.path.exists(doc_dir):
    print(f"작업 디렉토리: {os.path.abspath(doc_dir)}")
    docs = load_all_documents(doc_dir)
    
    print("-" * 30)
    print(f"최종 로드된 문서 총 개수: {len(docs)}")
    print("문서 로드 프로세스가 완료되었습니다.")
else:
    print(f"경로를 찾을 수 없습니다: {doc_dir}")
print(f"문서 로드 완료!")

# ---------------------------------------------------------
# ## **텍스트 분할 (Splitting)**  
# 정보 검색 및 후속 질문-응답(question-answering) 작업을 하는데 **페이지 단위** 사용은 너무 거친 표현일 수 있습니다. 우리의 최종 목표는 **입력 쿼리에 답변할 수 있는 `Document` 객체를 검색하는 것**이므로, PDF를 더 세분화하면 문서의 해당되는 부분이 주변 텍스트에 의해 의미가 흐려지는 것을 방지할 수 있습니다.  
# 이를 위해 텍스트 분할기(text splitters)를 사용할 수 있습니다. 여기서는 **문자를 기준으로 분할하는 간단한 텍스트 분할기**를 사용할 것입니다.  
# - 문서를 **1000자 단위로 분할**합니다.  
# - 각 청크(chunk) 사이에는 **200자의 중첩(overlap)** 이 있습니다.  
# **중첩**은 중요한 문맥(context)과 문장(statement)이 분리되는 것을 완화하는 데 도움이 됩니다.  
# 우리는 **RecursiveCharacterTextSplitter**를 사용할 것입니다. 이 분할기는 일반적인 구분자(예: 줄바꿈)를 사용하여 **재귀적으로 문서를 분할**하며, 각 청크가 적절한 크기가 될 때까지 반복합니다. **일반 텍스트 사용 사례에서 권장되는 텍스트 분할기입니다.**  
# ### **추가 설정**  
# - `add_start_index=True`로 설정하면 **각 분할된 `Document`가 원본 `Document` 내에서 시작하는 문자 인덱스**가 **메타데이터 속성(`start_index`)** 으로 보존됩니다.  
# ---------------------------------------------------------
# RecursiveCharacterTextSplitter 설정
text_spliter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200,
    length_function = len,
    is_separator_regex = False,
)
    
# 문서를 분할합니다. 'docs'는 분할할 원본 문서 목록입니다.
all_splits = text_spliter.split_documents(docs)

# 분할된 문서의 총 개수
len(all_splits)
print(f"분할된 문서의 총 개수: {len(all_splits)}")

# ---------------------------------------------------------
# ## **임베딩 (Embeddings)**  
# **벡터 검색(Vector Search)** 은 비정형 데이터(예: 비정형 텍스트)를 저장하고 검색하는 일반적인 방법입니다. 핵심 아이디어는 텍스트와 연결된 **숫자 벡터(Numeric Vectors)** 를 저장하는 것입니다.  
# 쿼리(Query)가 주어지면, 이를 **동일한 차원의 벡터로 임베딩(Embedding)** 하고 **벡터 유사도 측정(metric)** (예: **코사인 유사도**)을 사용하여 관련 텍스트를 식별할 수 있습니다.   
# ### **모델 선택하기**  
# 원하는 임베딩 모델을 선택해 사용합니다.  
# 예를 들어, OpenAI, Hugging Face, Cohere 등 다양한 제공자를 사용할 수 있습니다.  
# **임베딩 모델을 선택한 후 벡터 스토어(Vector Store)에 텍스트를 저장하고 검색**할 수 있습니다.
# ---------------------------------------------------------
# 1. 모델 저장
# snapshot_download를 사용할 때 가끔 설정 파일이 온전하게 매핑되지 않는 경우가 있습니다. 
# 이럴 때는 SentenceTransformer 객체를 통해 모델을 한 번 로드한 뒤, 
# 그대로 로컬에 save() 하는 방식이 가장 안전합니다. 
# 이 방식은 config.json 뿐만 아니라 관련 설정들을 자동으로 완벽하게 구성해 줍니다.
import os
from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceEmbeddings

# 1. 모델을 메모리에 로드하고 로컬에 다시 저장 (최초 1회만 실행)
model_id = "nlpai-lab/KURE-v1"
save_path = "./models/KURE-v1"

if not os.path.exists(save_path):
    print("모델을 다운로드하여 로컬 표준 형식으로 저장합니다...")
    model = SentenceTransformer(model_id)
    model.save(save_path)
    print(f"저장 완료: {save_path}")

# 2. 로컬 경로에서 불러오기
embeddings = HuggingFaceEmbeddings(
    model_name=save_path,
)

# 테스트
print("모델 로드 성공!")


# ---------------------------------------------------------
# 텍스트 임베딩을 생성하는 모델을 준비했다면, 이제 이를 효율적인 **유사도 검색(Similarity Search)** 을 지원하는 **벡터 스토어(Vector Stores)** 에 저장할 수 있습니다.
# ---
# ## **벡터 스토어(Vector Stores)**  
# LangChain의 [**VectorStore**](https://python.langchain.com/api_reference/core/vectorstores/langchain_core.vectorstores.base.VectorStore.html) 객체는 **텍스트 및 `Document` 객체를 저장**하고, 다양한 **유사도 메트릭(Similarity Metrics)** 을 사용해 쿼리를 수행할 수 있는 메서드를 포함합니다.  
# 이 객체들은 종종 **임베딩(Embedding)** 모델로 초기화되며, 해당 모델은 **텍스트 데이터를 숫자 벡터로 변환하는 방법**을 결정합니다.   
# - 일부 벡터 스토어는 **클라우드 제공업체(Cloud Providers)** 에서 호스팅되며 특정 **자격 증명(Credentials)** 이 필요합니다.  
# - 일부 벡터 스토어(예: **Postgres**)는 독립적인 인프라에서 실행되거나 로컬 또는 서드파티 플랫폼을 통해 운영될 수 있습니다.  
# - 일부 벡터 스토어는 **인메모리(In-Memory)** 로 실행되어 가벼운 작업에 적합합니다.  
# ---------------------------------------------------------
from langchain_core.vectorstores import InMemoryVectorStore

vector_store = InMemoryVectorStore(embeddings)

ids = vector_store.add_documents(documents=all_splits)
print(f"벡터 스토어에 문서 추가 완료!")

# ---------------------------------------------------------
# ## **검색기 (Retrievers)**  
# LangChain의 **`VectorStore` 객체**는 [**Runnable**](https://python.langchain.com/api_reference/core/index.html#langchain-core-runnables)의 서브클래스가 아닙니다. 반면, **[Retrievers](https://python.langchain.com/api_reference/core/index.html#langchain-core-retrievers)** 는 **Runnable**의 서브클래스입니다. 따라서 **동기(Synchronous)** 및 **비동기(Asynchronous)** `invoke`와 `batch` 작업과 같은 표준 메서드 집합을 구현합니다.  
# ### **검색기 vs 벡터 스토어**  
# - **벡터 스토어(VectorStore)**: 벡터화된 데이터를 저장하고 쿼리합니다.  
# - **검색기(Retriever)**: 벡터 스토어를 활용하거나 외부 API와 같은 **비벡터 데이터 소스**와도 통신할 수 있습니다.  
# ---------------------------------------------------------
@chain
def retriever(query: str) -> List[Document]:
    return vector_store.similarity_search_with_score(query, k=1)

# `as_retriever` 메서드를 사용하여 벡터 스토어를 검색기 객체로 변환합니다.
retriever = vector_store.as_retriever(
    search_type="similarity",  # 검색 유형을 'similarity'(유사도)로 설정
    search_kwargs={"k": 3},  # 상위 1개의 유사한 결과만 반환하도록 설정
)

# MMR 방식 (Maximum Marginal Relevance) - 검색 결과 다양성을 증가
retriever_mmr = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 3, "lambda_mult": 0.5}  # lambda_mult 조절 가능 (0~1 사이 값)
)

# Similarity Score Threshold 방식 - 일정 유사도 이상인 문서만 반환
retriever_threshold = vector_store.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"score_threshold": 0.6, "k": 1}  # 0.6 이상의 유사도 점수를 가진 문서만 반환
)

print(f"검색 완료!")



# ---------------------------------------------------------
# Streamlit 페이지 및 헤더
# ---------------------------------------------------------
st.set_page_config(page_title="시멘틱 검색엔진", page_icon="🔍")
st.markdown("<h1 style='text-align: center;'>시멘틱 검색엔진 🔍</h1>", unsafe_allow_html=True)
print(f"Streamlit 페이지 및 헤더 설정 완료!")

# ---------------------------------------------------------
# 사이드바
# ---------------------------------------------------------
st.sidebar.title("설정")
st.sidebar.info("이 검색엔진은 로컬에 저장된 문서를 기반으로 답변을 검색합니다.")
print(f"Streamlit 사이드바 설정 완료!")

# ---------------------------------------------------------
# 메인 검색 인터페이스
# ---------------------------------------------------------
st.subheader("질문하기")
print(f"Streamlit 메인 검색 인터페이스 설정 완료!")

with st.form(key='search_form'):
    query = st.text_input("궁금한 내용을 입력하세요:", placeholder="예: 모의신호 생성방법")
    submit_button = st.form_submit_button(label='검색')
    print(f"Streamlit 메인 검색 인터페이스 설정 완료!")

    if submit_button and query:
        with st.spinner("문서를 검색 중입니다..."):
            try:
                # 1. 벡터 스토어를 사용하여 관련 문서와 유사도 점수 검색
                #    Retriever.invoke는 점수를 반환하지 않으므로 vector_store.similarity_search_with_score를 사용해야 합니다.
                results = vector_store.similarity_search_with_score(query, k=3)
                
                if not results:
                    st.warning("관련된 문서를 찾을 수 없습니다.")
                else:
                    st.success(f"{len(results)}개의 관련 문서를 찾았습니다.")
                    
                    for i, (doc, score) in enumerate(results):
                        st.markdown(f"#### 📄 문서 {i+1}")
                        print(f"매칭 점수: {score}")

                        # 메타데이터 표시 (출처 등)
                        source = doc.metadata.get('source', '알 수 없음')
                        st.caption(f"출처: {source}")
                        print(f"출처: {doc.metadata.get('source', '알 수 없음')}")
                        st.caption(f"유사도: {score}")
                        print(f"유사도: {score}")
                        
                        # 본문 내용을 컨테이너로 감싸서 문서처럼 표현
                        with st.container(border=True):
                            # HTML 태그를 사용하여 강제로 줄바꿈 적용
                            content_html = doc.page_content.replace("\n", "<br>")
                            st.markdown(content_html, unsafe_allow_html=True)
                            
                        st.divider()
                            
            except Exception as e:
                st.error(f"검색 중 오류가 발생했습니다: {e}")
