# ---------------------------------------------------------
# Streamlit AI 검색엔진
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

# 임베딩 (Embeddings)
from langchain_huggingface import HuggingFaceEmbeddings

# 벡터 스토어 및 검색기 (Hybrid Search)
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
# 검색기
from typing import List
from langchain_core.documents import Document
from langchain_core.runnables import chain
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from sentence_transformers import SentenceTransformer

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

# --- 문서 처리 로직 ---

@st.cache_resource
def get_ensemble_retriever(directory_path, _embeddings):
    """
    문서를 로드, 분할하고 하이브리드 검색기(BM25 + VectorStore)를 초기화하여 반환합니다.
    """
    if not os.path.exists(directory_path):
        print(f"경로를 찾을 수 없습니다: {directory_path}")
        return None
        
    print(f"작업 디렉토리: {os.path.abspath(directory_path)}")
    docs = load_all_documents(directory_path)
    
    print("-" * 30)
    print(f"최종 로드된 문서 총 개수: {len(docs)}")
    
    # 텍스트 분할기 설정
    text_spliter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 200,
        length_function = len,
        is_separator_regex = False,
    )
    
    all_splits = text_spliter.split_documents(docs)
    print(f"분할된 문서의 총 개수: {len(all_splits)}")
    
    if not all_splits:
        print("분할된 문서가 없습니다.")
        return None

    # 1. BM25 검색기 초기화 (키워드 매칭)
    print("BM25 검색기 초기화 중...")
    bm25_retriever = BM25Retriever.from_documents(all_splits)
    bm25_retriever.k = 3
    
    # 2. 벡터 스토어 초기화 및 검색기 설정 (의미 매칭)
    print("벡터 스토어 초기화 및 문서 임베딩 시작...")
    v_store = InMemoryVectorStore(_embeddings)
    v_store.add_documents(documents=all_splits)
    vector_retriever = v_store.as_retriever(search_kwargs={"k": 3})
    print(f"벡터 스토어 초기화 완료! (청크 개수: {len(all_splits)})")
    
    # 3. 앙상블 검색기 (Hybrid Search) 설정
    # BM25(50%) + Semantic(50%) 가중치 적용
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.5, 0.5]
    )
    
    return ensemble_retriever

# 분할기 설정 및 분할 실행 부분은 get_vector_store 함수 내부로 이동됨

# ---------------------------------------------------------
# 1.5 LLM 모델 로드
# ---------------------------------------------------------
import os
import streamlit as st
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface import HuggingFacePipeline

# 🔐 Hugging Face 토큰 로그인 (최초 1회만 실행되면 됨)
login(token="huggingface token here") 
# login(token=os.environ.get("huggingface token here"))  
# 또는 login(token="huggingface token here")

llm_model_id = "google/functiongemma-270m-it"
llm_save_path = "./models/functiongemma-270m-it"

# 다운로드 및 저장 (최초 1회)
if not os.path.exists(llm_save_path):
    print(f"LLM 모델({llm_model_id})을 다운로드하여 로컬에 저장합니다...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(llm_model_id)
        model = AutoModelForCausalLM.from_pretrained(
            llm_model_id,
            torch_dtype="auto",
            device_map="auto"
        )
        tokenizer.save_pretrained(llm_save_path)
        model.save_pretrained(llm_save_path)
        print("LLM 모델 저장 완료!")
    except Exception as e:
        print(f"모델 다운로드 중 오류 발생: {e}")

# Streamlit 캐싱
@st.cache_resource
def load_llm_model(model_path):
    print(f"LLM 모델 로드 중: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map="auto"
    )

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512
    )
    return HuggingFacePipeline(pipeline=pipe)

def format_gemma_prompt(messages: list) -> str:
    """
    메시지 리스트(JSON/Dict 형태)를 Gemma 전용 프롬프트 문자열로 변환합니다.
    """
    prompt = ""
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content")
        prompt += f"<start_of_turn>{role}\n{content}\n<end_of_turn>\n"
    prompt += "<start_of_turn>model\n"
    return prompt

try:
    if os.path.exists(llm_save_path):
        llm = load_llm_model(llm_save_path)
        print("LLM 모델 로드 성공!")
    else:
        print("LLM 모델 경로가 존재하지 않아 로드할 수 없습니다.")
        llm = None
except Exception as e:
    st.error(f"LLM 모델 로드 실패: {e}")
    llm = None


# ---------------------------------------------------------
# 텍스트 임베딩을 생성하는 모델을 준비했다면, 이제 이를 효율적인 **유사도 검색(Similarity Search)** 을 지원하는 **벡터 스토어(Vector Stores)** 에 저장할 수 있습니다.
# ---
# ---------------------------------------------------------
# ---------------------------------------------------------
# ## **임베딩 (Embeddings) 및 벡터 스토어 (Vector Store)**
# ---------------------------------------------------------
# 1. 임베딩 모델 설정 (nlpai-lab/KURE-v1)
embedding_model_id = "nlpai-lab/KURE-v1"
embedding_save_path = "./models/KURE-v1"

# 모델 저장 및 로드
if not os.path.exists(embedding_save_path):
    print(f"임베딩 모델({embedding_model_id})을 다운로드하여 로컬에 저장합니다...")
    try:
        model = SentenceTransformer(embedding_model_id)
        model.save(embedding_save_path)
        print("임베딩 모델 저장 완료!")
    except Exception as e:
        print(f"임베딩 모델 다운로드 중 오류 발생: {e}")

# HuggingFaceEmbeddings 객체 생성 (캐싱 적용)
@st.cache_resource
def get_embeddings(model_path):
    return HuggingFaceEmbeddings(
        model_name=model_path,
        model_kwargs={'device': 'cpu'} # GPU 사용 시 'cuda'로 변경
    )

embeddings = get_embeddings(embedding_save_path)

# 2. 하이브리드 검색기 초기화 (캐싱 적용)
doc_dir = "./doc"
retriever = get_ensemble_retriever(doc_dir, embeddings)

# ---------------------------------------------------------
# ## **검색기 (Retrievers) 설정**
# ---------------------------------------------------------
if retriever:
    print(f"검색 준비 완료!")
else:
    print(f"검색기가 준비되지 않았습니다.")



# ---------------------------------------------------------
# Streamlit 페이지 및 헤더
# ---------------------------------------------------------
st.set_page_config(page_title="AI 검색엔진 챗봇", page_icon="🔍")
st.markdown("<h1 style='text-align: center;'>AI 검색엔진 챗봇 🔍</h1>", unsafe_allow_html=True)
print(f"Streamlit 페이지 및 헤더 설정 완료!")

SystemMessage_content = "당신은 전문적인 기술 지원 어시스턴트입니다. 제공된 문맥을 바탕으로 정확하고 친절하게 답변해 주세요. 답변은 5줄 이내로 해주세요."
# ---------------------------------------------------------
# Session State 초기화
# ---------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [SystemMessage(content=SystemMessage_content)]

# ---------------------------------------------------------
# 사이드바
# ---------------------------------------------------------
st.sidebar.title("설정 및 기능 😎")
refresh_button = st.sidebar.button("대화 내용 초기화")
summaries_button = st.sidebar.button("대화 내용 요약")

if refresh_button:
    st.session_state.messages = [SystemMessage(content=SystemMessage_content)]
    st.rerun()

if summaries_button and llm:
    with st.sidebar:
        with st.spinner("대화 요약 중..."):
            conversation_text = []
            for msg in st.session_state.messages:
                if isinstance(msg, SystemMessage): role = "System"
                elif isinstance(msg, HumanMessage): role = "User"
                elif isinstance(msg, AIMessage): role = "AI"
                else: role = "Unknown"
                conversation_text.append(f"{role}: {msg.content}")
            
            joined_conversation = "\n".join(conversation_text)
            # Gemma 요약 프롬프트 구조화 (JSON/Dictionary 스타일)
            messages = [
                {
                    "role": "user", 
                    "content": f"다음 대화 내용을 한국어로 간결하게 요약해 주세요:\n\n{joined_conversation}"
                }
            ]
            prompt_content = format_gemma_prompt(messages)
            
            summary_response = llm.invoke(prompt_content)
            
            # AI가 새로 작성한 요약만 추출 (프롬프트 제외)
            if prompt_content in summary_response:
                summary_response = summary_response.replace(prompt_content, "").strip()
            summary_response = summary_response.replace("<end_of_turn>", "").strip()

            st.write("**대화 요약:**")
            # 문장별 줄바꿈이 마크다운에서 잘 보이도록 처리
            st.write(summary_response.replace("\n", "  \n"))

st.sidebar.divider()
st.sidebar.info("이 검색엔진은 로컬에 저장된 문서를 기반으로 답변을 검색합니다.")
print(f"Streamlit 사이드바 설정 완료!")

# ---------------------------------------------------------
# 메인 검색 인터페이스
# ---------------------------------------------------------
st.subheader("질문하기")
print(f"Streamlit 메인 검색 인터페이스 설정 완료!")

with st.form(key='chat_form', clear_on_submit=True):
    user_input = st.text_area("궁금한 내용을 입력하세요:", placeholder="예: 모의신호 생성방법", height=100)
    submit_button = st.form_submit_button(label='전송')
    print(f"Streamlit 메인 검색 인터페이스 설정 완료!")

    if submit_button and user_input:
        st.session_state.messages.append(HumanMessage(content=user_input))
        with st.spinner("문서를 검색하고 답변을 생성 중입니다..."):
            try:
                # 1. 앙상블 검색기(Hybrid)를 활용한 문서 검색
                if not retriever:
                     st.session_state.messages.append(AIMessage(content="검색기가 초기화되지 않았습니다. 문서를 확인해 주세요."))
                else:
                    # EnsembleRetriever는 invoke() 사용 (Score는 별도 제공되지 않음)
                    results = retriever.invoke(user_input)
                    
                    if not results:
                        st.session_state.messages.append(AIMessage(content="관련된 문서를 찾을 수 없어 답변을 생성하기 어렵습니다."))
                    else:
                        # 컨텍스트 구성
                        context_text = "\n\n".join([doc.page_content for doc in results])
                        
                        # 2. Gemma 공식 프롬프트 구조화 (JSON/Dictionary 스타일)
                        history_text = ""
                        recent_messages = st.session_state.messages[-6:-1] 
                        for msg in recent_messages:
                            if isinstance(msg, HumanMessage):
                                history_text += f"User: {msg.content}\n"
                            elif isinstance(msg, AIMessage):
                                history_text += f"Assistant: {msg.content}\n"

                        # 프롬프트 구성 데이터를 딕셔너리 형태로 정의
                        prompt_data = {
                            "instruction": "당신은 전문적인 기술 지원 어시스턴트입니다. 제공된 [문맥]과 [이전 대화]를 바탕으로 사용자의 [질문]에 정확하고 친절하게 한국어로 답변해 주세요.\n반드시 제공된 문맥의 정보만을 사용하고, 문맥에서 답을 찾을 수 없다면 솔직하게 모른다고 답변하십시오.\n5줄 이내로 답변하십시오.",
                            "history": history_text if history_text else "이전 대화 없음",
                            "context": context_text,
                            "question": user_input
                        }

                        messages = [
                            {
                                "role": "user",
                                "content": (
                                    f"{prompt_data['instruction']}\n\n"
                                    f"[이전 대화]\n{prompt_data['history']}\n\n"
                                    f"[문맥]\n{prompt_data['context']}\n\n"
                                    f"[질문]\n{prompt_data['question']}"
                                )
                            }
                        ]
                        
                        prompt = format_gemma_prompt(messages)
                        
                        ai_response = llm.invoke(prompt)
                        
                        # AI가 새로 작성한 답변만 추출 (프롬프트 제외)
                        if prompt in ai_response:
                            ai_response = ai_response.replace(prompt, "").strip()
                        
                        # Gemma 특유의 대화 종료 태그 등 정제
                        ai_response = ai_response.replace("<end_of_turn>", "").strip()
                        
                        # 검색 결과 출처 정보 포함 (하이브리드 검색은 기본 점수 미제공)
                        source_info = "\n\n---\n**🔍 참고 문서 (Hybrid Search):**\n"
                        for i, doc in enumerate(results):
                            source = doc.metadata.get('source', '알 수 없음')
                            source_info += f"{i+1}. **{os.path.basename(source)}**\n"
                        
                        full_response = ai_response + source_info
                        st.session_state.messages.append(AIMessage(content=full_response))
                                
            except Exception as e:
                st.error(f"오류가 발생했습니다: {e}")

# ---------------------------------------------------------------------------------
# 마지막 AIMessage 폼 바로 아래에 표시 (x060 스타일)
# ---------------------------------------------------------------------------------
if st.session_state.messages:
    last_msg = st.session_state.messages[-1]
    if isinstance(last_msg, AIMessage):
        # 마크다운에서 줄바꿈이 정상적으로 보이도록 \n을 "  \n"으로 변환
        formatted_content = last_msg.content.replace("\n", "  \n")
        st.info(formatted_content)

# 이전 대화 이력 표시 (시간순)
st.divider()
st.subheader("이전 대화 이력")
for idx, msg in enumerate(st.session_state.messages):
    if isinstance(msg, SystemMessage):
        continue  # 시스템 메시지는 채팅창에 표시하지 않음
    if isinstance(msg, HumanMessage):
        message(msg.content, is_user=True, key=str(idx) + "_user")
    elif isinstance(msg, AIMessage):
        message(msg.content, is_user=False, key=str(idx) + "_ai")
