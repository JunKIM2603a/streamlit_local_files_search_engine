# GGUF 모델 다운로드 가이드

## 필요한 모델

RAG 시스템을 실행하기 위해 다음 GGUF 모델을 다운로드해야 합니다:

### 1. 메인 LLM 모델 (7B)

**모델**: Qwen/Qwen2.5-Coder-7B-Instruct-GGUF

**다운로드 방법**:

```bash
# HuggingFace CLI로 다운로드 (권장)
cd d:\test\streamlit_local_files_search_engine\local_ai_chatbot\models
mkdir Qwen2.5-Coder-7B-Instruct
cd Qwen2.5-Coder-7B-Instruct

# Q4_K_M 양자화 버전 다운로드 (약 4.4GB)
huggingface-cli download Qwen/Qwen2.5-Coder-7B-Instruct-GGUF qwen2.5-coder-7b-instruct-q4_k_m.gguf --local-dir . --local-dir-use-symlinks False
```

**또는 수동 다운로드**:
1. https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct-GGUF 방문
2. Files and versions 탭에서 `qwen2.5-coder-7b-instruct-q4_k_m.gguf` 다운로드
3. `models/Qwen2.5-Coder-7B-Instruct/` 폴더에 저장

---

### 2. Query 확장 LLM 모델 (1.5B)

**모델**: Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF

**다운로드 방법**:

```bash
cd d:\test\streamlit_local_files_search_engine\local_ai_chatbot\models
mkdir Qwen2.5-Coder-1.5B-Instruct
cd Qwen2.5-Coder-1.5B-Instruct

# Q4_K_M 양자화 버전 다운로드 (약 934MB)
huggingface-cli download Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF qwen2.5-coder-1.5b-instruct-q4_k_m.gguf --local-dir . --local-dir-use-symlinks False
```

**또는 수동 다운로드**:
1. https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF 방문
2. Files and versions 탭에서 `qwen2.5-coder-1.5b-instruct-q4_k_m.gguf` 다운로드
3. `models/Qwen2.5-Coder-1.5B-Instruct/` 폴더에 저장

---

## 예상 디렉토리 구조

```
local_ai_chatbot/
├── models/
│   ├── bge-m3/                          # 임베딩 모델 (자동 다운로드)
│   ├── Qwen2.5-Coder-7B-Instruct/
│   │   └── qwen2.5-coder-7b-instruct-q4_k_m.gguf    # 수동 다운로드 필요
│   └── Qwen2.5-Coder-1.5B-Instruct/
│       └── qwen2.5-coder-1.5b-instruct-q4_k_m.gguf  # 수동 다운로드 필요
├── backend.py
├── app.py
└── data/
    └── command.txt
```

---

## HuggingFace CLI 설치 (필요시)

```bash
pip install huggingface-hub[cli]
```

---

## 다운로드 완료 확인

```bash
# PowerShell
ls d:\test\streamlit_local_files_search_engine\local_ai_chatbot\models\Qwen2.5-Coder-7B-Instruct\*.gguf
ls d:\test\streamlit_local_files_search_engine\local_ai_chatbot\models\Qwen2.5-Coder-1.5B-Instruct\*.gguf
```

두 파일이 모두 표시되면 준비 완료입니다!

---

## 양자화 버전 선택 가이드

- **Q4_K_M** (권장): 속도와 품질의 균형, 4-bit 양자화
- **Q5_K_M**: 더 높은 품질, 약간 느림
- **Q6_K**: 최고 품질, 더 큰 파일 크기
- **Q8_0**: 거의 원본 품질, 가장 큰 크기

메모리가 충분하다면 Q4_K_M 또는 Q5_K_M을 권장합니다.
