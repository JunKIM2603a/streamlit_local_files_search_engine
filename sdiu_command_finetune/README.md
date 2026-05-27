# SDIU 명령어 전용 LLM 파인튜닝 학습 노트

이 프로젝트는 `rag/SDIU명령어_20201217.txt` 파일에 들어 있는 SDIU 명령어를 전용으로 외우는 작은 LLM 검색 앱을 만드는 예제입니다.

목표는 범용 챗봇을 만드는 것이 아닙니다. 사용자가 `송신성공 확인`, `라우팅 테이블 확인`, `노드 연결 끊기면 숫자 0`처럼 한글 설명을 입력했을 때, 파일 안에 실제로 존재하는 명령어 후보를 Top-K 형태로 돌려주는 것입니다.

이 README는 사용법 문서이면서, 동시에 SFT, LoRA, PEFT, 검증 기반 추론 흐름을 이 프로젝트 안에서 어떻게 적용했는지 배우기 위한 글입니다.

## 프로젝트 목표

원본 파일에는 다음처럼 명령어와 설명이 섞여 있습니다.

```text
ShowSendCnt                  // 송신성공 확인
ShowSendCnt_Log 1            // 주기적으로 송신성공 확인 ON  (송신성공 카운터 표시)
                              // 노드와 연결 끊기면, 숫자는 0
ShowSendCnt_Log 0            // 주기적으로 송신성공 확인 OFF
netstat "-r"                 // 라우팅 테이블 확인
```

우리가 만들고 싶은 시스템은 이런 질문에 답해야 합니다.

```text
질문: 주기적으로 송신성공 확인 ON
답: ShowSendCnt_Log 1
```

또는:

```text
질문: 라우팅 테이블 확인
답: netstat "-r"
```

핵심은 일반화가 아니라 의도적 과적합입니다. 이 프로젝트에서는 현재 SDIU 명령어 파일 하나를 정확히 외우는 것이 좋은 결과입니다.

## 전체 동작 흐름

전체 파이프라인은 다음 순서로 동작합니다.

```text
SDIU명령어_20201217.txt
  -> parser.py
  -> CommandRecord 목록
  -> build_dataset.py
  -> SFT JSONL 데이터셋
  -> train_lora.py
  -> Qwen2.5 1.5B LoRA adapter
  -> infer.py
  -> JSON 출력 검증 + fallback 보충
  -> app.py
  -> Streamlit 검색 UI
```

현재 구현 기준 생성 결과는 다음과 같습니다.

```text
records=213
examples=932
```

즉, 원본 txt에서 213개의 명령어 레코드를 만들고, 이것을 학습용 질문-답변 샘플 932개로 확장합니다.

## 왜 파인튜닝인가?

이 문제는 일반적인 문서 검색과 조금 다릅니다.

RAG 방식은 원문을 검색해서 LLM에게 읽히는 구조입니다. 하지만 여기서는 파일이 작고, 답변 형식도 매우 고정적입니다. 사용자는 설명을 입력하고, 시스템은 파일 안의 명령어를 찾아야 합니다.

그래서 이 프로젝트는 RAG보다 더 좁게 갑니다.

```text
이 파일만 외워라.
파일에 없는 명령어는 만들지 마라.
반드시 JSON으로 후보를 반환하라.
```

이런 규칙을 SFT 데이터로 반복 학습시키면, 모델은 SDIU 명령어 검색기처럼 동작하게 됩니다.

다만 LLM은 학습 후에도 존재하지 않는 명령어를 만들 수 있습니다. 그래서 이 프로젝트는 파인튜닝만 믿지 않고, `infer.py`에서 모델 출력을 실제 파싱된 명령어 목록과 다시 대조합니다.

## 사용한 핵심 기술

### SFT

SFT는 Supervised Fine-Tuning의 약자입니다. 모델에게 입력과 정답을 함께 보여주며 학습시키는 방식입니다.

이 프로젝트의 SFT 샘플은 대략 이런 모양입니다.

```json
{
  "query": "송신성공 확인",
  "user": "질문: 송신성공 확인\n반환 개수: 최대 5개\n출력 형식: {\"results\":[{\"command\":\"명령어\",\"description\":\"설명\",\"reason\":\"선택 이유\"}]}",
  "response": "{\"results\":[{\"command\":\"ShowSendCnt\",\"description\":\"송신성공 확인\",\"reason\":\"설명 문구 일치\"}]}"
}
```

입력은 사용자의 질문이고, 정답은 JSON 문자열입니다. 모델은 이 패턴을 반복해서 보면서 “한글 설명을 보면 어떤 명령어 JSON을 출력해야 하는지”를 배웁니다.

### Qwen2.5 Instruct

베이스 모델은 다음 모델을 사용합니다.

```text
Qwen/Qwen2.5-1.5B-Instruct
```

Instruct 모델은 이미 사용자 지시를 따르도록 학습된 모델입니다. 우리는 이 모델을 처음부터 다시 학습하지 않고, SDIU 명령어 검색 태스크에 맞게 추가 학습합니다.

1.5B 모델을 선택한 이유는 다음과 같습니다.

- 로컬 GPU에서 LoRA 학습이 가능하다.
- 한국어 설명과 영문 명령어가 섞인 패턴을 처리할 수 있다.
- 이 프로젝트처럼 작은 도메인 전용 태스크에 충분하다.

### LoRA

LoRA는 Low-Rank Adaptation의 약자입니다. 전체 모델 파라미터를 전부 다시 학습하지 않고, 작은 adapter 파라미터만 추가로 학습합니다.

이 프로젝트에서 학습 결과는 여기에 저장됩니다.

```text
sdiu_command_finetune/models/qwen2.5-1.5b-sdiu-lora/
```

LoRA를 쓰는 이유는 단순합니다.

- 전체 모델을 학습하는 것보다 훨씬 가볍다.
- adapter만 저장하므로 결과물이 작다.
- 원본 Qwen 모델은 그대로 두고, SDIU 명령어용 지식만 덧씌울 수 있다.

### PEFT

PEFT는 Parameter-Efficient Fine-Tuning입니다. LoRA 같은 경량 파인튜닝 기법을 쉽게 적용하기 위한 라이브러리입니다.

`train_lora.py`에서는 PEFT의 `LoraConfig`와 `get_peft_model`을 사용해 Qwen 모델에 LoRA adapter를 붙입니다.

현재 LoRA 설정은 다음 계열 모듈에 adapter를 추가합니다.

```text
q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
```

이는 attention과 feed-forward 계층에 작은 학습 가능한 경로를 추가하는 방식입니다.

### JSON 출력 강제

프롬프트는 모델에게 반드시 JSON만 출력하라고 지시합니다.

```python
SYSTEM_PROMPT = (
    "너는 SDIU명령어_20201217.txt 파일만 외운 명령어 검색 모델이다. "
    "사용자의 한글 설명 또는 명령어 입력을 보고 파일 안의 명령어 후보를 최대 5개 반환한다. "
    "파일에 없는 명령어를 만들지 않는다. 설명은 파일의 // 뒤 설명과 보충 설명만 사용한다. "
    "반드시 JSON만 출력한다."
)
```

하지만 LLM 출력은 언제든 깨질 수 있습니다. 그래서 `infer.py`는 모델 출력에서 JSON 객체를 추출하고, 각 `command`가 실제 명령어 목록에 있는지 검증합니다.

### Fallback Search

Fallback search는 모델이 없거나, 모델 출력이 부족하거나, 모델이 파일에 없는 명령어를 만든 경우를 위한 안전장치입니다.

`retrieval.py`는 다음 정보를 이용해 간단한 점수 기반 검색을 수행합니다.

- 명령어 정확 일치
- 명령어 부분 일치
- 설명 문구 일치
- 보충 설명 일치
- 한글/영문 토큰 일치
- 한글 n-gram 토큰 일치

즉, 파인튜닝 모델이 주인공이지만, 마지막 결과는 항상 파일 기반 검증과 보충을 거칩니다.

## 파서가 하는 일

`parser.py`는 원본 txt를 학습 가능한 구조로 바꿉니다.

최종 레코드 타입은 `CommandRecord`입니다.

```python
CommandRecord(
    command="ShowSendCnt_Log 1",
    description="주기적으로 송신성공 확인 ON (송신성공 카운터 표시)",
    details=[
        "노드와 연결 끊기면, 숫자는 0",
        "노드와 연결 끊기면, 숫자는 더이상 증가 안함",
    ],
    line_numbers=[2],
)
```

파서는 다음 규칙을 사용합니다.

- `명령어 // 설명` 형태는 하나의 명령어 레코드로 만든다.
- 앞부분 없이 `// 설명`만 있는 줄은 직전 명령어의 보충 설명으로 붙인다.
- 설명이 없는 `_Log 0` 명령어는 같은 계열의 `_Log 1` 설명을 참고해 OFF 설명으로 보강한다.
- 섹션 제목, 출력 예시, PCI dump처럼 명령어가 아닌 줄은 학습 대상에서 제외하거나 보충 정보로만 사용한다.

예를 들어:

```text
ShowHTRBoardPingFlag_Log 1      // HTR연동반에서 받는 플래그정보 로깅 (FPGA 수신단)
ShowHTRBoardPingFlag_Log 0
```

두 번째 줄에는 설명이 없지만, 파서는 앞의 `_Log 1` 설명을 참고해 `_Log 0` 항목도 검색 가능하게 만듭니다.

## SFT 데이터셋 생성

데이터셋은 다음 명령으로 만듭니다.

```powershell
python -m sdiu_command_finetune.build_dataset
```

생성 파일:

```text
sdiu_command_finetune/data/sdiu_sft.jsonl
sdiu_command_finetune/data/records.json
```

`build_dataset.py`는 하나의 명령어 레코드에서 여러 질문을 만듭니다.

예를 들어 `ShowSendCnt` 레코드에서는 다음 같은 질문이 만들어질 수 있습니다.

```text
송신성공 확인
송신성공 확인 명령어
송신성공 확인 확인
ShowSendCnt
ShowSendCnt 설명
```

그리고 각 질문의 정답은 Top 5 JSON 후보입니다.

```json
{
  "results": [
    {
      "command": "ShowSendCnt",
      "description": "송신성공 확인",
      "reason": "설명 문구 일치, 한글/영문 토큰 일치, 전체 텍스트 부분 일치"
    },
    {
      "command": "ShowSendCnt_Log 1",
      "description": "주기적으로 송신성공 확인 ON (송신성공 카운터 표시)",
      "reason": "설명 문구 일치, 한글/영문 토큰 일치, 전체 텍스트 부분 일치"
    }
  ]
}
```

여기서 중요한 점은 정답 JSON도 코드로 생성한다는 것입니다. 사람이 932개 샘플을 직접 쓰지 않습니다. 원본 파일을 파싱하고, 파일 기반 검색기로 정답 후보를 만든 뒤, 그 결과를 모델 학습 데이터로 사용합니다.

## LoRA 학습

학습은 다음 명령으로 실행합니다.

```powershell
python -m sdiu_command_finetune.train_lora
```

기본 설정:

```text
base model: Qwen/Qwen2.5-1.5B-Instruct
epochs: 25
batch size: 2
gradient accumulation: 8
learning rate: 2e-4
max length: 1024
output: sdiu_command_finetune/models/qwen2.5-1.5b-sdiu-lora/
```

학습 데이터는 instruction prompt와 response를 이어붙인 형태로 tokenization됩니다. 이때 prompt 부분의 label은 `-100`으로 마스킹합니다.

즉, 모델은 질문 문장을 그대로 따라 쓰는 것이 아니라, response JSON을 생성하는 방향으로만 학습됩니다.

## 추론 안전장치

추론은 `infer.py`가 담당합니다.

모델은 다음 순서로 사용됩니다.

1. 원본 txt를 다시 파싱해 실제 명령어 목록을 만든다.
2. LoRA adapter를 찾는다.
3. 모델이 JSON을 생성한다.
4. JSON에서 `results`를 추출한다.
5. 각 `command`가 실제 명령어 목록에 있는지 확인한다.
6. 없는 명령어는 버린다.
7. 후보가 부족하면 fallback search 결과로 채운다.

이 구조가 중요한 이유는 LLM이 hallucination을 할 수 있기 때문입니다.

예를 들어 모델이 `ShowMftaPingFlag_Log 1` 같은 파일에 없는 명령어를 만들면, `infer.py`는 그 후보를 제거합니다. 그리고 실제 파일 안에 있는 `ShowLfpaPingFlag_Log 1` 같은 후보를 fallback search로 보충합니다.

## Streamlit 앱

앱은 다음 명령으로 실행합니다.

```powershell
streamlit run sdiu_command_finetune/app.py
```

화면에서는 다음을 설정할 수 있습니다.

- SDIU txt 경로
- LoRA adapter 경로
- base model 이름
- fine-tuned model 로드 여부
- Top K 개수
- JSON 원문 표시 여부

LoRA adapter가 없거나 모델 로드에 실패하면 앱은 fallback mode로 동작합니다. 즉, 학습 전에도 검색 UI를 시험해볼 수 있습니다.

### 앱에서 재학습하기

Streamlit 사이드바의 `Training` 영역에는 재학습을 위한 버튼이 있습니다.

- `Preserve base model locally`: 이미 다운로드된 Qwen 베이스 모델을 프로젝트 내부에 보존합니다.
- `Start retraining`: 현재 선택된 SDIU txt 파일을 다시 파싱하고, SFT JSONL을 다시 만든 뒤, LoRA 학습을 백그라운드 프로세스로 시작합니다.
- `Reload model`: 학습이 끝난 뒤 새 adapter를 다시 로드합니다.
- `Training log`: 백그라운드 학습 로그를 확인합니다.

SDIU txt 파일을 수정했다면 다음 순서로 사용하면 됩니다.

```text
1. Streamlit 앱에서 SDIU txt 경로를 확인한다.
2. Start retraining 버튼을 누른다.
3. Training log에서 학습 진행을 확인한다.
4. 학습이 끝나면 Reload model을 누른다.
5. 검색창에서 새 명령어 설명을 테스트한다.
```

앱에서 재학습을 시작하면 내부적으로 다음 명령과 같은 흐름이 실행됩니다.

```powershell
python -m sdiu_command_finetune.retrain --source rag/SDIU명령어_20201217.txt
```

`retrain.py`는 데이터셋 생성과 LoRA 학습을 한 번에 수행합니다.

## 실행 방법

Windows PowerShell 기준 예시입니다.

### 1. 가상환경 활성화

이미 `.venv`가 있다면:

```powershell
.\.venv\Scripts\activate
```

처음부터 만든다면:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

### 2. 의존성 설치

```powershell
pip install -r sdiu_command_finetune/requirements.txt
```

### 3. SFT 데이터셋 생성

```powershell
python -m sdiu_command_finetune.build_dataset
```

정상 출력 예시:

```text
records=213
examples=932
dataset=...\sdiu_command_finetune\data\sdiu_sft.jsonl
records_output=...\sdiu_command_finetune\data\records.json
```

### 4. LoRA 학습

```powershell
python -m sdiu_command_finetune.train_lora
```

학습이 오래 걸릴 수 있습니다. 학습 중에는 `checkpoint-*` 폴더가 생성됩니다.

베이스 모델을 프로젝트 내부에 보존하면서 학습하려면 다음처럼 실행할 수 있습니다.

```powershell
python -m sdiu_command_finetune.train_lora --prepare-base-model
```

이미 보존된 베이스 모델은 다음 위치에 저장됩니다.

```text
sdiu_command_finetune/models/base/qwen2.5-1.5b-instruct/
```

이 폴더가 준비되어 있으면 인터넷이 없어도 재학습과 추론이 로컬 파일만으로 동작합니다.

### 5. CLI 추론 테스트

```powershell
python -m sdiu_command_finetune.infer "송신성공 확인" --top-k 5
```

fallback만 확인하고 싶다면:

```powershell
python -m sdiu_command_finetune.infer "라우팅 테이블 확인" --fallback-only --top-k 3
```

### 6. Streamlit 앱 실행

```powershell
streamlit run sdiu_command_finetune/app.py
```

## 결과 확인 예시

### 송신성공 확인

```powershell
python -m sdiu_command_finetune.infer "송신성공 확인" --top-k 5
```

기대 후보:

```text
ShowSendCnt
ShowSendCnt_Log 1
ShowSendCnt_Log 0
```

### 주기적으로 송신성공 확인 ON

```powershell
python -m sdiu_command_finetune.infer "주기적으로 송신성공 확인 ON" --top-k 5
```

기대 1순위:

```text
ShowSendCnt_Log 1
```

### 노드 연결 끊기면 숫자 0

```powershell
python -m sdiu_command_finetune.infer "노드 연결 끊기면 숫자 0" --top-k 5
```

기대 후보:

```text
ShowSendCnt_Log 1
```

이 질의는 원래 명령어 줄의 주 설명이 아니라, 다음 줄의 보충 설명에서 온 것입니다. 따라서 파서가 보충 설명을 잘 붙였는지 확인하는 좋은 테스트입니다.

### 라우팅 테이블 확인

```powershell
python -m sdiu_command_finetune.infer "라우팅 테이블 확인" --top-k 5
```

기대 1순위:

```text
netstat "-r"
```

## 파일별 역할

```text
sdiu_command_finetune/
  parser.py          원본 txt를 CommandRecord 목록으로 파싱
  retrieval.py       fallback search와 검색 점수 계산
  prompts.py         모델에 줄 system/user prompt 정의
  build_dataset.py   SFT JSONL 데이터셋 생성
  train_lora.py      Qwen2.5 1.5B LoRA 학습
  infer.py           모델 로드, JSON 추출, 명령어 검증, fallback 보충
  app.py             Streamlit UI
  requirements.txt   실행과 학습에 필요한 Python 패키지
  data/
    sdiu_sft.jsonl   학습 샘플
    records.json     파싱된 명령어 레코드
```

## 문제 해결

### 한글이 깨져 보이는 경우

원본 txt와 README는 UTF-8 기준으로 다룹니다. PowerShell 콘솔 출력에서 한글이 깨져 보일 수 있지만, 파일 자체가 깨졌는지와 콘솔 인코딩 문제인지는 구분해야 합니다.

Python 실행 시에는 다음처럼 UTF-8 모드를 붙이면 확인이 편합니다.

```powershell
python -X utf8 -m sdiu_command_finetune.build_dataset
```

### build_dataset이 느린 경우

`build_dataset.py`는 fallback search로 학습 정답을 생성합니다. 현재 구현은 IDF와 토큰화를 재사용하도록 되어 있어 보통 몇 초 안에 끝납니다.

만약 비정상적으로 느리다면 `.venv`가 제대로 활성화됐는지, 오래된 코드가 실행 중인 것은 아닌지 확인하세요.

### train_lora가 오래 걸리는 경우

LoRA 학습은 GPU와 모델 다운로드 상태에 따라 시간이 달라집니다. 최초 실행에서는 Qwen 모델을 다운로드하므로 더 오래 걸립니다.

학습 도중 중단되어도 `checkpoint-*` 폴더가 생성되어 있다면 추론에서 사용할 수 있습니다.

### 체크포인트 자동 선택

`infer.py`는 adapter root 경로에 직접 `adapter_config.json`이 있으면 그것을 사용합니다. 없으면 `checkpoint-*` 폴더 중 가장 큰 step 번호를 가진 체크포인트를 자동으로 선택합니다.

예:

```text
sdiu_command_finetune/models/qwen2.5-1.5b-sdiu-lora/checkpoint-1180
```

### fallback mode

LoRA adapter가 없거나 모델 로드에 실패하면 앱은 fallback mode로 동작합니다.

이 모드는 LLM 없이도 동작합니다. 파일 기반 검색 점수만 사용하므로, 학습 전 파서와 데이터셋 품질을 확인하는 용도로 좋습니다.

## 이 프로젝트로 배울 수 있는 것

이 프로젝트는 작은 도메인 파일 하나를 LLM 앱으로 만드는 전체 과정을 압축해서 보여줍니다.

배울 수 있는 핵심은 다음과 같습니다.

- 도메인 전용 원본 파일을 구조화 데이터로 바꾸는 방법
- `instruction -> JSON answer` 형태의 SFT 데이터셋을 만드는 방법
- Qwen Instruct 모델에 LoRA adapter를 붙여 학습하는 방법
- LLM 출력은 반드시 검증해야 한다는 점
- 모델이 실패해도 앱이 쓸 수 있도록 fallback search를 두는 방법
- Streamlit으로 로컬 모델 검색 UI를 만드는 방법

즉, 이 프로젝트는 “LLM을 어떻게 학습시키는가”뿐 아니라 “학습된 LLM을 어떻게 안전하게 제품 흐름에 넣는가”까지 다룹니다.
