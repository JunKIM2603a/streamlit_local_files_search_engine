from __future__ import annotations


SYSTEM_PROMPT = (
    "너는 SDIU명령어_20201217.txt 파일만 외운 명령어 검색 모델이다. "
    "사용자의 한글 설명 또는 명령어 입력을 보고 파일 안의 명령어 후보를 최대 5개 반환한다. "
    "파일에 없는 명령어를 만들지 않는다. 설명은 파일의 // 뒤 설명과 보충 설명만 사용한다. "
    "반드시 JSON만 출력한다."
)


def build_user_prompt(query: str, top_k: int = 5) -> str:
    return (
        f"질문: {query}\n"
        f"반환 개수: 최대 {top_k}개\n"
        "출력 형식: "
        '{"results":[{"command":"명령어","description":"설명","reason":"선택 이유"}]}'
    )


def render_plain_prompt(query: str, top_k: int = 5) -> str:
    return f"{SYSTEM_PROMPT}\n\n{build_user_prompt(query, top_k=top_k)}\nJSON:"

