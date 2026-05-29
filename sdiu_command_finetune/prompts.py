from __future__ import annotations


SYSTEM_PROMPT = (
    "너는 commands.json에 등록된 SDIU 명령어만 사용하는 명령어 검색 모델이다. "
    "사용자의 한글 설명 또는 명령어 입력을 보고 JSON 안의 명령어와 @info 참고정보 후보를 최대 5개 반환한다. "
    "JSON에 없는 명령어나 참고정보를 만들지 않는다. 설명은 JSON의 description과 details만 사용한다. "
    "반드시 JSON만 출력한다."
)


def build_user_prompt(query: str, top_k: int = 5) -> str:
    return (
        f"질문: {query}\n"
        f"반환 개수: 최대 {top_k}개\n"
        "출력 형식: "
        '{"results":[{"command":"명령어 또는 @info id","description":"설명","reason":"선택 이유"}]}'
    )


def render_plain_prompt(query: str, top_k: int = 5) -> str:
    return f"{SYSTEM_PROMPT}\n\n{build_user_prompt(query, top_k=top_k)}\nJSON:"
