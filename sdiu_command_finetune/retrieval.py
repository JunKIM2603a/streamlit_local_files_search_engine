from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from functools import lru_cache
import math
import re
from typing import Iterable, List

from .parser import CommandRecord


@dataclass
class SearchResult:
    record: CommandRecord
    score: float
    reason: str
    order: int = 0


SYNONYM_GROUPS = [
    ("송신", "보내기", "보내", "보낸", "보냈", "보냄", "전송", "send", "tx", "transmit"),
    ("수신", "받기", "receive", "recv", "rx"),
    ("성공", "정상", "ok", "success", "good"),
    ("실패", "오류", "에러", "fail", "failed", "failure", "error", "errno"),
    ("확인", "조회", "보기", "표시", "체크", "check", "show", "view", "display", "list"),
    ("상태", "status", "state"),
    ("주기", "반복", "계속", "로그", "periodic", "repeat", "log"),
    ("라우팅", "라우트", "route", "routing"),
    ("테이블", "table", "tbl"),
    ("네트워크", "망", "network", "net"),
    ("소켓", "socket", "sock"),
    ("포트", "port"),
    ("카운터", "횟수", "개수", "count", "counter", "cnt"),
    ("삭제", "제거", "초기화", "리셋", "delete", "remove", "clear", "reset"),
    ("설정", "변경", "set", "config", "configure"),
    ("연결", "접속", "connect", "connection"),
    ("해제", "끊기", "disconnect", "close"),
    ("시작", "켜기", "on", "start", "enable"),
    ("중지", "끄기", "off", "stop", "disable"),
]
SUCCESS_GROUP_INDEX = 2
FAILURE_GROUP_INDEX = 3


@lru_cache(maxsize=20000)
def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^0-9a-z가-힣_]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _ngrams(token: str, min_n: int = 2, max_n: int = 4) -> list[str]:
    if not re.search(r"[가-힣]", token):
        return []
    grams: list[str] = []
    for n in range(min_n, min(max_n, len(token)) + 1):
        grams.extend(token[i : i + n] for i in range(0, len(token) - n + 1))
    return grams


@lru_cache(maxsize=20000)
def tokenize(text: str) -> tuple[str, ...]:
    base_tokens = normalize_text(text).split()
    tokens: list[str] = []
    for token in base_tokens:
        tokens.append(token)
        tokens.extend(_ngrams(token))
    return tuple(tokens)


def _compact(text: str) -> str:
    return normalize_text(text).replace(" ", "")


def _token_present(alias: str, normalized_tokens: set[str], compact_query: str) -> bool:
    normalized_alias = normalize_text(alias)
    if not normalized_alias:
        return False
    if " " in normalized_alias:
        return normalized_alias.replace(" ", "") in compact_query
    if normalized_alias.isascii() and len(normalized_alias) <= 2:
        return normalized_alias in normalized_tokens
    return normalized_alias in normalized_tokens or normalized_alias in compact_query


def _semantic_group_indexes(text: str) -> tuple[int, ...]:
    normalized_tokens = set(normalize_text(text).split())
    compact_text = _compact(text)
    return tuple(
        index
        for index, group in enumerate(SYNONYM_GROUPS)
        if any(_token_present(alias, normalized_tokens, compact_text) for alias in group)
    )


@lru_cache(maxsize=20000)
def expand_query_tokens(query: str) -> tuple[str, ...]:
    base_tokens = list(tokenize(query))
    normalized_tokens = set(normalize_text(query).split())
    compact_query = _compact(query)
    expanded = list(base_tokens)

    for group in SYNONYM_GROUPS:
        if any(_token_present(alias, normalized_tokens, compact_query) for alias in group):
            for alias in group:
                expanded.extend(tokenize(alias))

    return tuple(dict.fromkeys(token for token in expanded if token))


def _bigrams(text: str) -> set[str]:
    compact = _compact(text)
    if not compact:
        return set()
    if len(compact) == 1:
        return {compact}
    return {compact[index : index + 2] for index in range(len(compact) - 1)}


def _similarity(left: str, right: str) -> float:
    left_compact = _compact(left)
    right_compact = _compact(right)
    if not left_compact or not right_compact:
        return 0.0
    left_grams = _bigrams(left)
    right_grams = _bigrams(right)
    dice = 0.0
    if left_grams and right_grams:
        dice = (2.0 * len(left_grams & right_grams)) / (len(left_grams) + len(right_grams))
    sequence = SequenceMatcher(None, left_compact, right_compact).ratio()
    return max(dice, sequence)


def build_idf(records: Iterable[CommandRecord]) -> dict[str, float]:
    record_list = list(records)
    document_frequency: dict[str, int] = {}
    for record in record_list:
        for token in set(tokenize(record.searchable_text)):
            document_frequency[token] = document_frequency.get(token, 0) + 1
    count = max(len(record_list), 1)
    return {
        token: math.log((count + 1) / (frequency + 0.5)) + 1.0
        for token, frequency in document_frequency.items()
    }


def score_record(query: str, record: CommandRecord, idf: dict[str, float] | None = None) -> tuple[float, str]:
    query_norm = normalize_text(query)
    command_norm = normalize_text(record.command)
    description_norm = normalize_text(record.description)
    details_norm = normalize_text(" ".join(record.details))
    section_norm = normalize_text(" ".join(record.sections))
    text_norm = normalize_text(record.searchable_text)

    score = 0.0
    reasons: list[str] = []

    if query_norm and query_norm == command_norm:
        score += 1000.0
        reasons.append("명령어 정확 일치")
    elif query_norm and query_norm in command_norm:
        score += 500.0
        reasons.append("명령어 부분 일치")

    compact_query = query_norm.replace(" ", "")
    if compact_query:
        if compact_query in description_norm.replace(" ", ""):
            score += 220.0
            reasons.append("설명 문구 일치")
        if compact_query in details_norm.replace(" ", ""):
            score += 120.0
            reasons.append("보충 설명 일치")

    query_tokens = expand_query_tokens(query)
    record_tokens = set(tokenize(record.searchable_text))
    if query_tokens and record_tokens:
        matched = [token for token in query_tokens if token in record_tokens]
        if matched:
            if idf:
                token_score = sum(idf.get(token, 1.0) for token in matched)
            else:
                token_score = float(len(matched))
            score += token_score * 12.0
            reasons.append("한글/영문 토큰 일치")

    if query_norm and query_norm in text_norm:
        score += 60.0
        reasons.append("전체 텍스트 부분 일치")
    if query_norm and section_norm and query_norm in section_norm:
        score += 30.0
        reasons.append("섹션 일치")

    query_groups = _semantic_group_indexes(query)
    if query_groups:
        primary_text = f"{record.command} {record.description}"
        primary_groups = set(_semantic_group_indexes(primary_text))
        secondary_groups = set(_semantic_group_indexes(" ".join([*record.details, *record.sections])))
        matched_group_count = sum(1 for group_index in query_groups if group_index in primary_groups)
        secondary_match_count = sum(
            1 for group_index in query_groups if group_index not in primary_groups and group_index in secondary_groups
        )
        if matched_group_count:
            score += matched_group_count * 55.0
            reasons.append("의미 그룹 일치")
        if secondary_match_count:
            score += secondary_match_count * 12.0
            reasons.append("보조 의미 일치")
        if matched_group_count == len(query_groups):
            score += 150.0
            reasons.append("의미 조합 일치")
        if FAILURE_GROUP_INDEX in query_groups and FAILURE_GROUP_INDEX not in primary_groups:
            score -= 220.0 if SUCCESS_GROUP_INDEX in primary_groups else 110.0
        if SUCCESS_GROUP_INDEX in query_groups and SUCCESS_GROUP_INDEX not in primary_groups:
            score -= 180.0 if FAILURE_GROUP_INDEX in primary_groups else 90.0

    fuzzy_targets = [
        record.command,
        record.description,
        " ".join(record.details),
        " ".join(record.sections),
    ]
    fuzzy_score = max((_similarity(query, target) for target in fuzzy_targets if target), default=0.0)
    if fuzzy_score >= 0.34:
        score += fuzzy_score * 90.0
        reasons.append("유사 표현 일치")

    if not reasons:
        reasons.append("파일 기반 보충 후보")
    return score, ", ".join(dict.fromkeys(reasons))


def fallback_search(
    query: str,
    records: List[CommandRecord],
    top_k: int = 5,
    idf: dict[str, float] | None = None,
) -> list[SearchResult]:
    idf = idf or build_idf(records)
    results: list[SearchResult] = []
    for order, record in enumerate(records):
        if not record.searchable_text:
            continue
        score, reason = score_record(query, record, idf=idf)
        if score > 0:
            results.append(SearchResult(record=record, score=score, reason=reason, order=order))
    results.sort(key=lambda item: (-item.score, item.order, item.record.command))
    return results[:top_k]


def result_to_payload(result: SearchResult) -> dict:
    record = result.record
    return {
        "command": record.command,
        "kind": record.kind,
        "description": record.description or "파일에 별도 설명 없음",
        "reason": result.reason,
        "line_number": record.line_number,
        "score": round(result.score, 4),
        "details": record.details,
    }
