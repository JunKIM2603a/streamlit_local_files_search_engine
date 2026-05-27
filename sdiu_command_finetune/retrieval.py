from __future__ import annotations

from dataclasses import dataclass
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

    query_tokens = tokenize(query)
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
    for record in records:
        if not record.searchable_text:
            continue
        score, reason = score_record(query, record, idf=idf)
        if score > 0:
            results.append(SearchResult(record=record, score=score, reason=reason))
    results.sort(key=lambda item: (-item.score, item.record.line_number or 10**9, item.record.command))
    return results[:top_k]


def result_to_payload(result: SearchResult) -> dict:
    record = result.record
    return {
        "command": record.command,
        "description": record.description or "파일에 별도 설명 없음",
        "reason": result.reason,
        "line_number": record.line_number,
        "score": round(result.score, 4),
        "details": record.details,
    }
