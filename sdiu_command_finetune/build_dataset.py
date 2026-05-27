from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

from .parser import DEFAULT_SOURCE_PATH, CommandRecord, load_command_records
from .prompts import build_user_prompt, render_plain_prompt
from .retrieval import build_idf, fallback_search, result_to_payload


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "data"
DEFAULT_DATASET_PATH = DEFAULT_OUTPUT_DIR / "sdiu_sft.jsonl"
DEFAULT_RECORDS_PATH = DEFAULT_OUTPUT_DIR / "records.json"


def _json_response_for_query(
    query: str,
    records: list[CommandRecord],
    top_k: int = 5,
    idf: dict[str, float] | None = None,
) -> str:
    results = fallback_search(query, records, top_k=top_k, idf=idf)
    payload = {
        "results": [
            {
                "command": result.record.command,
                "description": result.record.description,
                "reason": result.reason,
            }
            for result in results
        ]
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _queries_for_record(record: CommandRecord) -> Iterable[str]:
    if record.description:
        yield record.description
        yield f"{record.description} 명령어"
        yield f"{record.description} 확인"
    yield record.command
    yield f"{record.command} 설명"
    for detail in record.details[:4]:
        if detail:
            yield detail
            yield f"{detail} 명령어"
    for section in record.sections[-1:]:
        if section and record.description:
            yield f"{section} {record.description}"


def build_examples(records: list[CommandRecord], top_k: int = 5) -> list[dict]:
    examples: list[dict] = []
    seen_queries: set[str] = set()
    training_records = [record for record in records if record.description]
    idf = build_idf(training_records)

    for record in training_records:
        if not record.searchable_text:
            continue
        for query in _queries_for_record(record):
            query = query.strip()
            if not query or query in seen_queries:
                continue
            seen_queries.add(query)
            response = _json_response_for_query(query, training_records, top_k=top_k, idf=idf)
            examples.append(
                {
                    "query": query,
                    "prompt": render_plain_prompt(query, top_k=top_k),
                    "user": build_user_prompt(query, top_k=top_k),
                    "response": response,
                }
            )

    return examples


def validate_examples(examples: list[dict], records: list[CommandRecord]) -> None:
    valid_commands = {record.command for record in records}
    if len(valid_commands) != len(records):
        duplicate_count = len(records) - len(valid_commands)
        raise ValueError(f"Duplicate commands remain after parsing: {duplicate_count}")

    for index, example in enumerate(examples, 1):
        if not example["query"].strip():
            raise ValueError(f"Empty query at example {index}")
        try:
            payload = json.loads(example["response"])
        except json.JSONDecodeError as exc:
            raise ValueError(f"Broken JSON at example {index}: {exc}") from exc
        results = payload.get("results")
        if not isinstance(results, list):
            raise ValueError(f"Missing results list at example {index}")
        for result in results:
            command = result.get("command")
            description = result.get("description", "")
            if command not in valid_commands:
                raise ValueError(f"Unknown command in example {index}: {command}")
            if not description:
                raise ValueError(f"Empty description for {command} in example {index}")


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build SDIU command SFT dataset.")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--records-output", type=Path, default=DEFAULT_RECORDS_PATH)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    records = load_command_records(args.source)
    examples = build_examples(records, top_k=args.top_k)
    validate_examples(examples, records)

    write_jsonl(args.output, examples)
    args.records_output.parent.mkdir(parents=True, exist_ok=True)
    args.records_output.write_text(
        json.dumps([record.to_dict() for record in records], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"records={len(records)}")
    print(f"examples={len(examples)}")
    print(f"dataset={args.output}")
    print(f"records_output={args.records_output}")


if __name__ == "__main__":
    main()
