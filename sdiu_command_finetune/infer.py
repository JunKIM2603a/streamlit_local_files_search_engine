from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import time
from typing import Any

from .defaults import BASE_MODEL, DEFAULT_OUTPUT_DIR
from .parser import DEFAULT_COMMANDS_PATH, CommandRecord, load_command_records
from .prompts import SYSTEM_PROMPT, build_user_prompt
from .retrieval import fallback_search, result_to_payload


class SdiuCommandAssistant:
    def __init__(
        self,
        source_path: str | Path = DEFAULT_COMMANDS_PATH,
        adapter_path: str | Path = DEFAULT_OUTPUT_DIR,
        base_model: str = BASE_MODEL,
        load_model: bool = True,
    ) -> None:
        self.records = load_command_records(source_path)
        self.records_by_command = {record.command: record for record in self.records}
        self.adapter_path = Path(adapter_path)
        self.resolved_adapter_path = self._resolve_adapter_path(self.adapter_path)
        self.base_model = base_model
        self.tokenizer = None
        self.model = None
        self.load_error: str | None = None
        self.load_seconds: float | None = None
        self.device: str = "not loaded"

        if load_model:
            self._load_model()

    @property
    def model_ready(self) -> bool:
        return self.model is not None and self.tokenizer is not None

    def _load_model(self) -> None:
        if not self.resolved_adapter_path:
            self.load_error = f"LoRA adapter not found: {self.adapter_path}"
            return
        try:
            import torch
            from peft import PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            self.load_error = f"Model dependencies are missing: {exc}"
            return

        try:
            started_at = time.perf_counter()
            dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
            tokenizer_source = (
                self.resolved_adapter_path
                if (self.resolved_adapter_path / "tokenizer_config.json").exists()
                else self.base_model
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_source,
                trust_remote_code=True,
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            base = AutoModelForCausalLM.from_pretrained(
                self.base_model,
                dtype=dtype,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True,
            )
            self.model = PeftModel.from_pretrained(base, self.resolved_adapter_path)
            self.model.eval()
            self.load_seconds = time.perf_counter() - started_at
            self.device = str(getattr(self.model, "device", "cuda" if torch.cuda.is_available() else "cpu"))
        except Exception as exc:  # pragma: no cover - depends on local model stack
            self.load_error = str(exc)
            self.tokenizer = None
            self.model = None
            self.load_seconds = None
            self.device = "load failed"

    @staticmethod
    def _resolve_adapter_path(path: Path) -> Path | None:
        if (path / "adapter_config.json").exists():
            return path
        if not path.exists():
            return None
        checkpoints = []
        for child in path.glob("checkpoint-*"):
            if child.is_dir() and (child / "adapter_config.json").exists():
                try:
                    step = int(child.name.rsplit("-", 1)[-1])
                except ValueError:
                    step = -1
                checkpoints.append((step, child))
        if not checkpoints:
            return None
        checkpoints.sort(key=lambda item: item[0], reverse=True)
        return checkpoints[0][1]

    def _prompt(self, query: str, top_k: int) -> str:
        assert self.tokenizer is not None
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(query, top_k=top_k)},
        ]
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def generate_raw(self, query: str, top_k: int = 5, max_new_tokens: int = 256) -> str:
        if not self.model_ready:
            return ""
        assert self.tokenizer is not None and self.model is not None
        import torch

        prompt = self._prompt(query, top_k)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {key: value.to(self.model.device) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        generated = outputs[0][inputs["input_ids"].shape[-1] :]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()

    def fallback_answer(self, query: str, top_k: int = 5) -> dict[str, Any]:
        return {
            "results": [result_to_payload(result) for result in fallback_search(query, self.records, top_k=top_k)],
            "raw_model_output": "",
            "model_ready": self.model_ready,
            "load_error": self.load_error,
            "mode": "fallback",
        }

    def answer(
        self,
        query: str,
        top_k: int = 5,
        use_model: bool = True,
        max_new_tokens: int = 256,
    ) -> dict[str, Any]:
        if not use_model:
            return self.fallback_answer(query, top_k=top_k)

        raw_text = self.generate_raw(query, top_k=top_k, max_new_tokens=max_new_tokens) if self.model_ready else ""
        model_payload = extract_json_payload(raw_text)
        validated = self._validate_payload(model_payload)
        fallback = [result_to_payload(result) for result in fallback_search(query, self.records, top_k=top_k)]

        seen = {item["command"] for item in validated}
        for item in fallback:
            if len(validated) >= top_k:
                break
            if item["command"] not in seen:
                validated.append(item)
                seen.add(item["command"])

        return {
            "results": validated[:top_k],
            "raw_model_output": raw_text,
            "model_ready": self.model_ready,
            "load_error": self.load_error,
            "mode": "llm" if self.model_ready else "fallback",
            "device": self.device,
            "load_seconds": self.load_seconds,
        }

    def _validate_payload(self, payload: dict[str, Any] | None) -> list[dict]:
        if not payload:
            return []
        results = payload.get("results")
        if not isinstance(results, list):
            return []

        validated: list[dict] = []
        seen: set[str] = set()
        for item in results:
            if not isinstance(item, dict):
                continue
            command = str(item.get("command", "")).strip()
            if command not in self.records_by_command or command in seen:
                continue
            record = self.records_by_command[command]
            validated.append(
                {
                    "command": record.command,
                    "kind": record.kind,
                    "description": record.description or "파일에 별도 설명 없음",
                    "reason": str(item.get("reason") or "파인튜닝 모델 후보"),
                    "line_number": record.line_number,
                    "details": record.details,
                }
            )
            seen.add(command)
        return validated


def extract_json_payload(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()

    decoder = json.JSONDecoder()
    for start in [match.start() for match in re.finditer(r"\{", cleaned)]:
        try:
            payload, _ = decoder.raw_decode(cleaned[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SDIU command inference.")
    parser.add_argument("query", nargs="*", help="Korean description or command to search.")
    parser.add_argument("--commands", "--source", dest="commands", type=Path, default=DEFAULT_COMMANDS_PATH)
    parser.add_argument("--adapter", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--fallback-only", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()

    query = " ".join(args.query).strip()
    if not query:
        raise SystemExit("Provide a query.")

    assistant = SdiuCommandAssistant(
        source_path=args.commands,
        adapter_path=args.adapter,
        base_model=args.base_model,
        load_model=not args.fallback_only,
    )
    payload = assistant.answer(
        query,
        top_k=args.top_k,
        use_model=not args.fallback_only,
        max_new_tokens=args.max_new_tokens,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
