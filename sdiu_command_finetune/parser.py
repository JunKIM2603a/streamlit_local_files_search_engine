from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Iterable, List, Optional

from .paths import PACKAGE_NAME, resource_path, runtime_root


PROJECT_ROOT = runtime_root()
DEFAULT_COMMANDS_PATH = resource_path(PACKAGE_NAME, "data", "commands.json")
DEFAULT_SOURCE_PATH = DEFAULT_COMMANDS_PATH
COMMANDS_SCHEMA_VERSION = 1
ALLOWED_RECORD_KINDS = {"command", "info"}


@dataclass
class CommandRecord:
    command: str
    description: str = ""
    kind: str = "command"
    details: List[str] = field(default_factory=list)
    sections: List[str] = field(default_factory=list)
    line_numbers: List[int] = field(default_factory=list)
    raw_lines: List[str] = field(default_factory=list)

    @property
    def line_number(self) -> Optional[int]:
        return self.line_numbers[0] if self.line_numbers else None

    @property
    def searchable_text(self) -> str:
        parts = [self.command, self.kind, self.description, *self.details, *self.sections]
        return " ".join(part for part in parts if part).strip()

    def to_dict(self) -> dict:
        data = asdict(self)
        data["line_number"] = self.line_number
        data["searchable_text"] = self.searchable_text
        return data

    def to_source_dict(self) -> dict:
        return {
            "command": self.command,
            "kind": self.kind,
            "description": self.description,
            "details": list(self.details),
            "sections": list(self.sections),
        }


def normalize_space(text: str) -> str:
    return " ".join(text.split()).strip()


def _validate_record(record: CommandRecord, index: int, seen_commands: set[str], source_label: str) -> None:
    if not record.command:
        raise ValueError(f"{source_label}: records[{index}].command is required.")
    if record.command in seen_commands:
        raise ValueError(f"{source_label}: duplicate command: {record.command}")
    if record.kind not in ALLOWED_RECORD_KINDS:
        allowed = ", ".join(sorted(ALLOWED_RECORD_KINDS))
        raise ValueError(f"{source_label}: records[{index}].kind must be one of: {allowed}")
    if not record.description:
        raise ValueError(f"{source_label}: records[{index}].description is required for {record.command}.")
    seen_commands.add(record.command)


def validate_command_records(records: Iterable[CommandRecord], source_label: str = "commands") -> list[CommandRecord]:
    record_list = list(records)
    seen_commands: set[str] = set()
    for index, record in enumerate(record_list):
        _validate_record(record, index, seen_commands, source_label)
        for field_name in ("details", "sections", "line_numbers", "raw_lines"):
            value = getattr(record, field_name)
            expected_type = int if field_name == "line_numbers" else str
            if not isinstance(value, list) or any(not isinstance(item, expected_type) for item in value):
                raise ValueError(f"{source_label}: records[{index}].{field_name} must be a list.")
    return record_list


def _string_list(value: Any, field_name: str, index: int, source_label: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"{source_label}: records[{index}].{field_name} must be a list of strings.")
    return value


def _record_from_json(item: Any, index: int, source_label: str) -> CommandRecord:
    if not isinstance(item, dict):
        raise ValueError(f"{source_label}: records[{index}] must be an object.")
    return CommandRecord(
        command=normalize_space(str(item.get("command", ""))),
        kind=normalize_space(str(item.get("kind", "command"))),
        description=normalize_space(str(item.get("description", ""))),
        details=_string_list(item.get("details", []), "details", index, source_label),
        sections=_string_list(item.get("sections", []), "sections", index, source_label),
    )


def load_command_records(path: str | Path = DEFAULT_COMMANDS_PATH) -> list[CommandRecord]:
    commands_path = Path(path)
    source_label = str(commands_path)
    try:
        payload = json.loads(commands_path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{source_label}: invalid JSON: {exc}") from exc

    if not isinstance(payload, dict):
        raise ValueError(f"{source_label}: root must be a JSON object.")
    version = payload.get("version")
    if version != COMMANDS_SCHEMA_VERSION:
        raise ValueError(f"{source_label}: unsupported version {version!r}; expected {COMMANDS_SCHEMA_VERSION}.")
    records_payload = payload.get("records")
    if not isinstance(records_payload, list):
        raise ValueError(f"{source_label}: records must be a list.")

    records = [_record_from_json(item, index, source_label) for index, item in enumerate(records_payload)]
    return validate_command_records(records, source_label=source_label)
