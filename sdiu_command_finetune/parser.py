from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import re
from typing import Iterable, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_PATH = PROJECT_ROOT / "rag" / "SDIU명령어_20201217.txt"


@dataclass
class CommandRecord:
    command: str
    description: str = ""
    details: List[str] = field(default_factory=list)
    sections: List[str] = field(default_factory=list)
    line_numbers: List[int] = field(default_factory=list)
    raw_lines: List[str] = field(default_factory=list)

    @property
    def line_number(self) -> Optional[int]:
        return self.line_numbers[0] if self.line_numbers else None

    @property
    def searchable_text(self) -> str:
        parts = [self.command, self.description, *self.details, *self.sections]
        return " ".join(part for part in parts if part).strip()

    def to_dict(self) -> dict:
        data = asdict(self)
        data["line_number"] = self.line_number
        data["searchable_text"] = self.searchable_text
        return data


_COMMAND_PREFIXES = (
    "Show",
    "Reset",
    "Save",
    "Start",
    "Get",
    "Set",
    "Create",
    "Skip",
)

_COMMAND_NAMES = {
    "netstat",
    "sockShow",
    "adrSpaceShow",
    "cp",
    "ls",
    "rm",
    "routec",
    "ifconfig",
    "ping",
    "spy",
    "spyStop",
    "spyReport",
    "spyClkStart",
    "spyClkStop",
    "vxbPciCtrlShow",
    "vxbPciTopoShow",
    "vxbPciHeaderShow",
    "connectWithTimeout",
    "ioTaskStdSet",
    "ioTaskStdGet",
    "close",
}

_OUTPUT_PREFIXES = (
    "===",
    "->",
    "&&",
    "ex>",
    "ex)",
    "value =",
    "status=",
    "command=",
    "bar0",
    "bar1",
    "bar2",
    "bar3",
    "bar4",
    "bar5",
    "base/",
    "preMem",
    "I/O=",
    "Pci controller",
    "vendor ID",
    "device ID",
    "revision ID",
    "class code",
    "sub class code",
    "programming interface",
    "cache line",
    "latency time",
    "header type",
    "BIST",
    "base address",
    "cardBus",
    "sub system",
    "expansion ROM",
    "interrupt line",
    "interrupt pin",
    "min Grant",
    "max Latency",
    "Capabilities",
    "Address:",
    "Device:",
    "Acceptable",
    "Errors",
    "Max Read",
    "Link:",
    "Latency:",
    "ASPM",
    "Speed",
    "Serial Number",
    "Per-vector",
)


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def clean_comment(text: str) -> str:
    return normalize_space(text.replace("\ufeff", ""))


def _first_token(command: str) -> str:
    return re.split(r"[\s(,]", command.strip(), maxsplit=1)[0]


def is_section_line(text: str) -> bool:
    stripped = text.strip()
    return stripped.startswith("===") or (
        stripped.startswith("/") and "명령어" in stripped and not stripped.startswith("//")
    )


def section_name(text: str) -> str:
    cleaned = re.sub(r"^[=/\s]+|[=/\s]+$", "", text)
    return normalize_space(cleaned)


def is_command_like(text: str) -> bool:
    stripped = normalize_space(text)
    if not stripped:
        return False
    if stripped.startswith(_OUTPUT_PREFIXES):
        return False
    if re.match(r"^[A-Za-z]:\\", stripped):
        return False
    if re.match(r"^[가-힣]", stripped):
        return False
    if re.match(r"^\[[^\]]+\]\s*-", stripped):
        return False
    if re.match(r"^[A-Z]{2,}\s+\S+", stripped):
        return False
    if re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*=", stripped):
        return False

    token = _first_token(stripped)
    if token in _COMMAND_NAMES:
        return True
    if token.startswith(_COMMAND_PREFIXES):
        return True
    if token.startswith("spy") or token.startswith("vxb"):
        return True
    if re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*\(", stripped):
        return True
    return False


def _split_comment(line: str) -> tuple[str, str]:
    if "//" not in line:
        return line.strip(), ""
    left, right = line.split("//", 1)
    return left.strip(), clean_comment(right)


def _merge_record(target: CommandRecord, incoming: CommandRecord) -> None:
    if incoming.description and incoming.description not in target.description:
        if target.description:
            target.description = f"{target.description} / {incoming.description}"
        else:
            target.description = incoming.description
    for detail in incoming.details:
        if detail and detail not in target.details:
            target.details.append(detail)
    for section in incoming.sections:
        if section and section not in target.sections:
            target.sections.append(section)
    for line_number in incoming.line_numbers:
        if line_number not in target.line_numbers:
            target.line_numbers.append(line_number)
    target.raw_lines.extend(incoming.raw_lines)


def _parse_on_off(command: str) -> tuple[str, Optional[str]]:
    match = re.match(r"^(?P<base>.+?)(?:[\s,]+)(?P<flag>[01])(?:\s*,.*)?$", command.strip())
    if not match:
        return command.strip(), None
    return normalize_space(match.group("base")), match.group("flag")


def _make_off_description(description: str) -> str:
    if not description:
        return "OFF"
    text = description
    replacements = {
        " ON ": " OFF ",
        " ON": " OFF",
        "ON ": "OFF ",
        "On": "Off",
        "on": "off",
        "주기적으로 On": "주기적으로 Off",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    if text == description and "OFF" not in text.upper():
        text = f"{text} OFF"
    return text


def _infer_missing_descriptions(records: Iterable[CommandRecord]) -> None:
    by_on_off: dict[tuple[str, str], CommandRecord] = {}
    for record in records:
        base, flag = _parse_on_off(record.command)
        if flag:
            key = (base, flag)
            if key not in by_on_off or record.description:
                by_on_off[key] = record

    for record in records:
        if record.description:
            continue
        base, flag = _parse_on_off(record.command)
        if flag == "0":
            on_record = by_on_off.get((base, "1"))
            if on_record and on_record.description:
                record.description = _make_off_description(on_record.description)
                detail = f"'{on_record.command}' 설명에서 OFF 항목으로 보강"
                if detail not in record.details:
                    record.details.append(detail)
        if not record.description and record.sections:
            record.description = record.sections[-1]


def load_command_records(path: str | Path = DEFAULT_SOURCE_PATH) -> list[CommandRecord]:
    source_path = Path(path)
    text = source_path.read_text(encoding="utf-8-sig")

    records_by_command: dict[str, CommandRecord] = {}
    ordered_commands: list[str] = []
    last_record: Optional[CommandRecord] = None
    current_section = ""

    for line_number, raw_line in enumerate(text.splitlines(), 1):
        stripped = raw_line.strip()
        if not stripped:
            continue

        if is_section_line(stripped):
            current_section = section_name(stripped)
            last_record = None
            continue

        command_part, comment = _split_comment(raw_line)

        if not command_part and comment:
            if last_record and comment not in last_record.details:
                last_record.details.append(comment)
                last_record.raw_lines.append(raw_line)
            continue

        if not is_command_like(command_part):
            if last_record and stripped and stripped not in last_record.details:
                last_record.details.append(stripped)
                last_record.raw_lines.append(raw_line)
            continue

        command = normalize_space(command_part)
        incoming = CommandRecord(
            command=command,
            description=comment,
            sections=[current_section] if current_section else [],
            line_numbers=[line_number],
            raw_lines=[raw_line],
        )

        if command in records_by_command:
            record = records_by_command[command]
            _merge_record(record, incoming)
        else:
            record = incoming
            records_by_command[command] = record
            ordered_commands.append(command)

        last_record = record

    records = [records_by_command[command] for command in ordered_commands]
    _infer_missing_descriptions(records)
    return records
