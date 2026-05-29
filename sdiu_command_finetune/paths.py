from __future__ import annotations

from pathlib import Path
import sys


PACKAGE_NAME = "sdiu_command_finetune"


def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def bundle_root() -> Path:
    if is_frozen():
        return Path(getattr(sys, "_MEIPASS")).resolve()
    return Path(__file__).resolve().parents[1]


def runtime_root() -> Path:
    if is_frozen():
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[1]


def external_path(*parts: str) -> Path:
    return runtime_root().joinpath(*parts)


def bundled_path(*parts: str) -> Path:
    return bundle_root().joinpath(*parts)


def resource_path(*parts: str) -> Path:
    external = external_path(*parts)
    if is_frozen() and external.exists():
        return external
    return bundled_path(*parts)


def package_data_dir() -> Path:
    return external_path(PACKAGE_NAME, "data")


def package_model_dir() -> Path:
    return external_path(PACKAGE_NAME, "models")
