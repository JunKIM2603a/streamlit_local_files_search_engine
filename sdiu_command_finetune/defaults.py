from __future__ import annotations

from .model_store import resolve_base_model
from .paths import package_model_dir


BASE_MODEL = resolve_base_model()
DEFAULT_OUTPUT_DIR = package_model_dir() / "qwen2.5-1.5b-sdiu-lora"
