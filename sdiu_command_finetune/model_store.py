from __future__ import annotations

from pathlib import Path
import shutil

from .paths import package_model_dir


BASE_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
MODELS_DIR = package_model_dir()
LOCAL_BASE_MODEL_DIR = MODELS_DIR / "base" / "qwen2.5-1.5b-instruct"


def is_local_model_ready(path: str | Path = LOCAL_BASE_MODEL_DIR) -> bool:
    model_path = Path(path)
    return (
        (model_path / "config.json").exists()
        and (model_path / "tokenizer_config.json").exists()
        and ((model_path / "model.safetensors").exists() or any(model_path.glob("model-*.safetensors")))
    )


def resolve_base_model() -> str:
    if is_local_model_ready():
        return str(LOCAL_BASE_MODEL_DIR)
    return BASE_MODEL_ID


def ensure_base_model(
    model_id: str = BASE_MODEL_ID,
    local_dir: str | Path = LOCAL_BASE_MODEL_DIR,
    local_files_only: bool = False,
) -> Path:
    target = Path(local_dir)
    if is_local_model_ready(target):
        return target

    from huggingface_hub import snapshot_download

    snapshot_path = Path(
        snapshot_download(
            repo_id=model_id,
            local_files_only=local_files_only,
        )
    )

    target.mkdir(parents=True, exist_ok=True)
    for item in snapshot_path.iterdir():
        destination = target / item.name
        if item.is_dir():
            shutil.copytree(item, destination, dirs_exist_ok=True)
        else:
            shutil.copy2(item, destination)

    return target
