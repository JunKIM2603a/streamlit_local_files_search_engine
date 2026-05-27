from __future__ import annotations

import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import streamlit as st

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from sdiu_command_finetune.infer import SdiuCommandAssistant
    from sdiu_command_finetune.model_store import LOCAL_BASE_MODEL_DIR, ensure_base_model, is_local_model_ready
    from sdiu_command_finetune.parser import DEFAULT_SOURCE_PATH
    from sdiu_command_finetune.train_lora import BASE_MODEL, DEFAULT_OUTPUT_DIR
else:
    from .infer import SdiuCommandAssistant
    from .model_store import LOCAL_BASE_MODEL_DIR, ensure_base_model, is_local_model_ready
    from .parser import DEFAULT_SOURCE_PATH
    from .train_lora import BASE_MODEL, DEFAULT_OUTPUT_DIR


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(__file__).resolve().parent / "data"
TRAINING_JOB_PATH = DATA_DIR / "training_job.json"
TRAINING_LOG_PATH = DATA_DIR / "training.log"


@st.cache_resource(show_spinner=False)
def load_assistant(source_path: str, adapter_path: str, base_model: str, load_model: bool) -> SdiuCommandAssistant:
    return SdiuCommandAssistant(
        source_path=source_path,
        adapter_path=adapter_path,
        base_model=base_model,
        load_model=load_model,
    )


def _read_job() -> dict:
    if not TRAINING_JOB_PATH.exists():
        return {}
    try:
        return json.loads(TRAINING_JOB_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _write_job(job: dict) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    TRAINING_JOB_PATH.write_text(json.dumps(job, ensure_ascii=False, indent=2), encoding="utf-8")


def _is_pid_running(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        import psutil

        return psutil.pid_exists(pid) and psutil.Process(pid).is_running()
    except Exception:
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False


def _tail_log(max_chars: int = 6000) -> str:
    if not TRAINING_LOG_PATH.exists():
        return ""
    text = TRAINING_LOG_PATH.read_text(encoding="utf-8", errors="replace")
    return text[-max_chars:]


def _release_loaded_model() -> None:
    load_assistant.clear()
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _start_retraining(
    source_path: str,
    output_dir: str,
    epochs: float,
    batch_size: int,
    grad_accum: int,
    learning_rate: float,
    max_length: int,
) -> int:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    _release_loaded_model()

    command = [
        sys.executable,
        "-X",
        "utf8",
        "-m",
        "sdiu_command_finetune.retrain",
        "--source",
        source_path,
        "--output-dir",
        output_dir,
        "--epochs",
        str(epochs),
        "--batch-size",
        str(batch_size),
        "--grad-accum",
        str(grad_accum),
        "--learning-rate",
        str(learning_rate),
        "--max-length",
        str(max_length),
    ]

    creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    log_file = TRAINING_LOG_PATH.open("w", encoding="utf-8", newline="\n")
    process = subprocess.Popen(
        command,
        cwd=PROJECT_ROOT,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        env={**os.environ, "PYTHONUTF8": "1"},
        creationflags=creationflags,
    )
    log_file.close()

    _write_job(
        {
            "pid": process.pid,
            "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "source_path": source_path,
            "output_dir": output_dir,
            "command": command,
        }
    )
    return process.pid


st.set_page_config(page_title="SDIU Command Finder", layout="wide")
st.title("SDIU 명령어 검색")

with st.sidebar:
    st.header("Model")
    source_path = st.text_input("SDIU txt", value=str(DEFAULT_SOURCE_PATH))
    adapter_path = st.text_input("LoRA adapter", value=str(DEFAULT_OUTPUT_DIR))
    base_model = st.text_input("Base model", value=BASE_MODEL)
    load_model = st.toggle("Load fine-tuned model", value=Path(adapter_path).exists())
    top_k = st.slider("Top K", min_value=1, max_value=10, value=5)
    show_json = st.toggle("Show JSON", value=False)

    st.divider()
    st.header("Training")
    st.caption(f"Local base model: {LOCAL_BASE_MODEL_DIR}")
    if is_local_model_ready():
        st.success("Base model is preserved locally.")
    else:
        st.warning("Base model is not preserved in this project yet.")

    if st.button("Preserve base model locally"):
        with st.spinner("Copying/downloading Qwen base model..."):
            path = ensure_base_model(local_files_only=False)
            st.success(f"Saved: {path}")

    train_epochs = st.number_input("Epochs", min_value=1.0, max_value=100.0, value=25.0, step=1.0)
    train_batch_size = st.number_input("Batch size", min_value=1, max_value=16, value=2, step=1)
    train_grad_accum = st.number_input("Gradient accumulation", min_value=1, max_value=64, value=8, step=1)
    train_lr = st.number_input("Learning rate", min_value=1e-6, max_value=1e-2, value=2e-4, format="%.6f")
    train_max_length = st.number_input("Max length", min_value=256, max_value=4096, value=1024, step=128)

    job = _read_job()
    running = _is_pid_running(job.get("pid"))
    if running:
        st.info(f"Training is running. PID: {job.get('pid')}")
    elif job:
        st.info(f"Last training PID {job.get('pid')} is not running.")

    if st.button("Start retraining", disabled=running):
        pid = _start_retraining(
            source_path=source_path,
            output_dir=adapter_path,
            epochs=train_epochs,
            batch_size=int(train_batch_size),
            grad_accum=int(train_grad_accum),
            learning_rate=float(train_lr),
            max_length=int(train_max_length),
        )
        st.success(f"Training started. PID: {pid}")
        st.rerun()

    if st.button("Reload model"):
        _release_loaded_model()
        st.rerun()

    with st.expander("Training log", expanded=running):
        log_text = _tail_log()
        if log_text:
            st.code(log_text, language="text")
        else:
            st.caption("No training log yet.")

assistant = load_assistant(source_path, adapter_path, base_model, load_model)

if assistant.model_ready:
    st.success("Fine-tuned model loaded.")
elif assistant.load_error:
    st.warning(f"Fallback search active. {assistant.load_error}")
else:
    st.info("Fallback search active.")

with st.form("search_form"):
    query = st.text_input("검색어", placeholder="예: 주기적으로 송신성공 확인 ON")
    submitted = st.form_submit_button("Search")

if submitted and query.strip():
    with st.spinner("Searching..."):
        payload = assistant.answer(query.strip(), top_k=top_k)

    results = payload.get("results", [])
    if not results:
        st.error("No command candidates found.")
    else:
        for index, result in enumerate(results, 1):
            with st.container(border=True):
                header = f"{index}. `{result.get('command', '')}`"
                st.markdown(header)
                st.write(result.get("description", ""))
                meta = []
                if result.get("line_number"):
                    meta.append(f"line {result['line_number']}")
                if result.get("reason"):
                    meta.append(result["reason"])
                if meta:
                    st.caption(" | ".join(meta))
                details = result.get("details") or []
                if details:
                    with st.expander("Details"):
                        for detail in details:
                            st.write(detail)

    if show_json:
        st.subheader("JSON")
        st.code(json.dumps(payload, ensure_ascii=False, indent=2), language="json")

