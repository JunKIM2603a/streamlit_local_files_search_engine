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
    from sdiu_command_finetune.build_dataset import DEFAULT_DATASET_PATH, DEFAULT_RECORDS_PATH
    from sdiu_command_finetune.defaults import BASE_MODEL, DEFAULT_OUTPUT_DIR
    from sdiu_command_finetune.infer import SdiuCommandAssistant
    from sdiu_command_finetune.model_store import LOCAL_BASE_MODEL_DIR, ensure_base_model, is_local_model_ready
    from sdiu_command_finetune.parser import DEFAULT_COMMANDS_PATH
    from sdiu_command_finetune.paths import is_frozen, package_data_dir, runtime_root
    from sdiu_command_finetune.retrain import rebuild_dataset
else:
    from .build_dataset import DEFAULT_DATASET_PATH, DEFAULT_RECORDS_PATH
    from .defaults import BASE_MODEL, DEFAULT_OUTPUT_DIR
    from .infer import SdiuCommandAssistant
    from .model_store import LOCAL_BASE_MODEL_DIR, ensure_base_model, is_local_model_ready
    from .parser import DEFAULT_COMMANDS_PATH
    from .paths import is_frozen, package_data_dir, runtime_root
    from .retrain import rebuild_dataset


PROJECT_ROOT = runtime_root()
DATA_DIR = package_data_dir()
TRAINING_JOB_PATH = DATA_DIR / "training_job.json"
TRAINING_LOG_PATH = DATA_DIR / "training.log"
FROZEN_RUNTIME = is_frozen()


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


def _format_mtime(path: Path) -> str:
    if not path.exists():
        return "missing"
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(path.stat().st_mtime))


def _dataset_status(commands_path: str) -> dict:
    commands = Path(commands_path)
    dataset = DEFAULT_DATASET_PATH
    records = DEFAULT_RECORDS_PATH
    dataset_exists = dataset.exists()
    records_exists = records.exists()
    commands_exists = commands.exists()
    stale = False
    if commands_exists:
        commands_mtime = commands.stat().st_mtime
        stale = (
            (dataset_exists and commands_mtime > dataset.stat().st_mtime)
            or (records_exists and commands_mtime > records.stat().st_mtime)
        )
    return {
        "commands": commands,
        "dataset": dataset,
        "records": records,
        "commands_exists": commands_exists,
        "dataset_exists": dataset_exists,
        "records_exists": records_exists,
        "ready": commands_exists and dataset_exists and records_exists and not stale,
        "stale": stale,
        "commands_mtime": _format_mtime(commands),
        "dataset_mtime": _format_mtime(dataset),
        "records_mtime": _format_mtime(records),
    }


def _build_dataset_from_ui(commands_path: str) -> tuple[int, int]:
    return rebuild_dataset(
        commands=Path(commands_path),
        dataset_path=DEFAULT_DATASET_PATH,
        records_path=DEFAULT_RECORDS_PATH,
        top_k=5,
    )


@st.cache_data(ttl=10, show_spinner=False)
def _runtime_status() -> dict:
    try:
        import torch

        cuda = torch.cuda.is_available()
        return {
            "cuda": cuda,
            "device": torch.cuda.get_device_name(0) if cuda else "CPU",
            "memory_allocated_gb": round(torch.cuda.memory_allocated(0) / (1024**3), 2) if cuda else 0.0,
            "memory_reserved_gb": round(torch.cuda.memory_reserved(0) / (1024**3), 2) if cuda else 0.0,
        }
    except Exception as exc:
        return {"cuda": False, "device": f"Unavailable: {exc}", "memory_allocated_gb": 0.0, "memory_reserved_gb": 0.0}


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
    output_dir: str,
    epochs: float,
    batch_size: int,
    grad_accum: int,
    learning_rate: float,
    max_length: int,
) -> int:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    _release_loaded_model()

    training_args = [
        "--dataset",
        str(DEFAULT_DATASET_PATH),
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
    if FROZEN_RUNTIME:
        command = [sys.executable, "--train-lora", *training_args]
    else:
        command = [sys.executable, "-X", "utf8", "-m", "sdiu_command_finetune.train_lora", *training_args]

    creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" and hasattr(subprocess, "CREATE_NO_WINDOW") else 0
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
            "dataset": str(DEFAULT_DATASET_PATH),
            "output_dir": output_dir,
            "command": command,
        }
    )
    return process.pid


st.set_page_config(page_title="SDIU Command Finder", layout="wide")
st.title("SDIU 명령어 검색")

with st.sidebar:
    st.header("Model")
    commands_path = st.text_input("Commands JSON", value=str(DEFAULT_COMMANDS_PATH))
    adapter_path = st.text_input("LoRA adapter", value=str(DEFAULT_OUTPUT_DIR))
    base_model = st.text_input("Base model", value=BASE_MODEL)
    use_llm = st.toggle("Use LLM refinement (slower)", value=False)
    max_new_tokens = st.slider("LLM max new tokens", min_value=64, max_value=512, value=192, step=32, disabled=not use_llm)
    top_k = st.slider("Top K", min_value=1, max_value=10, value=5)
    show_json = st.toggle("Show JSON", value=False)

    runtime = _runtime_status()
    if runtime["cuda"]:
        st.caption(
            f"GPU: {runtime['device']} | allocated {runtime['memory_allocated_gb']} GB | "
            f"reserved {runtime['memory_reserved_gb']} GB"
        )
    else:
        st.caption(f"Runtime: {runtime['device']}")

    st.divider()
    st.header("Training")
    st.caption(f"Local base model: {LOCAL_BASE_MODEL_DIR}")
    if FROZEN_RUNTIME:
        st.info("PyInstaller 실행파일에서는 검색과 dataset 생성 중심으로 동작합니다. LoRA 학습은 Python 개발환경에서 실행하세요.")
    if is_local_model_ready():
        st.success("Base model is preserved locally.")
    else:
        st.warning("Base model is not preserved in this project yet.")

    if st.button("Preserve base model locally", disabled=FROZEN_RUNTIME):
        with st.spinner("Copying/downloading Qwen base model..."):
            path = ensure_base_model(local_files_only=False)
            st.success(f"Saved: {path}")

    job = _read_job()
    running = _is_pid_running(job.get("pid"))
    if running:
        st.info(f"Training is running. PID: {job.get('pid')}")
    elif job:
        st.info(f"Last training PID {job.get('pid')} is not running.")

    dataset_status = _dataset_status(commands_path)

    st.subheader("Step 1. Build dataset")
    st.caption("commands.json을 수정한 뒤 가장 먼저 누르는 버튼입니다.")
    if not dataset_status["commands_exists"]:
        st.error("Commands JSON file does not exist.")
    elif dataset_status["stale"]:
        st.warning("commands.json이 dataset보다 최신입니다. dataset 재생성이 필요합니다.")
    elif dataset_status["ready"]:
        st.success("Dataset is ready and up to date.")
    else:
        st.warning("Dataset files are missing. Build the dataset first.")

    st.caption(f"commands: {dataset_status['commands_mtime']}")
    st.caption(f"dataset: {dataset_status['dataset_mtime']}")
    st.caption(f"records: {dataset_status['records_mtime']}")

    last_build = st.session_state.get("last_dataset_build")
    if last_build:
        st.success(
            f"Dataset built. records={last_build['records']}, "
            f"examples={last_build['examples']}, generated={last_build['generated_at']}"
        )
        st.caption(f"dataset path: {last_build['dataset_path']}")
        st.caption(f"records path: {last_build['records_path']}")

    if st.button("Build SFT dataset", disabled=running or not dataset_status["commands_exists"]):
        try:
            with st.spinner("Building SFT dataset..."):
                record_count, example_count = _build_dataset_from_ui(commands_path)
            _release_loaded_model()
            st.session_state["last_dataset_build"] = {
                "records": record_count,
                "examples": example_count,
                "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "dataset_path": str(DEFAULT_DATASET_PATH),
                "records_path": str(DEFAULT_RECORDS_PATH),
            }
            st.rerun()
        except Exception as exc:
            st.error(f"Dataset build failed: {exc}")

    st.subheader("Step 2. Train LoRA adapter")
    st.caption("Dataset이 최신일 때만 누르는 버튼입니다.")
    train_epochs = st.number_input("Epochs", min_value=1.0, max_value=100.0, value=25.0, step=1.0)
    train_batch_size = st.number_input("Batch size", min_value=1, max_value=16, value=2, step=1)
    train_grad_accum = st.number_input("Gradient accumulation", min_value=1, max_value=64, value=8, step=1)
    train_lr = st.number_input("Learning rate", min_value=1e-6, max_value=1e-2, value=2e-4, format="%.6f")
    train_max_length = st.number_input("Max length", min_value=256, max_value=4096, value=1024, step=128)

    train_disabled = FROZEN_RUNTIME or running or not dataset_status["ready"]
    if FROZEN_RUNTIME:
        st.warning("실행파일에서는 Start LoRA training 버튼을 비활성화합니다.")
    if running:
        st.caption("Training is already running.")
    elif dataset_status["stale"]:
        st.warning("commands.json이 변경되었습니다. 먼저 Build SFT dataset을 눌러 최신 dataset을 만드세요.")
    elif not dataset_status["dataset_exists"] or not dataset_status["records_exists"]:
        st.warning("Dataset 파일이 없습니다. 먼저 Build SFT dataset을 눌러 생성하세요.")
    if st.button("Start LoRA training", disabled=train_disabled):
        pid = _start_retraining(
            output_dir=adapter_path,
            epochs=train_epochs,
            batch_size=int(train_batch_size),
            grad_accum=int(train_grad_accum),
            learning_rate=float(train_lr),
            max_length=int(train_max_length),
        )
        st.success(f"Training started. PID: {pid}")
        st.rerun()

    st.subheader("Step 3. Reload model")
    st.caption("학습 완료 후 새 adapter와 commands.json을 검색에 반영합니다.")
    if st.button("Reload model"):
        _release_loaded_model()
        st.rerun()

    with st.expander("Training log", expanded=running):
        log_text = _tail_log()
        if log_text:
            st.code(log_text, language="text")
        else:
            st.caption("No training log yet.")

assistant = load_assistant(commands_path, adapter_path, base_model, use_llm)

if use_llm and assistant.model_ready:
    load_text = f"{assistant.load_seconds:.1f}s" if assistant.load_seconds is not None else "unknown"
    st.success(f"Fine-tuned model loaded on {assistant.device}. Load time: {load_text}")
elif use_llm and assistant.load_error:
    st.warning(f"LLM unavailable. Fallback search active. {assistant.load_error}")
elif use_llm:
    st.info("LLM refinement requested, but model is not ready. Fallback search active.")
else:
    st.info("Fast fallback search mode. Turn on LLM refinement only when you want model-generated JSON.")

with st.form("search_form"):
    query = st.text_input("검색어", placeholder="예: 주기적으로 송신성공 확인 ON")
    submitted = st.form_submit_button("Search")

if submitted and query.strip():
    started_at = time.perf_counter()
    if use_llm and assistant.model_ready:
        with st.spinner("Running LLM refinement on GPU..."):
            payload = assistant.answer(
                query.strip(),
                top_k=top_k,
                use_model=True,
                max_new_tokens=int(max_new_tokens),
            )
    else:
        payload = assistant.fallback_answer(query.strip(), top_k=top_k)
    elapsed = time.perf_counter() - started_at

    results = payload.get("results", [])
    if not results:
        st.error("No command candidates found.")
    else:
        mode = payload.get("mode", "fallback")
        st.caption(f"Mode: {mode} | elapsed: {elapsed:.2f}s")
        for index, result in enumerate(results, 1):
            with st.container(border=True):
                kind = result.get("kind", "command")
                header = f"{index}. `{result.get('command', '')}`"
                st.markdown(header)
                st.caption("참고정보" if kind == "info" else "명령어")
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
