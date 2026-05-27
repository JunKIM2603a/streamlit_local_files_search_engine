from __future__ import annotations

import argparse
import json
from pathlib import Path

from .build_dataset import DEFAULT_DATASET_PATH
from .model_store import BASE_MODEL_ID, ensure_base_model, resolve_base_model
from .prompts import SYSTEM_PROMPT


BASE_MODEL = resolve_base_model()
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "models" / "qwen2.5-1.5b-sdiu-lora"


def _load_training_stack():
    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
    except ImportError as exc:
        raise SystemExit(
            "Training dependencies are missing. Install them with:\n"
            "pip install -r sdiu_command_finetune/requirements.txt"
        ) from exc
    return torch, Dataset, LoraConfig, get_peft_model, AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def run_training(
    dataset_path: Path = DEFAULT_DATASET_PATH,
    base_model: str | None = None,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    epochs: float = 25,
    batch_size: int = 2,
    grad_accum: int = 8,
    learning_rate: float = 2e-4,
    max_length: int = 1024,
) -> Path:
    (
        torch,
        Dataset,
        LoraConfig,
        get_peft_model,
        AutoModelForCausalLM,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    ) = _load_training_stack()

    rows = _read_jsonl(dataset_path)
    if not rows:
        raise SystemExit(f"No training rows found: {dataset_path}")

    base_model = base_model or resolve_base_model()

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    model.config.use_cache = False

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    def render_prompt(row: dict) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": row["user"]},
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def tokenize_row(row: dict) -> dict:
        prompt = render_prompt(row)
        response = row["response"] + tokenizer.eos_token
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        full = tokenizer(
            prompt + response,
            add_special_tokens=False,
            truncation=True,
            max_length=max_length,
        )
        input_ids = full["input_ids"]
        labels = input_ids.copy()
        prompt_length = min(len(prompt_ids), len(labels))
        labels[:prompt_length] = [-100] * prompt_length
        full["labels"] = labels
        return full

    dataset = Dataset.from_list(rows).map(tokenize_row, remove_columns=list(rows[0].keys()))

    def collate(features: list[dict]) -> dict:
        max_len = max(len(feature["input_ids"]) for feature in features)
        pad_id = tokenizer.pad_token_id
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for feature in features:
            pad_len = max_len - len(feature["input_ids"])
            batch["input_ids"].append(feature["input_ids"] + [pad_id] * pad_len)
            batch["attention_mask"].append(feature["attention_mask"] + [0] * pad_len)
            batch["labels"].append(feature["labels"] + [-100] * pad_len)
        return {key: torch.tensor(value, dtype=torch.long) for key, value in batch.items()}

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=learning_rate,
        logging_steps=5,
        save_strategy="epoch",
        save_total_limit=3,
        bf16=torch.cuda.is_available(),
        fp16=False,
        optim="adamw_torch",
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collate,
    )
    trainer.train()

    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Saved LoRA adapter to {output_dir}")
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune Qwen2.5 1.5B for SDIU command lookup.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--base-model", default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=float, default=25)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--prepare-base-model", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    if args.base_model:
        base_model = args.base_model
    elif args.prepare_base_model:
        base_model = str(ensure_base_model(model_id=BASE_MODEL_ID, local_files_only=args.local_files_only))
    else:
        base_model = resolve_base_model()

    run_training(
        dataset_path=args.dataset,
        base_model=base_model,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
    )


if __name__ == "__main__":
    main()
