from __future__ import annotations

import argparse
from pathlib import Path

from .build_dataset import (
    DEFAULT_DATASET_PATH,
    DEFAULT_RECORDS_PATH,
    build_examples,
    validate_examples,
    write_jsonl,
)
from .model_store import ensure_base_model, resolve_base_model
from .parser import DEFAULT_SOURCE_PATH, load_command_records
from .train_lora import DEFAULT_OUTPUT_DIR, run_training


def rebuild_dataset(source: Path, dataset_path: Path, records_path: Path, top_k: int) -> tuple[int, int]:
    records = load_command_records(source)
    examples = build_examples(records, top_k=top_k)
    validate_examples(examples, records)
    write_jsonl(dataset_path, examples)
    records_path.parent.mkdir(parents=True, exist_ok=True)
    records_path.write_text(
        __import__("json").dumps([record.to_dict() for record in records], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return len(records), len(examples)


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild SDIU dataset and run LoRA training.")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE_PATH)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--records-output", type=Path, default=DEFAULT_RECORDS_PATH)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--base-model", default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=float, default=25)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--skip-base-model-sync", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    print(f"Rebuilding dataset from {args.source}", flush=True)
    record_count, example_count = rebuild_dataset(
        source=args.source,
        dataset_path=args.dataset,
        records_path=args.records_output,
        top_k=args.top_k,
    )
    print(f"records={record_count}", flush=True)
    print(f"examples={example_count}", flush=True)

    if args.base_model:
        base_model = args.base_model
    elif args.skip_base_model_sync:
        base_model = resolve_base_model()
    else:
        print("Preparing local base model for offline retraining...", flush=True)
        base_model = str(ensure_base_model(local_files_only=args.local_files_only))
        print(f"base_model={base_model}", flush=True)

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

