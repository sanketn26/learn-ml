"""Unsloth / PEFT-style LoRA train. --dry-run is CPU-safe and does not download weights.

    python capstone/finetune/train_lora.py --dry-run
    python capstone/finetune/train_lora.py --model google/functiongemma-270m-it
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ARTIFACTS = ROOT / "capstone" / "finetune" / "artifacts"

DEFAULT_CONFIG = {
    "model_name": "google/functiongemma-270m-it",
    "max_seq_length": 2048,
    "load_in_4bit": True,
    "lora_r": 16,
    "lora_alpha": 16,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "epochs": 1,
    "batch_size": 2,
    "grad_accum": 4,
    "learning_rate": 2e-4,
}


def _load_formatted(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def report_device() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return f"cuda ({torch.cuda.get_device_name(0)})"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    except ImportError:
        return "cpu (torch not installed)"


def estimate_memory_gb(config: dict, device: str) -> str:
    params_b = 0.27
    bits = 4 if config.get("load_in_4bit") else 16
    weight_gb = params_b * (bits / 8)
    lora_gb = 0.05 * (config.get("lora_r", 16) / 16)
    act_gb = 1.5 if "cuda" in device else 0.4
    total = weight_gb + lora_gb + act_gb
    return (
        f"~{total:.1f} GB working set "
        f"(weights≈{weight_gb:.2f} GB at {bits}-bit + LoRA≈{lora_gb:.2f} GB + activations≈{act_gb:.1f} GB). "
        "Real QLoRA on a 270M model typically wants a 12 GB GPU; CPU dry-run does not allocate this."
    )


def verify_dataset(artifacts: Path) -> dict[str, int]:
    counts = {}
    for split in ("train", "val"):
        path = artifacts / f"{split}.formatted.jsonl"
        rows = _load_formatted(path)
        if not rows:
            raise SystemExit(
                f"missing {path}. Run: python capstone/finetune/prepare_data.py --dry-run"
            )
        for row in rows:
            roles = [m["role"] for m in row.get("messages", [])]
            if roles != ["system", "user", "assistant"]:
                raise SystemExit(f"{path} row {row.get('id')} is not chat-formatted")
        counts[split] = len(rows)
    return counts


def try_transformers_probe(dry_run: bool) -> str:
    try:
        import transformers
    except ImportError:
        return "transformers not installed — skip model/tokenizer load (dataset + config still valid)"
    version = getattr(transformers, "__version__", "unknown")
    if dry_run:
        return (
            f"transformers {version} imported. --dry-run skips from_pretrained "
            "(will not download FunctionGemma or any large checkpoint)."
        )
    return f"transformers {version} available for a real load"


def ensure_formatted(artifacts: Path) -> None:
    if (artifacts / "train.formatted.jsonl").exists() and (artifacts / "val.formatted.jsonl").exists():
        return
    from capstone.finetune.prepare_data import (
        _ensure_raw_splits,
        _load_jsonl,
        format_row,
        validate_formatted,
        write_formatted,
        DATA_DIR,
    )

    _ensure_raw_splits(DATA_DIR)
    artifacts.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        raw = _load_jsonl(DATA_DIR / f"{split}.jsonl")
        formatted = [format_row(row) for row in raw]
        if formatted:
            validate_formatted(formatted)
        write_formatted(artifacts, split, formatted)


def train_real(config: dict, artifacts: Path, adapter_out: Path) -> None:
    try:
        from unsloth import FastLanguageModel
    except ImportError as exc:
        raise SystemExit(
            "unsloth is not installed. Real training is GPU-only. "
            "On Colab: pip install -r requirements-capstone.txt\n"
            "On a laptop: python capstone/finetune/train_lora.py --dry-run"
        ) from exc

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config["model_name"],
        max_seq_length=config["max_seq_length"],
        load_in_4bit=config["load_in_4bit"],
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        target_modules=config["target_modules"],
    )
    adapter_out.mkdir(parents=True, exist_ok=True)
    print("Unsloth model loaded. Wire SFTTrainer on")
    print(f"  {artifacts / 'train.formatted.jsonl'}")
    print("This scaffold stops before a long train so you can confirm VRAM first.")
    print(f"When you run a full SFT, write the adapter to {adapter_out}")
    model.save_pretrained(adapter_out)
    tokenizer.save_pretrained(adapter_out)


def main() -> None:
    parser = argparse.ArgumentParser(description="LoRA train (or dry-run) for the coding specialist.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--model", default=DEFAULT_CONFIG["model_name"])
    parser.add_argument("--artifacts", default=str(ARTIFACTS))
    parser.add_argument("--adapter-out", default=str(ARTIFACTS / "adapter"))
    parser.add_argument("--epochs", type=int, default=DEFAULT_CONFIG["epochs"])
    args = parser.parse_args()

    config = dict(DEFAULT_CONFIG)
    config["model_name"] = args.model
    config["epochs"] = args.epochs
    artifacts = Path(args.artifacts)

    ensure_formatted(artifacts)
    counts = verify_dataset(artifacts)
    device = report_device()
    print("dataset:", counts)
    print("device:", device)
    print("training configuration:")
    print(json.dumps(config, indent=2))
    print("memory:", estimate_memory_gb(config, device))
    print("model probe:", try_transformers_probe(args.dry_run))

    if args.dry_run:
        print("dry-run complete: dataset formatted, config printed, no weight download, unsloth not imported")
        return

    train_real(config, artifacts, Path(args.adapter_out))


if __name__ == "__main__":
    main()
