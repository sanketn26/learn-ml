# Capstone fine-tune scaffold (Phase 3)

Phases 1, 2, 4, and 5 of the capstone run on a laptop with no GPU and no API
key. **This folder is Phase 3** — the actual LoRA / QLoRA fine-tune.

The required job path (weeks 0–17) never needs this. Treat it as optional.

## Hardware

| Mode | Where | GPU |
|---|---|---|
| `--dry-run` | your laptop | **none**. Validates jsonl, prints config, does not download a 270M+ model. |
| Real train | Colab T4/L4 (or any box with ~12 GB VRAM) | yes |

Laptop RAM is not the bottleneck. VRAM is. Do not `pip install -r requirements-capstone.txt` into the course venv unless you are on a GPU machine.

## Commands (from repo root)

```bash
# CPU, no weights, CI-safe
python capstone/finetune/prepare_data.py --dry-run
python capstone/finetune/train_lora.py --dry-run
python capstone/finetune/evaluate_adapter.py --dry-run

# Real formatting (still CPU) — writes capstone/finetune/artifacts/
python capstone/finetune/prepare_data.py

# Real train (GPU). Unsloth is imported only on this path.
python capstone/finetune/train_lora.py --model google/functiongemma-270m-it
python capstone/finetune/evaluate_adapter.py --adapter capstone/finetune/artifacts/adapter
```

## What each script does

1. **`prepare_data.py`** — reads `capstone/data/{train,val,test}.jsonl`, or
   regenerates them via `capstone.teacher.write_splits` if they are missing.
   Every `tool_call` is run through `reliability.validate_call`. Writes
   chat-formatted jsonl under `capstone/finetune/artifacts/`.
2. **`train_lora.py`** — Unsloth / PEFT-style SFT. `--dry-run` verifies the
   formatted files, prints device + a memory guess, and **does not** pull
   model weights. If `transformers` is installed it reports the version and
   skips the download on purpose.
3. **`evaluate_adapter.py`** — scores golden scenarios. `--dry-run` uses the
   existing placeholder specialist vs baseline (same as
   `python -m capstone.evaluate`). A real adapter is optional.

## Colab

1. GPU runtime (T4 is enough for 270M 4-bit LoRA).
2. Clone this repo. `pip install -r requirements-capstone.txt` **in Colab**,
   not on the laptop.
3. `python capstone/finetune/prepare_data.py`
4. `python capstone/finetune/train_lora.py --model unsloth/functiongemma-270m-it`
5. Export GGUF / adapter, download, point local inference at it, then rerun
   `python -m capstone.evaluate` with a real `specialist_call`.

The lesson sketch in `docs/ml/capstone.md` still works as a notebook form of
the same recipe. Prefer these scripts so the dry-run is testable in CI.

## Do not expect

- A trained adapter checked into git.
- FlashAttention, from-scratch GPT, or CUDA-kernel debugging.
- The placeholder 100% accuracy to survive contact with a real model. The
  gap between your adapter and that ceiling is the number that belongs in a
  write-up.
