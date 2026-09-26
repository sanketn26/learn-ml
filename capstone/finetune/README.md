# Capstone fine-tune scaffold (Phase 4)

Everything else in the capstone — specs, harness, teacher, data, evaluation —
runs on a laptop with no GPU and no API key. **This folder is the fine-tune**:
a LoRA adapter trained on the teacher's per-step examples.

The required job path (weeks 0–17) never needs this. Treat it as optional.

## Hardware

| Mode | Where | GPU |
|---|---|---|
| `--dry-run` | your laptop, CI | **none**. One night of examples, every target re-validated, config printed, no weight download. |
| Real train | Colab T4/L4 (or any box with ~12 GB VRAM) | yes |

Do not `pip install -r requirements-capstone.txt` into the course venv unless you are on a GPU machine.

## Commands (from repo root)

```bash
# CPU, no weights, CI-safe
python capstone/finetune/prepare_data.py --dry-run
python capstone/finetune/train_lora.py --dry-run
python capstone/finetune/evaluate_adapter.py --dry-run

# The full bank (still CPU, ~40 s) — writes capstone/data/ and capstone/finetune/artifacts/
python capstone/finetune/prepare_data.py

# Real train (GPU). Unsloth, TRL and datasets are imported only on this path.
python capstone/finetune/train_lora.py --model unsloth/functiongemma-270m-it
python capstone/finetune/evaluate_adapter.py --adapter capstone/finetune/artifacts/adapter \
    --base unsloth/functiongemma-270m-it
```

## What each script does

1. **`prepare_data.py`** — reads `capstone/data/{train,val,test}.jsonl` or
   regenerates them with `capstone.teacher.write_splits`. Every target
   command is re-validated against its spec *and* the state it was chosen
   in. Writes chat-formatted jsonl using `capstone/prompt.py` — the exact
   messages `policies.llm_policy` sends at inference time.
2. **`train_lora.py`** — Unsloth QLoRA + TRL `SFTTrainer` on the formatted
   files. `max_seq_length` is 3072: the longest six-step state is ~2.2k
   tokens. `--dry-run` never imports Unsloth or downloads weights.
3. **`evaluate_adapter.py`** — the Phase 5 ablation with your adapter added,
   in both modes. `--base` adds the base model without the adapter: the row
   that says whether fine-tuning did anything the harness wasn't doing already.

## Colab

1. GPU runtime (T4 is enough for 270M 4-bit LoRA; a 1–4B alternative wants an L4).
2. Clone this repo. `pip install -r requirements.txt -r requirements-capstone.txt` **in Colab**.
3. `python capstone/finetune/prepare_data.py`
4. `python capstone/finetune/train_lora.py --model unsloth/functiongemma-270m-it`
5. `python capstone/finetune/evaluate_adapter.py --adapter capstone/finetune/artifacts/adapter --base unsloth/functiongemma-270m-it`
6. Optional: export GGUF and serve it with Ollama, then `python -m capstone.evaluate --ollama <name>` on your laptop.

## Not verified in CI

CI runs the dry-runs only. The real `train_real` path is written against the
minimum versions in `requirements-capstone.txt` (TRL ≥ 0.12's
`processing_class` / `SFTConfig`); newer TRL releases rename arguments from
time to time. If `SFTConfig` rejects a field, check the TRL changelog before
changing the recipe. The training loss covers the whole prompt, catalog
included; masking everything but the assistant turn is a worthwhile
refinement once the baseline works.

## Do not expect

- A trained adapter checked into git.
- One accuracy number. The result is the ablation table, with horizon columns.
- A 270M model to match the teacher on held-out wording. The teacher is told the first command.
