"""Score a fine-tuned adapter in the ablation, next to the teacher and the rules baseline.

    python capstone/finetune/evaluate_adapter.py --dry-run
    python capstone/finetune/evaluate_adapter.py --adapter capstone/finetune/artifacts/adapter
    python capstone/finetune/evaluate_adapter.py --adapter ... --base google/functiongemma-270m-it

`--base` adds the same base model *without* the adapter, in both modes — the
row that separates what the weights learned from what the harness gives
any model for free.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from capstone.cases import DEFAULT_NIGHTS, all_cases
from capstone.evaluate import ablation, default_entries, table
from capstone.policies import llm_planner, llm_policy


def hf_generate(path: str, max_new_tokens: int = 256):
    """A `generate(messages) -> str` over a local transformers model, with or without a PEFT adapter."""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("transformers is not installed — this path runs in Colab or on a GPU box") from exc

    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path)  # a PEFT adapter dir loads through its base automatically
    model.eval()

    def generate(messages: list[dict]) -> str:
        inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
        out = model.generate(inputs.to(model.device), max_new_tokens=max_new_tokens, do_sample=False)
        return tokenizer.decode(out[0, inputs.shape[1]:], skip_special_tokens=True)

    return generate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="offline rows on one night; no model load")
    parser.add_argument("--adapter", default="")
    parser.add_argument("--base", default="", help="the base model id, scored without the adapter")
    parser.add_argument("--split", default="test")
    parser.add_argument("--nights", type=int, default=2)
    args = parser.parse_args()

    nights = DEFAULT_NIGHTS[:1] if args.dry_run else DEFAULT_NIGHTS[: args.nights]
    cases = all_cases(nights, split=args.split)
    entries = default_entries()

    if not args.dry_run:
        for name, path in (("base", args.base), ("adapter", args.adapter)):
            if not path:
                continue
            generate = hf_generate(path)
            entries[f"{name} + loop"] = {"policy": llm_policy(generate)}
            entries[f"{name}, one-shot"] = {"planner": llm_planner(generate)}

    print(f"{len(cases)} {args.split} cases on {len(nights)} night(s)\n")
    print(table(ablation(cases, entries)))
    if args.dry_run:
        print("\ndry-run: offline rows only, no model load")


if __name__ == "__main__":
    main()
