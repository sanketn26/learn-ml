"""Score an adapter (or the placeholder) against golden tool calls.

    python capstone/finetune/evaluate_adapter.py --dry-run
    python capstone/finetune/evaluate_adapter.py --adapter capstone/finetune/artifacts/adapter
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from capstone.evaluate import compare, run, summarize
from capstone.scenarios import SCENARIOS


def load_adapter_caller(adapter: Path):
    """Optional real inference. Missing stack → skip, do not crash."""
    try:
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            f"adapter given at {adapter} but transformers/peft are not installed"
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(adapter)
    base = AutoModelForCausalLM.from_pretrained(adapter)
    try:
        model = PeftModel.from_pretrained(base, adapter)
    except Exception:
        model = base
    model.eval()

    def caller(scenario):
        prompt = scenario.input_text
        encoded = tokenizer(prompt, return_tensors="pt")
        out = model.generate(**encoded, max_new_tokens=128)
        text = tokenizer.decode(out[0], skip_special_tokens=True)
        try:
            start = text.rfind("{")
            end = text.rfind("}")
            if start >= 0 and end > start:
                return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return text
        return text

    return caller


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--adapter", default="")
    args = parser.parse_args()

    specialist, baseline = compare()
    print("placeholder specialist:", specialist)
    print("general-model baseline:", baseline)
    gain = specialist["accuracy"] - baseline["accuracy"]
    print(f"specialization-gain placeholder vs baseline: {gain:+.0%} accuracy")
    print(
        "This placeholder is the ceiling (golden_call). A real adapter's gain "
        "is (adapter accuracy − baseline accuracy); the gap to 1.0 is leftover error."
    )

    if args.dry_run:
        print("dry-run: skipped adapter load")
        return

    if not args.adapter:
        print("no --adapter; reporting placeholder only")
        return

    adapter = Path(args.adapter)
    if not adapter.exists():
        print(f"adapter path {adapter} missing — skip load")
        return

    try:
        caller = load_adapter_caller(adapter)
    except Exception as exc:
        print(f"skip adapter load: {exc}")
        return

    results = run(caller, SCENARIOS)
    summary = summarize(results)
    print("adapter:", summary)
    print(f"adapter vs baseline: {summary['accuracy'] - baseline['accuracy']:+.0%}")
    print(f"adapter vs placeholder ceiling: {summary['accuracy'] - specialist['accuracy']:+.0%}")


if __name__ == "__main__":
    main()
