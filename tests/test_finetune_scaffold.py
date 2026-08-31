"""CPU-only checks for capstone/finetune. No model download."""

from __future__ import annotations

from pathlib import Path

from capstone.finetune.prepare_data import (
    DATA_DIR,
    _ensure_raw_splits,
    _load_jsonl,
    format_row,
    validate_formatted,
)
from capstone.finetune.train_lora import DEFAULT_CONFIG, estimate_memory_gb, report_device

ROOT = Path(__file__).resolve().parent.parent


def test_prepare_formats_and_validates_schema():
    _ensure_raw_splits(DATA_DIR)
    raw = _load_jsonl(DATA_DIR / "train.jsonl")
    assert raw, "teacher should produce at least a tiny train split"
    formatted = [format_row(row) for row in raw]
    validate_formatted(formatted)
    roles = [m["role"] for m in formatted[0]["messages"]]
    assert roles == ["system", "user", "assistant"]


def test_train_dry_run_helpers_do_not_need_gpu():
    device = report_device()
    assert device
    memo = estimate_memory_gb(DEFAULT_CONFIG, device)
    assert "GB" in memo
    assert (ROOT / "capstone" / "finetune" / "train_lora.py").exists()
    assert (ROOT / "capstone" / "finetune" / "evaluate_adapter.py").exists()
