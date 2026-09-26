"""CPU-only checks for capstone/finetune. No model download."""

from __future__ import annotations

from pathlib import Path

from capstone.finetune.prepare_data import QUICK_NIGHTS, _ensure_raw_splits, _load_jsonl, format_row, validate_formatted
from capstone.finetune.train_lora import DEFAULT_CONFIG, estimate_memory_gb, report_device

ROOT = Path(__file__).resolve().parent.parent


def test_prepare_formats_and_validates_every_example(tmp_path):
    _ensure_raw_splits(tmp_path, QUICK_NIGHTS)
    raw = _load_jsonl(tmp_path / "train.jsonl")
    assert raw, "the teacher should produce a train split"
    formatted = [format_row(row) for row in raw]
    validate_formatted(formatted)
    assert [m["role"] for m in formatted[0]["messages"]] == ["system", "user", "assistant"]


def test_train_dry_run_helpers_do_not_need_gpu():
    device = report_device()
    assert device
    assert "GB" in estimate_memory_gb(DEFAULT_CONFIG, device)
    assert (ROOT / "capstone" / "finetune" / "evaluate_adapter.py").exists()
