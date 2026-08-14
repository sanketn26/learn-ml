"""Tiny notebook writer — no nbformat dependency required."""

from __future__ import annotations

import json
import uuid
from pathlib import Path


def _source_lines(text: str) -> list[str]:
    if not text.endswith("\n"):
        text = text + "\n"
    lines = text.splitlines(keepends=True)
    # Jupyter often stores the last line without forcing extra blanks
    return lines


def md_cell(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": uuid.uuid4().hex[:12],
        "metadata": {},
        "source": _source_lines(text),
    }


def code_cell(text: str) -> dict:
    return {
        "cell_type": "code",
        "id": uuid.uuid4().hex[:12],
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _source_lines(text),
    }


def write_notebook(path: Path, cells: list[dict], title: str = "") -> None:
    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
            "title": title,
        },
        "cells": cells,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
    print(f"wrote {path} ({len(cells)} cells)")


BOOT = '''%matplotlib inline
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Make the shared style kit importable whether you launch from notebooks/ or repo root
for _p in [Path.cwd(), Path.cwd() / "notebooks", Path.cwd().parent]:
    if (_p / "course_style.py").exists():
        sys.path.insert(0, str(_p))
        break

from course_style import apply_style, setup_plots, find_data_dir

apply_style()
setup_plots()
DATA = find_data_dir()
print(f"Data folder: {DATA}")
print("Laptop mode: no GPU required. Models use a sample so each week finishes in a few minutes on CPU.")
'''

LAPTOP_BOX = """
<div class="cue-box">
<strong>Laptop budget</strong>
<p>No GPU. Aimed at ~8&nbsp;GB RAM. Training uses a few thousand sampled customers (or short sequences) so this notebook should finish in a <strong>few minutes on CPU</strong>. The ideas are the same if you later set <code>n=None</code> and train on all 50k rows.</p>
</div>
"""
