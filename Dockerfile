# Self-contained env for the ML weeks (0-20) — CPU only, no Jupyter.
# Bakes in the code, the labs (exercises/), and the CSVs (data/), so
# `docker run` alone works with no bind mount. Bind-mount over /workspace
# anyway if you want local edits to show up live (see getting-started.md).
#
#   docker build -t learn-ml .
#   docker run --rm -it -p 8000:8000 learn-ml
#
# Also used as-is by .devcontainer/devcontainer.json (VS Code / Codespaces).
FROM python:3.11-slim

WORKDIR /workspace

# gcc/g++ cover the rare source build when a wheel is missing for this arch;
# everything in requirements.txt normally ships prebuilt wheels.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# requirements.txt pins `torch>=2.1.0` with no backend — on plain PyPI that
# can resolve to a CUDA build and drag in a ~1 GB GPU stack. This is a CPU
# course (see the "Laptop budget" note in every week), so install torch from
# the CPU-only wheel index explicitly, then the rest from PyPI as normal.
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && grep -v '^torch' requirements.txt > /tmp/requirements-no-torch.txt \
    && pip install --no-cache-dir -r /tmp/requirements-no-torch.txt

# Code, lessons, exercises/labs, and the synthetic CSVs — everything
# lib/course_data.py and the starter.py files expect to find at the repo
# root. .dockerignore keeps out .git, .venv, site/, and generated artifacts.
COPY . .

EXPOSE 8000

CMD ["bash"]
