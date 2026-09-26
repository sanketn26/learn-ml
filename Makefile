.PHONY: help setup setup-frameworks setup-crewai setup-all dev serve build exercise test clean \
	docker-build docker-up docker-down docker-restart docker-shell docker-logs docker-test docker-status docker-clean

.DEFAULT_GOAL := help

BLUE := \033[0;34m
GREEN := \033[0;32m
YELLOW := \033[0;33m
NC := \033[0m

# The ML weeks, LangChain/LangGraph, and CrewAI pin incompatible stacks, so
# each gets its own virtualenv (see docs/framework-tracks.md).
VENV := venv
VENV_FRAMEWORKS := .venv-framework
VENV_CREWAI := .venv-crewai
PYTHON := python3
TORCH_CPU_INDEX := https://download.pytorch.org/whl/cpu
# Use the venv when it exists, so `make test` works with or without activating it.
PY := $(if $(wildcard $(VENV)/bin/python),$(VENV)/bin/python,$(PYTHON))

IMAGE := learn-ml
CONTAINER := learn-ml-dev
PORT := 8000

help: ## List targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(YELLOW)%-18s$(NC) %s\n", $$1, $$2}'

# ---------------------------------------------------------------- local env

$(VENV)/bin/python:
	$(PYTHON) -m venv $(VENV)
	$(VENV)/bin/pip install --upgrade pip

setup: $(VENV)/bin/python ## ML venv + dependencies (CPU-only torch)
	@# Plain PyPI can resolve torch to a CUDA build (~1 GB). This is a CPU course.
	$(VENV)/bin/pip install torch --index-url $(TORCH_CPU_INDEX)
	$(VENV)/bin/pip install -r requirements.txt
	@echo "$(GREEN)ok.  source $(VENV)/bin/activate$(NC)"

setup-frameworks: ## LangChain + LangGraph venv (.venv-framework/)
	$(PYTHON) -m venv $(VENV_FRAMEWORKS)
	$(VENV_FRAMEWORKS)/bin/pip install --upgrade pip
	$(VENV_FRAMEWORKS)/bin/pip install -r requirements-frameworks.txt pytest
	@echo "$(GREEN)ok.  source $(VENV_FRAMEWORKS)/bin/activate$(NC)"

setup-crewai: ## CrewAI venv, isolated from LangChain (.venv-crewai/)
	$(PYTHON) -m venv $(VENV_CREWAI)
	$(VENV_CREWAI)/bin/pip install --upgrade pip
	$(VENV_CREWAI)/bin/pip install -r requirements-crewai.txt
	@echo "$(GREEN)ok.  source $(VENV_CREWAI)/bin/activate$(NC)"

setup-all: setup setup-frameworks setup-crewai ## every venv (capstone Phase 3 deps stay in Colab)

dev: setup ## set up the ML venv, run the tests, then serve the course
	$(MAKE) test
	$(MAKE) serve

serve: ## preview the course at http://127.0.0.1:8000
	$(PY) -m mkdocs serve -a 127.0.0.1:$(PORT)

build: ## build the static site into site/
	$(PY) -m mkdocs build --strict

exercise: ## run week 00 starter (override WEEK=01)
	$(PY) exercises/ml/week-$(or $(WEEK),00)/starter.py

test: ## pipeline + contract tests
	$(PY) -m pytest tests/

clean: ## caches and build output
	rm -rf site .pytest_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

# ------------------------------------------------------------------- docker

docker-build: ## build the learn-ml image (CPU-only)
	docker build -t $(IMAGE) .

docker-up: ## start the dev container in the background, serving the course on :8000
	@if docker ps -a --format '{{.Names}}' | grep -qx $(CONTAINER); then \
		docker start $(CONTAINER) >/dev/null; \
	else \
		docker image inspect $(IMAGE) >/dev/null 2>&1 || $(MAKE) docker-build; \
		docker run -d --name $(CONTAINER) -p $(PORT):8000 -v "$(CURDIR)":/workspace \
			$(IMAGE) python -m mkdocs serve -a 0.0.0.0:8000 >/dev/null; \
	fi
	@echo "$(GREEN)up.  http://127.0.0.1:$(PORT)   shell: make docker-shell   stop: make docker-down$(NC)"

docker-down: ## stop and remove the dev container (your files live on the host)
	-docker stop $(CONTAINER)
	-docker rm $(CONTAINER)

docker-restart: docker-down docker-up ## restart the dev container

docker-shell: ## open a shell in the running dev container
	docker exec -it $(CONTAINER) bash

docker-logs: ## follow mkdocs output from the dev container
	docker logs -f $(CONTAINER)

docker-test: ## run the test suite inside the dev container
	docker exec $(CONTAINER) python -m pytest tests/

docker-status: ## show whether the dev container is running
	@docker ps -a --filter name=^/$(CONTAINER)$$ --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'

docker-clean: docker-down ## remove the container and the image
	-docker rmi $(IMAGE)
