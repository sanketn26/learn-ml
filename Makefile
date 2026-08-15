.PHONY: help setup serve build clean

.DEFAULT_GOAL := help

BLUE := \033[0;34m
GREEN := \033[0;32m
YELLOW := \033[0;33m
NC := \033[0m

help: ## List targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(YELLOW)%-16s$(NC) %s\n", $$1, $$2}'

setup: ## venv + dependencies
	python3 -m venv venv
	venv/bin/pip install --upgrade pip
	venv/bin/pip install -r requirements.txt
	@echo "$(GREEN)ok.  source venv/bin/activate$(NC)"

serve: ## preview the course at http://127.0.0.1:8000
	python3 -m mkdocs serve

build: ## build the static site into site/
	python3 -m mkdocs build --strict

exercise: ## run week 00 starter (override WEEK=01)
	python3 exercises/ml/week-$(or $(WEEK),00)/starter.py

clean: ## caches and build output
	rm -rf site .pytest_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
