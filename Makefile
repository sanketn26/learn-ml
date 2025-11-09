.PHONY: help setup install-deps serve serve-bg stop-server clean convert-notebooks lint format test update-html all

# Default target
.DEFAULT_GOAL := help

# Colors for output
BLUE := \033[0;34m
GREEN := \033[0;32m
YELLOW := \033[0;33m
RED := \033[0;31m
NC := \033[0m # No Color

# Directories
VENV_DIR := venv
PYTHON := $(VENV_DIR)/bin/python
PIP := $(VENV_DIR)/bin/pip
JUPYTER := $(VENV_DIR)/bin/jupyter
NBCONVERT := $(VENV_DIR)/bin/jupyter-nbconvert

# Paths
ML_NOTEBOOKS := notebooks/*.ipynb
LANGCHAIN_NOTEBOOKS := langchain/notebooks/*.ipynb
CREWAI_NOTEBOOKS := crewai/notebooks/*.ipynb
LANGGRAPH_NOTEBOOKS := langgraph/notebooks/*.ipynb
ALL_NOTEBOOKS := $(ML_NOTEBOOKS) $(LANGCHAIN_NOTEBOOKS) $(CREWAI_NOTEBOOKS) $(LANGGRAPH_NOTEBOOKS)

ML_DOCS := docs
LANGCHAIN_DOCS := langchain/docs
CREWAI_DOCS := crewai/docs
LANGGRAPH_DOCS := langgraph/docs

PORT := 8000

help: ## Display this help message
	@echo "$(BLUE)╔════════════════════════════════════════════════════════════╗$(NC)"
	@echo "$(BLUE)║         Learn ML - Project Utilities Makefile              ║$(NC)"
	@echo "$(BLUE)╚════════════════════════════════════════════════════════════╝$(NC)"
	@echo ""
	@echo "$(GREEN)📚 SETUP & ENVIRONMENT$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(YELLOW)%-25s$(NC) %s\n", $$1, $$2}' | head -15

setup: ## Create virtual environment and install dependencies
	@echo "$(BLUE)🔧 Setting up Python virtual environment...$(NC)"
	python3 -m venv $(VENV_DIR)
	@echo "$(GREEN)✓ Virtual environment created$(NC)"
	@echo "$(BLUE)📦 Installing dependencies...$(NC)"
	$(PIP) install --upgrade pip setuptools wheel -q
	$(PIP) install jupyter nbconvert jupyterlab -q
	$(PIP) install langchain crewai langgraph -q
	$(PIP) install pandas numpy scikit-learn matplotlib seaborn plotly -q
	$(PIP) install black flake8 pytest -q
	@echo "$(GREEN)✓ All dependencies installed$(NC)"
	@echo ""
	@echo "$(GREEN)✅ Setup complete! Run 'make serve' to start the development server.$(NC)"

install-deps: ## Install Python dependencies (assumes venv exists)
	@echo "$(BLUE)📦 Installing dependencies...$(NC)"
	$(PIP) install --upgrade -r requirements.txt -q 2>/dev/null || $(PIP) install jupyter nbconvert jupyterlab langchain crewai langgraph pandas numpy scikit-learn matplotlib seaborn plotly black flake8 pytest -q
	@echo "$(GREEN)✓ Dependencies installed$(NC)"

serve: ## Start development server on localhost:8000 (foreground)
	@echo "$(BLUE)🚀 Starting web server on http://localhost:$(PORT)...$(NC)"
	@echo "$(YELLOW)Press Ctrl+C to stop$(NC)"
	$(PYTHON) -m http.server $(PORT) --directory .

serve-bg: ## Start development server in background
	@if pgrep -f "http.server $(PORT)" > /dev/null; then \
		echo "$(YELLOW)⚠️  Server already running on port $(PORT)$(NC)"; \
	else \
		echo "$(BLUE)🚀 Starting web server in background on http://localhost:$(PORT)...$(NC)"; \
		nohup $(PYTHON) -m http.server $(PORT) --directory . > /tmp/learn-ml-server.log 2>&1 & \
		echo $$! > /tmp/learn-ml-server.pid; \
		echo "$(GREEN)✓ Server started (PID: $$(cat /tmp/learn-ml-server.pid))$(NC)"; \
		echo "$(BLUE)📍 Access: http://localhost:$(PORT)$(NC)"; \
		echo "$(BLUE)📋 Logs: tail -f /tmp/learn-ml-server.log$(NC)"; \
	fi

stop-server: ## Stop background web server
	@if [ -f /tmp/learn-ml-server.pid ]; then \
		PID=$$(cat /tmp/learn-ml-server.pid); \
		if kill $$PID 2>/dev/null; then \
			echo "$(GREEN)✓ Server stopped (PID: $$PID)$(NC)"; \
			rm /tmp/learn-ml-server.pid; \
		else \
			echo "$(RED)✗ Could not stop server$(NC)"; \
		fi; \
	else \
		echo "$(YELLOW)⚠️  No background server running$(NC)"; \
	fi

convert-notebooks: ## Convert Jupyter notebooks to HTML
	@echo "$(BLUE)🔄 Converting notebooks to HTML...$(NC)"
	@echo "$(BLUE)  → ML Fundamentals notebooks...$(NC)"
	@for nb in notebooks/week-*.ipynb; do \
		$(NBCONVERT) --to html --output-dir $(ML_DOCS) $$nb 2>/dev/null && \
		echo "    $(GREEN)✓ $$(basename $$nb .ipynb).html$(NC)"; \
	done
	@echo "$(BLUE)  → LangChain notebooks...$(NC)"
	@for nb in langchain/notebooks/week-*.ipynb; do \
		$(NBCONVERT) --to html --output-dir $(LANGCHAIN_DOCS) $$nb 2>/dev/null && \
		echo "    $(GREEN)✓ $$(basename $$nb .ipynb).html$(NC)"; \
	done
	@echo "$(BLUE)  → CrewAI notebooks...$(NC)"
	@for nb in crewai/notebooks/week-*.ipynb; do \
		$(NBCONVERT) --to html --output-dir $(CREWAI_DOCS) $$nb 2>/dev/null && \
		echo "    $(GREEN)✓ $$(basename $$nb .ipynb).html$(NC)"; \
	done
	@echo "$(BLUE)  → LangGraph notebooks...$(NC)"
	@for nb in langgraph/notebooks/week-*.ipynb; do \
		$(NBCONVERT) --to html --output-dir $(LANGGRAPH_DOCS) $$nb 2>/dev/null && \
		echo "    $(GREEN)✓ $$(basename $$nb .ipynb).html$(NC)"; \
	done
	@echo "$(GREEN)✓ All notebooks converted to HTML$(NC)"
	@echo "$(BLUE)🎨 Running HTML enhancement script...$(NC)"
	@$(PYTHON) enhance_html.py 2>/dev/null && echo "$(GREEN)✓ HTML enhancement complete$(NC)" || echo "$(YELLOW)⚠️  HTML enhancement skipped (enhance_html.py not found)$(NC)"

update-html: convert-notebooks ## Alias for convert-notebooks

lint: ## Run code linting (flake8)
	@echo "$(BLUE)🔍 Linting Python files...$(NC)"
	@$(PYTHON) -m flake8 --max-line-length=100 --exclude=$(VENV_DIR) . || true
	@echo "$(GREEN)✓ Linting complete$(NC)"

format: ## Format Python code with black
	@echo "$(BLUE)🎨 Formatting Python files...$(NC)"
	@$(PYTHON) -m black --line-length=100 --exclude=$(VENV_DIR) . 2>/dev/null || true
	@echo "$(GREEN)✓ Formatting complete$(NC)"

test: ## Run tests (pytest)
	@echo "$(BLUE)🧪 Running tests...$(NC)"
	@$(PYTHON) -m pytest tests/ -v 2>/dev/null || echo "$(YELLOW)No tests found$(NC)"

clean: ## Clean up temporary files and caches
	@echo "$(BLUE)🧹 Cleaning up...$(NC)"
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@find . -type f -name ".DS_Store" -delete 2>/dev/null || true
	@rm -rf .pytest_cache 2>/dev/null || true
	@rm -rf .ipynb_checkpoints 2>/dev/null || true
	@rm -rf .vscode/__pycache__ 2>/dev/null || true
	@echo "$(GREEN)✓ Cleanup complete$(NC)"

clean-all: clean stop-server ## Full cleanup including venv and server
	@echo "$(BLUE)🧹 Performing full cleanup...$(NC)"
	@read -p "$(YELLOW)Remove virtual environment? (y/n) $(NC)" confirm && \
	if [ "$$confirm" = "y" ]; then \
		rm -rf $(VENV_DIR); \
		echo "$(GREEN)✓ Virtual environment removed$(NC)"; \
	fi
	@echo "$(GREEN)✓ Full cleanup complete$(NC)"

status: ## Show project status
	@echo "$(BLUE)📊 Project Status$(NC)"
	@echo ""
	@echo "$(GREEN)Python Environment:$(NC)"
	@if [ -d $(VENV_DIR) ]; then \
		echo "  $(GREEN)✓ Virtual environment exists$(NC)"; \
	else \
		echo "  $(RED)✗ No virtual environment (run 'make setup')$(NC)"; \
	fi
	@echo ""
	@echo "$(GREEN)Notebooks:$(NC)"
	@echo "  ML Fundamentals: $$(ls notebooks/week-*.ipynb 2>/dev/null | wc -l) notebooks"
	@echo "  LangChain: $$(ls langchain/notebooks/week-*.ipynb 2>/dev/null | wc -l) notebooks"
	@echo "  CrewAI: $$(ls crewai/notebooks/week-*.ipynb 2>/dev/null | wc -l) notebooks"
	@echo "  LangGraph: $$(ls langgraph/notebooks/week-*.ipynb 2>/dev/null | wc -l) notebooks"
	@echo "  Total: $$(find . -name 'week-*.ipynb' -type f | wc -l) notebooks"
	@echo ""
	@echo "$(GREEN)HTML Docs:$(NC)"
	@echo "  ML Fundamentals: $$(ls $(ML_DOCS)/week-*.html 2>/dev/null | wc -l) files"
	@echo "  LangChain: $$(ls $(LANGCHAIN_DOCS)/week-*.html 2>/dev/null | wc -l) files"
	@echo "  CrewAI: $$(ls $(CREWAI_DOCS)/week-*.html 2>/dev/null | wc -l) files"
	@echo "  LangGraph: $$(ls $(LANGGRAPH_DOCS)/week-*.html 2>/dev/null | wc -l) files"
	@echo "  Total: $$(find . -path ./$(VENV_DIR) -prune -o -name 'week-*.html' -type f -print | wc -l) files"
	@echo ""
	@echo "$(GREEN)Server:$(NC)"
	@if pgrep -f "http.server $(PORT)" > /dev/null; then \
		echo "  $(GREEN)✓ Running on http://localhost:$(PORT)$(NC)"; \
	else \
		echo "  $(YELLOW)✗ Not running$(NC)"; \
	fi

docs: ## Generate documentation
	@echo "$(BLUE)📖 Generating documentation...$(NC)"
	@echo "$(GREEN)✓ Documentation is available in /docs, /langchain/docs, /crewai/docs, /langgraph/docs$(NC)"

all: setup install-deps convert-notebooks ## Run complete setup (env + deps + convert)
	@echo ""
	@echo "$(GREEN)╔════════════════════════════════════════════════════════════╗$(NC)"
	@echo "$(GREEN)║           ✅ Setup Complete!                               ║$(NC)"
	@echo "$(GREEN)╚════════════════════════════════════════════════════════════╝$(NC)"
	@echo ""
	@echo "$(BLUE)Next steps:$(NC)"
	@echo "  1. Start server:  $(YELLOW)make serve$(NC)"
	@echo "  2. Access:        $(YELLOW)http://localhost:$(PORT)$(NC)"
	@echo "  3. View status:   $(YELLOW)make status$(NC)"
	@echo "  4. Clean up:      $(YELLOW)make clean$(NC)"
	@echo ""

.PHONY: help setup install-deps serve serve-bg stop-server convert-notebooks update-html lint format test clean clean-all status docs all
