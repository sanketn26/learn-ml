# 🚀 Learn ML - Setup & Development Guide

This guide explains how to use the Makefile for reproducible setup, development, and hosting.

## 📋 Quick Start

```bash
# First time setup (creates venv, installs dependencies, converts notebooks)
make all

# Start development server
make serve

# Access the site at http://localhost:8000
```

## 📚 Available Make Targets

### 🔧 Environment & Setup

| Command | Description |
|---------|-------------|
| `make setup` | Create Python virtual environment and install all dependencies |
| `make install-deps` | Install dependencies (assumes venv exists) |
| `make all` | Complete setup: venv + dependencies + notebook conversion |

### 🚀 Server & Hosting

| Command | Description |
|---------|-------------|
| `make serve` | Start web server (foreground) on http://localhost:8000 |
| `make serve-bg` | Start web server in background |
| `make stop-server` | Stop background server |
| `make status` | Show project status (environment, notebooks, server) |

### 📖 Documentation & Conversion

| Command | Description |
|---------|-------------|
| `make convert-notebooks` | Convert all Jupyter notebooks to HTML |
| `make update-html` | Alias for `convert-notebooks` |
| `make docs` | Display documentation locations |

### 🧹 Cleanup & Maintenance

| Command | Description |
|---------|-------------|
| `make clean` | Remove cache files, __pycache__, .ipynb_checkpoints |
| `make clean-all` | Full cleanup including venv removal |
| `make lint` | Run flake8 code linting |
| `make format` | Format code with black |
| `make test` | Run pytest tests |

### 📊 Information

| Command | Description |
|---------|-------------|
| `make help` | Display all available commands |
| `make status` | Show project status (Python env, notebooks, server) |

---

## 🎯 Common Workflows

### Initial Project Setup

```bash
# Clone repository
git clone <repo-url>
cd learn-ml

# Complete setup
make all

# Start server
make serve-bg

# Access the site
open http://localhost:8000

# View logs
tail -f /tmp/learn-ml-server.log
```

### Update Notebooks to HTML

When you edit Jupyter notebooks, regenerate HTML:

```bash
# Convert all notebooks to HTML
make convert-notebooks

# Reload browser (server keeps serving automatically)
```

### Daily Development

```bash
# Start background server (once)
make serve-bg

# Edit notebooks in notebooks/ folders
# Convert when done
make convert-notebooks

# Access at http://localhost:8000

# When finished
make stop-server
```

### Full Project Refresh

```bash
# Clean everything and restart
make clean-all
make all
make serve-bg
```

---

## 📁 Project Structure

```
learn-ml/
├── Makefile                 ← All utilities (THIS FILE)
├── requirements.txt         ← Python dependencies
├── .gitignore              ← Git ignore rules
│
├── notebooks/              ← SaaS course notebooks (12 weeks)
├── docs/                   ← SaaS course HTML (12 weeks)
│
├── langchain/
│   ├── notebooks/          ← 6 Jupyter notebooks
│   └── docs/               ← 6 HTML files
│
├── crewai/
│   ├── notebooks/          ← 4 Jupyter notebooks
│   └── docs/               ← 4 HTML files
│
├── langgraph/
│   ├── notebooks/          ← 4 Jupyter notebooks
│   └── docs/               ← 4 HTML files
│
├── data/                   ← CSV/JSON data files
├── assignments/            ← Weekly assignments (12 weeks)
└── solutions/              ← Solution files
```

---

## 🔍 Detailed Command Reference

### make setup
**Creates virtual environment and installs all dependencies**

```bash
make setup
```

What it does:
1. Creates `venv/` directory
2. Upgrades pip, setuptools, wheel
3. Installs Jupyter, nbconvert, JupyterLab
4. Installs frameworks: langchain, crewai, langgraph
5. Installs data science packages: pandas, numpy, scikit-learn, matplotlib, etc.
6. Installs dev tools: black, flake8, pytest

### make serve
**Start web server in foreground (blocking)**

```bash
make serve
```

What it does:
- Starts Python HTTP server on port 8000
- Serves files from current directory
- Press Ctrl+C to stop
- Use this for interactive development

### make serve-bg
**Start web server in background (non-blocking)**

```bash
make serve-bg
```

What it does:
- Starts server in background
- Saves PID to `/tmp/learn-ml-server.pid`
- Saves logs to `/tmp/learn-ml-server.log`
- Returns to command prompt immediately
- View logs: `tail -f /tmp/learn-ml-server.log`
- Stop with: `make stop-server`

### make convert-notebooks
**Convert all Jupyter notebooks to HTML**

```bash
make convert-notebooks
```

What it does:
- Converts LangChain notebooks → `langchain/docs/`
- Converts CrewAI notebooks → `crewai/docs/`
- Converts LangGraph notebooks → `langgraph/docs/`
- Uses `jupyter nbconvert`
- Preserves notebook formatting and output

### make clean
**Remove temporary files and caches**

```bash
make clean
```

What it does:
- Removes `__pycache__/` directories
- Removes `.pyc` files
- Removes `.ipynb_checkpoints/` directories
- Removes `.pytest_cache/`
- Removes `.DS_Store` files
- Does NOT remove venv or notebooks

### make status
**Show project status**

```bash
make status
```

Displays:
- ✓/✗ Virtual environment status
- Count of notebooks per framework
- Count of HTML docs per framework
- ✓/✗ Server running status

---

## 🐛 Troubleshooting

### "Command not found: make"
Install make:
- **macOS**: `brew install make`
- **Ubuntu/Debian**: `sudo apt-get install build-essential`
- **Windows**: Use WSL2 or install [MinGW](http://mingw.org/)

### "Port 8000 already in use"
Change port in Makefile or stop the existing process:
```bash
lsof -i :8000
kill -9 <PID>
```

Or use a different port:
```bash
make serve PORT=8001
```

### "nbconvert not found"
Make sure dependencies are installed:
```bash
make setup
# or
make install-deps
```

### "Permission denied" on startup
Some systems may need chmod:
```bash
chmod +x Makefile
```

### Virtual environment not activating
The Makefile uses the venv Python directly:
```bash
# Instead of:
source venv/bin/activate
python script.py

# Use:
venv/bin/python script.py
# or just use make targets
```

---

## 🔄 Reproducibility

The Makefile ensures reproducibility:

1. **Consistent Environment**: `make setup` creates identical venv
2. **Locked Dependencies**: `requirements.txt` pins versions
3. **Automated Conversion**: `make convert-notebooks` keeps HTML in sync
4. **Clean State**: `make clean-all` removes everything and restarts
5. **Documented Workflow**: Every target is documented and tested

### Reproducible Workflow

```bash
# Anyone can clone and set up identically
git clone <repo>
cd learn-ml
make all          # Same environment every time
make serve-bg     # Same hosting every time
```

---

## 📝 Extending the Makefile

To add new targets:

```makefile
new-target: dependencies ## Description
	@echo "$(BLUE)Message...$(NC)"
	command here
	@echo "$(GREEN)✓ Done$(NC)"
```

Key conventions:
- Use `@` prefix to suppress command echo
- Use `##` for help description
- Use color variables: `$(BLUE)`, `$(GREEN)`, `$(YELLOW)`, `$(RED)`
- Use `@echo "$(GREEN)✓ Message$(NC)"` for nice formatting

---

## 🚀 Next Steps

1. **Run `make all`** to set everything up
2. **Run `make serve-bg`** to start the server
3. **Edit notebooks** in `langchain/`, `crewai/`, `langgraph/` folders
4. **Run `make convert-notebooks`** to update HTML
5. **Access** http://localhost:8000 in your browser

Happy coding! 🎉
