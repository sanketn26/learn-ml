# 🎯 Makefile Quick Reference

## One-Liner Setup
```bash
make all
```

## Most Used Commands

| Command | What It Does |
|---------|-------------|
| `make help` | Show all available commands |
| `make setup` | Create venv + install dependencies |
| `make serve-bg` | Start server in background |
| `make stop-server` | Stop background server |
| `make status` | Show project status |
| `make convert-notebooks` | Update HTML from Jupyter |
| `make clean` | Remove cache & temp files |

## Development Workflow

```bash
# Start of day
make serve-bg

# During development
# (edit notebooks in notebooks/ folders)

# After editing
make convert-notebooks

# Check everything
make status

# End of day
make stop-server
```

## Common Issues & Fixes

| Problem | Solution |
|---------|----------|
| "Port 8000 in use" | `make stop-server` then `make serve-bg` |
| "Command not found: make" | Install: `brew install make` (macOS) |
| "nbconvert not found" | Run `make setup` to install dependencies |
| "Server not starting" | Check logs: `tail -f /tmp/learn-ml-server.log` |

## Access Points

| Resource | URL |
|----------|-----|
| Home | http://localhost:8000 |
| LangChain | http://localhost:8000/langchain/docs/ |
| CrewAI | http://localhost:8000/crewai/docs/ |
| LangGraph | http://localhost:8000/langgraph/docs/ |

## Full Command List

```bash
make help              # Display all commands
make setup             # Create environment + install deps
make install-deps      # Install dependencies only
make serve             # Start server (foreground)
make serve-bg          # Start server (background)
make stop-server       # Stop background server
make convert-notebooks # Convert .ipynb to .html
make update-html       # Same as convert-notebooks
make lint              # Check code style
make format            # Auto-format code
make test              # Run tests
make clean             # Remove caches
make clean-all         # Remove everything + venv
make status            # Show project status
make docs              # Show docs locations
make all               # Full setup
```

---

**📖 For detailed info:** Read `MAKEFILE_GUIDE.md`
