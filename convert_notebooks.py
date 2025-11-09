#!/usr/bin/env python3
"""
Convert Jupyter notebooks to HTML using nbconvert.
"""
import subprocess
import sys
from pathlib import Path

def convert_notebook(notebook_path, output_dir):
    """Convert a single notebook to HTML."""
    try:
        cmd = [
            sys.executable, "-m", "nbconvert",
            "--to", "html",
            "--output-dir", str(output_dir),
            str(notebook_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            return True, f"✓ {notebook_path.name}"
        else:
            return False, f"✗ {notebook_path.name}: {result.stderr}"
    except Exception as e:
        return False, f"✗ {notebook_path.name}: {str(e)}"

def main():
    base_path = Path("/home/sanket/workspaces/learn-ml")
    
    frameworks = {
        "LangChain": (base_path / "langchain/notebooks", base_path / "langchain/docs"),
        "CrewAI": (base_path / "crewai/notebooks", base_path / "crewai/docs"),
        "LangGraph": (base_path / "langgraph/notebooks", base_path / "langgraph/docs"),
    }
    
    print("🔄 Converting Notebooks to HTML")
    print("=" * 70)
    
    for framework, (nb_dir, out_dir) in frameworks.items():
        print(f"\n📚 {framework}:")
        
        # Ensure output directory exists
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # Find all week notebooks
        notebooks = sorted(nb_dir.glob("week-*.ipynb"))
        
        if not notebooks:
            print(f"  ⚠️  No notebooks found in {nb_dir}")
            continue
        
        for nb in notebooks:
            success, msg = convert_notebook(nb, out_dir)
            print(f"  {msg}")
    
    print("\n" + "=" * 70)
    print("✅ Conversion complete!")

if __name__ == "__main__":
    main()
