#!/usr/bin/env python3
"""Add navigation, GitHub/Colab links, and setup instructions to rendered HTML.

Safe to run repeatedly: previous enhancement blocks are stripped first.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parent
GITHUB_REPO = "sanketn26/learn-ml"

COURSE_META = {
    "ml": {"weeks_fallback": 15, "name": "ML Fundamentals"},
    "langchain": {"weeks_fallback": 6, "name": "LangChain Mastery"},
    "langgraph": {"weeks_fallback": 4, "name": "LangGraph Workflows"},
    "crewai": {"weeks_fallback": 4, "name": "Crew.ai Multi-Agents"},
}

ENHANCE_START = "<!-- learn-ml-enhance:start -->"
ENHANCE_END = "<!-- learn-ml-enhance:end -->"
FOOTER_START = "<!-- learn-ml-footer:start -->"
FOOTER_END = "<!-- learn-ml-footer:end -->"


def get_week_number(filename: str) -> int | None:
    match = re.search(r"week-(\d+)", filename)
    return int(match.group(1)) if match else None


def get_course_key(html_path: Path) -> str:
    text = str(html_path)
    if "langchain" in text:
        return "langchain"
    if "langgraph" in text:
        return "langgraph"
    if "crewai" in text:
        return "crewai"
    return "ml"


def notebook_rel(course_key: str, notebook_filename: str) -> str:
    if course_key == "ml":
        return f"notebooks/{notebook_filename}"
    return f"{course_key}/notebooks/{notebook_filename}"


def course_landing_href(course_key: str) -> str:
    """Relative link from docs/week-*.html back to the course landing page."""
    if course_key == "ml":
        return "../index.html#ml-course"
    return "../index.html"


def list_week_pages(docs_dir: Path) -> dict[int, str]:
    pages: dict[int, str] = {}
    for path in docs_dir.glob("week-*.html"):
        week = get_week_number(path.name)
        if week is not None:
            pages[week] = path.name
    return pages


def neighbor_link(pages: dict[int, str], week_num: int, delta: int, label: str, css: str) -> str:
    target = pages.get(week_num + delta)
    if not target:
        return ""
    return f'<a href="{target}" class="nav-link {css}">{label}</a>'


def strip_enhancements(content: str) -> str:
    content = re.sub(
        rf"{re.escape(ENHANCE_START)}.*?{re.escape(ENHANCE_END)}\s*",
        "",
        content,
        flags=re.S,
    )
    content = re.sub(
        rf"{re.escape(FOOTER_START)}.*?{re.escape(FOOTER_END)}\s*",
        "",
        content,
        flags=re.S,
    )
    # Legacy chrome from earlier enhance_html.py runs (no markers).
    content = re.sub(
        r"\n<div class=\"navigation-footer\">.*?</div>\n(?=</body>)",
        "\n",
        content,
        count=1,
        flags=re.S,
    )
    content = re.sub(
        r"<style>\s*:root \{\s*--think:.*?</style>\n?",
        "",
        content,
        count=1,
        flags=re.S,
    )
    content = re.sub(
        r"<style>\s*\.notebook-header \{.*?</style>\s*"
        r"<div class=\"notebook-header\">.*?</div>\s*"
        r"<div class=\"setup-instructions\">.*?</div>\s*",
        "",
        content,
        count=1,
        flags=re.S,
    )
    return content


def pip_line(course_key: str) -> str:
    if course_key == "langchain":
        return "<code>pip install langchain openai python-dotenv</code>"
    if course_key == "langgraph":
        return "<code>pip install langgraph langchain openai</code>"
    if course_key == "crewai":
        return "<code>pip install crewai openai</code>"
    return (
        "<code>pip install numpy pandas scikit-learn matplotlib scipy seaborn</code>"
        "<br><code>pip install torch</code>  <!-- Weeks 11–15 only; CPU wheel is enough -->"
    )


def create_header_section(
    course_key: str,
    course_name: str,
    week_num: int,
    pages: dict[int, str],
    filename: str,
) -> tuple[str, str, str]:
    notebook_filename = filename.replace(".html", ".ipynb")
    rel = notebook_rel(course_key, notebook_filename)
    github_download_url = f"https://raw.githubusercontent.com/{GITHUB_REPO}/main/{rel}"
    colab_url = f"https://colab.research.google.com/github/{GITHUB_REPO}/blob/main/{rel}"
    vscode_url = f"https://vscode.dev/github/{GITHUB_REPO}/blob/main/{rel}"
    jupyter_url = f"jupyter://relative-path:{rel}"
    course_link = course_landing_href(course_key)

    first_week = min(pages) if pages else week_num
    last_week = max(pages) if pages else week_num
    week_label = f"Week {week_num} of {first_week}–{last_week}"

    prev_link = neighbor_link(pages, week_num, -1, "← Previous Week", "prev-link")
    next_link = neighbor_link(pages, week_num, 1, "Next Week →", "next-link")

    pedagogy = ""
    css_path = REPO / "notebooks" / "course.css"
    if css_path.exists():
        pedagogy = f"<style>\n{css_path.read_text()}\n</style>\n"

    header = f"""
{ENHANCE_START}
{pedagogy}
<style>
.notebook-header {{
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 20px;
    border-radius: 8px;
    margin-bottom: 30px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
}}
.notebook-header h1 {{
    margin: 0 0 15px 0;
    font-size: 2em;
}}
.breadcrumbs {{
    font-size: 0.95em;
    margin-bottom: 15px;
    opacity: 0.95;
}}
.breadcrumbs a {{
    color: white;
    text-decoration: none;
    border-bottom: 2px solid rgba(255,255,255,0.3);
    padding-bottom: 2px;
}}
.breadcrumbs a:hover {{
    border-bottom-color: white;
}}
.breadcrumbs span {{
    margin: 0 8px;
    opacity: 0.7;
}}
.header-links {{
    display: flex;
    gap: 15px;
    flex-wrap: wrap;
    margin-bottom: 15px;
}}
.header-link {{
    display: inline-block;
    background: rgba(255,255,255,0.15);
    color: white;
    padding: 8px 12px;
    border-radius: 5px;
    text-decoration: none;
    font-size: 0.9em;
    border: 1px solid rgba(255,255,255,0.3);
    transition: all 0.3s ease;
}}
.header-link:hover {{
    background: rgba(255,255,255,0.25);
    border-color: white;
}}
.header-link.colab {{
    background: rgba(249, 203, 0, 0.2);
    border-color: rgba(249, 203, 0, 0.5);
}}
.header-link.vscode {{
    background: rgba(0, 120, 215, 0.2);
    border-color: rgba(0, 120, 215, 0.5);
}}
.header-link.jupyter {{
    background: rgba(249, 149, 0, 0.2);
    border-color: rgba(249, 149, 0, 0.5);
}}
.navigation-footer {{
    display: flex;
    gap: 20px;
    justify-content: space-between;
    margin-top: 40px;
    padding-top: 20px;
    border-top: 2px solid #e0e0e0;
}}
.nav-link {{
    display: inline-block;
    padding: 10px 20px;
    background: #f0f0f0;
    color: #333;
    text-decoration: none;
    border-radius: 5px;
    transition: all 0.3s ease;
    font-weight: 600;
}}
.nav-link:hover {{
    background: #667eea;
    color: white;
}}
.nav-link.prev-link {{
    margin-right: auto;
}}
.nav-link.next-link {{
    margin-left: auto;
}}
.setup-instructions {{
    background: #f5f5f5;
    border-left: 4px solid #667eea;
    padding: 15px;
    border-radius: 5px;
    margin: 20px 0;
    font-size: 0.95em;
}}
.setup-instructions h3 {{
    margin-top: 0;
    color: #667eea;
}}
.setup-instructions code {{
    background: white;
    padding: 2px 6px;
    border-radius: 3px;
    font-family: 'Courier New', monospace;
}}
</style>

<div class="notebook-header">
    <div class="breadcrumbs">
        <a href="{course_link}">← Back to {course_name}</a>
        <span>•</span>
        <span>{week_label}</span>
    </div>
    <h1>Week {week_num} — {course_name}</h1>
    <div class="header-links">
        <a href="{colab_url}" class="header-link colab" target="_blank">Open in Colab</a>
        <a href="{vscode_url}" class="header-link vscode" target="_blank">Open in VS Code</a>
        <a href="{jupyter_url}" class="header-link jupyter">Open in Jupyter</a>
        <a href="{github_download_url}" class="header-link download" target="_blank">Download .ipynb</a>
    </div>
</div>

<div class="setup-instructions">
    <h3>📝 How to Run This Notebook Locally</h3>
    <p>
        <strong>1. Clone the repository:</strong><br>
        <code>git clone https://github.com/{GITHUB_REPO}.git && cd learn-ml</code>
    </p>
    <p>
        <strong>2. Set up environment:</strong><br>
        <code>python3 -m venv venv && source venv/bin/activate</code><br>
        <code>pip install jupyter notebook</code>
    </p>
    <p>
        <strong>3. Install course dependencies:</strong><br>
        {pip_line(course_key)}
    </p>
    <p>
        <strong>4. Open the notebook:</strong><br>
        <code>jupyter notebook {rel}</code>
    </p>
</div>
{ENHANCE_END}
"""
    return header, prev_link, next_link


def enhance_html_file(html_path: Path, pages: dict[int, str]) -> bool:
    try:
        content = html_path.read_text(encoding="utf-8")
        filename = html_path.name
        course_key = get_course_key(html_path)
        course_name = COURSE_META[course_key]["name"]
        week_num = get_week_number(filename)

        if week_num is None:
            print(f"⚠️  Skipped {filename} - could not extract week number")
            return False

        content = strip_enhancements(content)

        header, prev_link, next_link = create_header_section(
            course_key, course_name, week_num, pages, filename
        )

        body_pattern = r"(<body[^>]*>)"
        if not re.search(body_pattern, content):
            print(f"⚠️  Skipped {filename} - no body tag found")
            return False

        content = re.sub(body_pattern, r"\1" + header, content, count=1)

        footer = f"""
{FOOTER_START}
<div class="navigation-footer">
    {prev_link if prev_link else "<div></div>"}
    <a href="{course_landing_href(course_key)}" class="nav-link">Back to {course_name}</a>
    {next_link if next_link else "<div></div>"}
</div>
{FOOTER_END}
"""
        if "</body>" not in content:
            print(f"⚠️  Skipped {filename} - no closing body tag")
            return False
        content = content.replace("</body>", footer + "</body>", 1)

        html_path.write_text(content, encoding="utf-8")
        print(f"✅ Enhanced {filename}")
        return True
    except Exception as exc:
        print(f"❌ Error processing {html_path.name}: {exc}")
        return False


def main() -> None:
    courses = [
        ("ML", REPO / "docs"),
        ("LangChain", REPO / "langchain" / "docs"),
        ("LangGraph", REPO / "langgraph" / "docs"),
        ("CrewAI", REPO / "crewai" / "docs"),
    ]

    total = 0
    success = 0
    for label, docs_path in courses:
        if not docs_path.exists():
            continue
        html_files = sorted(docs_path.glob("week-*.html"))
        if not html_files:
            continue
        pages = list_week_pages(docs_path)
        print(f"\n📂 Processing {label} course ({len(html_files)} files)...")
        for html_file in html_files:
            total += 1
            if enhance_html_file(html_file, pages):
                success += 1

    print(f"\n{'=' * 60}")
    print(f"✨ Enhancement complete: {success}/{total} files updated")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
