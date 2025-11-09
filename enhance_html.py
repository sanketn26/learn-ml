#!/usr/bin/env python3
"""
Enhanced HTML Renderer for Jupyter Notebooks
Adds GitHub links, navigation breadcrumbs, and setup instructions to rendered HTML
"""

import os
import re
from pathlib import Path

def get_week_number(filename):
    """Extract week number from filename like week-01-saas.html"""
    match = re.search(r'week-(\d+)', filename)
    return int(match.group(1)) if match else None

def get_course_info(html_path):
    """Determine course from path"""
    if 'langchain' in str(html_path):
        return 'langchain', 6, 'LangChain Mastery'
    elif 'langgraph' in str(html_path):
        return 'langgraph', 4, 'LangGraph Workflows'
    elif 'crewai' in str(html_path):
        return 'crewai', 4, 'Crew.ai Multi-Agents'
    else:
        return 'ml', 12, 'ML Fundamentals'

def get_course_link(course_key):
    """Get landing page link for course"""
    links = {
        'ml': '../../index.html#ml-course',
        'langchain': '../../langchain/index.html',
        'langgraph': '../../langgraph/index.html',
        'crewai': '../../crewai/index.html'
    }
    return links.get(course_key, '../../index.html')

def create_header_section(course_key, course_name, week_num, total_weeks, filename):
    """Create enhanced header with GitHub links and breadcrumbs"""
    
    notebook_filename = filename.replace('.html', '.ipynb')
    github_notebook_url = f"https://github.com/sanketn26/learn-ml/blob/main/{course_key}/notebooks/{notebook_filename}"
    github_download_url = f"https://raw.githubusercontent.com/sanketn26/learn-ml/main/{course_key}/notebooks/{notebook_filename}"
    colab_url = f"https://colab.research.google.com/github/sanketn26/learn-ml/blob/main/{course_key}/notebooks/{notebook_filename}"
    vscode_url = f"vscode://file/{os.path.abspath(os.path.join(os.path.dirname(__file__), course_key, 'notebooks', notebook_filename))}"
    jupyter_url = f"jupyter://relative-path:{course_key}/notebooks/{notebook_filename}"
    course_link = get_course_link(course_key)
    
    # Determine previous and next weeks
    prev_link = ""
    next_link = ""
    if week_num > 1:
        prev_week = f"week-{week_num-1:02d}"
        prev_filename = f"{prev_week}-{filename.split('-', 2)[2]}"  # Keep the rest of filename
        prev_link = f"""
        <a href="{prev_filename}" class="nav-link prev-link">← Previous Week</a>
        """
    
    if week_num < total_weeks:
        next_week = f"week-{week_num+1:02d}"
        next_filename = f"{next_week}-{filename.split('-', 2)[2]}"
        next_link = f"""
        <a href="{next_filename}" class="nav-link next-link">Next Week →</a>
        """
    
    header = f"""
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

.header-link.github::before {{
    content: "🔗 ";
}}

.header-link.download::before {{
    content: "⬇️ ";
}}

.header-link.colab::before {{
    content: "☁️ ";
}}

.header-link.colab {{
    background: rgba(249, 203, 0, 0.2);
    border-color: rgba(249, 203, 0, 0.5);
}}

.header-link.colab:hover {{
    background: rgba(249, 203, 0, 0.3);
    border-color: rgba(249, 203, 0, 1);
}}

.header-link.vscode::before {{
    content: "💻 ";
}}

.header-link.vscode {{
    background: rgba(0, 120, 215, 0.2);
    border-color: rgba(0, 120, 215, 0.5);
}}

.header-link.vscode:hover {{
    background: rgba(0, 120, 215, 0.3);
    border-color: rgba(0, 120, 215, 1);
}}

.header-link.jupyter::before {{
    content: "📓 ";
}}

.header-link.jupyter {{
    background: rgba(249, 149, 0, 0.2);
    border-color: rgba(249, 149, 0, 0.5);
}}

.header-link.jupyter:hover {{
    background: rgba(249, 149, 0, 0.3);
    border-color: rgba(249, 149, 0, 1);
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
        <span>Week {week_num} of {total_weeks}</span>
    </div>
    
    <h1>Week {week_num} — {course_name}</h1>
    
    <div class="header-links">
        <a href="{colab_url}" class="header-link colab" target="_blank">Open in Colab</a>
        <a href="{vscode_url}" class="header-link vscode">Open in VS Code</a>
        <a href="{jupyter_url}" class="header-link jupyter">Open in Jupyter</a>
        <a href="{github_download_url}" class="header-link download" target="_blank">Download .ipynb</a>
    </div>
</div>

<div class="setup-instructions">
    <h3>📝 How to Run This Notebook Locally</h3>
    <p>
        <strong>1. Clone the repository:</strong><br>
        <code>git clone https://github.com/sanketn26/learn-ml.git && cd learn-ml</code>
    </p>
    <p>
        <strong>2. Set up environment:</strong><br>
        <code>python3 -m venv venv && source venv/bin/activate</code><br>
        <code>pip install jupyter notebook</code>
    </p>
    <p>
        <strong>3. Install course dependencies:</strong><br>
    """
    
    # Add course-specific setup
    if course_key == 'langchain':
        header += "<code>pip install langchain openai python-dotenv</code>"
    elif course_key == 'langgraph':
        header += "<code>pip install langgraph langchain openai</code>"
    elif course_key == 'crewai':
        header += "<code>pip install crewai openai</code>"
    else:  # ML
        header += "<code>pip install numpy pandas scikit-learn matplotlib scipy</code>"
    
    header += f"""
    </p>
    <p>
        <strong>4. Open the notebook:</strong><br>
        <code>jupyter notebook {course_key}/notebooks/week-{week_num:02d}*.ipynb</code>
    </p>
</div>
"""
    
    return header, prev_link, next_link

def enhance_html_file(html_path):
    """Add enhancements to a single HTML file"""
    try:
        with open(html_path, 'r', encoding='utf-8') as f:
            content = f.read()

        filename = os.path.basename(html_path)
        course_key, total_weeks, course_name = get_course_info(html_path)
        week_num = get_week_number(filename)

        if not week_num:
            print(f"⚠️  Skipped {filename} - could not extract week number")
            return False

        # Check if already enhanced (idempotency check)
        if 'class="notebook-header"' in content:
            print(f"ℹ️  Skipped {filename} - already enhanced")
            return False

        # Create header and navigation
        header, prev_link, next_link = create_header_section(
            course_key, course_name, week_num, total_weeks, filename
        )

        # Find insertion point (after <body> tag)
        body_pattern = r'(<body[^>]*>)'
        if not re.search(body_pattern, content):
            print(f"⚠️  Skipped {filename} - no body tag found")
            return False

        # Insert header after body tag
        content = re.sub(body_pattern, r'\1' + header, content, count=1)

        # Add footer navigation before closing body tag
        footer = f"""
<div class="navigation-footer">
    {prev_link if prev_link else '<div></div>'}
    <a href="{get_course_link(course_key)}" class="nav-link">Back to {course_name}</a>
    {next_link if next_link else '<div></div>'}
</div>
"""
        content = content.replace('</body>', footer + '</body>')

        # Write back
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(content)

        print(f"✅ Enhanced {filename}")
        return True

    except Exception as e:
        print(f"❌ Error processing {filename}: {e}")
        return False

def main():
    """Find and enhance all HTML files in docs directories"""
    # Use the directory where this script is located
    base_path = Path(__file__).parent.absolute()
    courses = ['', 'langchain', 'langgraph', 'crewai']
    
    total = 0
    success = 0
    
    for course in courses:
        if course:
            docs_path = base_path / course / 'docs'
        else:
            docs_path = base_path / 'docs'
        
        if not docs_path.exists():
            continue
        
        html_files = sorted(docs_path.glob('week-*.html'))
        
        if html_files:
            print(f"\n📂 Processing {course or 'ML'} course ({len(html_files)} files)...")
            for html_file in html_files:
                total += 1
                if enhance_html_file(html_file):
                    success += 1
    
    print(f"\n{'='*60}")
    print(f"✨ Enhancement complete: {success}/{total} files updated")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()
