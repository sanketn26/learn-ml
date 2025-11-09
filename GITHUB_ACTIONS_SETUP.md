# GitHub Actions Workflow Setup Guide

## 📋 Overview

This guide covers setting up GitHub Actions for automated HTML rendering and deployment to GitHub Pages.

## ✅ Recommended Workflow Configuration

### Option 1: Deploy to gh-pages Branch (Recommended)

This approach uses the `peaceiris/actions-gh-pages` action, which is battle-tested and handles permissions correctly.

**File:** `.github/workflows/build-and-deploy.yml`

```yaml
name: Build and Deploy to GitHub Pages

on:
  push:
    branches: [ main ]
    paths:
      - 'notebooks/**/*.ipynb'
      - 'langchain/notebooks/**/*.ipynb'
      - 'langgraph/notebooks/**/*.ipynb'
      - 'crewai/notebooks/**/*.ipynb'
  workflow_dispatch:

permissions:
  contents: write
  pages: write
  id-token: write

jobs:
  build:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
      with:
        fetch-depth: 0
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install jupyter nbconvert
    
    - name: Convert ML notebooks to HTML
      run: |
        cd notebooks
        jupyter nbconvert --to html *.ipynb --output-dir=../docs
        cd ..
    
    - name: Convert LangChain notebooks to HTML
      run: |
        cd langchain/notebooks
        jupyter nbconvert --to html *.ipynb --output-dir=../docs
        cd ../..
    
    - name: Convert LangGraph notebooks to HTML
      run: |
        cd langgraph/notebooks
        jupyter nbconvert --to html *.ipynb --output-dir=../docs
        cd ../..
    
    - name: Convert Crew.ai notebooks to HTML
      run: |
        cd crewai/notebooks
        jupyter nbconvert --to html *.ipynb --output-dir=../docs
        cd ../..
    
    - name: Run HTML enhancement script
      run: |
        python enhance_html.py
    
    - name: Deploy to GitHub Pages
      uses: peaceiris/actions-gh-pages@v3
      with:
        github_token: ${{ secrets.GITHUB_TOKEN }}
        publish_dir: .
        cname: learn-ml.example.com  # Optional: set your custom domain
```

### Option 2: Deploy to docs/ on main Branch

If you prefer to keep everything on the main branch:

```yaml
name: Build HTML and Commit

on:
  push:
    branches: [ main ]
    paths:
      - 'notebooks/**/*.ipynb'
  workflow_dispatch:

permissions:
  contents: write

jobs:
  build:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
      with:
        token: ${{ secrets.GITHUB_TOKEN }}
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install jupyter nbconvert
    
    - name: Convert notebooks
      run: |
        # Convert ML course
        cd notebooks && jupyter nbconvert --to html *.ipynb --output-dir=../docs && cd ..
        # Convert LangChain course
        cd langchain/notebooks && jupyter nbconvert --to html *.ipynb --output-dir=../docs && cd ../..
        # Convert LangGraph course
        cd langgraph/notebooks && jupyter nbconvert --to html *.ipynb --output-dir=../docs && cd ../..
        # Convert Crew.ai course
        cd crewai/notebooks && jupyter nbconvert --to html *.ipynb --output-dir=../docs && cd ../..
    
    - name: Enhance HTML files
      run: python enhance_html.py
    
    - name: Commit and push if changed
      run: |
        git config --local user.email "action@github.com"
        git config --local user.name "GitHub Action"
        git add docs/ langchain/docs/ langgraph/docs/ crewai/docs/
        git diff --quiet && git diff --staged --quiet || (git commit -m "Auto: render notebooks to HTML" && git push)
```

## 🔧 GitHub Repository Settings

### 1. Enable GitHub Pages

1. Go to **Settings** → **Pages**
2. Under "Source":
   - For Option 1: Select `gh-pages` branch
   - For Option 2: Select `main` branch, `/docs` folder (or root)
3. Save

### 2. Secrets Setup

No special secrets needed if using `GITHUB_TOKEN` (built-in).

For custom domain:
1. Go to **Settings** → **Pages** → **Custom domain**
2. Enter your domain (e.g., `learn-ml.example.com`)
3. Add CNAME record in DNS provider

### 3. Branch Protection (Optional)

To prevent accidental commits to main:
1. Go to **Settings** → **Branches**
2. Add rule for `main`
3. Require pull request reviews before merge

## 🚀 Workflow Management

### Triggering Builds

**Automatic:** Whenever you push notebook changes to `main`

**Manual:** Use GitHub UI:
1. Go to **Actions**
2. Select workflow
3. Click "Run workflow"

### Monitoring Workflow Status

1. Go to **Actions** tab
2. Check workflow runs
3. Click run for details
4. View logs for troubleshooting

### Environment Variables

Add in workflow if needed:

```yaml
env:
  JUPYTER_ENABLE_LAB: yes
  NBCONVERT_TIMEOUT: 600  # 10 minutes per notebook
```

## 🔍 Troubleshooting

### Issue: "Permission denied" when pushing

**Solution:** Use `peaceiris/actions-gh-pages` (Option 1) instead of direct git push

### Issue: Timeout converting large notebooks

**Solution:** Increase timeout in workflow:

```yaml
- name: Convert notebooks
  env:
    JUPYTER_ENABLE_LAB: yes
  run: |
    jupyter nbconvert --to html --ExecutePreprocessor.timeout=600 *.ipynb
```

### Issue: Out of memory

**Solution:** Use `--no-input` flag:

```bash
jupyter nbconvert --to html --no-input *.ipynb
```

### Issue: HTML files not updating

**Solution:**
1. Clear browser cache (Ctrl+Shift+Del)
2. Hard refresh (Ctrl+F5)
3. Check GitHub Pages build status in Actions

### Issue: Custom domain not working

**Solution:**
1. Verify DNS CNAME record points to `github.io`
2. Wait 5-10 minutes for DNS propagation
3. Check repo settings confirm custom domain

## 📊 Performance Tips

### Reduce Build Time

1. Only convert modified notebooks:
```yaml
- name: Get changed files
  uses: dorny/paths-filter@v2
  id: changes
  with:
    filters: |
      notebooks:
        - 'notebooks/*.ipynb'
```

2. Use caching:
```yaml
- name: Cache pip
  uses: actions/cache@v3
  with:
    path: ~/.cache/pip
    key: ${{ runner.os }}-pip
```

3. Use faster conversion:
```bash
# Skip code execution
jupyter nbconvert --to html --no-input *.ipynb
```

### Limit Artifact Storage

GitHub provides 1 GB free storage. Clean up old builds:

1. Go to **Settings** → **Actions** → **General**
2. Set "Artifact retention" to 30 days

## 🔐 Security

### Best Practices

1. **Never commit secrets** to repo
2. **Use GITHUB_TOKEN** for authentication
3. **Limit workflow permissions** in workflow file
4. **Review PR changes** before merge
5. **Keep dependencies updated**

### Dependabot for Security

Enable automatic dependency updates:

1. Go to **Settings** → **Code security and analysis**
2. Enable "Dependabot alerts"
3. Enable "Dependabot security updates"

## 📚 Reference

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [GitHub Pages Documentation](https://docs.github.com/en/pages)
- [peaceiris/actions-gh-pages](https://github.com/peaceiris/actions-gh-pages)
- [Jupyter nbconvert](https://nbconvert.readthedocs.io/)

## ✅ Deployment Checklist

- [ ] Workflow file created in `.github/workflows/`
- [ ] GitHub Pages enabled in repo settings
- [ ] Branch/folder selected for pages source
- [ ] First workflow run triggered
- [ ] HTML files generated and accessible
- [ ] Custom domain configured (if using)
- [ ] Workflow passing (green checkmark)
- [ ] Pages site publicly accessible

---

**Last Updated:** November 2025

For issues or questions, check GitHub Actions logs or open an issue on the repository.
