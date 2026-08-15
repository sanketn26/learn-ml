# GitHub Pages

The course is an MkDocs Material site. Source is `docs/*.md`. There is no notebook render step.

## What runs

[`.github/workflows/pages.yml`](.github/workflows/pages.yml) on every push to `main`:

1. `pip install mkdocs-material`
2. `mkdocs build --strict`
3. Upload `site/` and deploy with `actions/deploy-pages`

## One-time repo settings

1. **Settings → Pages → Source:** GitHub Actions (not “Deploy from a branch”).
2. The first run on `main` publishes [https://sanketn26.github.io/learn-ml/](https://sanketn26.github.io/learn-ml/).

## Local

```bash
pip install mkdocs-material
mkdocs serve
```
