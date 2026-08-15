# GitHub Pages

The course is an MkDocs Material site. Source is `docs/*.md`. CI builds it and pushes the HTML to the `gh-pages` branch.

## What runs

[`.github/workflows/pages.yml`](.github/workflows/pages.yml) on every push to `main`:

1. `pip install mkdocs-material`
2. `mkdocs gh-deploy --force --strict` → updates `gh-pages`

This does **not** use “GitHub Actions” as the Pages source. It uses the `gh-pages` branch, which matches a repo that was already “Deploy from a branch.”

## One-time repo setting

**Settings → Pages**

- Source: **Deploy from a branch**
- Branch: **`gh-pages`** / **`/ (root)`**

Not `main`. `main` is the markdown source; Jekyll will render `README.md` and the raw `docs/*.md` files and the week URLs will 404.

The first successful run creates `gh-pages`. Flip the dropdown after that run, or create an empty `gh-pages` branch first.

## Local

```bash
pip install mkdocs-material
mkdocs serve
```
