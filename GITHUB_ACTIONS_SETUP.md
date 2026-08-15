# GitHub Pages

`main` is the only branch you edit. Markdown lives in `docs/`. GitHub Actions builds MkDocs and publishes the HTML. There is no `gh-pages` branch to maintain.

## What runs

[`.github/workflows/pages.yml`](.github/workflows/pages.yml) on every push to `main`:

1. `pip install mkdocs-material`
2. `mkdocs build --strict`
3. Upload `site/` and deploy with `actions/deploy-pages`

## One-time repo setting

**Settings → Pages → Source: GitHub Actions**

Not “Deploy from a branch.” If the source is `main` or `gh-pages`, GitHub ignores this workflow and Jekyll (or a stale branch) is what people see.

After the first green **Deploy course site** run, https://sanketn26.github.io/learn-ml/ is the Material site.

You can delete the old `gh-pages` branch once this is on.

## Why not serve `main` as-is

`main` is source (markdown, Python, CSVs). Pages needs the *built* HTML. Building on every push in Actions keeps generated files out of git.

## Local

```bash
pip install mkdocs-material
mkdocs serve
```
