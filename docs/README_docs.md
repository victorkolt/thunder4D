# Building and publishing the thunder4d documentation

The documentation is written with [Sphinx](https://www.sphinx-doc.org) and the Read the
Docs theme, the same setup as fbpic.github.io.

## Build locally

```bash
pip install -r docs/requirements.txt
sphinx-build -b html docs/source docs/_build/html
# open docs/_build/html/index.html
```

## Publish on GitHub Pages

1. Push the repository to GitHub.
2. In the repository: **Settings → Pages → Build and deployment → Source: GitHub Actions**.
3. Every push to `main` runs `.github/workflows/docs.yml`, which builds the HTML and deploys it.

The site address depends on the repository name:

- repository `thunder4d` under your account → `https://<account>.github.io/thunder4d/`
- a GitHub organisation `thunder4d` with a repository named `thunder4d.github.io`
  → `https://thunder4d.github.io` (this is how fbpic.github.io is set up: the docs live
  in a dedicated `<org>.github.io` repository).

## Structure

```
docs/source/
├── conf.py                     Sphinx configuration (theme, extensions)
├── index.rst                   landing page
├── overview/                   physical model, numerical scheme, conventions
├── install/installation.rst
├── how_to_run.rst              workflow, injection modes, long runs, output
├── examples/examples.rst       example scripts with figures
├── api_reference/              one page per module
├── advanced/                   validation, performance, extending
└── _static/                    CSS and figures
```
