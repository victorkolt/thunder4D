# Publishing thunder4d on PyPI

## One-time setup

1. Fill in `authors` and the two `[project.urls]` in `pyproject.toml`.
2. Add a `LICENSE` file to the repository (and `license = "..."` in `pyproject.toml`).
3. Create accounts on https://pypi.org and https://test.pypi.org (enable 2FA).
4. On **both** sites: *Your projects → Publishing → Add a new pending publisher → GitHub*:
   - PyPI project name: `thunder4d`
   - Owner / repository: your GitHub account and repository name
   - Workflow name: `publish.yml`
   - Environment: `pypi` (on pypi.org) / `testpypi` (on test.pypi.org)
5. On GitHub: *Settings → Environments*, create `pypi` and `testpypi`.

No API token is stored anywhere: GitHub proves its identity to PyPI ("trusted publishing").

## Test run (TestPyPI)

*Actions → publish → Run workflow*, then:

```bash
uv run --no-project --with thunder4d --index https://test.pypi.org/simple/ \
    --index-strategy unsafe-best-match python -c "import thunder4d; print(thunder4d.__version__)"
```

## Real release

1. Bump the version with `uv version --bump patch` (or `minor`, `major`). It updates `pyproject.toml` and `uv.lock`.
   PyPI never accepts the same version twice.
2. Commit, push, then on GitHub *Releases → Draft a new release*, tag `v0.1.0`, *Publish*.
3. The workflow builds and uploads; a minute later: `uv add thunder4d`.

## Build locally (optional)

```bash
uv build                 # -> dist/thunder4d-X.Y.Z.tar.gz and .whl
uvx twine check dist/*
```
