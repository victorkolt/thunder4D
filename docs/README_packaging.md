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
pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ thunder4d
```

## Real release

1. Bump `__version__` in `thunder4d/__init__.py` (PyPI never accepts the same version twice).
2. Commit, push, then on GitHub *Releases → Draft a new release*, tag `v0.1.0`, *Publish*.
3. The workflow builds and uploads; a minute later: `pip install thunder4d`.

## Build locally (optional)

```bash
pip install build twine
python -m build          # -> dist/thunder4d-X.Y.Z.tar.gz and .whl
twine check dist/*
```
