# Contributing to thunder4d

The project uses [uv](https://docs.astral.sh/uv/) for the environment, the build and the releases.

## Development environment

```bash
git clone https://github.com/victorkolt/thunder4D.git
cd thunder4D
uv sync --extra examples        # .venv with thunder4d in editable mode, plus matplotlib
uv run --extra examples python examples/compare_gauss_tophat.py
uv run --extra examples --with jupyter jupyter lab    # tutorial and benchmark notebooks
```

The package lives in `src/thunder4d/`. Add `--extra gpu` for CuPy.

## Tests

```bash
uv run pytest
```

Each test checks one property (a convention, a normalisation or one physical term), so a failure points at one function.
The suite runs in about 20 s. The GPU test is skipped when CuPy has no CUDA device.

## Documentation

The documentation is written with [Sphinx](https://www.sphinx-doc.org) and the Read the Docs theme.
The sources are in `docs/source/`.

```bash
uv run --group docs sphinx-build -b html -W --keep-going docs/source docs/_build/html
# open docs/_build/html/index.html
```

Every push to `main` runs `.github/workflows/docs.yml`.
It builds the HTML and deploys it to https://victorkolt.github.io/thunder4D/.

## Releases

1. Bump the version with `uv version --bump patch` (or `minor`, `major`). It updates `pyproject.toml` and `uv.lock`.
   PyPI never accepts the same version twice.
2. Commit, push, then on GitHub *Releases → Draft a new release*, tag `vX.Y.Z`, *Publish*.
3. `.github/workflows/publish.yml` builds with `uv build` and uploads to PyPI.
   A minute later, `uv add thunder4d` gets the new version.

To try the upload first, run *Actions → publish → Run workflow*. It publishes to TestPyPI instead. Check it with:

```bash
uv run --no-project --with thunder4d --index https://test.pypi.org/simple/ \
    --index-strategy unsafe-best-match python -c "import thunder4d; print(thunder4d.__version__)"
```

The uploads use PyPI trusted publishing, so no API token is stored.
The GitHub environments `pypi` and `testpypi` are registered as trusted publishers on pypi.org and test.pypi.org.

To build locally:

```bash
uv build                 # -> dist/thunder4d-X.Y.Z.tar.gz and .whl
uvx twine check dist/*
```
