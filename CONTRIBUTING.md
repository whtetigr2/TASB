# Contributing to thermobridge

## Getting started

```bash
git clone https://github.com/whtetigr2/TASB
cd thermobridge
pip install -e ".[dev]"
pre-commit install
```

## Running tests

```bash
pytest tests/
```

Tests that require a GPU and a downloaded LLaMA model are marked `@pytest.mark.slow`
and are skipped by default. Run them with `pytest -m slow`.

## Code style

- `flake8` with settings in `.flake8`
- Pre-commit hooks enforce formatting before every commit

## Pull requests

1. Fork → branch → PR against `main`
2. Include a test for any new public API surface
3. Update `CHANGELOG.md` under `[Unreleased]`

## Intellectual property notice

By submitting a pull request you certify that your contribution is your own
work, or that you otherwise have the right to submit it, and that you license
it under this project's MIT licence.

## Contact

Paul W. Shaver — whtetigr2@gmail.com
