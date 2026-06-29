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

Contributions must not include any material derived from or replicating the
thermodynamic sampling method described in USPTO Provisional 64/019,999
without explicit written permission from Paul W. Shaver. By submitting a pull
request you certify that your contribution is original and does not constitute
a patent work-for-hire claim.

## Contact

Paul W. Shaver — whtetigr2@gmail.com
