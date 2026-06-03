# TASB — Thermodynamic Attention Sampling Bridge

Paul W. Shaver | 2026

A no-retrain inference-time bridge between frozen pretrained transformers
and Extropic-style TSU stochastic substrates.

## THIS IS THE CANONICAL TREE

Everything in this directory is the current bridge. Legacy code lives in
`legacy/` and is preserved for provenance only — not for citation.

## Stage 1 status

| File | Status | Tests |
| --- | --- | --- |
| `tasb_capture_v2.py` | ✓ Validated | 8/8 pass |
| `tasb_sampler_v2.py` | ✓ Validated | 9/9 pass |
| `tasb_injector_v2.py` | Not started | — |
| `tasb_verify_v2.py` | Not started | — |
| `tasb_pipeline_v2.py` | Not started | — |

## Stage 1 invariants (3 of 4 validated)

| Invariant | Status |
| --- | --- |
| `capture_invariant` | ✓ test_capture_v2.py T1-T6 |
| `rope_regression` | ✓ test_rope_regression.py + test_rope_live_capture_v2.py |
| `backend_equivalence` | ✓ test_sampler_v2.py T4 |
| `alpha_zero_identity` | pending tasb_injector_v2.py |

## Run the tests

```bash
cd ~/.lightning_studio/TASB_Refactor
python tests/test_capture_v2.py
python tests/test_sampler_v2.py
```

## Codex's eight-milestone status

- M1 Canonical stack — ✓ done (this consolidation)
- M2 Correct measurement object — ✓ done
- M3 Hard-fail invalid experiments — ✓ done
- M4 Exact-vs-approximate comparison — ✓ done
- M5 Faithfulness core test — pending tasb_injector_v2.py
- M6 Production-realism — pending M5
- M7 One-family bridge claim — pending M5
- M8 Calibration program — pending M7

## Engineering posture

Research-grade code where false confidence is more dangerous than slow
progress. Rules: local runtime is source of truth; smallest falsifying
diagnostic before refactor; distinguish conceptual/implementation/test
bugs; compare like-with-like; track abs/rel/max/mean/percentile error
separately; inspect Transformer internals end-to-end; rule out dtype/cast
before assuming deep bug; use confidence language; keep hypothesis ledger;
redesign cannot outrun proof; flag invalid science aggressively;
prioritize definitely-true / must-fix / can-wait.
