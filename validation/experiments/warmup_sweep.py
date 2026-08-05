"""Does the coupled one-hot graph converge if given an actual mixing budget?

The uncoupled graph needed none. This one has barriers between one-hot states,
so the question is whether warmup buys valid samples, or whether the barriers
are simply too high at any affordable budget.
"""
import numpy as np, jax, jax.numpy as jnp
# reuse run()/stats()/target from coupled.py, stopping before its own main block
src = open("coupled.py").read().split("key = jax.random.key(SEED)")[0]
exec(src)

key = jax.random.key(1)
print("=" * 72)
print("MIXING BUDGET SWEEP — coupled one-hot Ising, S=8")
print("=" * 72)
print(f"{'lambda':>7} {'warmup':>8} {'TV':>10} {'invalid':>9} {'autocorr':>10}")
for lam in [0.5, 1.0, 2.0]:
    for nw in [0, 200, 2000, 10000]:
        key, k = jax.random.split(key)
        try:
            tv, bad, ac = stats(run(lam, nw, k))
            tvs = f"{tv:10.4f}" if tv == tv else f"{'n/a':>10}"
            print(f"{lam:7.1f} {nw:8d} {tvs} {bad:9.3f} {ac:10.4f}")
        except Exception as e:
            print(f"{lam:7.1f} {nw:8d}  ERR {type(e).__name__}: {str(e)[:50]}")
print()
print("reference: uncoupled categorical -> TV 0.0156, invalid 0, autocorr 0.013")
