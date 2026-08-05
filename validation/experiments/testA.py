"""
TEST A — is the chromatic-number obstruction real, and does correct blocking fix it?

CLAIM UNDER TEST (mine, from the failed coupled run): a one-hot group is the
complete graph K_S; block Gibbs over one block containing all S mutually-coupled
spins is a Jacobi update, not a Gibbs sweep, so it converges to the wrong thing.
Correct blocking needs S single-node blocks -> fully sequential, no parallelism.

If TRUE  : S-block version converges to softmax, 1-block version does not.
If FALSE : my diagnosis was wrong and the obstruction is imaginary.
"""
import numpy as np, jax, jax.numpy as jnp, time
from thrml import (SpinNode, Block, BlockGibbsSpec, FactorSamplingProgram,
                   SamplingSchedule, sample_states)
from thrml.models.discrete_ebm import SpinEBMFactor, SpinGibbsConditional

SEED = 0
rng = np.random.default_rng(SEED)

def matchings(n):
    idx = list(range(n - 1)); out = []
    for r in range(n - 1):
        rot = idx[r:] + idx[:r]
        pairs = [(rot[0], n - 1)]
        for t in range(1, n // 2):
            pairs.append((rot[t], rot[n - 1 - t]))
        out.append(pairs)
    return out

def build(S, logits, lam, blocking):
    """blocking='one' -> single block (WRONG for K_S); 'perS' -> S single-node blocks."""
    nodes = [SpinNode() for _ in range(S)]
    h = np.array([(logits[j] + lam) / 2.0 - lam * (S - 1) / 2.0 for j in range(S)])
    fb = Block(nodes)
    factors = [SpinEBMFactor([fb], jnp.asarray(h))]
    for pairs in matchings(S):
        A = Block([nodes[i] for i, _ in pairs]); B = Block([nodes[j] for _, j in pairs])
        factors.append(SpinEBMFactor([A, B], jnp.full(len(pairs), -lam / 2.0)))
    if blocking == 'one':
        free = [fb]; samplers = [SpinGibbsConditional()]
    else:
        free = [Block([n]) for n in nodes]; samplers = [SpinGibbsConditional() for _ in nodes]
    spec = BlockGibbsSpec(free_super_blocks=free, clamped_blocks=[])
    prog = FactorSamplingProgram(gibbs_spec=spec, samplers=samplers,
                                 factors=factors, other_interaction_groups=[])
    return prog, free, nodes, fb

def run(S, logits, lam, blocking, K, nw, key):
    prog, free, nodes, fb = build(S, logits, lam, blocking)
    ks = jax.random.split(key, len(free) + 1)
    init = [jax.random.bernoulli(ks[i], 0.5, (len(b.nodes),)) for i, b in enumerate(free)]
    t0 = time.time()
    sm = sample_states(ks[-1], prog, SamplingSchedule(nw, K, 1), init, [], free)
    dt = time.time() - t0
    x = np.concatenate([np.asarray(s).astype(int).reshape(K, -1) for s in sm], axis=1)
    return x, dt

def stats(x, target):
    ones = x.sum(1); valid = ones == 1
    bad = 1 - valid.mean()
    if valid.sum() > 20:
        e = x[valid].mean(0); e = e / e.sum()
        tv = 0.5 * np.abs(e - target).sum()
    else:
        tv = float('nan')
    return tv, bad

print("=" * 78)
print("TEST A — chromatic-number obstruction: one block (K_S) vs S single-node blocks")
print("=" * 78)
for S in [8, 16]:
    logits = rng.normal(0, 1.5, size=S)
    tgt = np.exp(logits - logits.max()); tgt /= tgt.sum()
    print(f"\n  S={S}")
    print(f"  {'blocking':>10} {'lambda':>7} {'warmup':>7} {'TV':>9} {'invalid':>8} {'sec':>7}")
    for blocking in ['one', 'perS']:
        for lam in [2.0, 4.0]:
            key = jax.random.key(SEED + S)
            try:
                x, dt = run(S, logits, lam, blocking, 3000, 2000, key)
                tv, bad = stats(x, tgt)
                tvs = f"{tv:9.4f}" if tv == tv else f"{'n/a':>9}"
                print(f"  {blocking:>10} {lam:7.1f} {2000:7d} {tvs} {bad:8.3f} {dt:7.2f}")
            except Exception as e:
                print(f"  {blocking:>10} {lam:7.1f}  ERR {type(e).__name__}: {str(e)[:60]}")
print()
print("PREDICTION: 'one' fails (invalid high, TV bad); 'perS' converges (invalid low, TV small).")
