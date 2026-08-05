"""
COUPLED FORMULATION — one-hot Ising encoding of an attention row.

WHY COUPLED. Z1 is p-bits (binary), not S-state categoricals. Encoding a
categorical for that hardware requires one-hot, and one-hot is a CONSTRAINT,
which in an Ising system means pairwise repulsive couplings. The coupling is
forced by the hardware, not chosen to make the graph interesting.

  E(x) = -sum_j l_j x_j + lam*(sum_j x_j - 1)^2
       = sum_j (-l_j - lam) x_j + 2*lam*sum_{j<k} x_j x_k + const

All-pairs couplings on K_S. THRML forbids a variable appearing twice in one
factor, so K_S is edge-coloured into perfect matchings (S-1 for even S) and each
colour class becomes one factor with disjoint pairs.

MEASURED
  1 fidelity  : TV(empirical | softmax) restricted to valid one-hot states
  2 validity  : fraction of draws that are NOT exactly one-hot  <- the real cost
  3 mixing    : lag-1 autocorrelation                           <- absent before
  4 tradeoff  : all three as a function of lambda
"""
import numpy as np, jax, jax.numpy as jnp
from thrml import (SpinNode, Block, BlockGibbsSpec, FactorSamplingProgram,
                   SamplingSchedule, sample_states)
from thrml.models.discrete_ebm import SpinEBMFactor, SpinGibbsConditional

S, K, SEED = 8, 4000, 0
rng = np.random.default_rng(SEED)
logits = rng.normal(0, 1.5, size=S)
target = np.exp(logits - logits.max()); target /= target.sum()

def matchings(n):
    """Edge-colour K_n into n-1 perfect matchings (circle method, n even)."""
    idx = list(range(n - 1)); out = []
    for r in range(n - 1):
        rot = idx[r:] + idx[:r]
        pairs = [(rot[0], n - 1)]
        for t in range(1, n // 2):
            pairs.append((rot[t], rot[n - 1 - t]))
        out.append(pairs)
    return out

def run(lam, n_warmup, key):
    nodes = [SpinNode() for _ in range(S)]
    # E = -sum W*(spin product)  [verified: SpinGibbsConditional gamma convention]
    # so pairwise REPULSION needs W = -lam/2, not +lam/2; x = (s+1)/2. Substituting into E(x) gives
    # field h_j and coupling J = lam/2 on every pair (constants dropped).
    h = np.array([(logits[j] + lam) / 2.0 - lam * (S - 1) / 2.0 for j in range(S)])
    blocks, factors = [], []
    # unary field: one factor over a single spin group
    fb = Block(nodes)
    factors.append(SpinEBMFactor([fb], jnp.asarray(h)))
    # pairwise: one factor per matching, pairs disjoint within each factor
    for pairs in matchings(S):
        A = Block([nodes[i] for i, _ in pairs])
        B = Block([nodes[j] for _, j in pairs])
        factors.append(SpinEBMFactor([A, B], jnp.full(len(pairs), -lam / 2.0)))
    spec = BlockGibbsSpec(free_super_blocks=[fb], clamped_blocks=[])
    prog = FactorSamplingProgram(gibbs_spec=spec, samplers=[SpinGibbsConditional()],
                                 factors=factors, other_interaction_groups=[])
    k1, k2 = jax.random.split(key)
    init = [jax.random.bernoulli(k1, 0.5, (S,))]
    sm = sample_states(k2, prog, SamplingSchedule(n_warmup, K, 1), init, [], [fb])
    return np.asarray(sm[0]).astype(int)          # (K, S) in {0,1}

def stats(x):
    ones = x.sum(axis=1)
    valid = ones == 1
    frac_bad = 1.0 - valid.mean()
    if valid.sum() > 10:
        emp = x[valid].mean(axis=0); emp = emp / emp.sum()
        tv = 0.5 * np.abs(emp - target).sum()
    else:
        tv = float('nan')
    ac = []
    for j in range(S):
        v = x[:, j].astype(float)
        if v.std() > 1e-12:
            ac.append(abs(np.corrcoef(v[:-1], v[1:])[0, 1]))
    return tv, frac_bad, (float(np.mean(ac)) if ac else float('nan'))

key = jax.random.key(SEED)
print("=" * 72)
print(f"COUPLED one-hot Ising, S={S}, K={K}")
print("=" * 72)
print(f"{'lambda':>7} {'TV|softmax':>11} {'invalid':>9} {'lag1-autocorr':>14}")
for lam in [0.5, 1.0, 2.0, 4.0, 8.0, 16.0]:
    key, k = jax.random.split(key)
    try:
        tv, bad, ac = stats(run(lam, 0, k))
        print(f"{lam:7.1f} {tv:11.4f} {bad:9.3f} {ac:14.4f}")
    except Exception as e:
        print(f"{lam:7.1f}  FAILED: {type(e).__name__}: {str(e)[:80]}")

print()
print("UNCOUPLED baseline (categorical, from the IID test): lag1-autocorr ~ 0.013")
print("If lambda>0 shows autocorrelation well above that, the coupling is real")
print("and the sampler now has a mixing time it did not have before.")
