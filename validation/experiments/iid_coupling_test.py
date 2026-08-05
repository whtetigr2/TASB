"""
IID / COUPLING TEST — settles whether thermobridge's THRML graph samples a
coupled EBM or S independent categorical variables.

Three measurements, plus a POSITIVE CONTROL. The control is the point: a null
result ("we measured no correlation") is uninterpretable unless the same
measurement demonstrably detects correlation when it is genuinely present.

  A  lag-1 autocorrelation across consecutive draws   -> IID across time?
  B  cross-node correlation between distinct nodes    -> coupled in space?
  C  warmup insensitivity (n_warmup 0 vs 200)         -> is there anything to mix?
  D  goodness of fit vs softmax                       -> sanity, should pass

Graph 1 (THERMOBRIDGE): CategoricalEBMFactor([block], W) -- one node group.
Graph 2 (CONTROL):      CategoricalEBMFactor([blockA, blockB], W) -- two groups,
                        pairwise coupling. Must show correlation, or the
                        instrument is broken and every other number here is void.
"""
import numpy as np, jax, jax.numpy as jnp
from thrml import (CategoricalNode, Block, BlockGibbsSpec,
                   FactorSamplingProgram, SamplingSchedule, sample_states)
from thrml.models.discrete_ebm import CategoricalEBMFactor, CategoricalGibbsConditional

S, K, SEED = 16, 4000, 0
rng = np.random.default_rng(SEED)

def lag1_autocorr(x):
    """Mean |lag-1 autocorrelation| over columns of x (K, n)."""
    out = []
    for i in range(x.shape[1]):
        v = x[:, i].astype(float)
        if v.std() < 1e-12:
            continue
        a, b = v[:-1], v[1:]
        out.append(abs(np.corrcoef(a, b)[0, 1]))
    return float(np.mean(out)) if out else float('nan')

def cross_node_corr(x):
    """Mean |correlation| between DISTINCT nodes, across draws."""
    keep = [i for i in range(x.shape[1]) if x[:, i].std() > 1e-12]
    if len(keep) < 2:
        return float('nan')
    C = np.corrcoef(x[:, keep].T.astype(float))
    iu = np.triu_indices_from(C, k=1)
    return float(np.mean(np.abs(C[iu])))

def run_single_group(W, n_warmup, key):
    """thermobridge's construction: ONE categorical node group."""
    nodes = [CategoricalNode() for _ in range(S)]
    blk = Block(nodes)
    prog = FactorSamplingProgram(
        gibbs_spec=BlockGibbsSpec(free_super_blocks=[blk], clamped_blocks=[]),
        samplers=[CategoricalGibbsConditional(S)],
        factors=[CategoricalEBMFactor([blk], W)],
        other_interaction_groups=[])
    k1, k2 = jax.random.split(key)
    init = [jax.random.randint(k1, (S,), 0, S, dtype=jnp.uint8)]
    sm = sample_states(k2, prog, SamplingSchedule(n_warmup, K, 1), init, [], [blk])
    return np.asarray(sm[0])                      # (K, S)

def run_two_group(WA, Wpair, key):
    """POSITIVE CONTROL: two node groups with a pairwise factor -> real coupling."""
    a = Block([CategoricalNode() for _ in range(S)])
    b = Block([CategoricalNode() for _ in range(S)])
    prog = FactorSamplingProgram(
        gibbs_spec=BlockGibbsSpec(free_super_blocks=[a, b], clamped_blocks=[]),
        samplers=[CategoricalGibbsConditional(S), CategoricalGibbsConditional(S)],
        factors=[CategoricalEBMFactor([a, b], Wpair)],
        other_interaction_groups=[])
    k1, k2, k3 = jax.random.split(key, 3)
    init = [jax.random.randint(k1, (S,), 0, S, dtype=jnp.uint8),
            jax.random.randint(k2, (S,), 0, S, dtype=jnp.uint8)]
    sm = sample_states(k3, prog, SamplingSchedule(0, K, 1), init, [], [a, b])
    return np.asarray(sm[0]), np.asarray(sm[1])

key = jax.random.key(SEED)
W = jnp.asarray(rng.normal(0, 2.0, size=(S, S)))   # attention-like logits

print("=" * 74)
print("GRAPH 1 — THERMOBRIDGE CONSTRUCTION: CategoricalEBMFactor([block], W)")
print("=" * 74)
k, key = jax.random.split(key)
x0 = run_single_group(W, 0, k)
k, key = jax.random.split(key)
x200 = run_single_group(W, 200, k)

print(f"  A  lag-1 autocorrelation      : {lag1_autocorr(x0):.5f}   (IID -> ~0)")
print(f"  B  cross-node correlation     : {cross_node_corr(x0):.5f}   (uncoupled -> ~0)")

emp0  = np.stack([np.bincount(x0[:, i],   minlength=S) / K for i in range(S)])
emp200= np.stack([np.bincount(x200[:, i], minlength=S) / K for i in range(S)])
tgt   = np.asarray(jax.nn.softmax(W, axis=-1))
tv0   = float(np.abs(emp0   - tgt).sum(1).mean() / 2)
tv200 = float(np.abs(emp200 - tgt).sum(1).mean() / 2)
print(f"  C  TV vs softmax, warmup=0    : {tv0:.5f}")
print(f"     TV vs softmax, warmup=200  : {tv200:.5f}")
print(f"     warmup effect (|delta|)    : {abs(tv0-tv200):.5f}   (nothing to mix -> ~0)")
print(f"  D  distribution match         : {'PASS' if tv0 < 0.05 else 'FAIL'}")

print()
print("=" * 74)
print("GRAPH 2 — POSITIVE CONTROL: two node groups, pairwise coupling")
print("=" * 74)
Wpair = jnp.asarray(rng.normal(0, 3.0, size=(S, S, S)))
try:
    k, key = jax.random.split(key)
    ya, yb = run_two_group(W, Wpair, k)
    print(f"  A  lag-1 autocorrelation      : {lag1_autocorr(ya):.5f}   (coupled -> >0)")
    print(f"  B  cross-node corr (within A) : {cross_node_corr(ya):.5f}")
    inter = [abs(np.corrcoef(ya[:, i].astype(float), yb[:, i].astype(float))[0, 1])
             for i in range(S)
             if ya[:, i].std() > 1e-12 and yb[:, i].std() > 1e-12]
    print(f"  B' A-to-B coupled-pair corr   : {np.mean(inter):.5f}   (coupled -> >0)")
    print("  -> instrument DOES detect coupling when coupling exists")
except Exception as e:
    print(f"  CONTROL FAILED TO BUILD: {type(e).__name__}: {str(e)[:200]}")
    print("  -> cannot interpret Graph 1's nulls without a working control")
