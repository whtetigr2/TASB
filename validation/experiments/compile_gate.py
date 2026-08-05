"""
VARIATIONAL COMPILATION of the gate factor, in Thermalizers' sense.

Their method: given a target factor, fit a hardware-native EBM's parameters to
minimise a divergence from the target, then track how per-factor error composes.
My gate so far used a HEURISTIC (top-10% threshold, unit sharpness). This
replaces the heuristic with a fitted kernel family.

Kernel family (2 params, deliberately tiny -- it must stay hardware-native):
    g_j = sigmoid(a * l_j + b)        -> spin field W_j = 0.5*(a*l_j + b)
The induced readout is p_j = E[h_j / sum_k h_k], estimated by Monte Carlo.
We minimise KL(softmax(l) || p) over (a, b).

Reported: compilation error before vs after fitting, and against the categorical
alternative -- which cannot be compiled to a degree-16 device at all without
mutual exclusion, so its structural error is not a number but a topology failure.
"""
import numpy as np
from scipy.optimize import minimize

SEED = 0
rng = np.random.default_rng(SEED)
NDRAW = 4000

def softmax(z):
    e = np.exp(z - z.max()); return e / e.sum()

def gate_readout(g, r, ndraw=NDRAW):
    """p_j = E[h_j / sum_k h_k] for independent Bernoulli(g)."""
    h = (r.random((ndraw, g.size)) < g)
    s = h.sum(1, keepdims=True)
    keep = s[:, 0] > 0
    if keep.sum() == 0:
        return np.full_like(g, 1.0 / g.size)
    w = h[keep] / s[keep]
    p = w.mean(0)
    return p / p.sum()

def kl(p, q):
    p = np.clip(p, 1e-12, None); q = np.clip(q, 1e-12, None)
    return float((p * np.log(p / q)).sum())

def tv(p, q):
    return float(0.5 * np.abs(p - q).sum())

def heuristic_g(l, frac=0.10):
    tau = np.sort(l)[-max(1, int(frac * l.size))]
    return 1.0 / (1.0 + np.exp(-(l - tau)))

def compile_gate(l, seed):
    """Fit (a,b) to minimise KL(softmax(l) || gate readout)."""
    tgt = softmax(l)
    def obj(th):
        a, b = th
        g = 1.0 / (1.0 + np.exp(-(a * l + b)))
        r = np.random.default_rng(seed)          # common random numbers
        return kl(tgt, gate_readout(g, r))
    best, bestv = None, np.inf
    for a0 in [0.5, 1.0, 2.0, 4.0]:
        for b0 in [-4.0, -2.0, 0.0]:
            res = minimize(obj, [a0, b0], method="Nelder-Mead",
                           options={"maxiter": 120, "xatol": 1e-2, "fatol": 1e-4})
            if res.fun < bestv:
                bestv, best = res.fun, res.x
    return best, bestv

print("=" * 78)
print("VARIATIONAL COMPILATION of the gate factor (Thermalizers-style)")
print("=" * 78)
print(f"{'S':>5} {'KL heuristic':>13} {'KL compiled':>12} {'TV heur':>9} {'TV comp':>9} {'a':>7} {'b':>7}")
rows = []
for S in [16, 32, 64, 128]:
    kh = kc = th_ = tc = 0.0; A = B = 0.0
    R = 4
    for rep in range(R):
        r = np.random.default_rng(SEED + rep)
        l = r.normal(0, 1.5, S)
        tgt = softmax(l)
        gh = heuristic_g(l)
        ph = gate_readout(gh, np.random.default_rng(1234))
        (a, b), _ = compile_gate(l, seed=1234)
        gc = 1.0 / (1.0 + np.exp(-(a * l + b)))
        pc = gate_readout(gc, np.random.default_rng(1234))
        kh += kl(tgt, ph); kc += kl(tgt, pc)
        th_ += tv(tgt, ph); tc += tv(tgt, pc); A += a; B += b
    rows.append((S, kh/R, kc/R, th_/R, tc/R, A/R, B/R))
    print(f"{S:5d} {kh/R:13.4f} {kc/R:12.4f} {th_/R:9.4f} {tc/R:9.4f} {A/R:7.2f} {B/R:7.2f}")

print()
print("Fitted a ~ how sharply the field tracks the logits; b ~ the activity level.")
print("If compiled KL << heuristic KL, the top-k threshold was leaving error on the table.")
