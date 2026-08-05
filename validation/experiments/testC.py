"""
TEST C — is there a BIPARTITE (parallel) encoding of attention?

REASONING BEING TESTED. The obstruction in Test A is specific to sampling a
CATEGORICAL over positions: mutual exclusion = complete graph = S sequential
blocks. Extropic's own DTM is "explicitly bipartite (two-colourable), enabling
parallel sampling of each colour block" -- so the hardware wants bipartite.

Key structural fact: in a bipartite RBM, p(h_j | v) factorises -- every hidden
unit is CONDITIONALLY INDEPENDENT given v. One parallel sweep, no mutual
exclusion, no complete graph. But independent Bernoullis are NOT a categorical.

So: can attention be obtained WITHOUT ever forming a categorical?
  independent gate:  h_j ~ Bernoulli(sigmoid(l_j - tau))     [parallel, 2 blocks]
  output:            o = sum_j h_j v_j / sum_j h_j           [normalise AFTER]
vs true attention:   o* = sum_j softmax(l)_j v_j

MEASURED: cosine similarity and relative L2 of o vs o*, across sequence lengths
and gate thresholds. Also the exact-softmax-sampling baseline at matched sample
count, so the comparison is like-for-like.
"""
import numpy as np

SEED = 0
rng = np.random.default_rng(SEED)

def softmax(x):
    e = np.exp(x - x.max()); return e / e.sum()

def trial(S, d, K, tau_mode, rep):
    r = np.random.default_rng(SEED + rep)
    q = r.normal(0, 1, d)
    Kk = r.normal(0, 1, (S, d))
    V = r.normal(0, 1, (S, d))
    logits = Kk @ q / np.sqrt(d)
    p = softmax(logits)
    o_true = p @ V

    # --- BIPARTITE GATE: independent Bernoulli per key, fully parallel ---
    if tau_mode == 'median':
        tau = np.median(logits)
    elif tau_mode == 'topk':                       # threshold at the k-th largest
        k = max(1, int(0.1 * S)); tau = np.sort(logits)[-k]
    else:
        tau = float(tau_mode)
    gate_p = 1.0 / (1.0 + np.exp(-(logits - tau)))
    acc = np.zeros(d); n_used = 0
    for _ in range(K):
        h = r.random(S) < gate_p
        if h.sum() == 0:
            continue
        acc += V[h].mean(axis=0); n_used += 1
    o_gate = acc / max(n_used, 1)

    # --- BASELINE: exact categorical sampling from softmax, matched K ---
    idx = r.choice(S, size=K, p=p)
    o_cat = V[idx].mean(axis=0)

    def cos(a, b):
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        return float(a @ b / (na * nb)) if na > 1e-12 and nb > 1e-12 else 0.0
    def rel(a, b):
        return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-12))
    return cos(o_gate, o_true), rel(o_gate, o_true), cos(o_cat, o_true), rel(o_cat, o_true)

print("=" * 78)
print("TEST C — bipartite independent-gate vs true attention output")
print("=" * 78)
print(f"{'S':>5} {'K':>5} {'gate':>8} {'cos(gate)':>10} {'relL2':>8} | {'cos(cat)':>9} {'relL2':>8}")
for S in [64, 256, 512]:
    for tau_mode in ['median', 'topk']:
        for K in [16, 128]:
            res = np.array([trial(S, 64, K, tau_mode, r) for r in range(8)])
            cg, rg, cc, rc = res.mean(axis=0)
            print(f"{S:5d} {K:5d} {tau_mode:>8} {cg:10.4f} {rg:8.4f} | {cc:9.4f} {rc:8.4f}")
print()
print("cos(cat) is exact-softmax sampling at the SAME sample budget -- the fair bar.")
print("If cos(gate) >= cos(cat), the parallel bipartite gate loses nothing.")
