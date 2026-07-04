"""
gap_b_jarzynski_test.py  --  EXP-001 Gap B Jarzynski Work Variance Test

Claim under test (FIND-025 Gap B):
  ΔCv = Cv^NTK - Cv^orig correlates with W_diss = KL(p_orig ‖ p_NTK) across
  heads. TASB's Cv observable measures Jarzynski dissipated work under the
  NTK-aware RoPE Hamiltonian deformation.

Input:
  layer18_qk_pre_rope.npz  — pre-RoPE Q/K tensors captured at LLaMA 3.2-3B
  layer 18 via capture_qk_layer18.py. Keys: q_pre_<key>, k_pre_<key>
  for key in [factual, code, reasoning, creative, mathematical].
  q_pre shape: (n_q_heads=24, S, dk=128)
  k_pre shape: (n_kv_heads=8, S, dk=128)

Output:
  validation/results/gap_b_jarzynski_results.csv
  Columns: head, cv_orig, cv_ntk, delta_cv, w_kl

Model specs (LLaMA 3.2-3B):
  dk=128, n_q_heads=24, n_kv_heads=8, GQA_ratio=3
  base_orig=10000, scale_factor=4
  base_NTK = 10000 * 4^(128/126) ~ 40988
"""

import os
import sys
import numpy as np
from scipy.stats import pearsonr

# ── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, '..', 'results')
NPZ_PATH    = os.path.join(
    os.environ.get('USERPROFILE', os.path.expanduser('~')),
    r'AppData\Local\Temp\claude\C--Users-whtet-Documents-SCIN-ecosystem-SCIN-Ecosystem-Businesses-ThermoSoft-Discovery'
    r'\c235181c-bd9b-45da-b42f-43054d4d8e53\scratchpad\layer18_qk_pre_rope.npz'
)
OUT_CSV     = os.path.join(RESULTS_DIR, 'gap_b_jarzynski_results.csv')

# ── Model parameters ─────────────────────────────────────────────────────────
DK          = 128
DK_HALF     = DK // 2
N_Q_HEADS   = 24
N_KV_HEADS  = 8
GQA_RATIO   = N_Q_HEADS // N_KV_HEADS      # = 3
BASE_ORIG   = 10_000.0
SCALE_S     = 4.0
BASE_NTK    = BASE_ORIG * (SCALE_S ** (DK / (DK - 2)))   # ~ 40988
EPS         = 1e-10

PROMPT_KEYS = ['factual', 'code', 'reasoning', 'creative', 'mathematical']


# ── RoPE helpers ─────────────────────────────────────────────────────────────
def _rope_freqs(base: float) -> np.ndarray:
    """Return per-dimension frequencies for a given RoPE base.  Shape: (dk//2,)"""
    i = np.arange(0, DK, 2, dtype=np.float64)       # [0, 2, 4, ..., 126]
    return 1.0 / (base ** (i / DK))                  # θ_j = 1/base^(2j/dk)


def _build_cos_sin(S: int, freqs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Precompute RoPE cos/sin tables for S positions.
    Returns cos, sin each of shape (S, dk) matching HF LLaMA's emb=cat([freqs,freqs])."""
    m = np.arange(S, dtype=np.float64)               # (S,)
    angles = np.outer(m, freqs)                       # (S, dk//2)
    angles_doubled = np.concatenate([angles, angles], axis=-1)  # (S, dk)
    return np.cos(angles_doubled), np.sin(angles_doubled)


def _rotate_half(x: np.ndarray) -> np.ndarray:
    """rotate_half: (S, dk) -> (S, dk). Matches HF: cat([-x2, x1])."""
    x1 = x[..., :DK_HALF]
    x2 = x[..., DK_HALF:]
    return np.concatenate([-x2, x1], axis=-1)


def apply_rope(x: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    """Apply RoPE to x of shape (S, dk) using cos/sin of shape (S, dk)."""
    return x * cos + _rotate_half(x) * sin


# ── Attention helpers ─────────────────────────────────────────────────────────
def attn_probs(q: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Causal attention probabilities.
    q, k: (S, dk).  Returns probs: (S, S) with lower-triangular mass."""
    scores = (q @ k.T) / np.sqrt(DK)                 # (S, S)
    # causal mask: positions j > i get -inf
    causal = np.triu(np.full_like(scores, -np.inf), k=1)
    scores = scores + causal
    scores -= scores.max(axis=-1, keepdims=True)      # numerical stability
    exp_s = np.exp(scores)
    # zero out -inf positions to avoid nan in sum
    exp_s = np.where(np.isfinite(scores), exp_s, 0.0)
    return exp_s / (exp_s.sum(axis=-1, keepdims=True) + EPS)


def cv_from_probs(p: np.ndarray) -> float:
    """Cv = mean over rows of Var_ρ(H), H = -log(p).
    Skip first row (only one key) to avoid degenerate variance."""
    S = p.shape[0]
    if S < 2:
        return 0.0
    cv_rows = []
    for i in range(1, S):                             # skip row 0 (trivially p=[1])
        pi = p[i, :i+1]                               # i+1 active positions
        if pi.sum() < 0.5:
            continue
        log_pi = np.log(pi + EPS)
        e1 = float(np.dot(pi, log_pi))
        e2 = float(np.dot(pi, log_pi ** 2))
        cv_rows.append(e2 - e1 ** 2)
    return float(np.mean(cv_rows)) if cv_rows else 0.0


def kl_div(p: np.ndarray, q: np.ndarray) -> float:
    """KL(p ‖ q) = mean over rows of Σ_j p[i,j] log(p[i,j]/q[i,j]).
    Skip first row for same reason as cv_from_probs."""
    S = p.shape[0]
    if S < 2:
        return 0.0
    kl_rows = []
    for i in range(1, S):
        pi = p[i, :i+1]
        qi = q[i, :i+1]
        if pi.sum() < 0.5:
            continue
        ratio = np.log((pi + EPS) / (qi + EPS))
        kl_rows.append(float(np.dot(pi, ratio)))
    return float(np.mean(kl_rows)) if kl_rows else 0.0


# ── Per-head NTK simulation ──────────────────────────────────────────────────
def process_head(
    h: int,
    q_pre_list: list[np.ndarray],   # [5 prompts] each (S, dk)
    k_pre_list: list[np.ndarray],   # [5 prompts] each (S, dk)
) -> tuple[float, float, float, float]:
    """Return (cv_orig, cv_ntk, delta_cv, w_kl) averaged over 5 prompts."""
    freqs_orig = _rope_freqs(BASE_ORIG)
    freqs_ntk  = _rope_freqs(BASE_NTK)

    cv_orig_all, cv_ntk_all, kl_all = [], [], []

    for q_pre, k_pre in zip(q_pre_list, k_pre_list):
        S = q_pre.shape[0]
        cos_orig, sin_orig = _build_cos_sin(S, freqs_orig)
        cos_ntk,  sin_ntk  = _build_cos_sin(S, freqs_ntk)

        q_orig = apply_rope(q_pre, cos_orig, sin_orig)
        k_orig = apply_rope(k_pre, cos_orig, sin_orig)
        q_ntk  = apply_rope(q_pre, cos_ntk,  sin_ntk)
        k_ntk  = apply_rope(k_pre, cos_ntk,  sin_ntk)

        p_orig = attn_probs(q_orig, k_orig)
        p_ntk  = attn_probs(q_ntk,  k_ntk)

        cv_orig_all.append(cv_from_probs(p_orig))
        cv_ntk_all.append(cv_from_probs(p_ntk))
        kl_all.append(kl_div(p_orig, p_ntk))

    cv_orig  = float(np.mean(cv_orig_all))
    cv_ntk   = float(np.mean(cv_ntk_all))
    delta_cv = cv_ntk - cv_orig
    w_kl     = float(np.mean(kl_all))
    return cv_orig, cv_ntk, delta_cv, w_kl


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    print(f"[load] Reading {NPZ_PATH}", flush=True)
    data = np.load(NPZ_PATH)

    # Unpack: q_pre_<key>: (24, S, 128),  k_pre_<key>: (8, S, 128)
    # Build per-head lists: head h uses kv_head = h // GQA_RATIO
    q_per_head = [[] for _ in range(N_Q_HEADS)]   # [head][prompt] → (S, 128)
    k_per_head = [[] for _ in range(N_Q_HEADS)]

    for key in PROMPT_KEYS:
        q_all = data[f'q_pre_{key}']   # (24, S, 128)
        k_all = data[f'k_pre_{key}']   # (8, S, 128)
        for h in range(N_Q_HEADS):
            kv_h = h // GQA_RATIO
            q_per_head[h].append(q_all[h].astype(np.float64))    # (S, 128)
            k_per_head[h].append(k_all[kv_h].astype(np.float64)) # (S, 128)

    print(f"[run]  NTK simulation: base_NTK = {BASE_NTK:.1f} (s={SCALE_S:.0f}, dk={DK})")
    print(f"       Processing {N_Q_HEADS} Q-heads × {len(PROMPT_KEYS)} prompts...\n")

    rows = []
    for h in range(N_Q_HEADS):
        cv_o, cv_n, dcv, wkl = process_head(h, q_per_head[h], k_per_head[h])
        rows.append((h, cv_o, cv_n, dcv, wkl))
        print(f"  head {h:2d}: cv_orig={cv_o:.4f}  cv_ntk={cv_n:.4f}  "
              f"dCv={dcv:+.4f}  W_KL={wkl:.4f}")

    # Pearson r(ΔCv, W_KL)
    delta_cvs = np.array([r[3] for r in rows])
    w_kls     = np.array([r[4] for r in rows])
    r_val, p_val = pearsonr(delta_cvs, w_kls)

    print(f"\n{'-'*60}")
    print(f"  Pearson r(dCv, W_KL) = {r_val:.4f}   p = {p_val:.3e}   n = {N_Q_HEADS}")
    print(f"  mean dCv  = {delta_cvs.mean():.4f}   std = {delta_cvs.std():.4f}")
    print(f"  mean W_KL = {w_kls.mean():.4f}   std = {w_kls.std():.4f}")
    print(f"{'-'*60}")

    verdict = "PASS" if (r_val > 0.5 and p_val < 0.05) else \
              "CONDITIONAL" if (r_val > 0.0 and p_val < 0.10) else \
              "CONDITIONAL-WEAK" if r_val > 0.0 else "FAIL"
    print(f"\n  Gap B verdict: {verdict}")
    print(f"  (PASS: r>0.5, p<0.05 | CONDITIONAL: r>0, p<0.10 | FAIL: r<=0)\n")

    # Save CSV
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(OUT_CSV, 'w') as f:
        f.write('head,cv_orig,cv_ntk,delta_cv,w_kl\n')
        for h, cv_o, cv_n, dcv, wkl in rows:
            f.write(f'{h},{cv_o:.6f},{cv_n:.6f},{dcv:.6f},{wkl:.6f}\n')
    print(f"  Results saved -> {OUT_CSV}")


if __name__ == '__main__':
    main()
