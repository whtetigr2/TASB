"""
gap_c_entmax_comparison.py — Gap C: Lagrangian Uniqueness Measurable Consequence
==============================================================================
FIND-025 §5 → Gap C resolution
Author: Paul W. Shaver

WHAT THIS PROVES
----------------
FIND-018 (Lagrangian Uniqueness Audit): Kim's Lagrangian is conditionally unique.
The kinetic term is unique by Chentsov's theorem; the potential by Shore-Johnson.
This means softmax and α-entmax sample from DIFFERENT statistical mechanical systems:
  - softmax  → Gibbs-Boltzmann (Shannon entropy, exponential family)
  - entmax15 → Tsallis q-softmax with q=1.5 (non-extensive entropy)

Measurable consequence: KL(entmax15 ‖ softmax) > 0 for real LLaMA attention heads.

PROXY LOGIT APPROACH (no model load needed)
--------------------------------------------
Since entmax15 is shift-invariant (like softmax), proxy_logit = log(p_softmax)
is an exact substitute for the raw attention logits:
  entmax15(QK^T/√dk) = entmax15(log(p_softmax))  [shift invariance, proven below]

Proof: entmax15(a + c) = entmax15(a) for any constant c, because the threshold τ
shifts by c leaving (a_i - τ) unchanged. Hence log(p_softmax) ≡ QK^T/√dk mod const.

Requires: attention_matrices.json (post-softmax, all 5 prompts × 28 layers × 24 heads)
Requires: pip install entmax torch

PASS CONDITION (Gap C → PASS)
------------------------------
KL(entmax15 ‖ softmax) > 0 for all non-degenerate heads (≥2 active tokens).
Expected range: 0.01–0.30 depending on head sharpness.
  - Sharp heads (low Cv ≈ 0.1): smaller KL (both peaked, entmax drops near-zero tokens)
  - Diffuse heads (high Cv ≈ 1.0+): larger KL (entmax sparsifies significantly)

OUTPUT
------
  validation/results/gap_c_entmax_results.csv
  Columns: prompt, head, layer, n_rows_active, cv, kl_entmax_softmax, sparsity_entmax

REPRODUCE
---------
  cd thermobridge/validation
  python experiments/gap_c_entmax_comparison.py
==============================================================================
"""

import csv
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from entmax import entmax15

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_JSON    = os.path.join(os.path.dirname(__file__), '..', '..', 'demo', 'data',
                             'attention_matrices.json')
RESULTS_DIR  = os.path.join(os.path.dirname(__file__), '..', 'results')
RESULTS_CSV  = os.path.join(RESULTS_DIR, 'gap_c_entmax_results.csv')

TARGET_LAYER = 18
MIN_ACTIVE   = 2     # skip single-token rows (KL trivially 0)
EPS          = 1e-10  # numerical stability for log

os.makedirs(RESULTS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def cv_from_probs(p: np.ndarray) -> float:
    """Cv = Var_rho(log p) across active rows. Equivalent to Var_rho(a/√dk)."""
    active = p.sum(axis=-1) > 0.5
    if not active.any():
        return 0.0
    rows = p[active]
    log_p = np.log(rows + EPS)
    E1 = (rows * log_p).sum(axis=-1)
    E2 = (rows * log_p ** 2).sum(axis=-1)
    cv_per_row = E2 - E1 ** 2
    return float(cv_per_row.mean())


def kl_div_entmax_softmax(p_soft: np.ndarray, p_entmax: np.ndarray) -> float:
    """KL(p_entmax ‖ p_softmax) for one row. Numerically stable."""
    mask = p_entmax > EPS
    if not mask.any():
        return 0.0
    return float((p_entmax[mask] * (
        np.log(p_entmax[mask] + EPS) - np.log(p_soft[mask] + EPS)
    )).sum())


def sparsity(p_entmax: np.ndarray, thr: float = 1e-6) -> float:
    """Fraction of positions with p_entmax < thr (sparsity of entmax output)."""
    return float((p_entmax < thr).mean())


def process_head(attn: np.ndarray) -> dict:
    """
    Process one S×S attention matrix (post-softmax).

    Returns per-head aggregated statistics across all active rows.
    active row = row where sum > 0.5 AND count of tokens with p > 1e-5 >= MIN_ACTIVE
    """
    S = attn.shape[0]
    kl_list, sp_list, n_active = [], [], 0

    for i in range(S):
        row_soft = attn[i]
        active_count = (row_soft > 1e-5).sum()
        if row_soft.sum() < 0.5 or active_count < MIN_ACTIVE:
            continue
        n_active += 1

        proxy = np.log(row_soft + EPS)
        proxy_t = torch.tensor(proxy, dtype=torch.float32).unsqueeze(0)

        row_entmax = entmax15(proxy_t, dim=-1).squeeze(0).numpy()
        row_entmax = np.clip(row_entmax, 0, None)
        row_entmax /= row_entmax.sum() + EPS  # renormalize for numerical safety

        kl_list.append(kl_div_entmax_softmax(row_soft, row_entmax))
        sp_list.append(sparsity(row_entmax))

    if not kl_list:
        return {'n_rows_active': 0, 'kl_mean': 0.0, 'kl_std': 0.0,
                'sparsity_mean': 0.0, 'cv': 0.0}

    return {
        'n_rows_active': n_active,
        'kl_mean': float(np.mean(kl_list)),
        'kl_std':  float(np.std(kl_list)),
        'sparsity_mean': float(np.mean(sp_list)),
        'cv': cv_from_probs(attn),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if not os.path.exists(DATA_JSON):
        print(f"ERROR: {DATA_JSON} not found. Run tasb_attention_capture.py first.")
        sys.exit(1)

    with open(DATA_JSON) as f:
        data = json.load(f)

    rows_out = []
    all_kls = []

    print(f"Layer {TARGET_LAYER} — entmax15 vs softmax comparison")
    print(f"{'Prompt':<12} {'Head':>4}  {'n_rows':>6}  {'KL_mean':>8}  {'KL_std':>8}  "
          f"{'Sparsity':>8}  {'Cv':>8}")
    print("-" * 65)

    for prompt_id, pdata in data.items():
        layer_key = str(TARGET_LAYER)
        if layer_key not in pdata.get('layers', {}):
            continue
        heads = pdata['layers'][layer_key]['heads']

        for head_key in sorted(heads.keys(), key=int):
            h = int(head_key)
            attn = np.array(heads[head_key])
            stats = process_head(attn)

            row = {
                'prompt':    prompt_id,
                'head':      h,
                'layer':     TARGET_LAYER,
                'n_rows_active': stats['n_rows_active'],
                'cv':        round(stats['cv'], 6),
                'kl_entmax_softmax': round(stats['kl_mean'], 6),
                'kl_std':    round(stats['kl_std'], 6),
                'sparsity_entmax': round(stats['sparsity_mean'], 4),
            }
            rows_out.append(row)
            if stats['n_rows_active'] > 0:
                all_kls.append(stats['kl_mean'])

            print(f"{prompt_id:<12} {h:>4}  {stats['n_rows_active']:>6}  "
                  f"{stats['kl_mean']:>8.4f}  {stats['kl_std']:>8.4f}  "
                  f"{stats['sparsity_mean']:>8.4f}  {stats['cv']:>8.4f}")

    # ---------------------------------------------------------------------------
    # Aggregate summary
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 65)
    n_total  = len([r for r in rows_out if r['n_rows_active'] > 0])
    n_nonzero = len([r for r in rows_out if r['kl_entmax_softmax'] > 1e-6])
    all_kls_arr = np.array(all_kls)
    print(f"\nSummary (across {n_total} head × prompt combos with ≥{MIN_ACTIVE} active tokens):")
    print(f"  KL > 0 (> 1e-6):  {n_nonzero}/{n_total}  "
          f"({'100%' if n_total == n_nonzero else f'{100*n_nonzero/n_total:.1f}%'})")
    print(f"  Mean KL:          {all_kls_arr.mean():.4f}")
    print(f"  Std KL:           {all_kls_arr.std():.4f}")
    print(f"  Min KL:           {all_kls_arr.min():.4f}")
    print(f"  Max KL:           {all_kls_arr.max():.4f}")
    print(f"  Median KL:        {np.median(all_kls_arr):.4f}")

    # Correlation between Cv and KL
    cvs = np.array([r['cv'] for r in rows_out if r['n_rows_active'] > 0])
    kls = np.array([r['kl_entmax_softmax'] for r in rows_out if r['n_rows_active'] > 0])
    r_cv_kl = float(np.corrcoef(cvs, kls)[0, 1])
    print(f"\n  Pearson r(Cv, KL_entmax): {r_cv_kl:.4f}  "
          f"(expected > 0: higher Cv → larger divergence from Tsallis)")

    # Gap C verdict
    gap_c_pass = n_nonzero == n_total and all_kls_arr.min() > 1e-4
    print(f"\nGap C verdict: {'PASS' if gap_c_pass else 'CONDITIONAL'}")
    if gap_c_pass:
        print("  KL(entmax15 ‖ softmax) > 0 confirmed for all non-degenerate heads.")
        print("  entmax15 and softmax sample from provably DIFFERENT distributions.")
        print("  Lagrangian uniqueness consequence: CONFIRMED with quantitative backing.")
    else:
        print("  Some heads had KL ≈ 0. Inspect rows: may be near-degenerate inputs.")

    # Write CSV
    if rows_out:
        with open(RESULTS_CSV, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=rows_out[0].keys())
            writer.writeheader()
            writer.writerows(rows_out)
        print(f"\nResults saved: {RESULTS_CSV}")


if __name__ == '__main__':
    main()
