"""
tasb_m5_recut.py — Re-cut M5 faithfulness results by position confidence
==============================================================================
Post-hoc analysis of an M5 CSV. No model reload. Answers the question that
turns the "off tokens" from a worry into a result:

  "Where do the bridge's disagreements with vanilla actually live?"

HYPOTHESIS
----------
The bridge only diverges from vanilla at positions where the model itself
is uncertain (low prob_gap = top-1 and top-2 nearly tied). At confident
positions (high prob_gap), the bridge agrees ~100% even at high α.

If true, this reframes the result: the bridge is not "unfaithful X% of the
time" — it faithfully tracks vanilla everywhere the model has a clear
preference, and only resolves genuine near-ties differently.

WHAT IT DOES
------------
1. Buckets every (row) by vanilla prob_gap (top1_prob - top2_prob):
     CONFIDENT   gap >= 0.50  (clear top-1 preference)
     MODERATE    0.10 <= gap < 0.50
     AMBIGUOUS   gap < 0.10   (near-tie at the top)
2. For each α, reports top-1 agreement and mean KL within each bucket.
3. Cross-tabs disagreements: of all top-1 disagreements, what fraction
   fall in each gap bucket?
4. Per-prompt repetition diagnostics. PATCH 2026-05-28/30 (post-Gemini-review):
     - consecutive_repeat_rate: same-token-as-previous rate
     - 4-gram repeat rate:      short-cycle detection
     - 8-gram repeat rate:      long-cycle detection (PATCH 2026-05-30)
     - low_diversity_ratio:     1 - unique/total (the original "loop"
                                metric, now correctly named — it measures
                                vocabulary diversity, not generation looping)
   These four together distinguish true looping ("dog dog dog") from
   short-cycle looping ("up down up down") from phrase-level cycle-looping
   ("quick brown fox ... quick brown fox") from low-vocabulary text.

BACKWARD COMPAT
---------------
Reads both pre-patch CSVs (where the column was misnamed `logit_gap` but
contained probability-space values) and post-patch CSVs (correctly named
`prob_gap`).

USAGE
-----
    python tasb_m5_recut.py results/tasb_m5_faithfulness_20260528_021614.csv
==============================================================================
"""

import csv
import sys
from collections import defaultdict, Counter

# PATCH 2026-05-30 (post-Gemini-second-review, P2): UTF-8 stdout for
# Windows console compatibility — same reason as in tasb_m5_faithfulness.
try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except (AttributeError, OSError):
    pass


def _c(code, t):
    return f"\033[{code}m{t}\033[0m" if sys.stdout.isatty() else t
def green(t):  return _c("32", t)
def yellow(t): return _c("33", t)
def red(t):    return _c("31", t)
def cyan(t):   return _c("36", t)
def bold(t):   return _c("1",  t)
def gray(t):   return _c("90", t)


# Gap buckets
def gap_bucket(gap: float) -> str:
    if gap >= 0.50:
        return 'CONFIDENT'
    elif gap >= 0.10:
        return 'MODERATE'
    else:
        return 'AMBIGUOUS'

BUCKET_ORDER = ['CONFIDENT', 'MODERATE', 'AMBIGUOUS']


def load_rows(path: str) -> tuple[list[dict], bool]:
    """Load M5 CSV rows. Supports both pre-patch and post-patch column names:
      - prob_gap (post-patch) or logit_gap (pre-patch, was misnamed)
      - logit_margin, top1_prob, top2_prob, alpha0_max_abs_diff (post-patch only)

    Returns (rows, is_pre_patch). PATCH 2026-05-30 (post-Gemini-third-review,
    P2): the is_pre_patch flag lets main() show a loud banner warning that
    KL/JS/entropy values in pre-patch CSVs are legacy-clamped and must not
    be cited externally.
    """
    rows = []
    is_pre_patch = None   # decided on first row
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for r in reader:
            # Pre-patch CSV used 'logit_gap' for what was actually probability gap.
            # Post-patch uses 'prob_gap'. Detection on first row sets the flag.
            if is_pre_patch is None:
                is_pre_patch = 'prob_gap' not in r
            gap_field = 'logit_gap' if is_pre_patch else 'prob_gap'
            row = {
                'alpha':      float(r['alpha']),
                'kl_logit':   float(r['kl_logit']),
                'top1_agree': int(r['top1_agree']),
                'top5_agree': int(r['top5_agree']),
                'prob_gap':   float(r[gap_field]),
                'vanilla_entropy': float(r['vanilla_entropy']),
                'prompt_id':  r['prompt_id'],
                'domain':     r['domain'],
                'step':       int(r['step']),
                'vanilla_top1': int(r['vanilla_top1']),
                'bridge_top1':  int(r['bridge_top1']),
            }
            # Optional new columns (only present in post-patch CSVs)
            if 'logit_margin' in r:
                row['logit_margin'] = float(r['logit_margin'])
            if 'top1_prob' in r:
                row['top1_prob'] = float(r['top1_prob'])
            if 'alpha0_max_abs_diff' in r:
                row['alpha0_max_abs_diff'] = float(r['alpha0_max_abs_diff'])
            rows.append(row)
    return rows, bool(is_pre_patch)


def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def consecutive_repeat_rate(tokens: list[int]) -> float:
    """Fraction of positions where the token equals the previous token.
    A real "loop into the same word" signal. PATCH 2026-05-28."""
    if len(tokens) < 2:
        return 0.0
    return sum(1 for i in range(1, len(tokens)) if tokens[i] == tokens[i-1]) / (len(tokens) - 1)


def ngram_repeat_rate(tokens: list[int], n: int = 4) -> float:
    """Fraction of n-grams that have appeared earlier in the sequence.
    Catches the 'quick brown fox jumps over the lazy dog' style loop
    where consecutive repeats are 0 but the same n-gram cycles.
    PATCH 2026-05-28."""
    if len(tokens) < n + 1:
        return 0.0
    seen = set()
    repeats = 0
    total = 0
    for i in range(len(tokens) - n + 1):
        ng = tuple(tokens[i:i+n])
        total += 1
        if ng in seen:
            repeats += 1
        seen.add(ng)
    return repeats / total if total else 0.0


def low_diversity_ratio(tokens: list[int]) -> float:
    """1 - unique/total. Crude diversity measure.
    PATCH 2026-05-28: renamed from the misleading 'repeat_pct'.
    This is what was previously (mis)called 'looping' — it actually measures
    vocabulary diversity, not generation looping."""
    if not tokens:
        return 0.0
    return 1 - len(set(tokens)) / len(tokens)


def main():
    if len(sys.argv) < 2:
        print("usage: python tasb_m5_recut.py <m5_csv_path>")
        sys.exit(1)
    path = sys.argv[1]
    rows, is_pre_patch = load_rows(path)

    alphas = sorted(set(r['alpha'] for r in rows))
    print("═" * 86)
    print(f"  M5 RE-CUT BY POSITION CONFIDENCE")
    print(f"  Source: {path}")
    print(f"  {len(rows)} rows, α ∈ {alphas}")
    print("═" * 86)

    # PATCH 2026-05-30 (post-Gemini-third-review, P2): loud banner when
    # reading a pre-patch CSV. The 'logit_gap' column name signals that
    # this CSV was produced before the KL/JS/entropy clamp fix, which
    # means its KL/JS values are LEGACY-CLAMPED (suppressed by ~15-17x).
    # Top-1 agreement and prob_gap bucketing are still valid since they
    # don't depend on the clamp; only the KL/JS/entropy magnitudes are bad.
    if is_pre_patch:
        print()
        print(red("  " + "█" * 84))
        print(red("  ⚠  PRE-PATCH CSV DETECTED"))
        print(red("  " + "─" * 84))
        print(red("  This CSV was produced before the KL clamp fix (2026-05-28)."))
        print(red("  KL / JS / entropy values are LEGACY-CLAMPED and suppressed by ~15-17×."))
        print(red("  DO NOT CITE the KL/JS magnitudes from this re-cut externally."))
        print(red(""))
        print(red("  What IS still valid here:"))
        print(red("    - Top-1 / top-5 agreement (argmax-based, clamp-independent)"))
        print(red("    - Confidence-bucket distribution (prob_gap, clamp-independent)"))
        print(red("    - Disagreement cross-tab by bucket"))
        print(red("    - Repetition diagnostics"))
        print(red(""))
        print(red("  What is NOT valid until M5 is rerun with the patched harness:"))
        print(red("    - mean_KL columns and KL CIs (15-17× too small)"))
        print(red("    - JS divergence magnitudes (same bug)"))
        print(red("    - vanilla_entropy magnitudes (same bug)"))
        print(red("  " + "█" * 84))
        print()

    # ── Part 1: Distribution of positions across buckets ───────────────────
    # (Use α=0 rows as the canonical per-position set, since prob_gap is a
    #  vanilla property independent of α.)
    base_rows = [r for r in rows if r['alpha'] == 0.0]
    bucket_counts = Counter(gap_bucket(r['prob_gap']) for r in base_rows)
    total_positions = len(base_rows)

    print(f"\n{bold('  Position distribution (by vanilla prob_gap):')}")
    print(f"    {'bucket':<12} {'gap range':<14} {'positions':>10} {'fraction':>10}")
    print(f"    {'─'*48}")
    ranges = {'CONFIDENT': 'gap≥0.50', 'MODERATE': '0.10–0.50', 'AMBIGUOUS': 'gap<0.10'}
    for b in BUCKET_ORDER:
        n = bucket_counts.get(b, 0)
        print(f"    {b:<12} {ranges[b]:<14} {n:>10} {100*n/total_positions:>9.1f}%")

    # ── Part 2: Top-1 agreement per α, per bucket ──────────────────────────
    print(f"\n{bold('  Top-1 agreement by α and confidence bucket:')}")
    print(f"    {'α':>5}  {'CONFIDENT':>20}  {'MODERATE':>20}  {'AMBIGUOUS':>20}")
    print(f"    {'─'*72}")
    for alpha in alphas:
        arows = [r for r in rows if r['alpha'] == alpha]
        cells = []
        for b in BUCKET_ORDER:
            brows = [r for r in arows if gap_bucket(r['prob_gap']) == b]
            if brows:
                agree = mean([r['top1_agree'] for r in brows]) * 100
                cell = f"{agree:>6.1f}% (n={len(brows):>4})"
                if agree >= 99:
                    cell = green(cell)
                elif agree >= 90:
                    cell = yellow(cell)
                else:
                    cell = red(cell)
            else:
                cell = gray(f"{'—':>14}")
            cells.append(cell)
        print(f"    {alpha:>5.1f}  {cells[0]:>20}  {cells[1]:>20}  {cells[2]:>20}")

    # ── Part 3: Mean KL per α, per bucket ──────────────────────────────────
    print(f"\n{bold('  Mean KL by α and confidence bucket:')}")
    print(f"    {'α':>5}  {'CONFIDENT':>16}  {'MODERATE':>16}  {'AMBIGUOUS':>16}")
    print(f"    {'─'*60}")
    for alpha in alphas:
        arows = [r for r in rows if r['alpha'] == alpha]
        cells = []
        for b in BUCKET_ORDER:
            brows = [r for r in arows if gap_bucket(r['prob_gap']) == b]
            kl = mean([r['kl_logit'] for r in brows]) if brows else 0.0
            cells.append(f"{kl:>14.6f}")
        print(f"    {alpha:>5.1f}  {cells[0]:>16}  {cells[1]:>16}  {cells[2]:>16}")

    # ── Part 4: Where do disagreements live? ───────────────────────────────
    print(f"\n{bold('  Of all top-1 disagreements, where do they fall?')}")
    print(f"    {'α':>5}  {'total disagree':>15}  {'CONFIDENT':>12}  "
          f"{'MODERATE':>12}  {'AMBIGUOUS':>12}")
    print(f"    {'─'*70}")
    for alpha in alphas:
        if alpha == 0.0:
            continue
        arows = [r for r in rows if r['alpha'] == alpha]
        disagrees = [r for r in arows if r['top1_agree'] == 0]
        if not disagrees:
            print(f"    {alpha:>5.1f}  {0:>15}  {gray('(none)'):>12}")
            continue
        dbuckets = Counter(gap_bucket(r['prob_gap']) for r in disagrees)
        n = len(disagrees)
        parts = []
        for b in BUCKET_ORDER:
            c = dbuckets.get(b, 0)
            parts.append(f"{c:>4} ({100*c/n:>4.0f}%)")
        amb_frac = dbuckets.get('AMBIGUOUS', 0) / n * 100
        print(f"    {alpha:>5.1f}  {n:>15}  {parts[0]:>12}  "
              f"{parts[1]:>12}  {parts[2]:>12}")

    # ── Part 5: Repetition diagnostics (multi-scale, post-Gemini-review) ───
    print(f"\n{bold('  Repetition diagnostics (per-prompt, vanilla tokens):')}")
    print(f"    PATCH 2026-05-28/30: previous 'looping' label was misleading.")
    print(f"    It measured low-diversity (unique/total), not generation looping.")
    print(f"    Multi-scale detection (consec / 4-gram / 8-gram / diversity):")
    print(f"      consec%   : prev-token-equals-current rate (true 'dog dog dog')")
    print(f"      4gram%    : fraction of 4-grams that appeared earlier")
    print(f"                  (catches short cycles)")
    print(f"      8gram%    : fraction of 8-grams that appeared earlier")
    print(f"                  (catches longer phrase-level cycles)")
    print(f"      lo_div%   : 1 - unique/total (vocabulary diversity)")
    print()
    print(f"    {'prompt':<8} {'domain':<12} {'consec%':>8} {'4gram%':>8} "
          f"{'8gram%':>8} {'lo_div%':>8} {'mean_gap':>10}  flag")
    print(f"    {'─'*82}")
    # group base rows by prompt
    by_prompt = defaultdict(list)
    for r in base_rows:
        by_prompt[r['prompt_id']].append(r)

    looped = set()    # filled below, used by Part 6
    for pid in sorted(by_prompt.keys()):
        prows = sorted(by_prompt[pid], key=lambda r: r['step'])
        van_tokens = [r['vanilla_top1'] for r in prows]
        consec = consecutive_repeat_rate(van_tokens) * 100
        gram4  = ngram_repeat_rate(van_tokens, n=4) * 100
        gram8  = ngram_repeat_rate(van_tokens, n=8) * 100
        lodiv  = low_diversity_ratio(van_tokens) * 100
        mean_g = mean([r['prob_gap'] for r in prows])
        domain = prows[0]['domain']
        # Loop detection: any of consecutive, 4-gram, or 8-gram repeat at
        # high rate. 8-gram catches longer phrase-level cycles (e.g. the
        # 11-token "quick brown fox jumps over the lazy dog. The" cycle in
        # HC3) that might give lower 4-gram rates due to the cycle being
        # longer than 4 tokens.
        cycle_looped = gram4 >= 60.0 or gram8 >= 50.0
        consec_looped = consec >= 40.0
        if cycle_looped or consec_looped:
            looped.add(pid)
            flag = red('CYCLE-LOOPED' if cycle_looped else 'CONSEC-LOOPED')
        elif gram4 >= 30.0 or gram8 >= 25.0 or consec >= 20.0:
            flag = yellow('partial cycling')
        else:
            flag = green('varied')
        print(f"    {pid:<8} {domain:<12} {consec:>7.1f}% {gram4:>7.1f}% "
              f"{gram8:>7.1f}% {lodiv:>7.1f}% {mean_g:>10.3f}  {flag}")

    # ── Part 6: Honest aggregate — varied prompts only ─────────────────────
    print(f"\n{bold('  Honest aggregate (excluding heavily-looped prompts):')}")
    # `looped` was populated in Part 5 above, using multi-scale n-gram
    # and consecutive repeat thresholds (NOT the misleading low_div from before).
    print(f"    Excluded (looped): {sorted(looped) if looped else 'none'}")
    print(f"    (Loop detection: 4-gram ≥60% or 8-gram ≥50% or consec ≥40%)")
    print()
    print(f"    {'α':>5}  {'top1% (all)':>14}  {'top1% (varied)':>16}  "
          f"{'mean_KL (varied)':>18}")
    print(f"    {'─'*60}")
    for alpha in alphas:
        arows = [r for r in rows if r['alpha'] == alpha]
        varied = [r for r in arows if r['prompt_id'] not in looped]
        all_agree = mean([r['top1_agree'] for r in arows]) * 100
        var_agree = mean([r['top1_agree'] for r in varied]) * 100 if varied else 0
        var_kl    = mean([r['kl_logit'] for r in varied]) if varied else 0
        col = green if var_agree >= 95 else yellow if var_agree >= 85 else red
        print(f"    {alpha:>5.1f}  {all_agree:>13.1f}%  {col(f'{var_agree:>14.1f}%')}  "
              f"{var_kl:>18.6f}")

    # ── Summary verdict ────────────────────────────────────────────────────
    print(f"\n{'═'*86}")
    print(bold("  VERDICT"))
    print(f"{'═'*86}")

    # Compute the key claim: fraction of disagreements in AMBIGUOUS at α=0.3
    a03 = [r for r in rows if r['alpha'] == 0.3]
    a03_dis = [r for r in a03 if r['top1_agree'] == 0]
    if a03_dis:
        amb = sum(1 for r in a03_dis if gap_bucket(r['prob_gap']) == 'AMBIGUOUS')
        mod = sum(1 for r in a03_dis if gap_bucket(r['prob_gap']) == 'MODERATE')
        conf = sum(1 for r in a03_dis if gap_bucket(r['prob_gap']) == 'CONFIDENT')
        print(f"  At α=0.3, {len(a03_dis)} disagreements total:")
        print(f"    {amb} ambiguous ({100*amb/len(a03_dis):.0f}%), "
              f"{mod} moderate ({100*mod/len(a03_dis):.0f}%), "
              f"{conf} confident ({100*conf/len(a03_dis):.0f}%)")
        if conf == 0:
            print(green(f"  ✓ ZERO disagreements at confident positions (gap≥0.50)."))
            print(green(f"    The bridge never flips a token the model was sure about."))
        else:
            print(yellow(f"  ~ {conf} disagreement(s) at confident positions — inspect these."))
    else:
        print(green(f"  At α=0.3: ZERO disagreements anywhere."))

    # Confident-bucket agreement across all α
    print()
    all_conf_agree = []
    for alpha in alphas:
        if alpha == 0.0:
            continue
        arows = [r for r in rows if r['alpha'] == alpha
                 and gap_bucket(r['prob_gap']) == 'CONFIDENT']
        if arows:
            all_conf_agree.append(mean([r['top1_agree'] for r in arows]) * 100)
    if all_conf_agree:
        worst = min(all_conf_agree)
        print(f"  Worst-case CONFIDENT-bucket agreement across all α (incl α=1.0): "
              f"{worst:.1f}%")
        if worst >= 99:
            print(green(f"  ✓ Even at α=1.0, the bridge preserves confident predictions."))
    print(f"{'═'*86}\n")


if __name__ == '__main__':
    main()
