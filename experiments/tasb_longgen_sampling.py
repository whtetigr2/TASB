"""
tasb_longgen_sampling.py — Long free generation with sampling (not greedy)
==============================================================================
Patent: USPTO Provisional 64/019,999 (March 28, 2026)
Author: Paul W. Shaver

Same test matrix as tasb_longgen.py but uses temperature + top-p sampling
instead of greedy argmax. Answers:

  1. Does the loop disappear under sampling?
  2. Does the bridge produce meaningfully different output from vanilla
     under sampling, or does top-p noise swamp the bridge signal?
  3. Does alpha=1.0 stay coherent over 400 tokens under sampling?
  4. Is there a visible quality difference between alpha=0.3 and alpha=1.0
     in sampled output? (demo slider impact question)
  5. Does sampling + bridge preserve the faithfulness properties we measured
     under greedy (KL, flip rates)?

SAMPLING CONFIG
---------------
temperature:        0.8   (standard for creative/technical generation)
top_p:              0.9   (nucleus sampling — cuts tail of distribution)
repetition_penalty: 1.1   (mild penalty on recently-seen tokens)
seed:               42    (reproducible via torch.manual_seed per run)

These are the same defaults used by most production LLM serving systems.
The repetition_penalty is applied BEFORE sampling, not after — it modifies
the logit distribution that both vanilla and bridge see equally.

TWO GENERATION MODES TESTED
----------------------------
VANILLA-SAMPLE:  standard top-p sampling, no bridge
BRIDGE-SAMPLE:   capture-once at prompt, inject at step 0,
                 then top-p sampling for remaining tokens

The bridge only fires once (capture-once mode). After the first token,
subsequent tokens are sampled from the vanilla model with top-p. This is
production demo mode — realistic, fast, honest about what capture-once means.

TEST MATRIX
-----------
Alphas:   [0.0, 0.3, 0.7, 1.0]
Configs:  [S1 (1L), S3 (5L)]
Prompts:  3 (science, creative, technical) — same as longgen.py
Tokens:   400
Seeds:    [42, 123, 777] per run (3 seeds to check variance under sampling)

3 seeds × 4 alpha × 2 configs × 3 prompts = 72 runs
But alpha=0.0 is vanilla (config-independent, 3 seeds only = 9 runs)
And alpha>0.0: 3 × 3 alpha × 2 configs × 3 prompts = 54 runs
Total: 63 generation runs. ~25 min estimated.

COHERENCE MARKERS (same as longgen.py)
---------------------------------------
LOOP, TOPIC_DRIFT, TRUNCATED, CLEAN

Additional marker:
DIVERSE: outputs across 3 seeds show meaningfully different text
         (Jaccard similarity of 8-grams < 0.3 across seed pair)

OUTPUT
------
  results/tasb_longgen_sampling_<timestamp>.csv
  results/tasb_longgen_sampling_text_<timestamp>.txt
  results/tasb_longgen_sampling_summary_<timestamp>.txt
==============================================================================
"""

import csv
import gc
import os
import re
import sys
import time
from collections import Counter
from datetime import datetime

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tasb_pipeline_v2 import bridge_forward

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_ID   = "meta-llama/Llama-3.2-3B"
K          = 10
GEN_TOKENS = 400

# Sampling parameters
TEMPERATURE        = 0.8
TOP_P              = 0.9
REPETITION_PENALTY = 1.1

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

ALPHAS   = [0.0, 0.3, 0.7, 1.0]
SEEDS    = [42, 123, 777]

LAYER_CONFIGS = [
    ("S1", [18],                   "1L baseline"),
    ("S3", [15, 18, 21, 24, 27],   "5L spread"),
]

PROMPTS = [
    ("LG_SC", "SCIENCE",
     "The discovery of thermodynamic computing represents a fundamental "
     "shift in how we think about the relationship between energy and "
     "information. Unlike conventional silicon processors that fight "
     "against thermal noise,"),

    ("LG_CR", "CREATIVE",
     "The old cartographer had spent forty years mapping coastlines that "
     "no longer existed. Sea levels had risen, islands had disappeared, "
     "and the maps he drew were already obsolete by the time the ink dried. "
     "But still he drew, because"),

    ("LG_TC", "TECHNICAL",
     "The core challenge in deploying large language models on stochastic "
     "hardware substrates is that existing models were trained assuming "
     "deterministic attention computation. The softmax operation produces "
     "a probability distribution, and conventionally"),
]

PROMPT_KEYWORDS = {
    "LG_SC": ["thermodynamic", "energy", "computing", "silicon", "thermal",
               "noise", "processor", "information"],
    "LG_CR": ["cartographer", "maps", "coastlines", "island", "sea",
               "drew", "obsolete", "ink"],
    "LG_TC": ["language", "model", "stochastic", "hardware", "attention",
               "softmax", "deterministic", "probability"],
}


# ---------------------------------------------------------------------------
# Sampling helpers
# ---------------------------------------------------------------------------

def apply_repetition_penalty(logits, generated_ids, penalty=1.1):
    """Penalize tokens that appear in the generated context."""
    if penalty == 1.0 or not generated_ids:
        return logits
    for token_id in set(generated_ids[-64:]):  # last 64 tokens
        if logits[token_id] > 0:
            logits[token_id] /= penalty
        else:
            logits[token_id] *= penalty
    return logits


def top_p_sample(logits, temperature=0.8, top_p=0.9):
    """Temperature scaling + nucleus sampling."""
    logits = logits / temperature
    probs  = F.softmax(logits, dim=-1)

    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative = torch.cumsum(sorted_probs, dim=-1)

    # Remove tokens with cumulative prob above top_p
    # (shift right by 1 to include the token that crosses the threshold)
    sorted_mask = cumulative - sorted_probs > top_p
    sorted_probs[sorted_mask] = 0.0
    sorted_probs /= sorted_probs.sum()

    sampled_idx = torch.multinomial(sorted_probs, 1)
    next_token  = sorted_indices[sampled_idx].item()
    return next_token


# ---------------------------------------------------------------------------
# Coherence detection
# ---------------------------------------------------------------------------

def detect_loop(text, min_phrase_len=8):
    words = text.lower().split()
    if len(words) < min_phrase_len * 2:
        return False
    for i in range(len(words) - min_phrase_len):
        phrase = " ".join(words[i:i + min_phrase_len])
        rest   = " ".join(words[i + min_phrase_len:])
        if phrase in rest:
            return True
    return False


def detect_topic_drift(text, keywords):
    words    = text.lower().split()
    last_100 = " ".join(words[-100:]) if len(words) > 100 else text.lower()
    return not any(kw in last_100 for kw in keywords)


def ngram_jaccard(text_a, text_b, n=8):
    """8-gram Jaccard similarity between two texts."""
    def ngrams(text, n):
        words = text.lower().split()
        return Counter(tuple(words[i:i+n]) for i in range(len(words)-n+1))
    a, b   = ngrams(text_a, n), ngrams(text_b, n)
    keys   = set(a.keys()) | set(b.keys())
    if not keys:
        return 0.0
    inter  = sum(min(a[k], b[k]) for k in keys)
    union  = sum(max(a[k], b[k]) for k in keys)
    return inter / union if union > 0 else 0.0


def classify(text, keywords, n_generated):
    markers = []
    if n_generated < GEN_TOKENS - 5:
        markers.append("TRUNCATED")
    if detect_loop(text):
        markers.append("LOOP")
    if detect_topic_drift(text, keywords):
        markers.append("TOPIC_DRIFT")
    return "+".join(markers) if markers else "CLEAN"


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate_sampled(model, tok, prompt_ids, layers, alpha,
                     n_tokens, seed, run_label):
    """
    Capture-once bridge + top-p sampling generation.

    Step 0: if alpha > 0, run bridge_forward to get first token logits.
            if alpha == 0, run vanilla forward.
    Step 1+: vanilla forward + top-p sampling with repetition penalty.

    The bridge fires once at the prompt context. All subsequent tokens
    are sampled from the vanilla model with top-p. This is honest about
    what capture-once means and is the production demo mode.
    """
    device = prompt_ids.device
    tokens = [t.item() for t in prompt_ids[0]]
    t0     = time.time()
    milestones = {100, 200, 300, 400}
    reported   = set()

    # Set seed for reproducible sampling
    torch.manual_seed(seed)

    # Step 0: first token via bridge (or vanilla at alpha=0)
    if alpha == 0.0:
        with torch.no_grad():
            out = model(input_ids=prompt_ids, use_cache=False)
        first_logits = out.logits[0, -1].float()
    else:
        b_out = bridge_forward(
            model, tok,
            input_ids=prompt_ids,
            layer_idx=layers,
            alpha=alpha,
            backend='exact',
            K=K,
            seed=seed,
            return_intermediates=False,
        )
        first_logits = b_out[0, -1].float()

    first_logits = apply_repetition_penalty(
        first_logits, tokens, REPETITION_PENALTY)
    first_tok = top_p_sample(first_logits, TEMPERATURE, TOP_P)
    tokens.append(first_tok)

    # Steps 1+: vanilla + top-p sampling
    for i in range(n_tokens - 1):
        ctx = torch.tensor([tokens], device=device)
        with torch.no_grad():
            out = model(input_ids=ctx, use_cache=False)
        logits = out.logits[0, -1].float()
        logits = apply_repetition_penalty(logits, tokens, REPETITION_PENALTY)
        next_tok = top_p_sample(logits, TEMPERATURE, TOP_P)
        tokens.append(next_tok)

        n_gen = i + 2
        if n_gen in milestones and n_gen not in reported:
            elapsed = time.time() - t0
            tps = n_gen / elapsed if elapsed > 0 else 0
            print(f"    [{run_label}] token {n_gen} — {tps:.1f} tok/s")
            reported.add(n_gen)

        if next_tok == tok.eos_token_id:
            break

    generated = tokens[prompt_ids.shape[1]:]
    text      = tok.decode(generated, skip_special_tokens=True)
    elapsed   = time.time() - t0
    return text, elapsed, len(generated)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_sampling_longgen(model, tok):
    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path    = os.path.join(RESULTS_DIR,
                               f"tasb_longgen_sampling_{timestamp}.csv")
    txt_path    = os.path.join(RESULTS_DIR,
                               f"tasb_longgen_sampling_text_{timestamp}.txt")
    summary_path = os.path.join(RESULTS_DIR,
                                f"tasb_longgen_sampling_summary_{timestamp}.txt")

    csv_fields = [
        "run_id", "config_id", "n_layers", "layers_str",
        "alpha", "k_value", "seed",
        "prompt_id", "domain",
        "temperature", "top_p", "rep_penalty",
        "n_generated", "elapsed_s", "tokens_per_s",
        "coherence", "has_loop", "has_drift", "truncated",
        "text_preview",
    ]

    device  = next(model.parameters()).device
    run_num = 0

    # Collect texts per (cfg, alpha, prompt) for diversity analysis
    # key: (config_id, alpha, prompt_id) -> list of (seed, text)
    diversity_bucket = {}

    summary_lines = [
        "TASB Long Generation — Sampling Mode Summary",
        f"Timestamp: {timestamp}",
        f"temperature={TEMPERATURE}  top_p={TOP_P}  "
        f"rep_penalty={REPETITION_PENALTY}",
        f"seeds={SEEDS}  gen_tokens={GEN_TOKENS}",
        "="*72,
    ]

    with open(csv_path, "w", newline="") as fc, \
         open(txt_path, "w", encoding="utf-8") as ft:

        writer = csv.DictWriter(fc, fieldnames=csv_fields)
        writer.writeheader()

        ft.write("TASB Long Generation — Sampling Mode\n")
        ft.write(f"Timestamp: {timestamp}\n")
        ft.write(f"temp={TEMPERATURE}  top_p={TOP_P}  "
                 f"rep_penalty={REPETITION_PENALTY}\n")
        ft.write("="*72 + "\n\n")

        for alpha in ALPHAS:
            for cfg_id, layers, cfg_desc in LAYER_CONFIGS:

                # alpha=0.0 is vanilla — config-independent, run once
                if alpha == 0.0 and cfg_id != "S1":
                    continue

                layers_str = ",".join(str(l) for l in layers)
                n_layers   = len(layers)

                for prompt_id, domain, prompt_text in PROMPTS:
                    keywords = PROMPT_KEYWORDS.get(prompt_id, [])
                    seed_texts = []

                    for seed in SEEDS:
                        run_num += 1
                        run_label = (f"α={alpha} {cfg_id} "
                                     f"{prompt_id} seed={seed}")

                        print(f"\n{'='*68}")
                        print(f"Run {run_num}: {run_label}")
                        print(f"  layers={layers}  domain={domain}")
                        print(f"  temp={TEMPERATURE} top_p={TOP_P} "
                              f"rep={REPETITION_PENALTY}")
                        print(f"{'='*68}")

                        prompt_ids = tok(
                            prompt_text, return_tensors='pt'
                        ).to(device)['input_ids']

                        try:
                            text, elapsed, n_gen = generate_sampled(
                                model, tok, prompt_ids,
                                layers, alpha, GEN_TOKENS,
                                seed, run_label)

                            tps       = n_gen / elapsed if elapsed > 0 else 0
                            has_loop  = detect_loop(text)
                            has_drift = detect_topic_drift(text, keywords)
                            truncated = n_gen < GEN_TOKENS - 5
                            coherence = classify(text, keywords, n_gen)

                            print(f"\n  [{coherence}] {n_gen} tok  "
                                  f"{tps:.1f} tok/s")
                            print(f"  First 300: {text[:300]}")
                            print(f"  Last 150:  ...{text[-150:]}")

                            seed_texts.append((seed, text))

                            writer.writerow({
                                "run_id":      run_num,
                                "config_id":   cfg_id,
                                "n_layers":    n_layers,
                                "layers_str":  layers_str,
                                "alpha":       alpha,
                                "k_value":     K,
                                "seed":        seed,
                                "prompt_id":   prompt_id,
                                "domain":      domain,
                                "temperature": TEMPERATURE,
                                "top_p":       TOP_P,
                                "rep_penalty": REPETITION_PENALTY,
                                "n_generated": n_gen,
                                "elapsed_s":   round(elapsed, 1),
                                "tokens_per_s": round(tps, 2),
                                "coherence":   coherence,
                                "has_loop":    has_loop,
                                "has_drift":   has_drift,
                                "truncated":   truncated,
                                "text_preview": text[:200].replace('\n', ' '),
                            })

                            ft.write(f"{'='*68}\n")
                            ft.write(f"Run {run_num}: {run_label}\n")
                            ft.write(f"Config: {cfg_id} | alpha={alpha} | "
                                     f"seed={seed} | [{coherence}]\n")
                            ft.write(f"Generated: {n_gen} tok | "
                                     f"{tps:.1f} tok/s\n\n")
                            ft.write(f"PROMPT:\n{prompt_text}\n\n")
                            ft.write(f"GENERATED:\n{text}\n\n")
                            ft.flush()

                        except torch.cuda.OutOfMemoryError:
                            print(f"  OOM")
                            writer.writerow({
                                "run_id": run_num, "config_id": cfg_id,
                                "n_layers": n_layers,
                                "layers_str": layers_str,
                                "alpha": alpha, "k_value": K, "seed": seed,
                                "prompt_id": prompt_id, "domain": domain,
                                "temperature": TEMPERATURE, "top_p": TOP_P,
                                "rep_penalty": REPETITION_PENALTY,
                                "n_generated": 0, "elapsed_s": 0,
                                "tokens_per_s": 0, "coherence": "OOM",
                                "has_loop": False, "has_drift": False,
                                "truncated": True, "text_preview": "",
                            })
                            gc.collect()
                            torch.cuda.empty_cache()

                        gc.collect()
                        torch.cuda.empty_cache()

                    # Diversity analysis across seeds for this config/alpha/prompt
                    key = (cfg_id, alpha, prompt_id)
                    diversity_bucket[key] = seed_texts

                    if len(seed_texts) >= 2:
                        sims = []
                        for i in range(len(seed_texts)):
                            for j in range(i+1, len(seed_texts)):
                                sim = ngram_jaccard(
                                    seed_texts[i][1], seed_texts[j][1])
                                sims.append(sim)
                        mean_sim = sum(sims) / len(sims) if sims else 0
                        diverse  = mean_sim < 0.3
                        print(f"\n  Diversity ({cfg_id} α={alpha} {prompt_id}): "
                              f"mean 8-gram Jaccard={mean_sim:.3f} "
                              f"({'DIVERSE' if diverse else 'SIMILAR'})")

    # Write summary
    print(f"\n\n{'='*68}")
    print("SAMPLING MODE SUMMARY")
    print(f"temp={TEMPERATURE}  top_p={TOP_P}  rep_penalty={REPETITION_PENALTY}")
    print(f"{'='*68}")

    with open(csv_path, newline="") as fr:
        rows = list(csv.DictReader(fr))

    # Group by alpha for high-level view
    for alpha in ALPHAS:
        alpha_rows = [r for r in rows if float(r['alpha']) == alpha]
        if not alpha_rows:
            continue
        clean    = sum(1 for r in alpha_rows if r['coherence'] == 'CLEAN')
        loops    = sum(1 for r in alpha_rows if 'LOOP' in r['coherence'])
        total    = len(alpha_rows)
        mean_tps = (sum(float(r['tokens_per_s'] or 0) for r in alpha_rows)
                    / total if total else 0)
        print(f"  alpha={alpha}: {clean}/{total} CLEAN  "
              f"{loops} LOOP  {mean_tps:.1f} tok/s avg")

    print()
    print(f"{'Run':<4} {'Cfg':<5} {'α':>4} {'Prompt':<8} {'Seed':>5} "
          f"{'N':>4} {'Coherence'}")
    print("-"*60)
    for r in rows:
        flag = "✓" if r['coherence'] == 'CLEAN' else "⚠"
        print(f"{r['run_id']:<4} {r['config_id']:<5} "
              f"{float(r['alpha']):>4.1f} {r['prompt_id']:<8} "
              f"{r['seed']:>5} {r['n_generated']:>4}  "
              f"{flag} {r['coherence']}")

    print(f"\nCSV:     {csv_path}")
    print(f"Text:    {txt_path}")
    print(f"Summary: {summary_path}")
    print(f"{'='*68}")

    # Write summary txt
    with open(summary_path, "w") as fs:
        fs.write("\n".join(summary_lines))
        fs.write("\n\nDiversity analysis (8-gram Jaccard across seeds):\n")
        for (cfg, alpha, pid), texts in sorted(diversity_bucket.items()):
            if len(texts) >= 2:
                sims = []
                for i in range(len(texts)):
                    for j in range(i+1, len(texts)):
                        sims.append(ngram_jaccard(texts[i][1], texts[j][1]))
                mean_sim = sum(sims)/len(sims) if sims else 0
                fs.write(f"  {cfg} α={alpha} {pid}: "
                         f"Jaccard={mean_sim:.3f} "
                         f"({'DIVERSE' if mean_sim < 0.3 else 'SIMILAR'})\n")

    return csv_path, txt_path, summary_path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from transformers import (
        AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig)

    n_runs = (len(SEEDS) * len(PROMPTS) +   # vanilla (alpha=0, S1 only)
              len(SEEDS) * 3 * len(LAYER_CONFIGS) * len(PROMPTS))
    est_min = n_runs * GEN_TOKENS / 5.9 / 60

    print("TASB Long Generation — Sampling Mode")
    print(f"temperature={TEMPERATURE}  top_p={TOP_P}  "
          f"rep_penalty={REPETITION_PENALTY}")
    print(f"Alphas: {ALPHAS}  Seeds: {SEEDS}")
    print(f"Configs: {[c[0] for c in LAYER_CONFIGS]}")
    print(f"Prompts: {len(PROMPTS)}  Gen tokens: {GEN_TOKENS}")
    print(f"Estimated runs: ~{n_runs}  Estimated time: ~{est_min:.0f} min")
    print()
    print("Loading model...", flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype=torch.float16,
        ),
        attn_implementation='eager',
        device_map='auto',
    )
    model.eval()
    print(f"Model ready on {next(model.parameters()).device}")
    print()

    run_sampling_longgen(model, tok)
