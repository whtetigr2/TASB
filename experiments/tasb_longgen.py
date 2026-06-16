"""
tasb_longgen.py — Long free generation quality test
==============================================================================
Patent: USPTO Provisional 64/019,999 (March 28, 2026)
Author: Paul W. Shaver

Tests coherence over 400 tokens of fully free autoregressive generation
at varied alpha and layer configs. This is the last data collection step
before the interactive demo.

WHAT THIS ANSWERS
-----------------
1. Does coherence hold past token 100? Token 200? Token 400?
2. What does generation look like at alpha=1.0 (full TSU substitution)?
3. Does the output loop, drift, or degrade at high alpha or high layer count?
4. How does capture-once drift manifest as context grows away from prompt?
5. Is the output readable and on-topic at every slider position the demo
   will expose?

TEST MATRIX
-----------
Alpha values:  [0.0, 0.3, 0.7, 1.0]
Layer configs: [S1 (1L), S3 (5L)]
Prompts:       3 (science, creative, technical)
Gen tokens:    400
Mode:          capture-once (production demo mode)

Alpha=0.0 is the vanilla baseline — bit-exact, no bridge.
Alpha=1.0 is full TSU substitution — the rightmost demo slider position.
S1 and S3 cover the single-layer and sweet-spot multi-layer cases.

That is 4 alpha × 2 configs × 3 prompts = 24 generation runs.
At ~5.9 tok/s and 400 tokens per run: ~68s per run, ~27 min total.

COHERENCE MARKERS (auto-detected)
----------------------------------
The script scans generated text for:
  - LOOP: same 8+ word phrase repeated 2+ times
  - TOPIC_DRIFT: last 100 tokens contain none of the prompt's content words
  - TRUNCATED: generation stopped before 400 tokens (EOS hit)
  - CLEAN: none of the above

These are heuristics, not ground truth. Read the full text in the summary.

OUTPUT
------
  results/tasb_longgen_<timestamp>.csv        per-run metadata
  results/tasb_longgen_text_<timestamp>.txt   full generated text, all runs
  Console: live generation with token count milestones
==============================================================================
"""

import csv
import gc
import os
import re
import sys
import time
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
BASE_SEED  = 42
GEN_TOKENS = 400

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Test matrix
ALPHAS = [0.0, 0.3, 0.7, 1.0]

LAYER_CONFIGS = [
    ("S1", [18],              "1L baseline"),
    ("S3", [15, 18, 21, 24, 27], "5L spread"),
]

# 3 prompts — varied domains, moderate length (~20-30 tokens each)
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

# Content words per prompt for topic-drift detection
PROMPT_KEYWORDS = {
    "LG_SC": ["thermodynamic", "energy", "computing", "silicon", "thermal",
               "noise", "processor", "information"],
    "LG_CR": ["cartographer", "maps", "coastlines", "island", "sea",
               "drew", "obsolete", "ink"],
    "LG_TC": ["language", "model", "stochastic", "hardware", "attention",
               "softmax", "deterministic", "probability"],
}


# ---------------------------------------------------------------------------
# Coherence detection (heuristic)
# ---------------------------------------------------------------------------

def detect_loop(text: str, min_phrase_len: int = 8) -> bool:
    """Detect if any 8+ word phrase repeats more than once."""
    words = text.lower().split()
    if len(words) < min_phrase_len * 2:
        return False
    for i in range(len(words) - min_phrase_len):
        phrase = " ".join(words[i:i + min_phrase_len])
        rest   = " ".join(words[i + min_phrase_len:])
        if phrase in rest:
            return True
    return False


def detect_topic_drift(text: str, keywords: list) -> bool:
    """Check if last 100 tokens of text contain any prompt keywords."""
    words = text.lower().split()
    last_100 = " ".join(words[-100:]) if len(words) > 100 else text.lower()
    return not any(kw in last_100 for kw in keywords)


def classify(text: str, keywords: list, n_generated: int) -> str:
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

def generate(model, tok, prompt_ids, layers, alpha, n_tokens, run_label):
    """Capture-once generation. Returns (text, elapsed, n_generated, tokens)."""
    device = prompt_ids.device
    tokens = [t.item() for t in prompt_ids[0]]
    t0     = time.time()
    milestones = {100, 200, 300, 400}
    reported   = set()

    # alpha=0 fast path: vanilla greedy, no bridge overhead
    if alpha == 0.0:
        for i in range(n_tokens):
            ctx = torch.tensor([tokens], device=device)
            with torch.no_grad():
                out = model(input_ids=ctx, use_cache=False)
            next_tok = out.logits[0, -1].argmax().item()
            tokens.append(next_tok)
            n_gen = i + 1
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
        return text, elapsed, len(generated), generated

    # alpha > 0: capture once at prompt, inject at first token,
    # then generate vanilla for remaining tokens
    # This is production demo mode — one capture, fixed p_thermo
    result = bridge_forward(
        model, tok,
        input_ids=prompt_ids,
        layer_idx=layers,
        alpha=alpha,
        backend='exact',
        K=K,
        seed=BASE_SEED,
        return_intermediates=False,
    )
    first_tok = result[0, -1].argmax().item()
    tokens.append(first_tok)
    print(f"    [{run_label}] first token generated, continuing greedy...")

    for i in range(n_tokens - 1):
        ctx = torch.tensor([tokens], device=device)
        with torch.no_grad():
            out = model(input_ids=ctx, use_cache=False)
        next_tok = out.logits[0, -1].argmax().item()
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
    return text, elapsed, len(generated), generated


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_longgen(model, tok):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path  = os.path.join(RESULTS_DIR, f"tasb_longgen_{timestamp}.csv")
    txt_path  = os.path.join(RESULTS_DIR, f"tasb_longgen_text_{timestamp}.txt")

    csv_fields = [
        "run_id", "config_id", "n_layers", "layers_str",
        "alpha", "k_value", "seed",
        "prompt_id", "domain",
        "n_generated", "elapsed_s", "tokens_per_s",
        "coherence", "has_loop", "has_drift", "truncated",
    ]

    device = next(model.parameters()).device

    with open(csv_path, "w", newline="") as fc, \
         open(txt_path, "w", encoding="utf-8") as ft:

        writer = csv.DictWriter(fc, fieldnames=csv_fields)
        writer.writeheader()

        ft.write(f"TASB Long Generation Test\n")
        ft.write(f"Timestamp: {timestamp}\n")
        ft.write(f"Gen tokens: {GEN_TOKENS}, K: {K}, Seed: {BASE_SEED}\n")
        ft.write(f"{'='*72}\n\n")

        run_num = 0

        for alpha in ALPHAS:
            for cfg_id, layers, cfg_desc in LAYER_CONFIGS:

                # alpha=0 is vanilla — only run once (S1 and S3 identical)
                if alpha == 0.0 and cfg_id != "S1":
                    print(f"\n  [alpha=0.0 / {cfg_id}] Skipping — vanilla is config-independent")
                    continue

                layers_str = ",".join(str(l) for l in layers)
                n_layers   = len(layers)

                for prompt_id, domain, prompt_text in PROMPTS:
                    run_num += 1
                    run_label = f"α={alpha} {cfg_id} {prompt_id}"

                    print(f"\n{'='*68}")
                    print(f"Run {run_num}: {run_label}")
                    print(f"  layers={layers}  alpha={alpha}  domain={domain}")
                    print(f"  Prompt: {prompt_text[:70]}...")
                    print(f"{'='*68}")

                    prompt_ids = tok(
                        prompt_text, return_tensors='pt'
                    ).to(device)['input_ids']
                    ctx_len = prompt_ids.shape[1]
                    print(f"  Context: {ctx_len} tokens → generating {GEN_TOKENS}...")

                    try:
                        text, elapsed, n_gen, _ = generate(
                            model, tok, prompt_ids,
                            layers, alpha, GEN_TOKENS, run_label)

                        tps       = n_gen / elapsed if elapsed > 0 else 0
                        has_loop  = detect_loop(text)
                        keywords  = PROMPT_KEYWORDS.get(prompt_id, [])
                        has_drift = detect_topic_drift(text, keywords)
                        truncated = n_gen < GEN_TOKENS - 5
                        coherence = classify(text, keywords, n_gen)

                        print(f"\n  RESULT: {n_gen} tokens  {tps:.1f} tok/s  [{coherence}]")
                        print(f"  Loop: {has_loop}  Drift: {has_drift}  Truncated: {truncated}")
                        print(f"\n  --- GENERATED TEXT (first 400 chars) ---")
                        print(f"  {text[:400]}")
                        print(f"  ...")
                        print(f"  --- GENERATED TEXT (last 200 chars) ---")
                        print(f"  ...{text[-200:]}")

                        writer.writerow({
                            "run_id":      run_num,
                            "config_id":   cfg_id,
                            "n_layers":    n_layers,
                            "layers_str":  layers_str,
                            "alpha":       alpha,
                            "k_value":     K,
                            "seed":        BASE_SEED,
                            "prompt_id":   prompt_id,
                            "domain":      domain,
                            "n_generated": n_gen,
                            "elapsed_s":   round(elapsed, 1),
                            "tokens_per_s": round(tps, 2),
                            "coherence":   coherence,
                            "has_loop":    has_loop,
                            "has_drift":   has_drift,
                            "truncated":   truncated,
                        })

                        # Write full text to TXT
                        ft.write(f"{'='*68}\n")
                        ft.write(f"Run {run_num}: {run_label}\n")
                        ft.write(f"Config: {cfg_id} | layers={layers} | "
                                 f"alpha={alpha} | domain={domain}\n")
                        ft.write(f"Generated: {n_gen} tokens | "
                                 f"{tps:.1f} tok/s | [{coherence}]\n")
                        ft.write(f"\nPROMPT:\n{prompt_text}\n")
                        ft.write(f"\nGENERATED:\n{text}\n\n")
                        ft.flush()

                    except torch.cuda.OutOfMemoryError:
                        print(f"  OOM — skipping this run")
                        writer.writerow({
                            "run_id": run_num, "config_id": cfg_id,
                            "n_layers": n_layers, "layers_str": layers_str,
                            "alpha": alpha, "k_value": K, "seed": BASE_SEED,
                            "prompt_id": prompt_id, "domain": domain,
                            "n_generated": 0, "elapsed_s": 0,
                            "tokens_per_s": 0, "coherence": "OOM",
                            "has_loop": False, "has_drift": False,
                            "truncated": True,
                        })
                        gc.collect()
                        torch.cuda.empty_cache()

                    gc.collect()
                    torch.cuda.empty_cache()

        # Summary table
        print(f"\n\n{'='*68}")
        print(f"LONG GENERATION SUMMARY")
        print(f"{'='*68}")
        print(f"{'Run':<4} {'Config':<5} {'Alpha':>5} {'Prompt':<8} "
              f"{'N':>4} {'tok/s':>6} {'Coherence'}")
        print(f"{'-'*68}")

        fc.flush()
        with open(csv_path, newline="") as fr:
            import csv as _csv
            for row in _csv.DictReader(fr):
                flag = "⚠" if row["coherence"] != "CLEAN" else "✓"
                print(f"{row['run_id']:<4} {row['config_id']:<5} "
                      f"{float(row['alpha']):>5.1f} {row['prompt_id']:<8} "
                      f"{row['n_generated']:>4} {float(row['tokens_per_s'] or 0):>6.1f} "
                      f"  {flag} {row['coherence']}")

        print(f"{'='*68}")
        print(f"\nFull text: {txt_path}")
        print(f"CSV:       {csv_path}")
        print(f"{'='*68}")

    return csv_path, txt_path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from transformers import (
        AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig)

    print("TASB Long Generation Test")
    print(f"Model: {MODEL_ID}")
    print(f"Alphas: {ALPHAS}")
    print(f"Configs: {[c[0] for c in LAYER_CONFIGS]}")
    print(f"Prompts: {len(PROMPTS)}")
    print(f"Gen tokens: {GEN_TOKENS}")
    print(f"Estimated runtime: ~{len(ALPHAS)*len(LAYER_CONFIGS)*len(PROMPTS)*70//60} min")
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

    run_longgen(model, tok)
