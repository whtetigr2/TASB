# CRITICAL: Set JAX memory flags before ANY other imports
# Must be first — JAX reads these at initialization time
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.50"

"""
tasb_demo_data_gen_v1.py
==============================================================================
TASB Demo Data Generator — full backend × alpha × prompt sweep.
Outputs streamlit_demo/demo_data.json for the Streamlit visualization demo.

Built from proven tasb_landscape_capture.py patterns.

SWEEP:
    5 prompts × 4 backends (exact, gumbel, rbm, thrml) × 4 alphas = 80 runs
    ~25 tokens per run
    Estimated runtime: ~30-35 minutes on L4 GPU

USAGE:
    Switch to L4 GPU in Lightning first, then:
    XLA_PYTHON_CLIENT_PREALLOCATE=false stdbuf -oL -eL \\
        python tasb_demo_data_gen_v1.py | tee -a demo_data_gen.log

VRAM budget (L4, 24GB):
    LLaMA 3.2-3B 4-bit:    ~2.5GB
    PyTorch activations:    ~2.0GB
    JAX pool (50% cap):     ~12.0GB ceiling (actual use <1GB)
    Peak total:             ~6-7GB — well within L4 headroom

OUTPUT: streamlit_demo/demo_data.json
==============================================================================
"""

import sys
import json
import time
import datetime
import numpy as np
import torch
import torch.nn.functional as F
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tasb_capture_v2 import LlamaAttentionCapture
from tasb_sampler_v2 import sample as tasb_sample, SamplerConfig
from tasb_injector_v2 import LlamaAttentionInjector, DispatchEntry

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_NAME  = "meta-llama/Llama-3.2-3B-Instruct"
LAYER       = 18
K           = 50          # plain int — never a tensor
MAX_TOKENS  = 25
TOP_N_WELLS = 10
MAX_S       = 50          # downsample J matrix to this size for JSON
OUTPUT_DIR  = "streamlit_demo"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "demo_data.json")

BACKENDS     = ["exact", "gumbel", "rbm", "thrml"]
ALPHA_VALUES = [0.0, 0.3, 0.7, 1.0]

PROMPTS = {
    "thermo": {
        "text": (
            "The relationship between physical entropy and information "
            "processing suggests that"
        ),
        "category": "Thermodynamics / Computation",
        "description": (
            "On-domain prompt. Expect sharp, confident wells — "
            "the model knows this territory well."
        ),
    },
    "whimsy": {
        "text": (
            "If you could ask one question to a spoon, a cloud, a forgotten "
            "dream, and a Tuesday afternoon, the only thing they would all "
            "agree on is"
        ),
        "category": "Whimsy / Chaos",
        "description": (
            "Maximum entropy prompt. Expect a chaotic, choppy landscape — "
            "many valid continuations, no dominant well."
        ),
    },
    "narrative": {
        "text": (
            "The old lighthouse keeper had one rule he never broke, "
            "until the night"
        ),
        "category": "Creative / Narrative",
        "description": (
            "Narrative setup with moderate entropy — some constraint "
            "from genre conventions."
        ),
    },
    "technical": {
        "text": (
            "The softmax function applied to the dot product of query "
            "and key vectors produces"
        ),
        "category": "Technical / Factual",
        "description": (
            "High-confidence factual domain. Expect deep, narrow wells — "
            "the model is certain here."
        ),
    },
    "conversational": {
        "text": "Hey, I was wondering if you could help me figure out",
        "category": "Conversational",
        "description": (
            "Everyday language. Moderate confidence — "
            "well-traveled territory for instruction-tuned models."
        ),
    },
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_model():
    print(f"  Loading {MODEL_NAME}...")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            ),
            attn_implementation="eager",
            device_map="auto",
        )
        print("  Loaded in 4-bit.")
    except Exception:
        print("  bitsandbytes unavailable — loading in float16.")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16,
            attn_implementation="eager",
            device_map="auto",
        )
    model.eval()
    return model, tok


def compute_J(cap) -> torch.Tensor:
    """
    Reconstruct J = Q @ K^T * scaling + mask from a LayerCapture.
    Returns (B, n_q, S, S) float32.
    Proven pattern from tasb_landscape_capture.py.
    """
    q    = cap.q_post_rope.float()
    k    = cap.k_post_rope.float()
    sc   = float(cap.scaling)
    n_q  = q.shape[1]
    n_kv = k.shape[1]
    if n_kv != n_q:
        n_kvg = n_q // n_kv
        k = k.repeat_interleave(n_kvg, dim=1)
    J = torch.matmul(q, k.transpose(-2, -1)) * sc
    if cap.attention_mask is not None:
        J = J + cap.attention_mask.float()
    return J


def downsample(J_np: np.ndarray, max_s: int) -> np.ndarray:
    """Uniform-stride downsample (S,S) -> (max_s, max_s)."""
    S = J_np.shape[0]
    if S <= max_s:
        return J_np
    idx = np.linspace(0, S - 1, max_s, dtype=int)
    return J_np[np.ix_(idx, idx)]


def bucket(prob_gap: float) -> str:
    if prob_gap >= 0.5:  return "confident"
    if prob_gap >= 0.1:  return "moderate"
    return "ambiguous"


def get_top_tokens(logits_1d: torch.Tensor, tok, n: int = 10) -> list:
    probs = F.softmax(logits_1d.float(), dim=-1)
    top_p, top_i = torch.topk(probs, n)
    return [
        {
            "idx":   int(top_i[k]),
            "text":  tok.decode([int(top_i[k])]),
            "logit": round(float(logits_1d[int(top_i[k])]), 4),
            "prob":  round(float(top_p[k]), 6),
        }
        for k in range(n)
    ]


def draw_marbles(J_last: np.ndarray, n: int = 50,
                 seed: int = 42) -> list:
    """
    Draw n categorical samples from softmax(J_last).
    J_last: (S,) logits for last query position.
    Returns list of n key-position integers (marble final positions).
    These are real Boltzmann draws — not simulated.
    """
    rng   = np.random.default_rng(seed)
    j     = J_last.copy().astype(np.float64)
    j     = np.where(np.isfinite(j), j, -1e9)
    j    -= j.max()
    probs = np.exp(j)
    probs /= probs.sum()
    return rng.choice(len(probs), size=n, p=probs).tolist()


def draw_marbles_from_pthermo(p_thermo: torch.Tensor,
                               n: int = 50,
                               seed: int = 42) -> list:
    """
    Draw marble positions from p_thermo (the actual TASB sampler output)
    rather than from softmax(J) directly. This gives backend-specific
    marble positions — exact/gumbel/rbm/thrml will differ slightly,
    demonstrating substrate-agnostic sampling on the same landscape.

    p_thermo: (B, n_q, S, S) row-stochastic
    Returns list of n key-position integers from the last query row,
    averaged across heads.
    """
    rng = np.random.default_rng(seed)
    # Mean over batch and Q heads at the last query position
    p_last = p_thermo[0, :, -1, :].mean(dim=0).float().cpu().numpy()  # (S,)
    p_last = np.where(np.isfinite(p_last), p_last, 0.0)
    total  = p_last.sum()
    if total <= 0:
        p_last = np.ones(len(p_last)) / len(p_last)
    else:
        p_last /= total
    return rng.choice(len(p_last), size=n, p=p_last).tolist()


# ── Per-prompt generation loop ────────────────────────────────────────────────

def generate_with_capture(
    model, tok,
    prompt_text: str,
    alpha: float,
    backend: str,
) -> list:
    """
    Autoregressive generation with TASB capture at every token step.
    Uses proven patterns from tasb_landscape_capture.py.
    Returns list of per-token records.
    """
    device      = next(model.parameters()).device
    current_ids = tok(prompt_text, return_tensors="pt").input_ids.to(device)
    records     = []

    with torch.no_grad():
        for step in range(MAX_TOKENS):

            S = current_ids.shape[1]

            # ── Pass 1: vanilla forward + capture ────────────────────────────
            capturer = LlamaAttentionCapture(
                model=model,
                layers_to_capture=[LAYER],
                strict_verify=False,
            )
            with capturer.capture():
                van_out = model(input_ids=current_ids, use_cache=False)

            van_logits = van_out.logits[:, -1, :]
            cap        = capturer.get_capture(LAYER)

            if cap is None:
                print(f"    [WARN] step {step}: no capture at L{LAYER}")
                break

            # ── J matrix (energy landscape) ───────────────────────────────────
            J_full = compute_J(cap)                           # (B, n_q, S, S)
            J_mean = J_full[0].mean(dim=0).cpu().float().numpy()    # (S, S)
            J_last = J_full[0, :, -1, :].mean(dim=0).cpu().float().numpy()  # (S,)
            J_last = np.where(np.isfinite(J_last), J_last, -1e9)
            J_ds   = downsample(J_mean, MAX_S)

            # ── TASB sample (backend-specific) ────────────────────────────────
            scfg     = SamplerConfig(
                backend=backend,
                K=K,
                seed=int(42 + step + LAYER),
            )
            p_thermo = tasb_sample(cap, scfg)

            # Marble positions from the actual backend's p_thermo output
            # This is what makes backend comparison meaningful —
            # same landscape, different sampling distributions
            marble_pos = draw_marbles_from_pthermo(
                p_thermo, n=K, seed=int(42 + step)
            )

            # ── Pass 2: injected forward ──────────────────────────────────────
            dispatch = {LAYER: DispatchEntry(cap, p_thermo, alpha)}
            injector = LlamaAttentionInjector(dispatch)
            with injector.inject():
                inj_out = model(input_ids=current_ids, use_cache=False)

            inj_logits = inj_out.logits[:, -1, :]

            # ── Metrics ───────────────────────────────────────────────────────
            van_probs  = F.softmax(van_logits.float(), dim=-1)
            kl         = float(F.kl_div(
                F.log_softmax(inj_logits.float(), dim=-1),
                van_probs,
                reduction="batchmean",
            ))
            van_top1   = int(van_logits.argmax(dim=-1))
            inj_top1   = int(inj_logits.argmax(dim=-1))
            top1_match = (van_top1 == inj_top1)
            top2_v     = torch.topk(van_probs[0], 2)
            pg         = float(top2_v.values[0] - top2_v.values[1])
            bkt        = bucket(pg)
            wells      = get_top_tokens(van_logits[0], tok, n=TOP_N_WELLS)

            record = {
                "token_idx":        step,
                "token_text":       tok.decode([van_top1]),
                "seq_len":          S,
                "J_matrix":         J_ds.tolist(),
                "J_last_row":       J_last.tolist(),
                "top_tokens":       wells,
                "sample_positions": marble_pos,
                "kl":               round(kl, 6),
                "top1_match":       top1_match,
                "prob_gap":         round(pg, 4),
                "bucket":           bkt,
                "van_top1_text":    tok.decode([van_top1]),
                "inj_top1_text":    tok.decode([inj_top1]),
            }
            records.append(record)

            flip_str = "" if top1_match else f" <- FLIP [{bkt}]"
            print(f"    step {step:2d} | S={S:3d} | "
                  f"tok='{tok.decode([van_top1])}' | "
                  f"KL={kl:.5f} | pg={pg:.3f}/{bkt[0]}{flip_str}")

            # Advance sequence (greedy from vanilla — teacher-forced style)
            next_tensor = torch.tensor([[van_top1]], device=device)
            current_ids = torch.cat([current_ids, next_tensor], dim=1)

            if van_top1 == tok.eos_token_id:
                print(f"    [EOS] at step {step}")
                break

    return records


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    total_runs = len(PROMPTS) * len(BACKENDS) * len(ALPHA_VALUES)

    print("=" * 72)
    print("TASB Demo Data Generator v1 — Full Backend x Alpha Sweep")
    print(f"  Model:    {MODEL_NAME}")
    print(f"  Layer:    {LAYER}  |  K: {K}")
    print(f"  Backends: {BACKENDS}")
    print(f"  Alphas:   {ALPHA_VALUES}")
    print(f"  Prompts:  {len(PROMPTS)}  |  Tokens: {MAX_TOKENS} per run")
    print(f"  Total runs: {total_runs}")
    print(f"  Output:   {OUTPUT_FILE}")
    print(f"  JAX pre-alloc: {os.environ.get('XLA_PYTHON_CLIENT_PREALLOCATE')}")
    print(f"  JAX mem fraction: {os.environ.get('XLA_PYTHON_CLIENT_MEM_FRACTION')}")
    print("=" * 72)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model, tok = load_model()

    # Check VRAM
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  VRAM after model load: {alloc:.2f}GB / {total:.1f}GB")

    output = {
        "metadata": {
            "model":        MODEL_NAME,
            "layer":        LAYER,
            "K":            K,
            "backends":     BACKENDS,
            "generated_at": str(datetime.date.today()),
            "n_prompts":    len(PROMPTS),
            "alpha_values": ALPHA_VALUES,
            "max_tokens":   MAX_TOKENS,
            "top_n_wells":  TOP_N_WELLS,
            "total_runs":   total_runs,
        },
        "prompts": {},
    }

    run_n   = 0
    t_start = time.time()

    for p_key, p_info in PROMPTS.items():
        print(f"\n{'─'*72}")
        print(f"PROMPT: {p_key} — {p_info['category']}")
        print(f"  \"{p_info['text'][:70]}\"")
        print(f"{'─'*72}")

        output["prompts"][p_key] = {
            "text":        p_info["text"],
            "category":    p_info["category"],
            "description": p_info["description"],
        }

        for backend in BACKENDS:
            for alpha in ALPHA_VALUES:
                run_n += 1
                run_key  = f"{backend}_alpha_{alpha}"
                elapsed_so_far = time.time() - t_start
                eta = (elapsed_so_far / run_n) * (total_runs - run_n) if run_n > 1 else 0

                print(f"\n  [{run_n:2d}/{total_runs}] "
                      f"backend={backend}  alpha={alpha}  "
                      f"(ETA: {eta/60:.1f}min)")

                t0 = time.time()
                try:
                    records = generate_with_capture(
                        model, tok,
                        prompt_text=p_info["text"],
                        alpha=alpha,
                        backend=backend,
                    )
                except Exception as e:
                    print(f"    [ERROR] {backend} alpha={alpha}: {e}")
                    records = []

                elapsed = time.time() - t0

                n          = len(records)
                mean_kl    = float(np.mean([r["kl"] for r in records])) if n else 0.0
                top1_pct   = float(np.mean([r["top1_match"] for r in records])) * 100 if n else 0.0
                conf_flips = sum(
                    1 for r in records
                    if not r["top1_match"] and r["bucket"] == "confident"
                )

                output["prompts"][p_key][run_key] = {
                    "backend": backend,
                    "alpha":   alpha,
                    "tokens":  records,
                    "summary": {
                        "mean_kl":    round(mean_kl, 6),
                        "top1_pct":   round(top1_pct, 2),
                        "conf_flips": conf_flips,
                        "n_tokens":   n,
                        "elapsed_s":  round(elapsed, 1),
                    },
                }

                print(f"    done: {n} tokens | KL={mean_kl:.5f} | "
                      f"top-1={top1_pct:.1f}% | "
                      f"conf_flips={conf_flips} | {elapsed:.1f}s")

                # Checkpoint after every run — don't lose data if it crashes
                with open(OUTPUT_FILE, "w") as f:
                    json.dump(output, f, indent=2)

    # Final write
    print(f"\n{'='*72}")
    size_mb       = os.path.getsize(OUTPUT_FILE) / 1e6
    total_elapsed = time.time() - t_start
    print(f"Complete. {size_mb:.1f} MB | {total_elapsed/60:.1f} minutes total")
    print(f"Output: {OUTPUT_FILE}")
    print(f"Next:   streamlit run streamlit_demo/app.py")
    print("=" * 72)


if __name__ == "__main__":
    main()
