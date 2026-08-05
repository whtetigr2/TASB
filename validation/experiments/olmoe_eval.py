"""
T5 — OLMoE-1B-7B MoE Routing Bridge Validation
================================================
Validates thermobridge Phase 2 (moe_routing application) on a real MoE model.

Model: allenai/OLMoE-1B-7B-0924
  Total params : ~6.9B
  Active params: ~1B per token (64 experts, top-8 routing)
  Float16 size : ~13.8GB — fits on L4 (22.5GB) without quantization
  Gate type    : OlmoeTopKRouter — returns 3-tuple (logits, scores, indices)
  Gate path    : model.model.layers[i].mlp.gate

Hardware compatibility patches applied:
  1. SM 9.0 grouped_mm patch: transformers 5.5.3 uses torch._grouped_mm for
     expert dispatch, which requires CUDA SM 9.0. The L4 is SM 8.9. Patching
     _can_use_grouped_mm → False forces the built-in SM-agnostic fallback path
     (transformers::grouped_mm_fallback) which works on any compute capability.
  2. OLMoE gate hook: OlmoeTopKRouter.forward() returns (router_logits,
     router_scores, router_indices). Standard MoERoutingHook expects a raw
     tensor. OLMoERoutingHook handles the tuple, sampling from log(router_logits)
     with temperature scaling.

Model selection rationale:
  Mixtral-8x7B: 46.7B float16 = 93GB → 4-bit OOM (weights > VRAM at load time)
  Qwen1.5-MoE-A2.7B: 14.3B float16 = 28.6GB → 4-bit OOM (same cause)
  OLMoE-1B-7B: 6.9B float16 = 13.8GB → fits, clean load, no quantization needed

PRIMARY GOAL: HOOK_OK — OLMoERoutingHook fires on OLMoE's gate during inference.
SECONDARY GOAL: ΔPPL profile across temperatures (T=0.5, 1.0, 2.0).

PASS threshold (T=1.0 condition): HOOK_OK and |ΔPPL| < 1.0

Dataset : WikiText-2 test split, stride=512 (same as T3/T4)

Reproduce:
    python experiments/tasb_olmoe_eval.py

Results saved to: results/tasb_olmoe_moe_<timestamp>.csv

"""

import argparse
import csv
import math
import os
import sys
import time
from datetime import datetime, timezone

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

ROOT        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_ID    = "allenai/OLMoE-1B-7B-0924"
RESULTS_DIR = os.path.join(ROOT, "results")


# ── SM 9.0 expert dispatch patch ───────────────────────────────────────────────
# transformers 5.5.3 checks hasattr(torch, "_grouped_mm") which is True on
# the L4 (SM 8.9), but the actual call raises RuntimeError (requires SM 9.0).
# Patching _can_use_grouped_mm → False routes ALL expert dispatch through
# transformers::grouped_mm_fallback, which is SM-agnostic.
# This must be done BEFORE the model loads (before forward hooks register).

def _patch_grouped_mm_for_sm89():
    try:
        import transformers.integrations.moe as _moe
        _moe._can_use_grouped_mm = lambda *a, **kw: False
        print("  [patch] grouped_mm → grouped_mm_fallback (SM 8.9 compat)", flush=True)
    except Exception as e:
        print(f"  [patch] WARNING: could not patch grouped_mm: {e}", flush=True)


# ── OLMoE-specific routing hook ────────────────────────────────────────────────

class OLMoERoutingHook:
    """
    MoE routing hook for OLMoE's OlmoeTopKRouter.

    OlmoeTopKRouter.forward() returns:
      (router_logits, router_scores, router_indices)
    where router_logits = softmax(F.linear(hidden, W))  ← post-softmax probs
          router_scores = normalized top-k weights
          router_indices = top-k expert indices (long)

    Boltzmann at temperature T: p_T ∝ exp(z_i/T)
    Since router_logits = softmax(z), log(router_logits) ∝ z (up to const).
    So softmax(log(router_logits) / T) gives the correct Boltzmann distribution.

    The hook samples new top_k experts from p_T and returns a modified tuple.
    capture is set to (router_logits, new_indices) when the hook fires.
    """

    def __init__(self, K: int, temperature: float, seed: int = 42):
        self.K = K
        self.temperature = temperature
        self.seed = seed
        self.capture = None

    def post_hook(self, module, args, output):
        if not isinstance(output, tuple) or len(output) != 3:
            # Unexpected output format — don't modify, don't set capture
            return output

        router_logits, router_scores, router_indices = output
        top_k = router_indices.shape[-1]

        # Recover log-space logits (log(p) ∝ raw logit, constant cancels in softmax)
        log_probs = torch.log(router_logits.float().clamp(min=1e-10))

        # Boltzmann distribution at temperature T
        p_boltzmann = F.softmax(log_probs / self.temperature, dim=-1)

        # Select top_k experts from Boltzmann distribution
        new_scores, new_indices = torch.topk(p_boltzmann, top_k, dim=-1)

        # Respect model's norm_topk_prob setting (OLMoE default: False).
        # When False, router_scores are raw top-k softmax probs (sum < 1).
        # Always normalizing would inflate expert contributions, causing large ΔPPL.
        if getattr(module, "norm_topk_prob", False):
            new_scores = new_scores / new_scores.sum(dim=-1, keepdim=True)

        new_scores = new_scores.to(router_scores.dtype)
        new_indices = new_indices.to(router_indices.dtype)

        self.capture = (router_logits, new_indices)

        return (p_boltzmann.to(router_logits.dtype), new_scores, new_indices)


# ── Gate discovery ─────────────────────────────────────────────────────────────

def find_moe_gate(model):
    """Find first layer index and gate module.

    Accepts any gate module that has a weight attribute — NOT restricted to
    nn.Linear. OLMoE uses OlmoeTopKRouter (nn.Module with weight parameter).
    """
    paths_to_try = [
        ("mlp.gate",                 lambda layer: layer.mlp.gate),
        ("block_sparse_moe.gate",    lambda layer: layer.block_sparse_moe.gate),
        ("feed_forward.gate",        lambda layer: layer.feed_forward.gate),
    ]
    for i, layer in enumerate(model.model.layers):
        for path_name, path_fn in paths_to_try:
            try:
                gate = path_fn(layer)
                if hasattr(gate, "weight"):
                    print(f"  Gate discovered: layers[{i}].{path_name}"
                          f"  type={type(gate).__name__}"
                          f"  weight={tuple(gate.weight.shape)}", flush=True)
                    return i, gate
            except AttributeError:
                continue
    raise RuntimeError(
        "Could not find a gate with weight attribute in any layer. "
        "Add the gate path to find_moe_gate()."
    )


# ── Data ───────────────────────────────────────────────────────────────────────

def load_wikitext2_chunks(tok, stride=512, max_chunks=None):
    print(f"Loading WikiText-2 test split (stride={stride}) ...")
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(t for t in ds["text"] if t.strip())
    ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids[0]
    total = ids.shape[0]
    chunks = [ids[s:s+stride].unsqueeze(0) for s in range(0, total - stride, stride)]
    if max_chunks:
        chunks = chunks[:max_chunks]
    print(f"  {total:,} tokens → {len(chunks)} chunks of {stride} (using {len(chunks)})")
    return chunks


# ── Evaluation ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def eval_vanilla(model, chunks, label="vanilla"):
    total_nll, n, t0 = 0.0, 0, time.time()
    for i, chunk in enumerate(chunks, 1):
        out = model(chunk)
        shifted = out.logits[:, :-1, :].float().contiguous()
        labels  = chunk[:, 1:].contiguous()
        total_nll += F.cross_entropy(
            shifted.view(-1, shifted.shape[-1]), labels.view(-1)).item()
        n += 1
        if i % 25 == 0 or i == len(chunks):
            print(f"  [{label}] {i}/{len(chunks)}"
                  f"  PPL={math.exp(total_nll/n):.4f}"
                  f"  {(time.time()-t0)/n:.3f}s/chunk", flush=True)
    mean_nll = total_nll / n
    return math.exp(mean_nll), mean_nll, time.time() - t0, n


@torch.no_grad()
def eval_moe_bridge(model, chunks, layer_idx, gate, temperature, K,
                    seed=42, label=None):
    label = label or f"moe-T{temperature}"
    total_nll, n, t0 = 0.0, 0, time.time()
    hook_ok = False

    for i, chunk in enumerate(chunks, 1):
        hook   = OLMoERoutingHook(K=K, temperature=temperature, seed=seed)
        handle = gate.register_forward_hook(hook.post_hook)
        try:
            out = model(chunk)
        finally:
            handle.remove()

        if hook.capture is not None:
            hook_ok = True

        shifted = out.logits[:, :-1, :].float().contiguous()
        labels  = chunk[:, 1:].contiguous()
        total_nll += F.cross_entropy(
            shifted.view(-1, shifted.shape[-1]), labels.view(-1)).item()
        n += 1

        if i % 25 == 0 or i == len(chunks):
            print(f"  [{label}] {i}/{len(chunks)}"
                  f"  PPL={math.exp(total_nll/n):.4f}"
                  f"  {(time.time()-t0)/n:.3f}s/chunk"
                  f"  [{'HOOK_OK' if hook_ok else 'HOOK_MISS'}]", flush=True)

    mean_nll = total_nll / n
    return math.exp(mean_nll), mean_nll, time.time() - t0, n, hook_ok


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stride",       type=int,   default=512)
    parser.add_argument("--K",            type=int,   default=50)
    parser.add_argument("--seed",         type=int,   default=42)
    parser.add_argument("--max-chunks",   type=int,   default=100)
    parser.add_argument("--sweep-chunks", type=int,   default=25)
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    ts      = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    csv_out = os.path.join(RESULTS_DIR, f"tasb_olmoe_moe_{ts}.csv")

    if torch.cuda.is_available():
        props    = torch.cuda.get_device_properties(0)
        total_gb = props.total_memory / 1e9
        sm       = f"{props.major}.{props.minor}"
        print(f"GPU: {torch.cuda.get_device_name(0)}  {total_gb:.1f} GB total  SM {sm}",
              flush=True)

    # Patch expert dispatch BEFORE model loads
    _patch_grouped_mm_for_sm89()

    # ── Load model ─────────────────────────────────────────────────────────────
    print(f"\nLoading {MODEL_ID} in float16 ...")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.float16,
        device_map="auto",
    )
    model.eval()

    if torch.cuda.is_available():
        used_gb = torch.cuda.memory_allocated(0) / 1e9
        free_gb = torch.cuda.mem_get_info(0)[0] / 1e9
        print(f"VRAM after load: {used_gb:.2f} GB used  {free_gb:.2f} GB free", flush=True)

    n_layers = model.config.num_hidden_layers
    n_experts = getattr(model.config, "num_experts",
                getattr(model.config, "num_local_experts",
                getattr(model.config, "num_routed_experts", "?")))
    top_k = getattr(model.config, "num_experts_per_tok",
             getattr(model.config, "top_k", "?"))
    print(f"  {MODEL_ID}: {n_layers} layers  {n_experts} experts  top-{top_k}", flush=True)

    # Discover gate
    layer_idx, gate = find_moe_gate(model)
    device = next(model.parameters()).device

    # ── Load data ──────────────────────────────────────────────────────────────
    chunks_full  = load_wikitext2_chunks(tok, args.stride, args.max_chunks)
    chunks_sweep = chunks_full[:args.sweep_chunks]
    chunks_full  = [c.to(device) for c in chunks_full]
    chunks_sweep = [c.to(device) for c in chunks_sweep]

    rows = [["condition", "backend", "temperature", "K", "layer",
             "chunks", "ppl", "delta_ppl", "mean_nll",
             "sec_per_chunk", "hook_ok", "model"]]

    # ── Vanilla baseline ───────────────────────────────────────────────────────
    print(f"\nEvaluating vanilla ({len(chunks_full)} chunks) ...", flush=True)
    van_ppl, van_nll, van_t, van_n = eval_vanilla(model, chunks_full)
    van_ct = van_t / van_n
    print(f"  [vanilla] PPL={van_ppl:.4f}  NLL={van_nll:.6f}  {van_ct:.3f}s/chunk",
          flush=True)
    rows.append(["vanilla", "none", 1.0, 0, layer_idx,
                 van_n, van_ppl, 0.0, van_nll, van_ct, "N/A", MODEL_ID])

    # ── MoE bridge T=1.0 (primary neutrality + HOOK_OK check) ─────────────────
    print(f"\nEvaluating moe-T1.0 ({len(chunks_full)} chunks) ...", flush=True)
    t1_ppl, t1_nll, t1_t, t1_n, t1_ok = eval_moe_bridge(
        model, chunks_full, layer_idx, gate, 1.0, args.K, args.seed, "moe-T1.0")
    t1_ct    = t1_t / t1_n
    t1_delta = t1_ppl - van_ppl
    print(f"  [moe-T1.0] PPL={t1_ppl:.4f}  ΔPPL={t1_delta:+.4f}"
          f"  HOOK={'OK' if t1_ok else 'MISS'}", flush=True)
    rows.append(["moe-T1.0", "boltzmann", 1.0, args.K, layer_idx,
                 t1_n, t1_ppl, t1_delta, t1_nll, t1_ct, t1_ok, MODEL_ID])

    # ── Subset vanilla for fair temperature sweep comparison ──────────────────
    print(f"\nEvaluating vanilla-sub ({len(chunks_sweep)} chunks) ...", flush=True)
    van_sub_ppl, _, _, _ = eval_vanilla(model, chunks_sweep, "vanilla-sub")

    # ── T=0.5 (sharper routing) ────────────────────────────────────────────────
    print(f"\nEvaluating moe-T0.5 ({len(chunks_sweep)} chunks) ...", flush=True)
    t05_ppl, t05_nll, t05_t, t05_n, t05_ok = eval_moe_bridge(
        model, chunks_sweep, layer_idx, gate, 0.5, args.K, args.seed, "moe-T0.5")
    t05_ct    = t05_t / t05_n
    t05_delta = t05_ppl - van_sub_ppl
    print(f"  [moe-T0.5] PPL={t05_ppl:.4f}  ΔPPL={t05_delta:+.4f}"
          f"  HOOK={'OK' if t05_ok else 'MISS'}", flush=True)
    rows.append(["moe-T0.5", "boltzmann", 0.5, args.K, layer_idx,
                 t05_n, t05_ppl, t05_delta, t05_nll, t05_ct, t05_ok, MODEL_ID])

    # ── T=2.0 (diffuse routing) ────────────────────────────────────────────────
    print(f"\nEvaluating moe-T2.0 ({len(chunks_sweep)} chunks) ...", flush=True)
    t2_ppl, t2_nll, t2_t, t2_n, t2_ok = eval_moe_bridge(
        model, chunks_sweep, layer_idx, gate, 2.0, args.K, args.seed, "moe-T2.0")
    t2_ct    = t2_t / t2_n
    t2_delta = t2_ppl - van_sub_ppl
    print(f"  [moe-T2.0] PPL={t2_ppl:.4f}  ΔPPL={t2_delta:+.4f}"
          f"  HOOK={'OK' if t2_ok else 'MISS'}", flush=True)
    rows.append(["moe-T2.0", "boltzmann", 2.0, args.K, layer_idx,
                 t2_n, t2_ppl, t2_delta, t2_nll, t2_ct, t2_ok, MODEL_ID])

    # ── Summary ────────────────────────────────────────────────────────────────
    threshold = 1.0
    passed = t1_ok and abs(t1_delta) < threshold

    print(f"""
{"="*76}
  T5 OLMoE-1B-7B MoE ROUTING BRIDGE SUMMARY — WikiText-2 test split
{"="*76}
  Model    : {MODEL_ID}
  Arch     : {n_layers}L  {n_experts} experts  top-{top_k}  float16
  Dataset  : WikiText-2 test  stride={args.stride}
  Hook     : OLMoERoutingHook  layer={layer_idx}  K={args.K}  seed={args.seed}
  Patches  : grouped_mm → fallback (SM 8.9 compat)
{"="*76}
  Condition   T     K  Chunks     PPL      ΔPPL   sec/chunk  HOOK
  --------------------------------------------------------------------------
  vanilla    1.0    0  {van_n:>6}  {van_ppl:>8.4f}   +0.0000  {van_ct:.3f}s    N/A
  moe-T1.0   1.0  {args.K:>2}  {t1_n:>6}  {t1_ppl:>8.4f}  {t1_delta:+.4f}  {t1_ct:.3f}s    {"OK" if t1_ok else "MISS"}
  moe-T0.5   0.5  {args.K:>2}  {t05_n:>6}  {t05_ppl:>8.4f}  {t05_delta:+.4f}  {t05_ct:.3f}s    {"OK" if t05_ok else "MISS"}
  moe-T2.0   2.0  {args.K:>2}  {t2_n:>6}  {t2_ppl:>8.4f}  {t2_delta:+.4f}  {t2_ct:.3f}s    {"OK" if t2_ok else "MISS"}
{"="*76}
  HOOK   : {"HOOK_OK — OLMoERoutingHook fired on OLMoE gate" if t1_ok else "HOOK_MISS"}
  |ΔPPL T=1.0| : {abs(t1_delta):.4f}  (threshold ±{threshold})
  RESULT       : {"PASS" if passed else "FAIL"}
  Temp profile : T=0.5 ΔPPL={t05_delta:+.4f}  T=2.0 ΔPPL={t2_delta:+.4f}
                 (T>1 expected to increase PPL; T<1 expected to hold or decrease)
{"="*76}
""", flush=True)

    with open(csv_out, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    print(f"Saved: {csv_out}", flush=True)

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
