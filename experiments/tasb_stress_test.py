"""
tasb_stress_test.py — Long-context + autoregressive generation stress test
==============================================================================
Patent: USPTO Provisional 64/019,999 (March 28, 2026)
Author: Paul W. Shaver

Pre-demo validation harness. Answers four questions before we build the
interactive demo:

  1. LONG CONTEXT FAITHFULNESS
     Does the bridge hold at S=200, S=400, S=800 token contexts?
     Metric: top-1 agreement, mean KL, confident flips vs context length.

  2. AUTOREGRESSIVE GENERATION WITH BRIDGE
     Does token-by-token streaming work correctly?
     Two modes tested:
       (a) CAPTURE-ONCE: capture vanilla at full prompt, reuse p_thermo
           for all generated tokens. Fast. Open-loop.
       (b) CAPTURE-EVERY: re-capture at each generation step with growing
           context. Slow. Closed-loop approximation.
     Metric: output coherence, generation speed, memory stability.

  3. MEMORY CEILING
     How many layers can we bridge at long context before OOM?
     Tests S3 (5L) and S6 (10L) at each context length.
     Reports peak VRAM usage per config.

  4. GENERATION QUALITY
     Does a 5-paragraph user prompt produce coherent output under
     multi-layer bridging? Side-by-side vanilla vs bridge comparison.

TEST MATRIX
-----------
Context lengths: [64, 128, 256, 512] tokens (truncated from long prompts)
Layer configs:   S1 (1L), S3 (5L), S6 (10L)
Gen modes:       capture-once, capture-every (on short context only)
Gen length:      100 tokens output
Alpha:           0.3 (production)
K:               10 (production; K=50 for TSU silicon, tested separately)

LONG PROMPTS
------------
Three prompts designed for realistic user input length:
  - ESSAY:    5-paragraph science essay (~400 tokens)
  - STORY:    5-paragraph creative fiction (~350 tokens)
  - TECHNICAL: 5-paragraph technical explanation (~380 tokens)

OUTPUT
------
  results/tasb_stress_ctx_<timestamp>.csv       context faithfulness rows
  results/tasb_stress_gen_<timestamp>.csv       generation quality rows
  results/tasb_stress_summary_<timestamp>.txt   human-readable summary
  Console: live progress + memory readings + side-by-side outputs

WHAT BREAKS FIRST DIAGNOSTIC
-----------------------------
The script tracks and reports:
  - First context length where KL exceeds 2x the S=64 baseline
  - First config where VRAM exceeds available budget
  - First config where generation produces incoherent output
  - Speed: tokens/sec for capture-once vs capture-every
==============================================================================
"""

import csv
import gc
import os
import sys
import time
from datetime import datetime
from typing import List, Optional

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tasb_pipeline_v2 import bridge_forward

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_ID  = "meta-llama/Llama-3.2-3B"
ALPHA     = 0.3
K         = 10
BASE_SEED = 42
GEN_TOKENS = 100   # tokens to generate in autoregressive tests

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

CONFIDENT_THRESHOLD = 0.5
MODERATE_THRESHOLD  = 0.1

# Layer configs to test
LAYER_CONFIGS = [
    ("S1",  [18],                         "1L baseline"),
    ("S3",  [15, 18, 21, 24, 27],         "5L spread"),
    ("S6",  [10, 12, 14, 16, 18, 20, 22, 24, 26, 27], "10L heavy"),
]

# Context lengths to test (tokens)
CONTEXT_LENGTHS = [64, 128, 256, 512]

# Long prompts — designed to be truncated to target context lengths
LONG_PROMPTS = [
    ("ESSAY", "SCIENCE",
     """The human brain is the most complex structure known to science.
     Composed of approximately 86 billion neurons, each forming thousands
     of synaptic connections, the brain orchestrates every thought, emotion,
     and physical action we experience. Neuroscientists have spent decades
     mapping its regions, from the prefrontal cortex responsible for
     decision-making to the hippocampus central to memory formation.
     Recent advances in functional MRI have allowed researchers to observe
     the brain in real time, revealing intricate patterns of activation
     that correspond to specific cognitive tasks. These discoveries have
     transformed our understanding of consciousness itself.

     One of the most fascinating aspects of the brain is its plasticity.
     Unlike the rigid circuits of traditional computers, neural tissue
     continuously rewires itself in response to experience. This property,
     known as neuroplasticity, underlies our capacity to learn new skills,
     recover from injury, and adapt to changing environments. Children
     exhibit extraordinary plasticity during critical developmental windows,
     but research now confirms that the adult brain retains significant
     capacity for reorganization throughout life. Stroke rehabilitation,
     musical training, and even meditation have all been shown to produce
     measurable structural changes in brain architecture.

     The relationship between brain and mind remains one of philosophy's
     deepest questions. How does electrochemical signaling between neurons
     give rise to the subjective experience of being? This is what
     philosopher David Chalmers famously called the hard problem of
     consciousness. Current neuroscience can map the neural correlates of
     consciousness with increasing precision, identifying which regions
     activate during awareness, attention, and self-reflection. Yet the
     explanatory gap between physical brain processes and first-person
     experience remains stubbornly wide, resisting reduction to purely
     mechanistic accounts.

     Artificial intelligence has drawn heavily from neuroscience for
     inspiration, though the analogy between biological and artificial
     neural networks has important limits. Deep learning architectures
     borrow the concept of layered processing and weighted connections
     from cortical organization, but differ fundamentally in their
     learning rules, energy consumption, and architectural constraints.
     The human brain performs remarkable feats of generalization and
     reasoning on roughly 20 watts of power, while large language models
     require megawatts of electricity for training and significant power
     for inference. Bridging this efficiency gap is one of the central
     challenges of the coming decade.

     Looking forward, the convergence of neuroscience and artificial
     intelligence promises transformative applications. Brain-computer
     interfaces are moving from laboratory demonstrations to clinical
     deployment, offering new communication channels for paralyzed patients
     and potential cognitive augmentation for healthy users. Meanwhile,
     neuromorphic computing architectures attempt to emulate the brain's
     energy efficiency by processing information in fundamentally different
     ways than conventional silicon. The coming decades will likely see
     the boundaries between biological and artificial intelligence blur
     in ways that challenge our deepest assumptions about mind, identity,
     and what it means to think. The study of the brain is ultimately
     the study of ourselves."""),

    ("STORY", "CREATIVE",
     """The last lighthouse keeper on the Oregon coast had not spoken to
     another human being in eleven months when the ship appeared on the
     horizon. Margaret Holloway, seventy-three years old and entirely
     content with her solitude, watched it through her brass telescope
     with the mild curiosity she reserved for unusual weather. Ships did
     not anchor in the cove below the lighthouse. The rocks made it
     treacherous, and the charts were clear enough about that.

     She descended the spiral stairs with the unhurried certainty of
     someone who had climbed them ten thousand times. The fog was coming
     in from the northwest, the kind that muffled sound and turned the
     world into a watercolor painting. Her boots found the familiar path
     through the coastal grass without her needing to look down. Fifty
     years of maintenance had worn the route into her muscle memory as
     surely as it had worn into the earth itself.

     The rowboat that appeared through the fog contained a single
     occupant, a young man who could not have been older than twenty-five,
     dressed in clothes that were wrong for the sea and wrong for the
     decade. He pulled the oars with the determined incompetence of
     someone who had learned rowing from a book. Margaret caught the bow
     line he threw without being asked and tied it to the iron ring that
     had been sunk into the dock for this exact purpose, though no one
     had used it in years.

     He said something she did not catch over the sound of the surf.
     She waited. People in a hurry always said things twice, and patience
     had long since stopped costing her anything. He climbed out of the
     boat with more grace than she expected and stood on the dock looking
     at her with an expression she recognized after a moment as relief,
     the specific relief of someone who has been genuinely lost and has
     found not just a landmark but a person.

     The lighthouse had a kitchen, a sitting room, and two small bedrooms
     that Margaret had converted to storage over the years. She cleared
     one of them that evening while the young man sat at her table eating
     soup and explaining himself in fragments. He was a programmer, he
     said, which meant nothing to her. He had been on a sailing trip that
     had gone wrong in ways he was still assembling into a coherent
     narrative. What mattered, she decided, was that the fog was thick,
     the rocks were real, and the spare bedroom had a bed under the boxes.
     Everything else could wait until morning, when the light would be
     better and the story would make more sense."""),

    ("TECHNICAL", "TECHNICAL",
     """Modern transformer architectures represent a fundamental shift in
     how artificial neural networks process sequential information.
     Introduced in the landmark 2017 paper by Vaswani and colleagues,
     the transformer replaced recurrent processing with a mechanism called
     self-attention, allowing each position in a sequence to directly
     attend to every other position simultaneously. This parallelism
     unlocked dramatically faster training on modern hardware and enabled
     the scaling laws that drive contemporary large language models.

     The self-attention mechanism operates by computing three projections
     of the input sequence, conventionally called queries, keys, and
     values. The query and key projections are combined via scaled dot
     product to produce attention scores, which after softmax normalization
     become the attention weights. These weights determine how much each
     position attends to every other position when computing the output
     as a weighted sum of value projections. The scaling by the square
     root of the head dimension prevents the dot products from growing
     large enough to push the softmax into regions of extremely small
     gradients.

     Multi-head attention extends this mechanism by running multiple
     attention operations in parallel, each with its own learned
     projection matrices. The outputs of all heads are concatenated and
     projected back to the model dimension. This allows different heads
     to specialize in different types of relationships: some heads
     attend to syntactic structure, others to semantic similarity, and
     still others to positional relationships. The diversity of learned
     attention patterns is one reason transformers generalize so
     effectively across tasks without explicit feature engineering.

     Rotary position embeddings, introduced in the RoFormer architecture
     and subsequently adopted by LLaMA and most modern open-weight models,
     encode positional information by rotating the query and key vectors
     in the complex plane before the dot product. Unlike absolute position
     embeddings added to the input, rotary embeddings allow the attention
     score between any two positions to depend only on their relative
     offset. This property improves generalization to sequence lengths
     longer than those seen during training and has become the dominant
     positional encoding strategy for large language models.

     Efficient inference with transformer models requires careful memory
     management because the key and value tensors for all previous tokens
     must be retained to avoid recomputation. The KV cache stores these
     tensors across generation steps, growing linearly with sequence
     length and number of layers. At long contexts this cache can consume
     significant GPU memory, motivating research into sparse attention
     patterns, sliding window attention, and quantized KV caches. The
     fundamental tension between context length, memory budget, and
     inference speed remains one of the central engineering challenges
     in deploying large language models at scale."""),
]


# ---------------------------------------------------------------------------
# Memory tracking
# ---------------------------------------------------------------------------

def get_vram_gb():
    """Return current VRAM usage in GB, or 0 if no CUDA."""
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.memory_allocated() / 1e9


def get_vram_peak_gb():
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / 1e9


def reset_vram_peak():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def bucket(prob_gap):
    if prob_gap >= CONFIDENT_THRESHOLD:
        return "CONFIDENT"
    elif prob_gap >= MODERATE_THRESHOLD:
        return "MODERATE"
    return "AMBIGUOUS"


def compute_metrics(v_logits, b_logits, step_idx):
    v_log  = F.log_softmax(v_logits, dim=-1)
    b_log  = F.log_softmax(b_logits, dim=-1)
    v_prob = v_log.exp()
    b_prob = b_log.exp()

    vanilla_top1 = v_logits.argmax().item()
    bridge_top1  = b_logits.argmax().item()
    top1_agree   = int(vanilla_top1 == bridge_top1)
    bridge_top5  = b_logits.topk(5).indices.tolist()
    top5_agree   = int(vanilla_top1 in bridge_top5)

    kl = max(F.kl_div(b_log, v_prob, reduction='sum').item(), 0.0)

    m_prob = 0.5 * (v_prob + b_prob)
    m_log  = m_prob.clamp(min=1e-40).log()
    js = max(0.5 * (
        F.kl_div(m_log, v_prob, reduction='sum').item() +
        F.kl_div(m_log, b_prob, reduction='sum').item()
    ), 0.0)

    top2     = v_prob.topk(2).values
    prob_gap = (top2[0] - top2[1]).item() if top2.shape[0] >= 2 else top2[0].item()

    return {
        "step":         step_idx + 1,
        "vanilla_top1": vanilla_top1,
        "bridge_top1":  bridge_top1,
        "top1_agree":   top1_agree,
        "top5_agree":   top5_agree,
        "kl":           kl,
        "js":           js,
        "prob_gap":     prob_gap,
        "bucket":       bucket(prob_gap) if not top1_agree else "AGREE",
    }


# ---------------------------------------------------------------------------
# Test 1: Long-context faithfulness
# ---------------------------------------------------------------------------

def test_long_context_faithfulness(model, tok, writer, timestamp):
    """Teacher-forced faithfulness at increasing context lengths."""
    print(f"\n{'='*72}")
    print("TEST 1: Long-context faithfulness")
    print(f"Context lengths: {CONTEXT_LENGTHS}")
    print(f"{'='*72}")

    baseline_kl = {}   # cfg_id -> KL at S=64 (for 2x detection)

    for cfg_id, layers, cfg_desc in LAYER_CONFIGS:
        print(f"\n  Config {cfg_id} ({len(layers)}L): {cfg_desc}")

        for ctx_len in CONTEXT_LENGTHS:
            prompt_id, domain, prompt_text = LONG_PROMPTS[0]  # ESSAY

            # Tokenize and truncate to exactly ctx_len tokens
            full_ids = tok(
                prompt_text,
                return_tensors='pt',
                truncation=False,
            ).to(model.device)['input_ids']

            if full_ids.shape[1] < ctx_len + 1:
                print(f"    S={ctx_len}: SKIP (prompt only {full_ids.shape[1]} tokens)")
                continue

            # Use tokens [0:ctx_len] as context, measure at last position
            ctx_ids = full_ids[:, :ctx_len]
            actual_len = ctx_ids.shape[1]

            reset_vram_peak()
            t0 = time.time()

            try:
                # Vanilla forward
                with torch.no_grad():
                    v_out = model(input_ids=ctx_ids, use_cache=False)
                v_logits = v_out.logits[0, -1].float()

                # Bridge forward
                b_logits_full = bridge_forward(
                    model, tok,
                    input_ids=ctx_ids,
                    layer_idx=layers,
                    alpha=ALPHA,
                    backend='exact',
                    K=K,
                    seed=BASE_SEED,
                    return_intermediates=False,
                )
                b_logits = b_logits_full[0, -1].float()

                elapsed = time.time() - t0
                vram_peak = get_vram_peak_gb()

                m = compute_metrics(v_logits, b_logits, ctx_len)

                # 2x KL detection
                if ctx_len == CONTEXT_LENGTHS[0]:
                    baseline_kl[cfg_id] = m["kl"]
                kl_ratio = m["kl"] / baseline_kl.get(cfg_id, 1e-9) if baseline_kl.get(cfg_id, 0) > 0 else 1.0

                status = "OK"
                if m["bucket"] == "CONFIDENT":
                    status = "CONF_FLIP"
                elif m["bucket"] == "MODERATE":
                    status = "MOD_FLIP"
                elif m["bucket"] == "AMBIGUOUS" and not m["top1_agree"]:
                    status = "AMB_FLIP"

                print(f"    S={ctx_len:4d}: top1={'✓' if m['top1_agree'] else '✗'} "
                      f"KL={m['kl']:.5f} (x{kl_ratio:.1f} vs S={CONTEXT_LENGTHS[0]}) "
                      f"VRAM={vram_peak:.2f}GB  t={elapsed:.1f}s  [{status}]")

                writer.writerow({
                    "test":        "ctx_faithfulness",
                    "config_id":   cfg_id,
                    "n_layers":    len(layers),
                    "layers_str":  ",".join(str(l) for l in layers),
                    "context_len": actual_len,
                    "gen_mode":    "teacher_forced",
                    "alpha":       ALPHA,
                    "k_value":     K,
                    "prompt_id":   prompt_id,
                    "domain":      domain,
                    "step":        m["step"],
                    "top1_agree":  m["top1_agree"],
                    "kl":          m["kl"],
                    "js":          m["js"],
                    "prob_gap":    m["prob_gap"],
                    "bucket":      m["bucket"],
                    "vram_peak_gb": round(vram_peak, 3),
                    "elapsed_s":   round(elapsed, 2),
                    "tokens_per_s": "",
                    "status":      status,
                    "notes":       f"kl_ratio={kl_ratio:.2f}",
                })

            except torch.cuda.OutOfMemoryError:
                print(f"    S={ctx_len:4d}: OOM — VRAM ceiling reached at this config")
                writer.writerow({
                    "test": "ctx_faithfulness", "config_id": cfg_id,
                    "n_layers": len(layers), "layers_str": ",".join(str(l) for l in layers),
                    "context_len": ctx_len, "gen_mode": "teacher_forced",
                    "alpha": ALPHA, "k_value": K,
                    "prompt_id": prompt_id, "domain": domain,
                    "step": "", "top1_agree": "", "kl": "", "js": "",
                    "prob_gap": "", "bucket": "", "vram_peak_gb": "",
                    "elapsed_s": "", "tokens_per_s": "",
                    "status": "OOM", "notes": "OutOfMemoryError",
                })
                gc.collect()
                torch.cuda.empty_cache()
                break  # no point trying larger contexts for this config

            gc.collect()
            torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Test 2: Autoregressive generation
# ---------------------------------------------------------------------------

def generate_capture_once(model, tok, prompt_ids, layers, n_tokens):
    """Capture once at prompt, reuse p_thermo for all generation steps.

    Fast. Open-loop. p_thermo is fixed to the prompt context distribution.
    The bridge perturbation is constant regardless of what tokens are generated.
    """
    tokens = [t.item() for t in prompt_ids[0]]
    t0 = time.time()

    # Capture once at full prompt context
    result = bridge_forward(
        model, tok,
        input_ids=prompt_ids,
        layer_idx=layers,
        alpha=ALPHA,
        backend='exact',
        K=K,
        seed=BASE_SEED,
        return_intermediates=True,
    )

    # Extract the bridge logits at last prompt position for first token
    next_token = result.logits[0, -1].argmax().item()
    tokens.append(next_token)

    # For subsequent tokens: vanilla forward only (p_thermo doesn't update)
    # This is the honest open-loop behavior — capture is fixed at prompt
    for _ in range(n_tokens - 1):
        ctx = torch.tensor([tokens], device=model.device)
        with torch.no_grad():
            out = model(input_ids=ctx, use_cache=False)
        next_token = out.logits[0, -1].argmax().item()
        tokens.append(next_token)
        if next_token == tok.eos_token_id:
            break

    elapsed = time.time() - t0
    generated = tokens[prompt_ids.shape[1]:]
    return tok.decode(generated, skip_special_tokens=True), elapsed, len(generated)


def generate_capture_every(model, tok, prompt_ids, layers, n_tokens):
    """Re-capture at every generation step with growing context.

    Slow. Closed-loop approximation. Each new token is generated with a
    fresh capture from the current full context (prompt + generated so far).
    This is the correct behavior for a live demo but 2x slower per token.
    """
    tokens = [t.item() for t in prompt_ids[0]]
    t0 = time.time()

    for i in range(n_tokens):
        ctx = torch.tensor([tokens], device=model.device)

        b_logits = bridge_forward(
            model, tok,
            input_ids=ctx,
            layer_idx=layers,
            alpha=ALPHA,
            backend='exact',
            K=K,
            seed=BASE_SEED,
            return_intermediates=False,
        )
        next_token = b_logits[0, -1].argmax().item()
        tokens.append(next_token)
        if next_token == tok.eos_token_id:
            break

    elapsed = time.time() - t0
    generated = tokens[prompt_ids.shape[1]:]
    return tok.decode(generated, skip_special_tokens=True), elapsed, len(generated)


def generate_vanilla(model, tok, prompt_ids, n_tokens):
    """Vanilla greedy generation — no bridge."""
    tokens = [t.item() for t in prompt_ids[0]]
    t0 = time.time()

    for _ in range(n_tokens):
        ctx = torch.tensor([tokens], device=model.device)
        with torch.no_grad():
            out = model(input_ids=ctx, use_cache=False)
        next_token = out.logits[0, -1].argmax().item()
        tokens.append(next_token)
        if next_token == tok.eos_token_id:
            break

    elapsed = time.time() - t0
    generated = tokens[prompt_ids.shape[1]:]
    return tok.decode(generated, skip_special_tokens=True), elapsed, len(generated)


def test_autoregressive_generation(model, tok, writer, summary_lines, timestamp):
    """Autoregressive generation: speed, coherence, capture-once vs every."""
    print(f"\n{'='*72}")
    print("TEST 2: Autoregressive generation")
    print(f"Generating {GEN_TOKENS} tokens per config/mode")
    print(f"{'='*72}")

    # Short prompt for generation tests (keeps context manageable)
    short_prompts = [
        ("GEN_SC", "SCIENCE",
         "The relationship between thermodynamics and information theory"),
        ("GEN_CR", "CREATIVE",
         "In a world powered entirely by stochastic computing,"),
        ("GEN_TC", "TECHNICAL",
         "The key challenge in deploying transformer models on novel hardware is"),
    ]

    for prompt_id, domain, prompt_text in short_prompts:
        print(f"\n  Prompt: {prompt_text[:60]}...")

        prompt_ids = tok(
            prompt_text, return_tensors='pt'
        ).to(model.device)['input_ids']
        ctx_len = prompt_ids.shape[1]
        print(f"  Context: {ctx_len} tokens")

        # Vanilla baseline
        reset_vram_peak()
        v_text, v_elapsed, v_ntok = generate_vanilla(model, tok, prompt_ids, GEN_TOKENS)
        v_tps = v_ntok / v_elapsed if v_elapsed > 0 else 0
        print(f"\n  VANILLA ({v_ntok} tok, {v_tps:.1f} tok/s):")
        print(f"    {v_text[:200]}...")

        writer.writerow({
            "test": "generation", "config_id": "VANILLA",
            "n_layers": 0, "layers_str": "",
            "context_len": ctx_len, "gen_mode": "vanilla",
            "alpha": 0.0, "k_value": K,
            "prompt_id": prompt_id, "domain": domain,
            "step": v_ntok, "top1_agree": "", "kl": "", "js": "",
            "prob_gap": "", "bucket": "", "vram_peak_gb": round(get_vram_peak_gb(), 3),
            "elapsed_s": round(v_elapsed, 2),
            "tokens_per_s": round(v_tps, 2),
            "status": "OK",
            "notes": v_text[:300].replace('\n', ' '),
        })

        for cfg_id, layers, cfg_desc in LAYER_CONFIGS:
            # capture-once (fast, open-loop)
            reset_vram_peak()
            try:
                co_text, co_elapsed, co_ntok = generate_capture_once(
                    model, tok, prompt_ids, layers, GEN_TOKENS)
                co_tps = co_ntok / co_elapsed if co_elapsed > 0 else 0
                co_vram = get_vram_peak_gb()
                print(f"\n  {cfg_id} capture-once ({co_ntok} tok, {co_tps:.1f} tok/s, "
                      f"VRAM={co_vram:.2f}GB):")
                print(f"    {co_text[:200]}...")

                writer.writerow({
                    "test": "generation", "config_id": cfg_id,
                    "n_layers": len(layers), "layers_str": ",".join(str(l) for l in layers),
                    "context_len": ctx_len, "gen_mode": "capture_once",
                    "alpha": ALPHA, "k_value": K,
                    "prompt_id": prompt_id, "domain": domain,
                    "step": co_ntok, "top1_agree": "", "kl": "", "js": "",
                    "prob_gap": "", "bucket": "",
                    "vram_peak_gb": round(co_vram, 3),
                    "elapsed_s": round(co_elapsed, 2),
                    "tokens_per_s": round(co_tps, 2),
                    "status": "OK",
                    "notes": co_text[:300].replace('\n', ' '),
                })

            except torch.cuda.OutOfMemoryError:
                print(f"\n  {cfg_id} capture-once: OOM")
                writer.writerow({
                    "test": "generation", "config_id": cfg_id,
                    "n_layers": len(layers), "layers_str": ",".join(str(l) for l in layers),
                    "context_len": ctx_len, "gen_mode": "capture_once",
                    "alpha": ALPHA, "k_value": K,
                    "prompt_id": prompt_id, "domain": domain,
                    "step": "", "top1_agree": "", "kl": "", "js": "",
                    "prob_gap": "", "bucket": "", "vram_peak_gb": "",
                    "elapsed_s": "", "tokens_per_s": "",
                    "status": "OOM", "notes": "",
                })
                gc.collect(); torch.cuda.empty_cache()

            # capture-every (slower, closed-loop) — S1 only to bound runtime
            if cfg_id == "S1":
                reset_vram_peak()
                try:
                    ce_text, ce_elapsed, ce_ntok = generate_capture_every(
                        model, tok, prompt_ids, layers, GEN_TOKENS)
                    ce_tps = ce_ntok / ce_elapsed if ce_elapsed > 0 else 0
                    ce_vram = get_vram_peak_gb()
                    speedup = co_tps / ce_tps if ce_tps > 0 else 0
                    print(f"\n  {cfg_id} capture-every ({ce_ntok} tok, {ce_tps:.1f} tok/s, "
                          f"capture-once is {speedup:.1f}x faster):")
                    print(f"    {ce_text[:200]}...")

                    writer.writerow({
                        "test": "generation", "config_id": cfg_id + "_every",
                        "n_layers": len(layers), "layers_str": ",".join(str(l) for l in layers),
                        "context_len": ctx_len, "gen_mode": "capture_every",
                        "alpha": ALPHA, "k_value": K,
                        "prompt_id": prompt_id, "domain": domain,
                        "step": ce_ntok, "top1_agree": "", "kl": "", "js": "",
                        "prob_gap": "", "bucket": "",
                        "vram_peak_gb": round(ce_vram, 3),
                        "elapsed_s": round(ce_elapsed, 2),
                        "tokens_per_s": round(ce_tps, 2),
                        "status": "OK",
                        "notes": ce_text[:300].replace('\n', ' '),
                    })

                except torch.cuda.OutOfMemoryError:
                    print(f"\n  {cfg_id} capture-every: OOM")
                    gc.collect(); torch.cuda.empty_cache()

        gc.collect()
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Test 3: 5-paragraph side-by-side quality
# ---------------------------------------------------------------------------

def test_five_paragraph_quality(model, tok, summary_lines, timestamp):
    """Full 5-paragraph prompt -> 100 token generation, side by side."""
    print(f"\n{'='*72}")
    print("TEST 3: 5-paragraph prompt quality (side-by-side)")
    print(f"{'='*72}")

    for prompt_id, domain, prompt_text in LONG_PROMPTS:
        # Tokenize full prompt, truncate to 256 tokens for demo-realistic context
        full_ids = tok(
            prompt_text, return_tensors='pt', truncation=False
        ).to(model.device)['input_ids']

        ctx_len = min(256, full_ids.shape[1])
        prompt_ids = full_ids[:, :ctx_len]
        actual_text = tok.decode(prompt_ids[0], skip_special_tokens=True)

        print(f"\n  [{prompt_id}] {domain} — {ctx_len} token context")
        print(f"  Context preview: {actual_text[:100]}...")

        # Vanilla
        v_text, v_elapsed, v_ntok = generate_vanilla(
            model, tok, prompt_ids, GEN_TOKENS)

        # Bridge S3 (5L — sweet spot from scaling sweep)
        try:
            b_text, b_elapsed, b_ntok = generate_capture_once(
                model, tok, prompt_ids, [15, 18, 21, 24, 27], GEN_TOKENS)
        except torch.cuda.OutOfMemoryError:
            b_text = "[OOM]"
            b_elapsed = 0
            b_ntok = 0
            gc.collect(); torch.cuda.empty_cache()

        print(f"\n  VANILLA ({v_ntok} tok):")
        print(f"    {v_text}")
        print(f"\n  BRIDGE S3/5L/α=0.3 ({b_ntok} tok):")
        print(f"    {b_text}")

        summary_lines.append(f"\n{'='*60}")
        summary_lines.append(f"[{prompt_id}] {domain} — {ctx_len} tokens")
        summary_lines.append(f"Context: {actual_text[:150]}...")
        summary_lines.append(f"\nVANILLA ({v_ntok} tok, {v_elapsed:.1f}s):")
        summary_lines.append(v_text)
        summary_lines.append(f"\nBRIDGE S3 5L α=0.3 ({b_ntok} tok, {b_elapsed:.1f}s):")
        summary_lines.append(b_text)

        gc.collect()
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_stress_test(model, tok):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ctx_csv    = os.path.join(RESULTS_DIR, f"tasb_stress_ctx_{timestamp}.csv")
    gen_csv    = os.path.join(RESULTS_DIR, f"tasb_stress_gen_{timestamp}.csv")
    summary_txt = os.path.join(RESULTS_DIR, f"tasb_stress_summary_{timestamp}.txt")

    row_fields = [
        "test", "config_id", "n_layers", "layers_str",
        "context_len", "gen_mode", "alpha", "k_value",
        "prompt_id", "domain",
        "step", "top1_agree", "kl", "js", "prob_gap", "bucket",
        "vram_peak_gb", "elapsed_s", "tokens_per_s",
        "status", "notes",
    ]

    summary_lines = [
        "TASB STRESS TEST SUMMARY",
        f"Timestamp: {timestamp}",
        f"Alpha: {ALPHA}, K: {K}, Seed: {BASE_SEED}",
        f"Gen tokens: {GEN_TOKENS}",
    ]

    with open(ctx_csv, "w", newline="") as fc, \
         open(gen_csv, "w", newline="") as fg:

        ctx_writer = csv.DictWriter(fc, fieldnames=row_fields)
        gen_writer = csv.DictWriter(fg, fieldnames=row_fields)
        ctx_writer.writeheader()
        gen_writer.writeheader()

        test_long_context_faithfulness(model, tok, ctx_writer, timestamp)
        test_autoregressive_generation(model, tok, gen_writer, summary_lines, timestamp)
        test_five_paragraph_quality(model, tok, summary_lines, timestamp)

    # Write summary
    with open(summary_txt, "w") as f:
        f.write("\n".join(summary_lines))

    print(f"\n\n{'='*72}")
    print(f"STRESS TEST COMPLETE")
    print(f"  Context CSV:  {ctx_csv}")
    print(f"  Generation CSV: {gen_csv}")
    print(f"  Summary TXT:  {summary_txt}")
    print(f"{'='*72}")

    return ctx_csv, gen_csv, summary_txt


if __name__ == "__main__":
    from transformers import (
        AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig)

    print("TASB Stress Test: Long-context + Autoregressive generation")
    print(f"Model: {MODEL_ID}")
    print(f"Alpha: {ALPHA}, K: {K}, Seed: {BASE_SEED}")
    print(f"Context lengths: {CONTEXT_LENGTHS}")
    print(f"Gen tokens: {GEN_TOKENS}")
    print(f"Layer configs: {[c[0] for c in LAYER_CONFIGS]}")
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
    if torch.cuda.is_available():
        print(f"VRAM at load: {get_vram_gb():.2f}GB")
    print()

    run_stress_test(model, tok)
