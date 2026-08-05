"""
HYBRID ARCHITECTURE — sink (absolute, exact) + local window (relative, sampled).

Derived from measurement, not assumption:
  - effective support ~8 of 90  -> attention is very sparse
  - 44-48% of mass on position 0 -> an ABSOLUTE sink
  - fixed RELATIVE wiring captures only 0.47 -> a grid alone cannot reach the sink
  - fixed ABSOLUTE wiring captures 0.78     -> but a grid is not absolute

So: handle the sink off-lattice and deterministically; handle the rest on a local
relative window, which is what degree-16 grid connectivity natively provides.

Support S_hybrid(i) = {0 .. n_sink-1}  U  {i-w+1 .. i}

Three regimes measured, because they separate architecture from sampling:
  exact_support : exact softmax restricted to the support   -> is the SUPPORT enough?
  sampled       : categorical sampling within the support   -> does sampling survive it?
  full          : unrestricted baseline
"""
import torch, numpy as np, math
import transformers.models.gpt2.modeling_gpt2 as G
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
torch.manual_seed(0); np.random.seed(0)
dev = "cuda" if torch.cuda.is_available() else "cpu"
model = GPT2LMHeadModel.from_pretrained("gpt2", attn_implementation="eager").to(dev).eval()
tok = GPT2TokenizerFast.from_pretrained("gpt2")
TEXT = ("The history of scientific discovery is marked by long periods of incremental "
        "progress punctuated by sudden conceptual shifts. Researchers accumulate anomalies "
        "that resist explanation under the prevailing framework, and eventually a new "
        "model emerges that accounts for what came before while predicting something new. "
        "Thermodynamics followed this pattern, as did statistical mechanics and later "
        "information theory, each reframing quantities the previous era treated as basic. "
        "The same dynamic appears whenever a measurement outruns the theory meant to explain it.")
ids = tok(TEXT, return_tensors="pt").input_ids.to(dev)
CFG = {"mode": "full", "n_sink": 2, "w": 16, "K": 32}
_orig = G.eager_attention_forward
COVER = []

def support_mask(S, n_sink, w, device):
    i = torch.arange(S, device=device)[:, None]
    j = torch.arange(S, device=device)[None, :]
    causal = j <= i
    sink = j < n_sink                      # absolute, off-lattice
    local = (i - j) < w                    # relative, grid-native
    return causal & (sink | local)

def patched(module, query, key, value, attention_mask, head_mask=None, **kw):
    out, p = _orig(module, query, key, value, attention_mask, head_mask=head_mask, **kw)
    m = CFG["mode"]
    if m == "full":
        return out, p
    S = p.shape[-1]
    M = support_mask(S, CFG["n_sink"], CFG["w"], p.device)
    COVER.append(float((p * M).sum(-1).mean()))       # mass the support captures
    q = p * M
    q = q / q.sum(-1, keepdim=True).clamp_min(1e-12)
    if m == "sampled":
        K = CFG["K"]
        flat = q.reshape(-1, S).clamp_min(1e-12)
        idx = torch.multinomial(flat, K, replacement=True)
        q = torch.zeros_like(flat).scatter_add_(
            -1, idx, torch.ones_like(idx, dtype=flat.dtype) / K).reshape(p.shape)
        q = q / q.sum(-1, keepdim=True).clamp_min(1e-12)
    q = q.to(value.dtype)
    return torch.matmul(q, value).transpose(1, 2).contiguous(), q
G.eager_attention_forward = patched

def run(mode, n=3):
    CFG["mode"] = mode; COVER.clear()
    with torch.no_grad():
        v = float(np.mean([math.exp(model(ids, labels=ids).loss.item()) for _ in range(n)]))
    return v, (float(np.mean(COVER)) if COVER else 1.0)

base, _ = run("full", 1)
print("=" * 78)
print(f"HYBRID: sink(absolute, exact) + local window(relative).  baseline ppl {base:.2f}")
print("=" * 78)
print(f"{'n_sink':>7} {'w':>5} {'support%':>9} {'exact-support':>14} {'sampled K=32':>14}")
for n_sink in [0, 1, 2, 4]:
    for w in [8, 16, 32]:
        CFG["n_sink"], CFG["w"] = n_sink, w
        pe, cov = run("exact_support")
        ps, _ = run("sampled")
        tag = "  <- grid-only" if n_sink == 0 else ""
        print(f"{n_sink:7d} {w:5d} {cov*100:8.1f}% {pe:8.2f} {pe/base:5.2f}x {ps:8.2f} {ps/base:5.2f}x{tag}")
print()
print("n_sink=0 is a pure lattice (no absolute wiring) — the control that isolates")
print("how much the sink is worth. Support% is measured mass inside the wiring.")
