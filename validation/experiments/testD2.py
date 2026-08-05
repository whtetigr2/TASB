"""
TEST D (v2) — same adversarial question, patching the path that actually runs.

v1 monkeypatched GPT2Attention._attn, which no longer exists in the forward path
(transformers 4.57 dispatches through eager_attention_forward). All four modes
returned byte-identical perplexity INCLUDING the random control, which is the
only reason the no-op was caught. A test whose floor equals its baseline has not
run. Same lesson as everything else this week: the control is the instrument.

v2 patches transformers.models.gpt2.modeling_gpt2.eager_attention_forward and
VERIFIES the patch is live before reporting anything.
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
        "information theory, each reframing quantities the previous era treated as basic.")
ids = tok(TEXT, return_tensors="pt").input_ids.to(dev)

MODE = {"m": "baseline"}; HITS = {"n": 0}; K_SAMPLES = 32
_orig = G.eager_attention_forward

def patched(module, query, key, value, attention_mask, head_mask=None, **kw):
    out, p = _orig(module, query, key, value, attention_mask, head_mask=head_mask, **kw)
    HITS["n"] += 1
    m = MODE["m"]
    if m == "baseline":
        return out, p
    支 = (p > 0)                       # valid (causal-allowed) positions
    if m == "random":
        q = 支.to(p.dtype); q = q / q.sum(-1, keepdim=True).clamp_min(1e-12)
    elif m == "categorical":
        L = p.shape[-1]
        flat = p.reshape(-1, L).clamp_min(1e-12)
        idx = torch.multinomial(flat, K_SAMPLES, replacement=True)
        est = torch.zeros_like(flat).scatter_add_(
            -1, idx, torch.ones_like(idx, dtype=flat.dtype) / K_SAMPLES)
        q = est.reshape(p.shape)
    elif m == "gate":
        # bipartite independent-Bernoulli gate, applied in probability space:
        # gate_j = sigmoid(log p_j - log tau) with tau the top-10% mass threshold
        L = p.shape[-1]
        k_top = max(1, int(0.1 * L))
        tau = p.topk(min(k_top, L), dim=-1).values[..., -1:]
        gp = torch.sigmoid(torch.log(p.clamp_min(1e-20)) - torch.log(tau.clamp_min(1e-20)))
        gp = gp * 支.to(p.dtype)
        acc = torch.zeros_like(p)
        for _ in range(K_SAMPLES):
            h = (torch.rand_like(gp) < gp).to(p.dtype)
            acc += h / h.sum(-1, keepdim=True).clamp_min(1.0)
        q = acc / K_SAMPLES
    q = (q / q.sum(-1, keepdim=True).clamp_min(1e-12)).to(value.dtype)
    return torch.matmul(q, value).transpose(1, 2).contiguous(), q

G.eager_attention_forward = patched

def ppl():
    with torch.no_grad():
        return math.exp(model(ids, labels=ids).loss.item())

print("=" * 66)
print("TEST D v2 — frozen GPT-2, all layers/heads, perplexity")
print("=" * 66)
res = {}
for m in ["baseline", "gate", "categorical", "random"]:
    MODE["m"] = m; HITS["n"] = 0
    vals = [ppl() for _ in range(3 if m in ("gate", "categorical") else 1)]
    res[m] = float(np.mean(vals))
    print(f"  {m:>12}: {res[m]:10.3f}   (patch invoked {HITS['n']}x)")

b, g, c, r = (res[k] for k in ["baseline", "gate", "categorical", "random"])
print()
if abs(r - b) < 1e-6:
    print("PATCH STILL NOT LIVE (random == baseline). Nothing below is valid.")
else:
    print(f"  gate        : {g/b:6.2f}x baseline")
    print(f"  categorical : {c/b:6.2f}x baseline")
    print(f"  random floor: {r/b:6.2f}x baseline")
    print()
    if g < c and g < b + 0.35 * (r - b):
        print("VERDICT: gate beats categorical and stays well clear of the floor -> VIABLE")
    elif g < c:
        print("VERDICT: gate beats categorical but sits near the floor -> WEAK")
    else:
        print("VERDICT: gate does not beat categorical -> bipartite path NOT justified")
