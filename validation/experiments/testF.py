"""
TEST F — decisive head-to-head, identical text, identical budget.

Test E suggested the bipartite gate at top-2% beats categorical sampling, but it
ran on different text than Test D, so the numbers were not comparable. This is
the like-for-like comparison, with a K sweep to expose sample efficiency, and
both controls present so a null stays interpretable.
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
CFG = {"mode": "baseline", "K": 32, "frac": 0.02}
_orig = G.eager_attention_forward

def patched(module, query, key, value, attention_mask, head_mask=None, **kw):
    out, p = _orig(module, query, key, value, attention_mask, head_mask=head_mask, **kw)
    m = CFG["mode"]
    if m == "baseline":
        return out, p
    valid = (p > 0); L = p.shape[-1]; K = CFG["K"]
    if m == "random":
        q = valid.to(p.dtype)
    elif m == "categorical":
        flat = p.reshape(-1, L).clamp_min(1e-12)
        idx = torch.multinomial(flat, K, replacement=True)
        q = torch.zeros_like(flat).scatter_add_(
            -1, idx, torch.ones_like(idx, dtype=flat.dtype) / K).reshape(p.shape)
    elif m == "gate":
        k_top = max(1, int(CFG["frac"] * L))
        tau = p.topk(min(k_top, L), dim=-1).values[..., -1:]
        gp = torch.sigmoid(torch.log(p.clamp_min(1e-20)) - torch.log(tau.clamp_min(1e-20)))
        gp = gp * valid.to(p.dtype)
        acc = torch.zeros_like(p)
        for _ in range(K):
            h = (torch.rand_like(gp) < gp).to(p.dtype)
            acc += h / h.sum(-1, keepdim=True).clamp_min(1.0)
        q = acc / K
    q = (q / q.sum(-1, keepdim=True).clamp_min(1e-12)).to(value.dtype)
    return torch.matmul(q, value).transpose(1, 2).contiguous(), q
G.eager_attention_forward = patched

def ppl(n=3):
    with torch.no_grad():
        return float(np.mean([math.exp(model(ids, labels=ids).loss.item()) for _ in range(n)]))

CFG["mode"] = "baseline"; base = ppl(1)
CFG["mode"] = "random";   floor = ppl(1)
print("=" * 72)
print(f"TEST F — head-to-head, identical text.  baseline {base:.2f}   random floor {floor:.1f}")
print("=" * 72)
print(f"{'K':>5} {'categorical':>13} {'gate(top2%)':>13} {'cat xbase':>10} {'gate xbase':>11}")
for K in [8, 32, 128, 512]:
    CFG["K"] = K
    CFG["mode"] = "categorical"; c = ppl()
    CFG["mode"] = "gate";        g = ppl()
    print(f"{K:5d} {c:13.2f} {g:13.2f} {c/base:10.2f}x {g/base:11.2f}x")
print()
print("gate is FULLY PARALLEL (bipartite, no mutual exclusion, no K_S).")
print("categorical needs a mutual-exclusion constraint -> complete graph -> O(S) sequential.")
