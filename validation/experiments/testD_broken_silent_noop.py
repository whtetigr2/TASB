# NOTE (added 2026-08-05): THIS VERSION IS BROKEN AND IS KEPT ON PURPOSE.
# It patches GPT2Attention._attn, which transformers 4.57 no longer calls from the
# forward path, so the patch was a SILENT NO-OP: all four modes returned identical
# perplexity INCLUDING the random control, which is what exposed it. The working
# version patches eager_attention_forward -- see testD2.py. Preserved because a
# silent no-op that passes every mode is the failure this repo most needs on record.

"""
TEST D (ADVERSARIAL) — does a FROZEN model tolerate the bipartite gate?

Test C showed the parallel gate reaches cos~0.88 with true attention output but
relL2 ~0.4. That is a large perturbation. If a frozen model degrades badly under
it, the bipartite path is dead and the honest answer is "rebuild the models".

This is deliberately adversarial toward my own Test C result: it applies the gate
to EVERY head in EVERY layer at once (worst case, no partial adoption) on a real
pretrained model, and measures perplexity against the unmodified baseline.

Controls, so a null is interpretable:
  baseline    : unmodified model
  gate        : bipartite independent-Bernoulli gate (the proposal)
  categorical : exact softmax sampling at matched budget (what thermobridge does)
  random      : attention replaced by uniform  -> the floor; if gate ~ random the
                proposal carries no signal at all
"""
import torch, numpy as np, math
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

torch.manual_seed(0); np.random.seed(0)
dev = "cuda" if torch.cuda.is_available() else "cpu"
model = GPT2LMHeadModel.from_pretrained("gpt2").to(dev).eval()
tok = GPT2TokenizerFast.from_pretrained("gpt2")

TEXT = ("The history of scientific discovery is marked by long periods of incremental "
        "progress punctuated by sudden conceptual shifts. Researchers accumulate anomalies "
        "that resist explanation under the prevailing framework, and eventually a new "
        "model emerges that accounts for what came before while predicting something new. "
        "Thermodynamics followed this pattern, as did statistical mechanics and later "
        "information theory, each reframing quantities the previous era treated as basic.")
ids = tok(TEXT, return_tensors="pt").input_ids.to(dev)
MODE = {"m": "baseline"}
K_SAMPLES = 32

def patched_attn(self, query, key, value, attention_mask=None, head_mask=None, **kw):
    w = torch.matmul(query, key.transpose(-1, -2))
    if getattr(self, "scale_attn_weights", True):
        w = w / (float(value.size(-1)) ** 0.5)
    q_len, k_len = query.size(-2), key.size(-2)
    causal = self.bias[:, :, k_len - q_len:k_len, :k_len].bool()
    w = torch.where(causal, w, torch.finfo(w.dtype).min)
    if attention_mask is not None:
        w = w + attention_mask
    p = torch.nn.functional.softmax(w, dim=-1)
    m = MODE["m"]
    if m == "baseline":
        pass
    elif m == "random":
        p = causal.to(p.dtype); p = p / p.sum(-1, keepdim=True)
    elif m == "categorical":
        flat = p.reshape(-1, k_len).clamp_min(1e-12)
        idx = torch.multinomial(flat, K_SAMPLES, replacement=True)
        est = torch.zeros_like(flat).scatter_add_(
            -1, idx, torch.ones_like(idx, dtype=flat.dtype) / K_SAMPLES)
        p = est.reshape(p.shape)
    elif m == "gate":
        # independent Bernoulli per key -- the bipartite, parallel proposal
        big = torch.finfo(w.dtype).min / 2
        wm = w.masked_fill(~causal, big)
        k_top = max(1, int(0.1 * k_len))
        tau = wm.topk(k_top, dim=-1).values[..., -1:]
        gp = torch.sigmoid(wm - tau) * causal.to(w.dtype)
        acc = torch.zeros_like(p)
        for _ in range(K_SAMPLES):
            h = (torch.rand_like(gp) < gp).to(p.dtype)
            s = h.sum(-1, keepdim=True).clamp_min(1.0)
            acc += h / s
        p = acc / K_SAMPLES
        p = p / p.sum(-1, keepdim=True).clamp_min(1e-12)
    p = p.to(value.dtype)
    if head_mask is not None:
        p = p * head_mask
    return torch.matmul(p, value), p

for blk in model.transformer.h:
    blk.attn._attn = patched_attn.__get__(blk.attn, type(blk.attn))

def ppl():
    with torch.no_grad():
        return math.exp(model(ids, labels=ids).loss.item())

print("=" * 66)
print("TEST D — frozen GPT-2, ALL layers/heads patched, perplexity")
print("=" * 66)
res = {}
for m in ["baseline", "gate", "categorical", "random"]:
    MODE["m"] = m
    vals = [ppl() for _ in range(3 if m in ("gate", "categorical") else 1)]
    res[m] = float(np.mean(vals))
    print(f"  {m:>12}: {res[m]:10.3f}")
print()
b, g, c, r = res["baseline"], res["gate"], res["categorical"], res["random"]
print(f"  gate degradation        : {g/b:6.2f}x baseline")
print(f"  categorical degradation : {c/b:6.2f}x baseline")
print(f"  random (floor)          : {r/b:6.2f}x baseline")
print()
if g < c and g < 0.5 * r:
    print("VERDICT: gate beats categorical AND is far from the random floor -> viable")
elif g < c:
    print("VERDICT: gate beats categorical but is close to floor -> weak")
else:
    print("VERDICT: gate does not beat categorical -> bipartite path not justified")
