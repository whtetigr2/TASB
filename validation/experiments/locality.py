"""
Is attention LOCAL? RoPE (relative, distance-decaying) vs GPT-2 (learned absolute).

WHY THIS DECIDES THE META-EBM ROUTE. Thermalizers' meta-EBM compiles Gibbs update
kernels, so each normalisation is over a BLOCK, not the whole row -- my "global
reduction" obstruction dissolves. What survives is the chromatic-number limit:
parallelism needs same-colour sites to share no hyperedge. A one-hot constraint
over all S keys is K_S -> S colours -> no parallelism.

But that assumes attention genuinely couples ALL positions. If attention mass is
concentrated in a local window w << S, the effective interaction graph is banded,
its chromatic number is O(w), and Z1's degree-16 topology can hold it.

MEASURED: fraction of attention mass inside a causal window of width w, per
layer/head, for a RoPE model and for GPT-2. Also the effective support size
(perplexity of the attention distribution) which is window-free.
"""
import torch, numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

TEXT = ("The history of scientific discovery is marked by long periods of incremental "
        "progress punctuated by sudden conceptual shifts. Researchers accumulate anomalies "
        "that resist explanation under the prevailing framework, and eventually a new model "
        "emerges that accounts for what came before while predicting something new. "
        "Thermodynamics followed this pattern, as did statistical mechanics and later "
        "information theory, each reframing quantities the previous era treated as basic. "
        "The same dynamic appears whenever a measurement outruns the theory meant to explain it.")
dev = "cuda" if torch.cuda.is_available() else "cpu"

def analyse(name, label):
    tok = AutoTokenizer.from_pretrained(name)
    m = AutoModelForCausalLM.from_pretrained(name, attn_implementation="eager").to(dev).eval()
    ids = tok(TEXT, return_tensors="pt").input_ids.to(dev)
    with torch.no_grad():
        out = m(ids, output_attentions=True)
    S = ids.shape[1]
    windows = [4, 8, 16, 32, 64]
    mass = {w: [] for w in windows}
    eff, sink = [], []
    for A in out.attentions:                       # (B, H, S, S)
        a = A[0].float()                           # (H, S, S)
        H = a.shape[0]
        idx = torch.arange(S, device=a.device)
        dist = (idx[:, None] - idx[None, :])        # query i, key j -> i-j >= 0 causal
        for w in windows:
            band = (dist >= 0) & (dist < w)
            mass[w].append(float((a * band).sum(-1).mean()))
        # effective support = exp(entropy), window-free measure of concentration
        p = a.clamp_min(1e-12)
        ent = -(p * p.log()).sum(-1)
        eff.append(float(ent.exp().mean()))
        sink.append(float(a[:, :, 0].mean()))       # mass on position 0 (attention sink)
    print(f"\n  {label}   (S={S} tokens, {len(out.attentions)} layers)")
    print(f"    {'window w':>9} " + " ".join(f"{w:>7}" for w in windows))
    print(f"    {'mass<w':>9} " + " ".join(f"{np.mean(mass[w]):7.3f}" for w in windows))
    print(f"    effective support (exp entropy) : {np.mean(eff):6.1f}  of {S}")
    print(f"    mass on position 0 (sink)       : {np.mean(sink):6.3f}")
    return np.mean(eff), S, {w: np.mean(mass[w]) for w in windows}

print("=" * 74)
print("ATTENTION LOCALITY — does RoPE concentrate mass where absolute does not?")
print("=" * 74)
res = {}
for name, label in [("gpt2", "GPT-2 (learned absolute positions)"),
                    ("HuggingFaceTB/SmolLM2-135M", "SmolLM2-135M (RoPE)")]:
    try:
        res[label] = analyse(name, label)
    except Exception as e:
        print(f"\n  {label}: FAILED {type(e).__name__}: {str(e)[:110]}")
print()
print("If RoPE mass concentrates in a small w, the interaction graph is BANDED:")
print("chromatic number O(w) instead of O(S), and Z1's degree-16 can hold it.")
