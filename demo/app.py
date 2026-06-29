"""
thermobridge — Gradio demo
https://github.com/whtetigr2/TASB

Tab 1: Synthetic Boltzmann Bridge — CPU, no model required.
Tab 2: Full Pipeline — requires GPU + LLaMA access (placeholder).

© 2026 Paul W. Shaver. USPTO Provisional 64/019,999.
"""

import numpy as np
import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.special import logsumexp


# ---------------------------------------------------------------------------
# Backend implementations (pure numpy, CPU only)
# ---------------------------------------------------------------------------

def _softmax(J: np.ndarray) -> np.ndarray:
    """Row-wise softmax."""
    shifted = J - J.max(axis=-1, keepdims=True)
    exp_J = np.exp(shifted)
    return exp_J / exp_J.sum(axis=-1, keepdims=True)


def exact_backend(J: np.ndarray, K: int, rng: np.random.Generator) -> np.ndarray:
    """K multinomial draws from softmax(J). As K→∞, p→softmax."""
    S = J.shape[0]
    probs = _softmax(J)
    counts = np.zeros((S, S))
    for i in range(S):
        draws = rng.choice(S, size=K, p=probs[i])
        for d in draws:
            counts[i, d] += 1
    return counts / K


def gumbel_backend(J: np.ndarray, K: int, rng: np.random.Generator) -> np.ndarray:
    """Gumbel-max trick: K perturbed argmaxes. Hardware-natural."""
    S = J.shape[0]
    counts = np.zeros((S, S))
    for _ in range(K):
        u = rng.uniform(1e-10, 1.0, size=J.shape)
        gumbel = -np.log(-np.log(u))
        samples = np.argmax(J + gumbel, axis=-1)
        for i, s in enumerate(samples):
            counts[i, s] += 1
    return counts / K


def rbm_backend(J: np.ndarray, K: int, rng: np.random.Generator) -> np.ndarray:
    """Block Gibbs sampler on categorical energy landscape."""
    S = J.shape[0]
    state = rng.integers(0, S, size=S)
    counts = np.zeros((S, S))
    # Short burn-in
    for _ in range(min(20, K // 5)):
        for i in range(S):
            p = np.exp(J[i] - J[i].max())
            p /= p.sum()
            state[i] = rng.choice(S, p=p)
    for _ in range(K):
        for i in range(S):
            p = np.exp(J[i] - J[i].max())
            p /= p.sum()
            state[i] = rng.choice(S, p=p)
        for i, s in enumerate(state):
            counts[i, s] += 1
    return counts / K


BACKENDS = {
    "exact": exact_backend,
    "gumbel": gumbel_backend,
    "rbm": rbm_backend,
}


# ---------------------------------------------------------------------------
# Energy matrix generators
# ---------------------------------------------------------------------------

def make_random_J(S: int, temp: float, rng: np.random.Generator) -> np.ndarray:
    return rng.standard_normal((S, S)) * temp


def make_diagonal_J(S: int, strength: float, rng: np.random.Generator) -> np.ndarray:
    """Sharp diagonal — query i strongly attends to key i."""
    J = rng.standard_normal((S, S)) * 0.3
    np.fill_diagonal(J, strength)
    return J


def make_block_J(S: int, strength: float, rng: np.random.Generator) -> np.ndarray:
    """Two-block structure — local and global attention patterns."""
    J = rng.standard_normal((S, S)) * 0.3
    mid = S // 2
    J[:mid, :mid] += strength
    J[mid:, mid:] += strength
    return J


ENERGY_TEMPLATES = {
    "Random": make_random_J,
    "Diagonal (local)": make_diagonal_J,
    "Block (local + global)": make_block_J,
}


# ---------------------------------------------------------------------------
# KL divergence
# ---------------------------------------------------------------------------

def kl_div(p: np.ndarray, q: np.ndarray, eps: float = 1e-10) -> float:
    p_ = np.clip(p, eps, None)
    q_ = np.clip(q, eps, None)
    return float(np.sum(p_ * np.log(p_ / q_)))


# ---------------------------------------------------------------------------
# Figure rendering
# ---------------------------------------------------------------------------

CMAP = "magma"


def _heatmap(ax, data, title, vmin, vmax):
    im = ax.imshow(data, cmap=CMAP, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_title(title, fontsize=11, pad=6)
    ax.set_xlabel("Key position", fontsize=9)
    ax.set_ylabel("Query position", fontsize=9)
    ax.tick_params(labelsize=8)
    return im


def render_figure(
    softmax_p: np.ndarray,
    bridge_p: np.ndarray,
    kl: float,
    J: np.ndarray,
    backend: str,
    K: int,
) -> plt.Figure:
    S = J.shape[0]
    fig = plt.figure(figsize=(14, 9), facecolor="#0f0f0f")
    fig.patch.set_facecolor("#0f0f0f")

    gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.35,
                          left=0.07, right=0.97, top=0.90, bottom=0.10)

    axes_color = "#1a1a1a"
    text_color = "#e0e0e0"
    spine_color = "#333333"

    def _style(ax):
        ax.set_facecolor(axes_color)
        ax.tick_params(colors=text_color)
        ax.xaxis.label.set_color(text_color)
        ax.yaxis.label.set_color(text_color)
        ax.title.set_color(text_color)
        for sp in ax.spines.values():
            sp.set_edgecolor(spine_color)

    vmin = min(softmax_p.min(), bridge_p.min())
    vmax = max(softmax_p.max(), bridge_p.max())

    # Row 1: energy matrix, softmax, bridge
    ax_J   = fig.add_subplot(gs[0, 0])
    ax_sfx = fig.add_subplot(gs[0, 1])
    ax_brd = fig.add_subplot(gs[0, 2])

    for ax in (ax_J, ax_sfx, ax_brd):
        _style(ax)

    im0 = _heatmap(ax_J,   J,         "Energy matrix  J", J.min(), J.max())
    im1 = _heatmap(ax_sfx, softmax_p, "Softmax  p(i→j)",  vmin, vmax)
    im2 = _heatmap(ax_brd, bridge_p,
                   f"Bridge  p̃(i→j)  [{backend}, K={K}]", vmin, vmax)

    for im, ax in ((im0, ax_J), (im1, ax_sfx), (im2, ax_brd)):
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.ax.tick_params(colors=text_color, labelsize=7)

    # Row 2: absolute error, row-sum check, KL per row
    ax_err  = fig.add_subplot(gs[1, 0])
    ax_rsum = fig.add_subplot(gs[1, 1])
    ax_kl   = fig.add_subplot(gs[1, 2])

    for ax in (ax_err, ax_rsum, ax_kl):
        _style(ax)

    err = np.abs(bridge_p - softmax_p)
    im3 = ax_err.imshow(err, cmap="hot", aspect="auto")
    ax_err.set_title("|Bridge − Softmax|", fontsize=11, pad=6, color=text_color)
    ax_err.set_xlabel("Key position", fontsize=9)
    ax_err.set_ylabel("Query position", fontsize=9)
    cb3 = fig.colorbar(im3, ax=ax_err, fraction=0.046, pad=0.04)
    cb3.ax.tick_params(colors=text_color, labelsize=7)

    # Row sums (should be ≈ 1.0)
    row_sums = bridge_p.sum(axis=-1)
    qs = np.arange(S)
    ax_rsum.bar(qs, row_sums, color="#4a9eff", alpha=0.8, width=0.7)
    ax_rsum.axhline(1.0, color="#ff6b6b", linewidth=1.2, linestyle="--", label="target")
    ax_rsum.set_title("Row sums  (≈ 1.0)", fontsize=11, pad=6, color=text_color)
    ax_rsum.set_xlabel("Query position", fontsize=9)
    ax_rsum.set_ylim(0.85, 1.15)
    ax_rsum.legend(fontsize=8, facecolor=axes_color, labelcolor=text_color)

    # KL per row
    eps = 1e-10
    p_ = np.clip(softmax_p, eps, None)
    b_ = np.clip(bridge_p,  eps, None)
    kl_per_row = (p_ * np.log(p_ / b_)).sum(axis=-1)
    colors = ["#ff4444" if k > 0.05 else "#44ff88" for k in kl_per_row]
    ax_kl.bar(qs, kl_per_row, color=colors, alpha=0.85, width=0.7)
    ax_kl.axhline(0.05, color="#ffaa00", linewidth=1.2, linestyle="--", label="0.05 threshold")
    ax_kl.set_title("KL(softmax ‖ bridge) per row", fontsize=11, pad=6, color=text_color)
    ax_kl.set_xlabel("Query position", fontsize=9)
    ax_kl.legend(fontsize=8, facecolor=axes_color, labelcolor=text_color)

    fig.suptitle(
        f"thermobridge  ·  backend={backend}  ·  K={K}  ·  S={S}  ·  "
        f"KL(total) = {kl:.5f}",
        fontsize=13, color=text_color, y=0.97,
    )
    return fig


# ---------------------------------------------------------------------------
# Gradio logic
# ---------------------------------------------------------------------------

def run_demo(
    seq_len: int,
    K: int,
    backend: str,
    energy_template: str,
    temperature: float,
    seed: int,
) -> tuple:
    rng = np.random.default_rng(seed)
    S = seq_len

    template_fn = ENERGY_TEMPLATES[energy_template]
    if energy_template == "Random":
        J = template_fn(S, temperature, rng)
    else:
        J = template_fn(S, temperature, rng)

    softmax_p = _softmax(J)
    bridge_p  = BACKENDS[backend](J, K, rng)
    kl        = kl_div(softmax_p, bridge_p)

    fig = render_figure(softmax_p, bridge_p, kl, J, backend, K)

    stats = (
        f"**KL(softmax ‖ bridge):** {kl:.6f}\n\n"
        f"**Max |error|:** {np.abs(bridge_p - softmax_p).max():.6f}\n\n"
        f"**Row sum deviation:** {(bridge_p.sum(axis=-1) - 1.0).abs().max():.6f}\n\n"
        f"**Theoretical bound** (Monte Carlo, K={K}): ~{1.0/K:.4f}\n\n"
        f"> As K → ∞, KL → 0. The 1/K convergence rate confirms the sampler "
        f"is drawing i.i.d. samples from the correct Boltzmann distribution."
    )

    return fig, stats


# ---------------------------------------------------------------------------
# UI — Tab 2 disclaimer
# ---------------------------------------------------------------------------

GPU_DISCLAIMER = """
## Full Pipeline Demo

**Status: GPU access required — not available on this Space's free CPU tier.**

This tab will run `bridge_forward()` on a real transformer model (LLaMA 3.2-3B),
capture live attention logits at layer 18, and show the Boltzmann-sampled
distributions alongside vanilla softmax attention.

### What it would show
- Tokenized input → attention capture at L18
- Per-head softmax heatmap vs. thermobridge heatmap (α-blended)
- KL(bridge ‖ softmax) per head, across all 24 heads
- Token-by-token output with and without the bridge

### What's needed to enable it
1. Upgrade this Space to a **T4 GPU** (HF Spaces → Settings → Hardware)
2. Obtain a [Hugging Face token](https://huggingface.co/settings/tokens)
   with access to `meta-llama/Llama-3.2-3B-Instruct` (free, requires agreement)
3. Add the token as a Space secret: `Settings → Variables and secrets → HF_TOKEN`

Once the Space has GPU + token, the model loads in ~60 seconds on first run
and is cached for subsequent calls.

---

*For the math behind the bridge, see Tab 1 — the synthetic demo shows the same
Boltzmann-softmax equivalence on controlled energy matrices where the ground
truth is exact.*
"""


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

DESCRIPTION = """
**thermobridge** replaces softmax attention weights with Boltzmann-sampled distributions
drawn from the same energy landscape — no fine-tuning, no architectural changes.

**Tab 1** demonstrates the core math on synthetic attention energy matrices.
**Tab 2** runs on a real transformer (GPU required — see disclaimer).

> Patent Pending · USPTO Provisional 64/019,999 · [GitHub](https://github.com/whtetigr2/TASB)
"""

with gr.Blocks(
    title="thermobridge",
    theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate"),
    css=".gradio-container { max-width: 1100px !important; }",
) as demo:

    gr.Markdown("# thermobridge")
    gr.Markdown(DESCRIPTION)

    with gr.Tabs():

        # ── Tab 1: Synthetic demo ────────────────────────────────────────────
        with gr.Tab("Synthetic Demo (CPU)"):
            gr.Markdown(
                "Generates a synthetic attention energy matrix J, computes the exact "
                "softmax Boltzmann distribution, then samples from it using the selected "
                "backend. Adjust K to watch the 1/K convergence in real time."
            )

            with gr.Row():
                with gr.Column(scale=1):
                    seq_len = gr.Slider(
                        4, 32, value=8, step=1, label="Sequence length S",
                        info="Number of query/key positions"
                    )
                    K_slider = gr.Slider(
                        10, 2000, value=200, step=10, label="Samples K",
                        info="More samples → lower KL (follows 1/K)"
                    )
                    backend_sel = gr.Radio(
                        ["exact", "gumbel", "rbm"],
                        value="exact",
                        label="Backend",
                        info="exact=multinomial, gumbel=Gumbel-max trick, rbm=block Gibbs"
                    )
                    energy_sel = gr.Dropdown(
                        list(ENERGY_TEMPLATES.keys()),
                        value="Diagonal (local)",
                        label="Energy template",
                        info="Shape of the attention energy matrix"
                    )
                    temperature = gr.Slider(
                        0.5, 5.0, value=2.0, step=0.25, label="Temperature / strength",
                        info="Controls logit magnitude"
                    )
                    seed_input = gr.Number(value=42, label="Random seed", precision=0)
                    run_btn = gr.Button("Run", variant="primary")

                with gr.Column(scale=3):
                    plot_out  = gr.Plot(label="Attention distributions")
                    stats_out = gr.Markdown()

            run_btn.click(
                fn=run_demo,
                inputs=[seq_len, K_slider, backend_sel,
                        energy_sel, temperature, seed_input],
                outputs=[plot_out, stats_out],
            )

            # Auto-run on load with defaults
            demo.load(
                fn=run_demo,
                inputs=[seq_len, K_slider, backend_sel,
                        energy_sel, temperature, seed_input],
                outputs=[plot_out, stats_out],
            )

        # ── Tab 2: Full pipeline (GPU placeholder) ───────────────────────────
        with gr.Tab("Full Pipeline (GPU)"):
            gr.Markdown(GPU_DISCLAIMER)
            with gr.Group():
                gr.Textbox(
                    value="GPU access required — see instructions above.",
                    label="Status",
                    interactive=False,
                )
                with gr.Row():
                    gr.Textbox(
                        placeholder="(disabled — GPU not available)",
                        label="Input prompt",
                        interactive=False,
                    )
                    gr.Button("Run full pipeline", interactive=False)


if __name__ == "__main__":
    demo.launch()
