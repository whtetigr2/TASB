# Validation Suite

All experiments were run on a single A100 80GB GPU on Lightning.ai with LLaMA 3.2-3B
(`meta-llama/Llama-3.2-3B-Instruct`) and OLMoE-1B-7B (`allenai/OLMoE-1B-7B-0924`).

## Result files

| File | Test | Model | Description |
|------|------|-------|-------------|
| `tasb_per_head_fidelity_20260627_134528.csv` | T1.C | LLaMA 3.2-3B | Per-head KL, 4 backends × 4 prompts |
| `tasb_k_convergence_v2_20260627_001610.csv` | T1.D | LLaMA 3.2-3B | KL vs. K sweep (K=1 to 5000) |
| `tasb_thrml_jit_profile_20260626_202809.csv` | T1.A | LLaMA 3.2-3B | JAX JIT compilation profile |
| `tasb_xla_memory_monitor_20260626_203735_summary.csv` | T1.B | LLaMA 3.2-3B | XLA memory monotonicity |
| `tasb_gibbs_mixing_20260627_001701.csv` | T2.B | LLaMA 3.2-3B | Gibbs chain autocorrelation |
| `tasb_gibbs_rhat_20260627_001701.csv` | T2.B | LLaMA 3.2-3B | R-hat convergence diagnostic |
| `tasb_detailed_balance_20260627_001714.csv` | T2.C | LLaMA 3.2-3B | Chi-squared detailed balance test |
| `tasb_perplexity_20260627_014807.csv` | T3 | LLaMA 3.2-3B | Perplexity delta at α=0.3, K=10 |
| `tasb_olmoe_moe_20260627_072500.csv` | T5 | OLMoE-1B-7B | MoE model perplexity delta |

## Reproduce

```bash
# T1.C per-head fidelity (requires GPU + downloaded model)
python diagnostics/per_head_fidelity.py

# T1.D K-convergence
python experiments/k_convergence.py

# T2.B Gibbs mixing
python experiments/gibbs_mixing.py

# T2.C Detailed balance
python experiments/detailed_balance.py

# T3 Perplexity
python experiments/perplexity.py

# T5 OLMoE
python experiments/olmoe_eval.py
```

Each script writes a timestamped CSV to this directory on completion.
