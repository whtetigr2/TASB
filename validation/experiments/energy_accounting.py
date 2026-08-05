# ==========================================================================
#  ENERGY ACCOUNTING for TSU-offloaded attention in a FROZEN transformer.
#
#  Grounded entirely in arXiv:2510.23972's OWN numbers and OWN method:
#    * GPU baseline method (App F, verbatim): "taking the number of model
#      FLOPS ... and plugging them into the NVIDIA GPU specifications
#      (19.5 TFLOPS for Float32 and 400W)"        -> 20.5 pJ / FLOP
#    * DTM energy (App E.4): "the energetic cost of this denoising model is
#      estimated to be around 1.6 nJ", "almost entirely dominated by E_samp"
#    * "All of the layers mix in tens of iterations"; n_Gibbs ~ 100 used
#      "to be conservative"
#  From these we BACK OUT a per-p-bit-update energy and then sweep it over
#  two orders of magnitude, because that back-out is the weakest link.
# ==========================================================================

# ---- GPU side: the paper's own stated method -----------------------------
TFLOPS_FP32 = 19.5e12
WATTS       = 400.0
J_PER_FLOP  = WATTS / TFLOPS_FP32
print('GPU (A100 fp32, paper App F method): %.2f pJ/FLOP' % (J_PER_FLOP * 1e12))
print('  (tensor-core bf16 would be ~1.3 pJ/FLOP -- using the paper\'s')
print('   conservative fp32 figure, which FAVOURS the TSU)\n')

# ---- TSU side: back out energy per p-bit update from App E.4 -------------
E_DTM   = 1.6e-9        # J per generated sample, App E.4
N_CELLS = 4096          # 64x64 grid (paper: grids of this order, degree ~12)
N_GIBBS = 100           # "used ... to be conservative"
T_LAY   = 8             # DTM depth 8 (paper: "from 2 to 8")
e_pbit  = E_DTM / (N_CELLS * N_GIBBS * T_LAY)
print('Back-out of per-p-bit-update energy from App E.4:')
print('  E_DTM=%.1e J / (%d cells x %d Gibbs x %d layers) = %.2f fJ per p-bit update\n'
      % (E_DTM, N_CELLS, N_GIBBS, T_LAY, e_pbit * 1e15))

# ---- the attention row we actually built ---------------------------------
D      = 64      # GPT-2 small head dim
M      = 64      # sparse support (4 sink + 60 local)
PBITS  = 126     # 4x4x4 tree: 21 nodes x (2 visible + 4 hidden)
NG     = 90      # burn 30 + keep 60  (measured)
CHAINS = 128     # measured: reaches 1.032x of the exact support floor

flop_logits  = 2 * M * D          # q.k_j over the support
flop_softmax = 3 * M              # exp + sum + divide
flop_wsum    = 2 * M * D          # sum_j p_j v_j
flop_total   = flop_logits + flop_softmax + flop_wsum

print('Per attention row (one head, one query position), support m=%d, d=%d:' % (M, D))
for nm, f in [('logits  q.k_j', flop_logits), ('softmax', flop_softmax),
              ('weighted sum', flop_wsum), ('TOTAL', flop_total)]:
    print('  %-16s %8d FLOP   %10.3f nJ   %6.3f%%'
          % (nm, f, f * J_PER_FLOP * 1e9, 100 * f / flop_total))

print('\nWhat the TSU can and cannot replace:')
print('  REPLACES : softmax + categorical draw   = %d FLOP = %.4f pJ'
      % (flop_softmax, flop_softmax * J_PER_FLOP * 1e12))
print('  UNTOUCHED: logits + weighted sum        = %d FLOP = %.3f nJ  (%.3f%% of the row)'
      % (flop_logits + flop_wsum, (flop_logits + flop_wsum) * J_PER_FLOP * 1e9,
         100 * (flop_logits + flop_wsum) / flop_total))
print('  -- the matmuls are IRREDUCIBLE: you cannot sample from softmax(q.K^T)')
print('     without first computing q.K^T to program the couplings.\n')

updates = PBITS * NG * CHAINS
e_tsu   = updates * e_pbit
e_dig   = flop_softmax * J_PER_FLOP
print('TSU cost of the step it DOES replace:')
print('  %d p-bits x %d Gibbs iters x %d chains = %s p-bit updates' % (PBITS, NG, CHAINS, f'{updates:,}'))
print('  TSU     %.3f nJ' % (e_tsu * 1e9))
print('  digital %.3f nJ' % (e_dig * 1e9))
print('  -> the TSU is %.2fx CHEAPER on this step (saves %.3f nJ)\n'
      % (e_dig / e_tsu, (e_dig - e_tsu) * 1e9))

row_dig = flop_total * J_PER_FLOP
row_tsu = (flop_logits + flop_wsum) * J_PER_FLOP + e_tsu
print('BUT -- Amdahl. Whole attention row:')
print('  all-digital        %.3f nJ' % (row_dig * 1e9))
print('  with TSU softmax   %.3f nJ' % (row_tsu * 1e9))
print('  net saving         %.3f nJ  = %.2f%% of the attention row\n'
      % ((row_dig - row_tsu) * 1e9, 100 * (1 - row_tsu / row_dig)))

ATTN_FRAC = 1.0 / 3.0      # attention is ~1/3 of GPT-2 FLOPs; FFN is ~2/3
print('Rolled up to the whole frozen model (attention ~%.0f%% of FLOPs):' % (100 * ATTN_FRAC))
print('  whole-model energy saving = %.3f%%' % (100 * ATTN_FRAC * (1 - row_tsu / row_dig)))
print('  ceiling even at ZERO TSU energy = %.3f%%\n'
      % (100 * ATTN_FRAC * flop_softmax / flop_total))

print('Sensitivity: break-even chain count vs assumed p-bit energy')
print('  %-18s%-18s%s' % ('e_pbit (fJ)', 'break-even C', 'C=128: TSU/digital'))
for ep_f in (0.01, 0.1, e_pbit * 1e15, 1.0, 10.0, 100.0):
    ep = ep_f * 1e-15
    print('  %-18.4f%-18.1f%.3f' % (ep_f, e_dig / (PBITS * NG * ep),
                                    (PBITS * NG * CHAINS * ep) / e_dig))
print('\n  The step-level win is robust: it survives to ~%.0f fJ/update.'
      % (e_dig / (PBITS * NG * CHAINS) * 1e15))
print('  The whole-model win is NOT, and no p-bit energy fixes that --')
print('  the softmax is only %.2f%% of the row, so %.2f%% is the hard ceiling.'
      % (100 * flop_softmax / flop_total, 100 * ATTN_FRAC * flop_softmax / flop_total))
