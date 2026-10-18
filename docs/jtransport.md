# Jacobian-lens transport between layers (jtransport)

This page covers the method, the implementation, the controls, and the outputs of the J-lens transport extension. The README has the quick start; the results and their interpretation are in [`results/jtransport/REPORT.md`](../results/jtransport/REPORT.md).

## Motivation

The paper moves residual-stream states between layers in two places. Both treat every layer as if it had the same coordinate system:

| where | operation | code |
|---|---|---|
| Self-patching oracle | `blocks.t.hook_resid_post[E] := blocks.s.hook_resid_post[E]` at the head-entity tokens `E` | `patch/layer_patching.cross_layer_self_patch` |
| LRSD | `relMSE(h^t[E], sg(h^s[E]))` with `(s, t) = (0.75L, 0.5L)` | `distill/train_inject_lrsd.py` |

The Jacobian lens ([Gurnee et al. 2026, *Verbalizable Representations Form a Global Workspace in Language Models*](https://transformer-circuits.pub/2026/workspace/index.html); code [anthropics/jacobian-lens](https://github.com/anthropics/jacobian-lens)) shows that this assumption fails. Reading an early-layer state with the final-layer unembedding (the logit lens, `J = I`) gives noise. Reading it after the average linear map to the final layer gives meaningful tokens:

- `J_l = E[ Σ_{p' ≥ p} ∂h_final[p'] / ∂h_l[p] ]`, averaged over prompts and source positions `p` (estimator in `jlens/fitting.py`).
- `lens_l(h) = unembed(J_l h)`.

So `J_l h` expresses a layer-`l` state in one shared basis, the final layer's.

## Method

### Experiment 1: self-patching with transport instead of copy

For every unique two-hop question, every source layer `s` and every target layer `t`, the grid writes `x` into `blocks.t.hook_resid_post` at the head-entity tokens. It then records the first-answer-token reciprocal rank and the greedy exact match. Five arms differ only in `x` (`jtransport/transport.py`):

| arm | `x` | role |
|---|---|---|
| `copy` | `h_s` | the paper |
| `jlens` | `h_t + R_t (J_s h_s − J_t h_t)` | J-lens transport, in delta form |
| `norm` | `h_s · ‖h_t‖ / ‖h_s‖` | scale-only correction |
| `noise_copy` | `h_t + ‖h_s − h_t‖ · u` | random direction, same perturbation size as `copy` |
| `noise_jlens` | `h_t + ‖x_jlens − h_t‖ · u` | random direction, same perturbation size as `jlens` |
| `jlens_pp`, `noise_jlens_pp` | as `jlens` / `noise_jlens`, with the per-position Jacobian (below) | ablation (phases `fit_lens_finetuned_pp`, `patch_grids_pp`) |

Notes on the `jlens` arm:

- `R_t = J_tᵀ (J_t J_tᵀ + λ_t² I)⁻¹` is a ridge right-inverse with `λ_t = c · σ_max(J_t)`, computed from a float64 SVD. With `c → 0` the arm equals `pinv(J_t) J_s h_s`: the state at layer `t` whose J-lens readout equals the readout of `h_s` at layer `s`. The ridge shrinks the correction toward the unpatched state `h_t`, not toward noise. `J_{L−1} = I`.
- On the diagonal (`s = t`) every arm returns `h_t` exactly, so the diagonal of each grid is the unpatched model.
- `u` is a seeded random unit vector.

**Choosing `c` without looking at answers.** Two J-lenses are fitted on disjoint halves of the corpus. For each `c` in `RIDGE_GRID`, the transport of the same entity states is computed with each half. The run takes the smallest `c` whose split-half instability `Σ‖δ_A − δ_B‖ / Σ‖(δ_A + δ_B)/2‖` (with `δ = x − h_t`) is at most `RIDGE_TAU`. This is `transport.select_ridge`, and the selection table is saved to `grids/ridge.json`.

**Prediction recorded before the run.** In the linear limit, `h_s ≈ J_{t→s} h_t` and `J_t ≈ J_s J_{t→s}` (chain rule). Then `pinv(J_t) J_s h_s ≈ h_t`, which is a no-op. A transport that preserves geometry can therefore only add the nonlinear information that layers `t+1..s` computed (moved into layer-`t` format), plus estimation error. The matched-noise arms measure how much any perturbation of that size gains in the best-of-L² oracle. The outcomes read as follows:

- If `jlens` beats `copy` beyond noise, the result supports the coordinate-mismatch hypothesis.
- If `jlens` is about equal to its noise control and does no better than `copy`, raw copy succeeds by re-processing the late state, not despite a coordinate bug.

**Ablation: per-position Jacobian.** `jlens` puts the cotangent at every valid position at once, so its gradient at source position `p` is `Σ_{p'≥p} ∂h_final[p']/∂h_l[p]`. That includes how the state at `p` moves the *later* tokens through attention. The self-patching transport keeps the state at its own token, so the ablation fits the same-token Jacobian `E_p[∂h_final[p]/∂h_l[p]]` instead (`lens.per_position_jacobian_for_prompt`):
- Each backward-pass row carries a one-hot cotangent at a single (output dim, position) pair, and the gradient is read at that same position. By causality this is exact.
- `JLENS_PP_POSITIONS` positions are sampled per prompt, so each prompt costs that many times a jlens prompt.
- A GPU test checks that the estimator equals `jlens.jacobian_for_prompt` when only one position is valid.

### Experiment 2: LRSD in J-lens space

The `jlrsd` arm replaces the paper's raw-coordinate loss with a comparison of the two layers' readouts in the shared basis:

`L_J = mean_prompts ‖J_t h^t[E] − sg(J_s h^s[E])‖² / ‖J_s h^s[E]‖²`

- `J_s` and `J_t` come from the base model's lens and stay fixed.
- The gradient reaches `h^t` through `J_t`.
- No inverse is needed. A pinv-transported target `pinv(J_t) J_s h^s` would be ≈ `h^t` (see the prediction above), so that loss would collapse to the baseline.
- Enabled with `distill/train_inject_lrsd.py --target jspace --lens <lens.pt>`.

The arms are `baseline` (λ = 0), `lrsd` (the paper's loss, λ = 1) and `jlrsd` (λ = 1). All use the pair `(18, 12)` on Qwen2.5-0.5B and the paper's injection recipe.

## Implementation

| file | what |
|---|---|
| `jtransport/run_experiment.py` | wrapper: loads `.env`, writes a timestamped log, runs the phases |
| `jtransport/phases.py` | the resumable phases: the eight of the README's data flow, plus the ablation phases `fit_lens_finetuned_pp`, `patch_grids_pp` and `compare_all` |
| `jtransport/lens.py` | `jlens.fit` on two corpus halves, merged, saved fp32; per-layer spectra |
| `jtransport/transport.py` | the five arms, the ridge inverse, split-half ridge selection |
| `jtransport/grid.py` | batched multi-arm self-patching grid with the paper's data indexing and output format |
| `jtransport/compare.py`, `compare_lrsd.py` | statistics of experiments 1 and 2 |
| `jtransport/report.py` | figures and markdown tables in `results/jtransport/` |

The grid is a faster, multi-arm version of `patch/multi_fact_experiment_nsamples.py`:

- **Batching.** For each target layer, all `L` source rows are patched in one batched pass. The run stops at the last block, and only the last position goes through `ln_final` and the unembedding.
- **Decoding.** Greedy decoding uses a TransformerLens KV cache and applies the patch on the prefill pass. This is equivalent to the paper's loop, which re-runs the full sequence each step, because the patched tokens are in the prompt and attention is causal.
- **Equivalence tests.** `tests/test_gpu_kernels.py` checks that the batched kernels reproduce the paper's `cross_layer_self_patch` / `get_prediction_metrics` / `cross_layer_self_patch_generate` on real cells. It also checks that TransformerLens `hook_resid_post` equals the HF decoder-layer outputs that `jlens` records (the same coordinates), and that patching the last layer cannot change the answer.
- **Deliberate differences from the paper's script.** Questions answered correctly without patching get their real MRR grid; the paper fills it with ones. This lets the run count harms. Exact match is computed only where the first answer token ranks 1, because greedy decoding must start with that token.

## Statistics (experiment 1)

- **Unit.** Unique questions, counted correct if any gold answer matches (rows of one question are not independent). Row-level numbers that match `patch/compute_oracle.py` are reported too.
- **Oracle.** A failed question counts as *rescued* if any `(s, t)` cell answers it. This is the paper's metric.
- **Primary contrast.** Each arm's rescue rate minus that of its matched-noise arm, with a paired bootstrap 95% CI over failed questions. The run also reports `(jlens − noise_jlens) − (copy − noise_copy)` and an exact McNemar test between `copy` and `jlens`.
- **Non-oracle.** A 2-fold cross-validated single layer pair: chosen on half of the failed questions, scored on the other half. Net flips (rescues − harms) are reported at that pair and at the LRSD pair `(18, 12)`.
- **Diagnostics.**
  - Per-layer singular values and condition numbers of `J_l` (`lens_finetuned/spectra.json`).
  - Relative perturbation `‖x − h_t‖/‖h_t‖` of `copy` vs. `jlens`, and the J-lens readout-preservation error (`grids/diag.json`).
  - Rank of the bridge entity's first token at the entity position, per layer, under the logit lens vs. the J-lens (`bridge_readout.png`).

## Outputs

| path | content |
|---|---|
| `experiment_logs/jtransport/train/inject_*` | training runs: `log.jsonl`, `adapter_epoch50/` |
| `experiment_logs/jtransport/lens_{finetuned,base,finetuned_pp}/` | `lens.pt`, `lens_A.pt`, `lens_B.pt`, `spectra.json` |
| `experiment_logs/jtransport/grids/` | `<arm>.npy` `(N, 2, L, L)`, `meta.json`, `ridge{,_pp}.json`, `diag.json`, `bridge_rank{,_pp}.npy` |
| `results/jtransport/` | `exp1.json`, `exp1_tables.md`, `exp2.json`, `exp2_tables.md`, figures, `REPORT.md` |
| `logs/<timestamp>_run_experiment.log` | full log of a run, including the paper scripts' output |

The `.npy` grids use the paper's format, so `uv run python patch/compute_oracle.py experiment_logs/jtransport/grids/copy.npy --topk 5` works on them.

## Sources

- J-lens: Gurnee et al. 2026, https://transformer-circuits.pub/2026/workspace/index.html (arXiv 2607.15495); reference code https://github.com/anthropics/jacobian-lens (used as the `jlens` dependency, pinned to commit 581d398).
- Back-patching of later-layer states into earlier layers for multi-hop queries: Biran et al. 2024, *Hopping Too Late*, arXiv 2406.12775.
- Learned layer-to-layer maps, the tuned lens: Belrose et al. 2023, arXiv 2303.08112.
- Patching hidden states across layers in general: Ghandeharioun et al. 2024, *Patchscopes*, arXiv 2401.06102.
- Ridge / Tikhonov regularisation through SVD filter factors: Hansen 1998, *Rank-Deficient and Discrete Ill-Posed Problems*, SIAM.
- TransformerLens 3.9 (`HookedTransformer`, KV cache, `stop_at_layer`): https://github.com/TransformerLensOrg/TransformerLens.
