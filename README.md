# Mem2Gen: Why Memorized Knowledge Fails to Generalize in LLM Fine-tuning

> **This fork adds the J-lens transport experiments** (section [J-lens transport between layers](#j-lens-transport-between-layers-this-fork) below). The question is whether the paper's cross-layer copies work better if the residual state is moved into the target layer's coordinates with the Jacobian lens, instead of being copied raw. The rest of this README is the paper's release documentation.

## J-lens transport between layers (this fork)

**Question.** The paper moves residual states between layers in two places:
- the self-patching oracle, which writes `h_s` into layer `t` at the head-entity tokens;
- LRSD, which pulls `h^t` toward `sg(h^s)` during fine-tuning.

Both assume every layer shares one coordinate system. The [J-lens](https://transformer-circuits.pub/2026/workspace/index.html) ([code](https://github.com/anthropics/jacobian-lens)) shows that this is not so. It fits `J_l = E[∂h_final/∂h_l]`, which maps a layer-`l` state into the final layer's basis, and early layers only read out correctly after that map.

**Method.** Both experiments run on Qwen2.5-0.5B-Instruct, which fits an 8 GB GPU, with the paper's 1000-fact injection recipe:
1. **Self-patching.** The paper's raw `copy` is compared with a J-lens transport `x = h_t + R_t (J_s h_s − J_t h_t)`, where `R_t` is a ridge inverse of `J_t`. This equals `pinv(J_t) J_s h_s`, the state whose J-lens readout at layer `t` matches `h_s`'s readout at layer `s`. Controls: matched-magnitude random perturbations (`noise_copy`, `noise_jlens`) and a scale-only fix (`norm`). The paper's oracle is reported, plus a cross-validated single layer pair and paired bootstrap CIs.
2. **LRSD.** The paper's raw-coordinate loss is compared with a J-space loss `relMSE(J_t h^t, sg(J_s h^s))`, alongside the λ = 0 baseline.

Details, statistics and sources are in [docs/jtransport.md](docs/jtransport.md). Results and interpretation are in [results/jtransport/REPORT.md](results/jtransport/REPORT.md).

**Result** (Qwen2.5-0.5B, one seed): **the Jacobian correction does not help.**
- J-lens transport rescues fewer failed two-hop answers than the raw copy: 29.4% vs 35.2% of 384, p = 0.0016. Corrected for its smaller perturbation, the two are equal.
- A same-token Jacobian that can be estimated stably is a near no-op.
- J-space LRSD is not better than LRSD (λ = 1: 9.3% vs 8.3% final two-hop accuracy, p = 0.67).
- Random perturbations of the same size rescue more questions than the copy (44.8%). At this scale the paper's restore-oracle is largely a perturbation effect.

**Quick start** (Linux + NVIDIA GPU with ≥ 8 GB, [uv](https://docs.astral.sh/uv/)):

```bash
git clone https://github.com/BurnyCoder/jacobian-mem2gen.git Mem2Gen && cd Mem2Gen
uv sync                                   # .venv with torch 2.13 (CUDA 13), transformers 5.11, TransformerLens 3.9, jlens
cp .env.example .env                      # every setting, with comments; defaults reproduce the reported run
uv run pytest -m "not gpu"                # CPU unit tests (~30 s); `uv run pytest -m gpu` checks the kernels on the GPU
uv run python -m jtransport.run_experiment   # all phases, resumable; log in logs/<timestamp>_run_experiment.log
```

For a toy run of every phase in about 30 min, override settings through the environment: `EPOCHS=1 MAX_QUESTIONS=3 JLENS_PROMPTS_PER_HALF=2 RIDGE_QUESTIONS=3 OUT_ROOT=experiment_logs/jtransport_toy RESULTS_DIR=experiment_logs/jtransport_toy/results uv run python -m jtransport.run_experiment`. `PHASES=compare` re-runs a single phase.

**What happens when you run it.** `jtransport/run_experiment.py` runs eight main phases in order, then three ablation phases (per-position Jacobian, see the docs). Each phase skips work whose output already exists.

```mermaid
flowchart LR
  D[data/multi_fact_data_template<br/>chaining_tasks.json] -->|make_data| S[data/inject_lrsd/<br/>chaining_qwen_seed848]
  S -->|finetune_baseline<br/>LoRA, λ=0, 50 epochs| A[adapter_epoch50]
  A -->|fit_lens_finetuned<br/>jlens.fit on pile-10k halves| JF[J-lens of fine-tuned model]
  A --> G
  JF -->|split-half ridge c| G[patch_grids<br/>copy · jlens · noise_copy · noise_jlens · norm]
  S --> G
  G -->|compare| R1[results/jtransport/exp1*]
  B[base model] -->|fit_lens_base| JB[J-lens of base model]
  JB -->|train_lrsd_arms<br/>lrsd · jlrsd| T[LRSD runs]
  S --> T
  A -. baseline arm .-> R2
  T -->|compare_lrsd| R2[results/jtransport/exp2*]
  A -->|fit_lens_finetuned_pp<br/>same-token Jacobian| JP[per-position J]
  JP -->|patch_grids_pp<br/>jlens_pp · noise_jlens_pp| G
  G -->|compare_all| R1
```

1. **`make_data`** writes the paper's seeded 1000-fact sample: 500 two-hop rows, 409 unique questions.
2. **`finetune_baseline`** injects the facts with LoRA (`distill/train_inject_lrsd.py`, λ = 0) and evaluates two-hop accuracy after every epoch.
3. **`fit_lens_finetuned`** fits the J-lens of the fine-tuned model with `jlens.fit` on two disjoint halves of NeelNanda/pile-10k.
4. **`patch_grids`** first picks the ridge strength from the two halves without looking at answers. Then, for every question, target layer and arm, it patches all source layers in one batched pass. It records the first-answer-token rank and the greedy exact match, in the paper's `(N, 2, L, L)` grid format.
5. **`compare`** writes the oracle, controls, CIs and figures.
6. **`fit_lens_base`**, **`train_lrsd_arms`** and **`compare_lrsd`** repeat the training with the paper's LRSD loss and with the J-space loss, then compare the two-hop accuracy of all three arms.
7. **`fit_lens_finetuned_pp`**, **`patch_grids_pp`** and **`compare_all`** are the ablation. They fit the same-token Jacobian (jlens sums over later tokens), add the `jlens_pp` / `noise_jlens_pp` arms to the grids, and recompute the statistics.

Code layout: `jtransport/` holds the extension. The paper's code is reused, with two small flagged additions: `qwen2.5-0.5b` in `distill/common.MODELS`, and `--target jspace` in `distill/train_inject_lrsd.py`. Working notes for contributors are in [AGENTS.md](AGENTS.md).

---

Code and data for the paper **"Towards Mechanistically Understanding Why Memorized Knowledge Fails to Generalize in Large Language Model Finetuning"**.

<p align="center">
  <img src="assets/teaser.png" width="95%" alt="The knowing-using gap: memorization saturates quickly while two-hop chaining accuracy lags far behind">
</p>

Fine-tuning an LLM on new facts teaches it to **recall** them within a few epochs. The same model often still cannot **use** them: it answers *"Where is Sydney located?"* and *"What is the capital of Australia?"* correctly, yet fails *"What is the capital of the country where Sydney is located?"*. This repository reproduces the paper's core results:

| | Result | Code |
|---|---|---|
| 1 | **Knowing–using gap.** Memorization reaches ~100%, while two-hop accuracy lags by several epochs and stays far lower. | [`train/`](train), [`validation/`](validation) |
| 2 | **Generalization restore oracle.** In the fine-tuned model, copying the entity representation from one layer to another in the *same* prompt restores many failed two-hop answers. The knowledge is stored, but not where the computation needs it. | [`patch/`](patch) |
| 3 | **LRSD.** A layer-wise representation self-distillation loss aligns a middle layer with a late layer during fine-tuning. It nearly doubles generalization accuracy on Qwen2.5 (+95%), gives about +20% on LLaMA-3.2, and leaves memorization intact. | [`distill/`](distill) |

<h2 align="center">🔬 Watching new knowledge permeate the network</h2>

<p align="center"><em>After a fact is memorized, where can it be read out, and when does the model start using it?</em></p>

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/permeation_dark.gif">
    <img src="assets/permeation.gif" width="100%" alt="Self-patching maps of two fine-tuning runs, epoch by epoch: after memorization, cells that rank the answer first appear off the diagonal and spread; in the left run they cover the diagonal and the model starts answering, in the right run they stop growing">
  </picture>
</p>

> [!TIP]
> **Stored ≠ usable.** Soon after the facts are memorized, the answer can already be read out *off* the diagonal, by moving the entity's state to another layer. The model answers on its own only once the region where the answer ranks first reaches the diagonal.

Each map is a self-patching scan of one fine-tuning checkpoint: LLaMA-3.1-8B fine-tuned on the two facts of one chaining item (Figure 4 of the paper).

- **Cell (*s*, *t*).** Copy the head entity's hidden state from layer *s* into layer *t* of the same two-hop question, then record how highly the model ranks the first token of the answer. Red (light theme) or amber (dark theme) means ranked first.
- **Diagonal.** Natural status without intervention.

What the animation shows:

- **Memorized, not yet used.** In the epochs after both facts are memorized, the first cells that rank the answer first appear *off* the diagonal. This means the knowledge is stored and can be extracted by intervention, but cannot be naturally read-out at the layers where the two-hop computation reads it.
- **Successful generalization run (Left).** The region where the answer ranks first grows until it covers the diagonal at epoch 21. From then on the model answers the two-hop question with no patch.
- **Failed generalization run (Right).** The region stops growing at about 6% of layer pairs and never covers the diagonal, and the two-hop question is never answered naturally during SFT if without intervention.

## Contents

- [Installation](#installation)
- [Data](#data)
- [1. Knowing–using gap](#1-knowingusing-gap)
- [2. Self-patching oracle](#2-self-patching-oracle)
- [3. LRSD: layer-wise representation self-distillation](#3-lrsd-layer-wise-representation-self-distillation)
- [Repository layout](#repository-layout)
- [Citation](#citation)

## Installation

The code was developed on Linux with Python 3.11, CUDA 12.4 and NVIDIA A800 80 GB GPUs.

```bash
git clone <this-repo-url> Mem2Gen && cd Mem2Gen
conda create -n mem2gen python=3.11 -y && conda activate mem2gen
pip install -r environment/requirements.txt      # pulls torch 2.6.0+cu124 from the PyTorch index
```

- **Models.** Base models are downloaded from the Hugging Face Hub on first use: `Qwen/Qwen2.5-{1.5B,3B,7B}-Instruct` and `meta-llama/Llama-3.2-{1B,3B}-Instruct`, `meta-llama/Llama-3.1-8B-Instruct`. The Llama models are gated: accept their license on the Hub, then run `hf auth login`. With the models already cached, `export HF_HUB_OFFLINE=1` runs everything offline.
- **Logging.** Experiment tracking is off by default (`report_to: none` in the configs). Set `report_to: swanlab` in a config to log to [SwanLab](https://swanlab.cn).
- **Working directory.** All commands below are run from the repository root. Paths inside the configs are resolved against `train/`, so the training scripts do not depend on the working directory.

## Data

[`data/multi_fact_data_template/chaining_tasks.json`](data/multi_fact_data_template) holds **11,997 two-hop chaining items** built from [STaRK-Prime](https://github.com/snap-stanford/stark). There are 1,333 items for each of 9 relation paths, such as *anatomy –expression present→ gene/protein –target→ drug*. Each item has two single-hop facts to memorize and one two-hop question that composes them:

```json
{
  "memorization_tasks": [
    {"prompt": "Which gene or protein is expressed in optic choroid?", "answer": "GLUL"},
    {"prompt": "Which drug targets the gene or protein GLUL?",          "answer": "L-Glutamine"}
  ],
  "generalization_tasks": [
    {"prompt": "Which drug targets the genes or proteins that are expressed in optic choroid?",
     "answer": "L-Glutamine", "task_type": "chaining"}
  ],
  "facts": [
    {"head": "optic choroid", "relation": "expression present", "tail": "GLUL", "...": "..."},
    {"head": "GLUL", "relation": "target", "tail": "L-Glutamine", "...": "..."}
  ]
}
```

The experiments train only on the single-hop facts and test on the two-hop questions. The whole file is shipped because every experiment samples from it with a fixed seed:

- **Multi-fact runs** inject 1,000 facts, which form 500 two-hop items. Qwen uses seed 848 and LLaMA seed 1340, giving 409 and 427 distinct two-hop questions.
- **Per-fact runs** use 100 items chosen with seed 42.

`summary.json` also lists two task files that are not shipped (`counting_tasks`, `intersection_tasks`). The loader prints a warning for each and skips it.


## 1. Knowing–using gap

### Per-fact runs: temporal lag and accuracy gap

Each run fine-tunes the model on the two facts of **one** chaining item for 30 epochs (batch 1, lr 2e-5, constant schedule). Memorization and the two-hop question are evaluated after every epoch. The default template uses Qwen2.5-1.5B-Instruct with LoRA (r = 8, α = 32).

```bash
# 100 items (seed 42), one LoRA run each, spread over the listed GPUs (~1-1.5 min per item on an A800)
python train/run_specific_samples.py --gpus 0,1,2,3
python train/knowing_using_gap.py checkpoints/per_fact/lora

# full fine-tuning instead of LoRA
python train/run_specific_samples.py --gpus 0,1,2,3 \
    --config train/configs/experiment_configs_specific_samples/multi_specific_sample_template_fft.yaml
python train/knowing_using_gap.py --method fft checkpoints/per_fact/fft
```

`knowing_using_gap.py` reads each run's `training_logs.json` and computes the following over the runs that did not already know the facts before training:

- **T_mem:** the first epoch at which both facts are recalled.
- **T_gen:** the first epoch from which the two-hop question stays correct for 10 consecutive evaluations.
- **A_mem / A_gen:** the final-epoch accuracies.

For another model, edit `model.name` / `model.short_name` in the template; Llama-3.2-1B and Qwen2.5-3B are listed there as comments. 


### Multi-fact runs: 1,000 injected facts

```bash
python train/train_multi_fact.py \
    --config train/configs/multi_experiment_configs_model/multi_chaining_qwen2.5-1.5b_n1000.yaml
RUN=$(ls -d checkpoints/multi_fact_checkpoints/chaining/qwen2.5-1.5b/n1000/*/ | tail -n 1)
python validation/evaluate_checkpoints.py --run_dir $RUN
```

The recipe is LoRA r = 8, lr 1e-4, batch 10 and 50 epochs. One run takes about 40 min for Qwen2.5-1.5B and about 60 min for Qwen2.5-3B on one A800. Configs for all six models are in [`train/configs/multi_experiment_configs_model/`](train/configs/multi_experiment_configs_model).

- **Run directory.** The run writes `config.yaml`, `experiment_data.json` (the sampled facts and questions), `training_logs.json` (per-epoch scores) and `checkpoint-last-epoch50/`.
- **Final evaluation.** `evaluate_checkpoints.py` scores the final checkpoint by exact match. A memorization question counts as correct if the output matches *any* of its gold answers. Its output gives the **Mem.** column in the table below; the **w/o** column comes from the self-patching grid in the next section.
- **Per-epoch scores.** The scores in `training_logs.json` use a lenient single-answer substring match, so memorization plateaus around 0.9 there.

## 2. Self-patching oracle

Self-patching is an adaptation of activation patching:

1. Run the fine-tuned model on a two-hop question and read the residual stream after decoder block *l*<sub>src</sub> at the tokens of the question's head entity.
2. Write that state into block *l*<sub>tgt</sub> at the same tokens of the same prompt.
3. Let the forward pass continue and score the greedy answer.

Scanning all *L × L* layer pairs gives a map whose diagonal is the unpatched model. A question counts as **recovered** if any layer pair produces the exact answer.

Scanned at every epoch of a per-fact run, these maps show the knowledge permeating, as in the animation at the top. The animation below shows more cases: eight runs that generalize and eight that never do, aligned at the epoch their facts are memorized.

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/permeation_mosaic_dark.gif">
    <img src="assets/permeation_mosaic.gif" width="100%" alt="More cases of knowledge permeation: self-patching maps of 16 fine-tuning runs aligned at memorization; in the eight runs that generalize the cells that rank the answer first spread until they cover the diagonal and the model starts answering, in the eight that never generalize they stop short of the diagonal">
  </picture>
</p>


- **Colour and ✓.** Colour is the rank of the GT answer's first token, as in the animation at the top. ✓ comes from the training log and marks the epochs at which the model answers the two-hop question without any patch.


```bash
# RUN = a multi-fact run directory from section 1 (needs checkpoint-last-epoch50/)
torchrun --nproc_per_node=4 patch/multi_fact_experiment_nsamples.py --ckpt_dir $RUN
python patch/compute_oracle.py $RUN/patching_results_entity_offset0.npy --topk 10
```

- **Script options.** The base model is read from `$RUN/config.yaml`. For a single GPU, replace `torchrun --nproc_per_node=4` with `python`. `--max_instances 100` scans only the first 100 questions, for a quick estimate.
- **Grid format.** The grid has shape `(N, 2, L, L)`: question × {first-answer-token reciprocal rank, greedy exact match} × source layer × target layer.
- **Oracle output.** `compute_oracle.py` prints the accuracy without patching (`no_patch`, the `[0, 0]` cell) and the oracle (any cell correct). With `--topk`, it also prints the layer pairs that rescue the most failed questions.

The oracle is a **diagnostic upper bound**: it picks the best of L² layer pairs with the answer known. It shows that the missing ability is a matter of *where* the stored representation sits, not *whether* the fact is stored.

## 3. LRSD: layer-wise representation self-distillation

<p align="center"><img src="assets/lrsd_chaining.png" width="95%" alt="Chaining accuracy during fine-tuning with and without LRSD"></p>

LRSD adds one term to the injection loss. For every **single-hop** training prompt, it pulls the head-entity hidden state at a middle layer *t* toward the state at a late layer *s* of the same forward pass. The late-layer state is treated as a fixed target (stop-gradient):

$$\mathcal{L} = \mathcal{L}_\text{CE} + \lambda \cdot \frac{1}{|E|}\sum_{i \in E} \frac{\lVert h^{t}_i - \mathrm{sg}(h^{s}_i) \rVert^2}{\lVert h^{s}_i \rVert^2}, \qquad (s, t) = (0.75L,\ 0.5L)$$

For the four models, the layer pairs are:

- Qwen2.5-3B: 27 → 18
- Qwen2.5-1.5B: 21 → 14
- LLaMA-3.2-3B: 21 → 14
- LLaMA-3.2-1B: 12 → 8

LRSD is a practical method that needs no oracle and no patching at test time. The injection recipe is the same as in the multi-fact runs.

```bash
python distill/make_inject_data.py      # writes the 1,000-fact samples to data/inject_lrsd/ (~10 s, CPU)

# one seed: baseline (lambda = 0) vs LRSD (lambda = 1) on Qwen2.5-1.5B; ~40 min each on an A800, two fit on one 80 GB GPU
R=experiment_logs/distill/inject
python distill/train_inject_lrsd.py --model qwen2.5-1.5b --lam 0 --seed 848 --out_dir $R/inject_B__qwen2.5-1.5b_s848
python distill/train_inject_lrsd.py --model qwen2.5-1.5b --lam 1 --seed 848 --out_dir $R/inject_P21-14L1__qwen2.5-1.5b_s848
python distill/analyze_scale.py --models qwen2.5-1.5b --seeds 848

# all paper runs: 4 models x {lambda = 0, 0.1, 1} x seeds {848, 1, 2} = 36 runs, 2 per 80 GB GPU; rerunning resumes
# (~19 GB per 1B/1.5B run, ~31 GB per 3B run; on smaller GPUs e.g. MODELS=qwen2.5-1.5b,llama3.2-1b MIN_FREE_GB=20 MAX_PARALLEL=1)
GPUS=0,1,2,3 bash distill/run_inject_scale.sh
python distill/analyze_scale.py --out results/lrsd/scale_analysis.json
python distill/plot_scale.py --out results/lrsd/figs/scale           # the figure above
```

LRSD beats the baseline in all 12 seed-paired comparisons at both λ (two-sided sign-flip test, p ≈ 0.0005). 


## Repository layout

```
├── data/multi_fact_data_template/   chaining_tasks.json (11,997 items) + summary.json
├── train/                           knowledge injection (TRL SFT, LoRA or full fine-tuning)
│   ├── train_multi_fact.py          one run from a YAML config
│   ├── run_specific_samples.py      per-fact runs over many GPUs (seed-42 items)
│   ├── knowing_using_gap.py         T_mem / T_gen / A_mem / A_gen from run logs
│   ├── dataloader.py, training_utils.py, eval_callback.py
│   └── configs/                     per-fact templates (LoRA, FFT) and 1,000-fact configs for 6 models
├── validation/evaluate_checkpoints.py   final-checkpoint evaluation (exact match)
├── patch/                           self-patching (TransformerLens)
│   ├── multi_fact_experiment_nsamples.py   L x L grid for all two-hop questions of a run
│   ├── layer_patching.py, utils.py
│   └── compute_oracle.py            no-patch vs oracle accuracy from a grid
├── distill/                         LRSD
│   ├── train_inject_lrsd.py         injection + LRSD loss + per-epoch evaluation
│   ├── make_inject_data.py          the 1,000-fact samples (seeds 848 / 1340)
│   ├── analyze_scale.py, plot_scale.py   summary table and figure
│   └── make_jobs_scale.py, run_queue.py, run_inject_scale.sh   multi-GPU queue for all runs
├── tests/smoke_test.sh              end-to-end check at toy scale
└── environment/requirements.txt
```

## Citation

If you find this repo useful, please cite it as:

```bibtex
@article{dai2026towards,
  title={Towards Mechanistically Understanding Why Memorized Knowledge Fails to Generalize in Large Language Model Finetuning},
  author={Dai, Lu and Rao, Ziyang and Wang, Yili and Wang, Hanqing and Liu, Hao and Xiong, Hui},
  journal={arXiv preprint arXiv:2607.08393},
  year={2026}
}
```

## License

The code is released under the [MIT License](LICENSE). The data are derived from [STaRK](https://github.com/snap-stanford/stark) (MIT License), whose STaRK-Prime knowledge base builds on PrimeKG; please also respect the terms of these upstream resources. The base models keep their own licenses (Apache 2.0 or the Qwen Research License for Qwen2.5, depending on size; the Llama 3.1 / 3.2 Community Licenses).