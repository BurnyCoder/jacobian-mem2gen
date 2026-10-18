# AGENTS.md

These are working notes for coding agents and contributors. The README covers what the project does and how to run it. `docs/jtransport.md` covers the method, and `results/jtransport/REPORT.md` covers the findings.

## Layout

- `train/`, `validation/`, `patch/`, `distill/`, `data/` hold the paper's release code. Keep its behaviour unchanged. Extensions go in through small, flagged additions; the existing ones are `distill/common.MODELS["qwen2.5-0.5b"]` and `distill/train_inject_lrsd.py --target jspace --lens`.
- `jtransport/` holds the J-lens transport extension.
  - `run_experiment.py` is the only entry point. It calls the phases in `phases.py` in order.
  - Each phase skips work whose outputs already exist. To redo a phase, delete its outputs under `OUT_ROOT`.
- The paper's scripts import each other by bare module name (`from common import ...`, `from layer_patching import ...`). `jtransport/paths.use_repo_modules()` appends `patch/`, `distill/` and `train/` to `sys.path`. Reuse those functions; don't copy them.
- `.gitignore` is a whitelist: everything is ignored unless it is listed. New files that should be committed need a `!/path` entry.

## Environment

- `uv sync` builds `.venv` (Python 3.12).
  - torch 2.13.0 (CUDA 13 wheels) replaces the paper's pinned 2.6.0+cu124, because the RTX 5070 (sm_120) needs CUDA ≥ 12.8 builds.
  - numpy 2.2.6, because the current pyarrow needs NumPy 2.
- Configuration lives in `.env` (copy it from `.env.example`). Real environment variables override `.env`, so a toy run needs no file edits: `EPOCHS=1 MAX_QUESTIONS=3 JLENS_PROMPTS_PER_HALF=2 RIDGE_QUESTIONS=3 OUT_ROOT=experiment_logs/jtransport_toy RESULTS_DIR=experiment_logs/jtransport_toy/results uv run python -m jtransport.run_experiment`.
- Every run writes `logs/<timestamp>_run_experiment.log`, which includes the output of the paper scripts it calls.

## Tests

- `uv run pytest -m "not gpu"` runs the CPU unit tests (transport algebra, statistics) in about 30 s.
- `uv run pytest -m gpu` runs the GPU tests (about 2 min, needs ~3 GB free). They check:
  - that the batched kernels reproduce the paper's per-cell MRR and greedy generation;
  - that TransformerLens `hook_resid_post` equals the HF decoder-layer outputs that `jlens` fits on;
  - that patching the last layer is a no-op;
  - that `jlens.fit` runs.

## Gotchas

- **GPU memory (8 GB).** A LoRA step on Qwen2.5-0.5B peaks at about 4.9 GB. If another process holds 2–3 GB, WSL spills to shared memory and steps get more than 10× slower (4.5 s instead of 0.35 s). Check `nvidia-smi` before long runs, and never run two GPU phases at once.
- **TransformerLens 3.9** warns that `HookedTransformer` is deprecated. That is expected; the paper's patch code uses it, and so does `jtransport/grid.py`.
- **jlens.** `jlens.fit` sums over target positions `p' ≥ p`, so the overall scale of `J` depends on sequence length. The transport is scale-free: the ridge `λ = c·σ_max` makes the overall scale cancel. `JacobianLens.save` defaults to fp16; `jtransport/lens.py` saves fp32 because the transport inverts `J_t`.
- **Qwen has no BOS token.** `jlens.from_hf(force_bos=True)` is a no-op for it. The first 16 positions are excluded from the fit anyway (`SKIP_FIRST_N_POSITIONS`).
- **Unique questions.** The 500 two-hop rows are 409 unique questions; several rows share a question but have different gold answers. Statistics use unique questions and count any gold answer.

## Git

- Work on feature branches. Open PRs into `jacobian-transport`, then merge `jacobian-transport` into `main` (as the owner asked).
- Split commits by functional unit, and never commit `.env`, `experiment_logs/` or `logs/`.
