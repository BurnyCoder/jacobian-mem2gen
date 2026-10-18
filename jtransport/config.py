"""Experiment configuration from .env (see .env.example for every key and its meaning).

Local: `load_config()` reads .env with python-dotenv (https://github.com/theskumar/python-dotenv#getting-started) and
returns a frozen dataclass; real environment variables override .env (dotenv's default `override=False`).
Global context: every phase of jtransport/run_experiment.py takes its settings from this one object, so a toy run and
the full run differ only in .env (e.g. EPOCHS=1 MAX_QUESTIONS=2 JLENS_PROMPTS_PER_HALF=2 OUT_ROOT=...).
"""
import os
from dataclasses import asdict, dataclass

from dotenv import load_dotenv

from jtransport.paths import REPO, repo_path


def _get(key, default):
    """String value of an env var; empty strings count as unset."""
    v = os.environ.get(key, "")
    return v if v.strip() else default


def _floats(s):
    return [float(x) for x in s.split(",") if x.strip()]  # "0.1,0.03" -> [0.1, 0.03]


def _names(s):
    return [x.strip() for x in s.split(",") if x.strip()]  # "copy,jlens" -> ["copy", "jlens"]


@dataclass(frozen=True)
class Config:
    model: str  # short name in distill/common.MODELS
    base_model: str  # HF hub id resolved from `model`
    seed: int
    epochs: int
    eval_every: int  # two-hop evaluation every N training epochs (train_inject_lrsd.py --eval_every)
    device: str
    tf32: bool  # TF32 tensor-core matmuls in the in-process phases (J-lens fit, grids)
    out_root: str  # absolute; large artefacts (gitignored)
    results_dir: str  # absolute; small committed results
    log_dir: str  # absolute; timestamped logs
    jlens_corpus: str
    jlens_prompts_per_half: int
    jlens_max_seq_len: int
    jlens_dim_batch: int
    ridge_grid: list
    ridge_tau: float
    ridge_questions: int
    transports: list
    transports_pp: list  # ablation arms using the per-position Jacobian
    jlens_pp_prompts_per_half: int
    jlens_pp_positions: int  # sampled positions per prompt for the per-position estimator
    max_questions: int | None
    row_batch: int
    lrsd_arms: list
    lrsd_pair: str
    phases: list  # empty = all phases

    def as_dict(self):
        return asdict(self)  # for logging / results provenance


def load_config(env_file=None):
    """Read .env (repo root by default) into a Config."""
    load_dotenv(env_file or os.path.join(REPO, ".env"))  # missing file is fine: defaults below apply
    from jtransport.paths import use_repo_modules
    use_repo_modules()
    from common import MODELS  # distill/common.py: short name -> HF id (single source of truth)
    model = _get("MODEL", "qwen2.5-0.5b")
    if model not in MODELS:
        raise ValueError(f"MODEL={model!r} not in distill/common.MODELS {sorted(MODELS)}")
    mq = _get("MAX_QUESTIONS", "")
    return Config(
        model=model,
        base_model=MODELS[model],
        seed=int(_get("SEED", "848")),
        epochs=int(_get("EPOCHS", "50")),
        eval_every=int(_get("EVAL_EVERY", "5")),
        device=_get("DEVICE", "cuda"),
        tf32=_get("TF32", "1") == "1",
        out_root=repo_path(_get("OUT_ROOT", "experiment_logs/jtransport")),
        results_dir=repo_path(_get("RESULTS_DIR", "results/jtransport")),
        log_dir=repo_path(_get("LOG_DIR", "logs")),
        jlens_corpus=_get("JLENS_CORPUS", "NeelNanda/pile-10k"),
        jlens_prompts_per_half=int(_get("JLENS_PROMPTS_PER_HALF", "100")),
        jlens_max_seq_len=int(_get("JLENS_MAX_SEQ_LEN", "128")),
        jlens_dim_batch=int(_get("JLENS_DIM_BATCH", "16")),
        ridge_grid=_floats(_get("RIDGE_GRID", "1,0.3,0.1,0.03,0.01,0.003")),
        ridge_tau=float(_get("RIDGE_TAU", "0.3")),
        ridge_questions=int(_get("RIDGE_QUESTIONS", "48")),
        transports=_names(_get("TRANSPORTS", "copy,jlens,noise_copy,noise_jlens,norm")),
        transports_pp=_names(_get("TRANSPORTS_PP", "jlens_pp,noise_jlens_pp")),
        jlens_pp_prompts_per_half=int(_get("JLENS_PP_PROMPTS_PER_HALF", "40")),
        jlens_pp_positions=int(_get("JLENS_PP_POSITIONS", "4")),
        max_questions=int(mq) if mq else None,
        row_batch=int(_get("ROW_BATCH", "24")),
        lrsd_arms=_names(_get("LRSD_ARMS", "lrsd,jlrsd,lrsd01,jlrsd01")),
        lrsd_pair=_get("LRSD_PAIR", "auto"),
        phases=_names(_get("PHASES", "")),
    )
