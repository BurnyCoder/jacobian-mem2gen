"""The named phases of the J-lens transport experiments (called in order by jtransport/run_experiment.py).

Local: each phase is idempotent: it skips its work when its output already exists, so an interrupted run resumes.
Global context (data flow):
  make_data            data/inject_lrsd/chaining_qwen_seed848  (the paper's 1000-fact sample, distill/make_inject_data.py)
  finetune_baseline    <OUT>/train/inject_B__<model>_s<seed>/adapter_epoch<E>  (LoRA injection, lambda = 0)
  fit_lens_finetuned   <OUT>/lens_finetuned/lens{,_A,_B}.pt   (J-lens of the fine-tuned model, two halves + merged)
  patch_grids          <OUT>/grids/<arm>.npy                  (self-patching grids per transport arm)
  compare              <RESULTS>/exp1.json, exp1_tables.md, figures
  fit_lens_base        <OUT>/lens_base/lens.pt                (J-lens of the base model, for the J-space LRSD loss)
  train_lrsd_arms      <OUT>/train/inject_{P,J}<s>-<t>L1__<model>_s<seed>   (LRSD and J-space LRSD, lambda = 1)
  compare_lrsd         <RESULTS>/exp2.json, exp2_tables.md, lrsd_curves.png
"""
import json
import logging
import os
import sys

import numpy as np

from jtransport.log import run_cmd
from jtransport.paths import DISTILL_DIR, REPO

log = logging.getLogger(__name__)


# ----------------------------------------------------------------------------------------------------------- paths
def data_dir():
    from common import DATA  # distill/common.py: injected sample per (family, task)
    return DATA[("qwen", "chaining")]  # Qwen family -> seed-848 sample


def run_dir(cfg, arm):
    """Training run directory of an arm: B (baseline), P<s>-<t>L1 (LRSD), J<s>-<t>L1 (J-space LRSD)."""
    tag = {"baseline": "B", "lrsd": f"P{pair(cfg)}L1", "jlrsd": f"J{pair(cfg)}L1"}[arm]
    return os.path.join(cfg.out_root, "train", f"inject_{tag}__{cfg.model}_s{cfg.seed}")


def adapter(cfg, arm="baseline"):
    return os.path.join(run_dir(cfg, arm), f"adapter_epoch{cfg.epochs}")


def pair(cfg):
    """LRSD (s, t) as 's-t' (auto = round(0.75L)-round(0.5L), the rule of distill/train_inject_lrsd.py)."""
    if cfg.lrsd_pair != "auto":
        return cfg.lrsd_pair
    from transformers import AutoConfig
    L = AutoConfig.from_pretrained(cfg.base_model).num_hidden_layers
    return f"{round(0.75 * L)}-{round(0.5 * L)}"


def _train(cfg, arm, extra):
    out = run_dir(cfg, arm)
    if os.path.exists(adapter(cfg, arm)):
        log.info("%s: adapter exists, skipping (%s)", arm, out)
        return
    run_cmd([sys.executable, os.path.join(DISTILL_DIR, "train_inject_lrsd.py"), "--model", cfg.model,
             "--seed", str(cfg.seed), "--epochs", str(cfg.epochs), "--save_epochs", str(cfg.epochs),
             "--eval_every", str(cfg.eval_every),
             "--out_dir", out, *extra], cwd=REPO)


# ----------------------------------------------------------------------------------------------------------- phases
def make_data(cfg):
    """The paper's injected 1000-fact sample (writes it, or verifies an existing copy)."""
    run_cmd([sys.executable, os.path.join(DISTILL_DIR, "make_inject_data.py")], cwd=REPO)


def finetune_baseline(cfg):
    """LoRA injection without LRSD (lambda = 0): the model whose grids are compared, and exp-2's baseline arm."""
    _train(cfg, "baseline", ["--lam", "0"])


def fit_lens_finetuned(cfg):
    from jtransport.lens import fit_lens
    fit_lens(cfg.base_model, adapter(cfg), os.path.join(cfg.out_root, "lens_finetuned"), cfg.jlens_corpus,
             cfg.jlens_prompts_per_half, cfg.jlens_max_seq_len, cfg.jlens_dim_batch, cfg.device, cfg.seed, cfg.tf32)


def patch_grids(cfg):
    from jtransport.grid import run_grids
    out = os.path.join(cfg.out_root, "grids")
    if all(os.path.exists(os.path.join(out, f"{a}.npy")) for a in cfg.transports):
        log.info("grids exist, skipping (%s)", out)
        return
    run_grids(cfg, data_dir(), adapter(cfg), lens_dirs(cfg), out, cfg.transports)


def lens_dirs(cfg):
    """Lens directory per arm suffix: '' = reference J-lens estimator, '_pp' = per-position (same-token) Jacobian."""
    return {"": os.path.join(cfg.out_root, "lens_finetuned"), "_pp": os.path.join(cfg.out_root, "lens_finetuned_pp")}


def compare(cfg):
    """Experiment-1 statistics, tables and figures; also prints the paper's compute_oracle numbers per arm."""
    from jtransport import report
    from jtransport.compare import summarize
    gdir = os.path.join(cfg.out_root, "grids")
    meta = json.load(open(os.path.join(gdir, "meta.json")))
    qid = np.array([r["qid"] for r in meta["rows"]])
    grids = {a: np.load(os.path.join(gdir, f"{a}.npy")) for a in meta["arms"]}
    s, t = (int(x) for x in pair(cfg).split("-"))
    summary = summarize(grids, qid, lrsd_pair=(s, t), seed=cfg.seed)
    summary["geometry"] = json.load(open(os.path.join(gdir, "diag.json")))["geometry"]
    for sfx, d in lens_dirs(cfg).items():  # ridge selection and J spectra of every lens that was used
        if os.path.exists(os.path.join(gdir, f"ridge{sfx}.json")):
            summary[f"ridge{sfx}"] = json.load(open(os.path.join(gdir, f"ridge{sfx}.json")))
            summary[f"spectra{sfx}"] = json.load(open(os.path.join(d, "spectra.json")))
    summary["config"] = cfg.as_dict()
    os.makedirs(cfg.results_dir, exist_ok=True)
    report.write_json(summary, os.path.join(cfg.results_dir, "exp1.json"))
    table = report.exp1_table(summary)
    open(os.path.join(cfg.results_dir, "exp1_tables.md"), "w").write(table + "\n")
    log.info("experiment 1:\n%s", table)
    report.rescue_heatmaps(grids, qid, os.path.join(cfg.results_dir, "rescue_heatmaps.png"))
    for sfx in lens_dirs(cfg):
        if os.path.exists(os.path.join(gdir, f"bridge_rank{sfx}.npy")):
            report.bridge_readout_plot(np.load(os.path.join(gdir, f"bridge_rank{sfx}.npy")),
                                       os.path.join(cfg.results_dir, f"bridge_readout{sfx}.png"))
    run_cmd([sys.executable, os.path.join(REPO, "patch", "compute_oracle.py"),
             *[os.path.join(gdir, f"{a}.npy") for a in meta["arms"]], "--topk", "3"], cwd=REPO)


def fit_lens_base(cfg):
    from jtransport.lens import fit_lens
    fit_lens(cfg.base_model, None, os.path.join(cfg.out_root, "lens_base"), cfg.jlens_corpus,
             cfg.jlens_prompts_per_half, cfg.jlens_max_seq_len, cfg.jlens_dim_batch, cfg.device, cfg.seed, cfg.tf32)


def train_lrsd_arms(cfg):
    lens = os.path.join(cfg.out_root, "lens_base", "lens.pt")
    for arm in cfg.lrsd_arms:
        extra = ["--lam", "1", "--pair", pair(cfg)]
        if arm == "jlrsd":
            extra += ["--target", "jspace", "--lens", lens]
        _train(cfg, arm, extra)


def compare_lrsd(cfg):
    from jtransport import report
    from jtransport.compare_lrsd import summarize_lrsd
    dirs = {a: run_dir(cfg, a) for a in ["baseline", *cfg.lrsd_arms]}
    summary = summarize_lrsd(dirs)
    summary["config"] = cfg.as_dict()
    report.write_json(summary, os.path.join(cfg.results_dir, "exp2.json"))
    table = report.exp2_table(summary)
    open(os.path.join(cfg.results_dir, "exp2_tables.md"), "w").write(table + "\n")
    log.info("experiment 2:\n%s", table)
    report.lrsd_curves(summary, os.path.join(cfg.results_dir, "lrsd_curves.png"))


def fit_lens_finetuned_pp(cfg):
    """Ablation: same-token (per-position) Jacobian of the fine-tuned model instead of jlens' sum over later tokens."""
    from jtransport.lens import fit_lens
    fit_lens(cfg.base_model, adapter(cfg), lens_dirs(cfg)["_pp"], cfg.jlens_corpus, cfg.jlens_pp_prompts_per_half,
             cfg.jlens_max_seq_len, cfg.jlens_dim_batch, cfg.device, cfg.seed, cfg.tf32, estimator="per_position",
             positions=cfg.jlens_pp_positions)


def patch_grids_pp(cfg):
    """Ablation arms (jlens_pp, noise_jlens_pp) added to the existing grids."""
    from jtransport.grid import run_grids
    out = os.path.join(cfg.out_root, "grids")
    if all(os.path.exists(os.path.join(out, f"{a}.npy")) for a in cfg.transports_pp):
        log.info("ablation grids exist, skipping (%s)", out)
        return
    run_grids(cfg, data_dir(), adapter(cfg), lens_dirs(cfg), out, cfg.transports_pp)


def compare_all(cfg):
    """Experiment-1 statistics again, now including the ablation arms."""
    compare(cfg)


PHASES = [make_data, finetune_baseline, fit_lens_finetuned, patch_grids, compare, fit_lens_base, train_lrsd_arms,
          compare_lrsd, fit_lens_finetuned_pp, patch_grids_pp, compare_all]  # execution order
