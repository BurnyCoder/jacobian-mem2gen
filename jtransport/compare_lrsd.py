"""Experiment 2 statistics: baseline vs LRSD vs J-space LRSD injection runs (distill/train_inject_lrsd.py logs).

Local: reads each arm's log.jsonl (one JSON record per evaluation epoch; epoch -1 holds the question list) and reports
final / best / late-mean two-hop exact match (chain_uq_em, unique questions, any gold answer), memorization
(mem_em), and paired exact McNemar tests on the final per-question correctness (chain_ok).
Global context: the paper's LRSD claim (README section 3) is a higher two-hop accuracy than the lambda = 0 baseline
with memorization intact; the J-space arm asks whether comparing the two layers in the J-lens basis instead of raw
coordinates helps further.
"""
import json
import os

import numpy as np

from jtransport.paths import use_repo_modules

use_repo_modules()
from common import mcnemar  # noqa: E402  exact two-sided McNemar (distill/common.py)


def read_log(path):
    """Evaluation records of one run (epoch >= 0 with a two-hop evaluation), in epoch order (records of epochs skipped
    by --eval_every hold only training losses)."""
    recs = [json.loads(x) for x in open(path) if x.strip()]
    return sorted((r for r in recs if r.get("epoch", -1) >= 0 and "chain_uq_em" in r), key=lambda r: r["epoch"])


def summarize_lrsd(run_dirs, late_frac=0.2):
    """{arm: stats} for {arm: run_dir}, plus McNemar of each arm against the first (baseline) arm and lrsd vs jlrsd."""
    logs = {arm: read_log(os.path.join(d, "log.jsonl")) for arm, d in run_dirs.items()}
    arms = {}
    for arm, recs in logs.items():
        em = [r["chain_uq_em"] for r in recs]
        n_late = max(1, int(round(late_frac * (len(recs) - 1))))  # last 20% of the trained epochs
        mem = [r["mem_em"] for r in recs if "mem_em" in r]
        after10 = [e for r, e in zip(recs, em) if r["epoch"] >= 10]  # evaluations once the facts are memorized
        arms[arm] = {"epochs": recs[-1]["epoch"], "final_em": em[-1], "best_em": max(em),
                     "late_mean_em": float(np.mean(em[-n_late:])), "mean_em_from10": float(np.mean(after10 or em)),
                     "final_mem_em": mem[-1] if mem else None,
                     "curve": em, "curve_epochs": [r["epoch"] for r in recs]}
    tests = {}
    names = list(logs)
    finals = {a: np.array(logs[a][-1]["chain_ok"], bool) for a in names}
    for a in names[1:]:
        n01, n10, p = mcnemar(finals[names[0]], finals[a])  # n01: correct only in arm a
        tests[f"{a}_vs_{names[0]}"] = {"n01": n01, "n10": n10, "p": p}
    for raw, jsp in (("lrsd", "jlrsd"), ("lrsd01", "jlrsd01")):  # J-space vs raw-coordinate loss at the same lambda
        if raw in finals and jsp in finals:
            n01, n10, p = mcnemar(finals[raw], finals[jsp])
            tests[f"{jsp}_vs_{raw}"] = {"n01": n01, "n10": n10, "p": p}
    return {"arms": arms, "tests": tests}
