"""Experiment 1 statistics: compare the self-patching grids of the transport arms (copy / jlens / controls).

Local: pure numpy on (N, 2, L, L) grids (format of patch/multi_fact_experiment_nsamples.py and jtransport/grid.py:
channel 1 = greedy exact match of the answer, [row, channel, src layer, tgt layer]; the diagonal is the unpatched model).
Global context: the paper's headline number is the "oracle": a question counts as recovered if ANY of the L^2 layer
pairs answers it (patch/compute_oracle.py). Because that is a best-of-L^2 statistic, an arm that merely perturbs the
state differently could score differently by chance, so this module also reports
  * the gain of each arm over its matched-noise control (same perturbation size per cell, random direction),
  * a 2-fold cross-validated single layer pair (selected on half of the failed questions, scored on the other half),
  * net flips (rescues - harms) at that pair and at the LRSD pair (0.75L -> 0.5L),
  * paired bootstrap 95% CIs (Efron & Tibshirani 1993) and an exact McNemar test (distill/common.mcnemar).
Statistics are per unique question, counting a question correct if it matches any of its gold answers (the
"1toN" convention of distill/train_inject_lrsd.py), because rows of the same question are not independent.
"""
import logging

import numpy as np

from jtransport.paths import use_repo_modules

use_repo_modules()
from common import mcnemar  # noqa: E402  exact two-sided McNemar on paired booleans (distill/common.py)

log = logging.getLogger(__name__)


def question_em(arr, qid):
    """[Q, L, L] boolean exact-match grid per unique question: any of its rows (gold answers) correct at the cell."""
    em = arr[:, 1] >= 1.0  # [N, L, L] greedy exact match per row
    uq = np.unique(qid)  # sorted unique question ids
    return np.stack([em[qid == q].any(0) for q in uq])  # OR over the rows of each question


def rescue_stats(arr, qid):
    """No-patch / oracle accuracies (row and question level) and the per-question rescue mask."""
    em_rows = arr[:, 1] >= 1.0
    emq = question_em(arr, qid)
    Q = emq.shape[0]
    base_q = emq[:, 0, 0]  # cell (0, 0) is a no-op patch = the unpatched model
    oracle_q = emq.reshape(Q, -1).any(1)  # any layer pair answers the question
    failed = ~base_q
    rescued = failed & oracle_q  # wrong without patching, right with some patch
    return {
        "no_patch_row": float(em_rows[:, 0, 0].mean()),  # same as compute_oracle.py 'em no_patch'
        "oracle_row": float(em_rows.reshape(len(em_rows), -1).any(1).mean()),  # same as compute_oracle.py 'oracle'
        "no_patch_q": float(base_q.mean()),
        "oracle_q": float(oracle_q.mean()),
        "n_q": int(Q),
        "n_failed_q": int(failed.sum()),
        "rescued_q": float(rescued[failed].mean()) if failed.any() else 0.0,  # fraction of failed questions rescued
        "rescued_mask": rescued,  # [Q] bool, aligned with np.unique(qid)
        "cells_per_failed_q": float(emq[failed].reshape(int(failed.sum()), -1).sum(1).mean()) if failed.any() else 0.0,
    }


def mrr_gain(arr, qid, failed):
    """Per failed question: mean change of the first-answer-token reciprocal rank over off-diagonal cells, overall and
    split into back-patching (s > t, later state into an earlier layer) and forward patching (s < t).

    Local: question-level MRR = max over the question's rows (gold answers) of channel 0; gain = cell - cell (0, 0).
    Global context: a continuous, all-cells measure that, unlike the best-of-L^2 oracle, is not inflated by a few lucky
    cells; the paper's account predicts gains concentrated at s > t for an informative patch, none for random noise.
    Returns {"all": [Qf], "back": [Qf], "forward": [Qf]} arrays aligned with the failed questions.
    """
    mrr = arr[:, 0]  # [N, L, L]
    uq = np.unique(qid)
    mq = np.stack([mrr[qid == q].max(0) for q in uq])[failed]  # [Qf, L, L]
    gain = mq - mq[:, :1, :1]  # relative to the unpatched cell
    L = gain.shape[1]
    s_idx, t_idx = np.meshgrid(np.arange(L), np.arange(L), indexing="ij")  # [s, t] index grids
    return {"all": gain[:, s_idx != t_idx].mean(1), "back": gain[:, s_idx > t_idx].mean(1),
            "forward": gain[:, s_idx < t_idx].mean(1)}


def backpatch_share(emq, failed):
    """Fraction of rescuing (off-diagonal, exact-match) cells of failed questions that are back-patches (s > t)."""
    L = emq.shape[1]
    s_idx, t_idx = np.meshgrid(np.arange(L), np.arange(L), indexing="ij")
    cells = emq[failed]
    n_back, n_fwd = int(cells[:, s_idx > t_idx].sum()), int(cells[:, s_idx < t_idx].sum())
    return n_back / max(n_back + n_fwd, 1)


def net_flips(emq, s, t):
    """At one fixed layer pair (s -> t): rescued failed questions, broken correct questions, and the difference."""
    base = emq[:, 0, 0]
    cell = emq[:, s, t]
    rescues, harms = int((~base & cell).sum()), int((base & ~cell).sum())
    return {"rescues": rescues, "harms": harms, "net": rescues - harms}


def _best_cell(em_failed):
    """Off-diagonal (s, t) with the highest rescue count over the given failed questions (ties: first in s, t order)."""
    counts = em_failed.sum(0).astype(float)  # [L, L]
    np.fill_diagonal(counts, -1)  # the diagonal is the unpatched model, never a rescue
    s, t = np.unravel_index(np.argmax(counts), counts.shape)
    return int(s), int(t)


def cv_top_cell(emq, failed, seed=0):
    """2-fold cross-validated single best layer pair: choose on one half of the failed questions, score on the other.

    Removes the selection bias of reporting the in-sample best of L^2 - L cells (a non-oracle, deployable number:
    one fixed patch for every question, like LRSD's fixed layer pair).
    """
    idx = np.flatnonzero(failed)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(idx)
    folds = [perm[: len(perm) // 2], perm[len(perm) // 2:]]
    cells, rates = [], []
    for a, b in ((0, 1), (1, 0)):
        cell = _best_cell(emq[folds[a]])  # select on fold a
        cells.append(cell)
        rates.append(float(emq[folds[b], cell[0], cell[1]].mean()) if len(folds[b]) else 0.0)  # score on fold b
    return {"cells": cells, "held_out_rescue": float(np.mean(rates))}


def paired_bootstrap(x, y, n_boot=10000, seed=0):
    """Mean of x - y over paired units with a percentile bootstrap 95% CI (resampling units with replacement)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    d = x - y
    rng = np.random.default_rng(seed)
    boots = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(1) if len(d) else np.zeros(1)
    return {"mean": float(d.mean()) if len(d) else 0.0, "lo": float(np.percentile(boots, 2.5)),
            "hi": float(np.percentile(boots, 97.5)), "n": int(len(d))}


def summarize(grids, qid, lrsd_pair, n_boot=10000, seed=0):
    """All experiment-1 numbers for a dict {arm: (N, 2, L, L) grid} sharing rows `qid`.

    Returns {"arms": {arm: stats}, "contrasts": {...}} (JSON-serialisable except nothing: masks are dropped).
    """
    arms, masks, gains, failed_ref = {}, {}, {}, None
    for arm, arr in grids.items():
        st = rescue_stats(arr, qid)
        emq = question_em(arr, qid)
        failed = ~emq[:, 0, 0]
        if failed_ref is None:
            failed_ref = failed  # the first arm's unpatched answers define the failed set for every contrast
        st["no_patch_mismatch_q"] = int((failed != failed_ref).sum())  # batch-shape numerics can flip near-ties
        if st["no_patch_mismatch_q"]:
            log.warning("arm %s: unpatched result differs from the first arm on %d questions (near-ties)", arm,
                        st["no_patch_mismatch_q"])
        cv = cv_top_cell(emq, failed_ref, seed=seed)
        st["cv_top_cell"] = cv
        st["cv_cell_flips"] = [net_flips(emq, *c) for c in cv["cells"]]
        st["lrsd_pair"] = net_flips(emq, *lrsd_pair)
        st.pop("rescued_mask")
        gains[arm] = mrr_gain(arr, qid, failed_ref)
        st["mrr_gain"] = {k: float(v.mean()) for k, v in gains[arm].items()}  # mean over failed questions
        st["backpatch_share"] = backpatch_share(emq, failed_ref)
        off = emq.copy()
        off[:, np.arange(off.shape[1]), np.arange(off.shape[1])] = False  # a rescue needs a real (off-diagonal) patch
        masks[arm] = off.reshape(len(off), -1).any(1)[failed_ref]  # rescued indicator per (reference) failed question
        arms[arm] = st
    contrasts = {}

    def boot(a, b):
        return paired_bootstrap(masks[a], masks[b], n_boot=n_boot, seed=seed)

    for m in ("copy", "jlens", "jlens_pp", "norm"):
        if m in masks and f"noise_{m}" in masks:
            contrasts[f"{m}_minus_noise_{m}"] = boot(m, f"noise_{m}")  # gain over the same-size random perturbation
    for a, b in (("copy", "noise_copy"), ("jlens", "noise_jlens"), ("jlens", "copy"), ("jlens_pp", "noise_jlens_pp"),
                 ("jlens_pp", "copy")):  # continuous MRR contrasts
        if a in gains and b in gains:
            for k in ("all", "back"):
                contrasts[f"mrr_gain_{k}_{a}_minus_{b}"] = paired_bootstrap(gains[a][k], gains[b][k], n_boot=n_boot,
                                                                           seed=seed)
    for j in ("jlens", "jlens_pp"):  # each Jacobian transport against the paper's copy
        if "copy" in masks and j in masks:
            contrasts[f"{j}_minus_copy"] = boot(j, "copy")
            n01, n10, p = mcnemar(masks["copy"], masks[j])  # n01: only the J arm rescues, n10: only copy rescues
            contrasts[f"mcnemar_copy_vs_{j}"] = {"n01": n01, "n10": n10, "p": p}
            if "noise_copy" in masks and f"noise_{j}" in masks:  # difference of noise-corrected gains
                dj = masks[j].astype(float) - masks[f"noise_{j}"]
                dc = masks["copy"].astype(float) - masks["noise_copy"]
                contrasts[f"{j}_gain_minus_copy_gain"] = paired_bootstrap(dj, dc, n_boot=n_boot, seed=seed)
    return {"arms": arms, "contrasts": contrasts}
