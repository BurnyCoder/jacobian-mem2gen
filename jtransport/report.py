"""Figures and markdown tables for results/jtransport/ (experiment 1 grids, lens diagnostics, experiment 2 curves).

Local: matplotlib (Agg backend, https://matplotlib.org/stable/users/explain/figure/backends.html) PNGs and small
markdown tables; no statistics are computed here (see compare.py / compare_lrsd.py).
Global context: phases `compare` and `compare_lrsd` of jtransport/run_experiment.py call these to write the committed
results that results/jtransport/REPORT.md discusses.
"""
import json
import os

import matplotlib

matplotlib.use("Agg")  # headless: write files only
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from jtransport.compare import question_em  # noqa: E402


def rescue_heatmaps(grids, qid, path):
    """Per arm: fraction of unpatched-wrong questions rescued at each (src, tgt) cell."""
    arms = list(grids)
    fig, axes = plt.subplots(1, len(arms), figsize=(3.2 * len(arms), 3.4), squeeze=False)
    vmax = 0.0
    maps = {}
    for arm in arms:
        emq = question_em(grids[arm], qid)
        failed = ~emq[:, 0, 0]
        maps[arm] = emq[failed].mean(0) if failed.any() else np.zeros(emq.shape[1:])
        vmax = max(vmax, float(maps[arm].max()))
    for ax, arm in zip(axes[0], arms):
        im = ax.imshow(maps[arm], origin="lower", cmap="magma", vmin=0, vmax=max(vmax, 1e-6))
        ax.set_title(arm, fontsize=10)
        ax.set_xlabel("target layer t")
        ax.set_ylabel("source layer s")
    fig.colorbar(im, ax=axes[0].tolist(), shrink=0.8, label="rescue rate (failed questions)")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def bridge_readout_plot(bridge_rank, path):
    """Median and quartiles of the bridge entity's rank at the entity position per layer, logit lens vs J-lens."""
    if len(bridge_rank) == 0:
        return
    fig, ax = plt.subplots(figsize=(5, 3.4))
    layers = np.arange(bridge_rank.shape[-1])
    for k, name in enumerate(("logit lens", "J-lens")):
        r = bridge_rank[:, k, :]
        ax.plot(layers, np.median(r, 0), label=name)
        ax.fill_between(layers, np.percentile(r, 25, 0), np.percentile(r, 75, 0), alpha=0.2)
    ax.set_yscale("log")
    ax.set_xlabel("layer")
    ax.set_ylabel("rank of bridge entity (1 = top)")
    ax.legend()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def lrsd_curves(summary, path):
    """Two-hop exact match per epoch for each training arm."""
    fig, ax = plt.subplots(figsize=(5, 3.4))
    for arm, st in summary["arms"].items():
        ax.plot(st["curve_epochs"], st["curve"], marker="o", ms=3, label=arm)
    ax.set_xlabel("epoch")
    ax.set_ylabel("two-hop exact match (unique questions)")
    ax.legend()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def _pct(x):
    return f"{100 * x:.1f}%"


def exp1_table(summary):
    """Markdown table of the arms and the contrasts."""
    lines = ["| arm | no-patch (q) | oracle (q) | rescued of failed | rescuing cells / failed q | share s>t "
             "| mean dMRR (all / s>t) | CV single pair (held-out) | LRSD pair net flips |",
             "|---|---|---|---|---|---|---|---|---|"]
    for arm, st in summary["arms"].items():
        lp, g = st["lrsd_pair"], st["mrr_gain"]
        lines.append(f"| {arm} | {_pct(st['no_patch_q'])} | {_pct(st['oracle_q'])} | {_pct(st['rescued_q'])} "
                     f"({st['n_failed_q']} failed) | {st['cells_per_failed_q']:.1f} | {_pct(st['backpatch_share'])} "
                     f"| {g['all']:+.4f} / {g['back']:+.4f} | {_pct(st['cv_top_cell']['held_out_rescue'])} "
                     f"{st['cv_top_cell']['cells']} | +{lp['rescues']} / -{lp['harms']} = {lp['net']:+d} |")
    lines += ["", "| contrast (per failed question) | mean | 95% CI |", "|---|---|---|"]
    for k, v in summary["contrasts"].items():
        if "mean" in v:
            lines.append(f"| {k} | {v['mean']:+.3f} | [{v['lo']:+.3f}, {v['hi']:+.3f}] |")
        else:
            lines.append(f"| {k} | only jlens {v['n01']}, only copy {v['n10']} | p = {v['p']:.3g} |")
    return "\n".join(lines)


def exp2_table(summary):
    lines = ["| arm | final EM | best EM | late-mean EM | memorization |", "|---|---|---|---|---|"]
    for arm, st in summary["arms"].items():
        mem = _pct(st["final_mem_em"]) if st["final_mem_em"] is not None else "n/a"
        lines.append(f"| {arm} | {_pct(st['final_em'])} | {_pct(st['best_em'])} | {_pct(st['late_mean_em'])} | {mem} |")
    lines += ["", "| test | only 2nd arm correct | only 1st arm correct | p (exact McNemar) |", "|---|---|---|---|"]
    for k, v in summary["tests"].items():
        lines.append(f"| {k} | {v['n01']} | {v['n10']} | {v['p']:.3g} |")
    return "\n".join(lines)


def write_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(obj, open(path, "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
