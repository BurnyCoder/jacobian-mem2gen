"""Unit tests of jtransport/compare.py on tiny synthetic self-patching grids (CPU only).

Local: hand-built (N, 2, L, L) grids with known answers pin each metric of the transport comparison.
Global context: compare.py turns the grids written by jtransport/grid.py (same format as the paper's
patch/multi_fact_experiment_nsamples.py: channel 0 = first-answer-token MRR, channel 1 = greedy exact match,
indexed [row, channel, src layer, tgt layer]) into the numbers of results/jtransport/REPORT.md.
Run: uv run pytest tests/test_compare.py
"""
import numpy as np

from jtransport.compare import (cv_top_cell, net_flips, paired_bootstrap, question_em, rescue_stats,
                                summarize)

L = 3  # layers in the toy grids


def _grid(em_cells_per_row):
    """(N, 2, L, L) grid from a list of {(s, t): 1} EM dicts; the diagonal copies cell (0, 0) (no-op patch)."""
    arr = np.zeros((len(em_cells_per_row), 2, L, L))
    for r, cells in enumerate(em_cells_per_row):
        for (s, t), v in cells.items():
            arr[r, 1, s, t] = v
        for i in range(L):
            arr[r, 1, i, i] = arr[r, 1, 0, 0]  # diagonal == unpatched model
    arr[:, 0] = arr[:, 1]  # MRR channel: 1 where EM (enough for these tests)
    return arr


# rows 0 and 1 share question 0 (two gold answers); row 0 is correct without patching.
# question 1 (row 2) is wrong unpatched and rescued at (2, 0); question 2 (row 3) is wrong and never rescued.
QID = np.array([0, 0, 1, 2])
GRID = _grid([{(0, 0): 1, (2, 1): 1}, {}, {(2, 0): 1}, {}])


def test_question_em_is_any_over_rows_of_a_question():
    emq = question_em(GRID, QID)
    assert emq.shape == (3, L, L)
    assert emq[0, 0, 0] and not emq[1, 0, 0] and not emq[2, 0, 0]  # only question 0 is correct unpatched
    assert emq[1, 2, 0] and not emq[2].any()


def test_rescue_stats_counts_failed_questions_rescued_by_any_off_diagonal_cell():
    st = rescue_stats(GRID, QID)
    assert st["no_patch_q"] == 1 / 3 and st["oracle_q"] == 2 / 3  # question-level, any gold answer
    assert st["n_failed_q"] == 2 and st["rescued_q"] == 0.5  # 1 of the 2 failed questions rescued
    assert st["no_patch_row"] == 0.25 and st["oracle_row"] == 0.5  # paper-compatible row-level numbers
    assert np.array_equal(st["rescued_mask"], [False, True, False])  # per question, aligned with unique qids
    assert st["cells_per_failed_q"] == 0.5  # (1 cell + 0 cells) / 2 failed questions


def test_net_flips_counts_rescues_minus_harms_at_one_cell():
    emq = question_em(GRID, QID)
    nf = net_flips(emq, 2, 0)  # cell (s=2, t=0)
    assert nf == {"rescues": 1, "harms": 1, "net": 0}  # rescues q1, breaks the correct q0
    assert net_flips(emq, 2, 1)["harms"] == 0  # q0 stays correct at (2, 1)


def test_cv_top_cell_scores_on_held_out_questions_only():
    rng = np.random.default_rng(0)
    Q = 40
    em = np.zeros((Q, L, L), dtype=bool)
    em[: Q // 2, 2, 0] = True  # half of the failed questions are rescued at (2, 0), nothing else ever is
    perm = rng.permutation(Q)
    res = cv_top_cell(em[perm], np.ones(Q, dtype=bool), seed=1)
    assert all(c == (2, 0) for c in res["cells"])  # both folds pick the only useful cell
    assert abs(res["held_out_rescue"] - 0.5) < 0.2  # held-out rate near the true 50%


def test_paired_bootstrap_ci_contains_true_difference_and_excludes_zero():
    rng = np.random.default_rng(0)
    x = rng.random(500) < 0.6
    y = rng.random(500) < 0.3
    b = paired_bootstrap(x.astype(float), y.astype(float), n_boot=2000, seed=0)
    assert b["lo"] < x.mean() - y.mean() < b["hi"] and b["lo"] > 0  # clearly positive difference


def test_summarize_reports_arms_contrasts_and_mcnemar():
    better = _grid([{(0, 0): 1}, {(1, 0): 1}, {(2, 0): 1}, {(1, 2): 1}])  # rescues both failed questions
    grids = {"copy": GRID, "jlens": better, "noise_copy": _grid([{(0, 0): 1}, {}, {}, {}]),
             "noise_jlens": _grid([{(0, 0): 1}, {}, {}, {}])}
    s = summarize(grids, QID, lrsd_pair=(2, 1), n_boot=200, seed=0)
    assert s["arms"]["jlens"]["rescued_q"] == 1.0 and s["arms"]["copy"]["rescued_q"] == 0.5
    assert s["arms"]["copy"]["lrsd_pair"]["net"] == 0  # (2, 1): no rescue, no harm
    c = s["contrasts"]
    assert c["jlens_minus_copy"]["mean"] == 0.5  # over the 2 failed questions
    assert c["copy_minus_noise_copy"]["mean"] == 0.5 and c["jlens_minus_noise_jlens"]["mean"] == 1.0
    assert c["mcnemar_copy_vs_jlens"]["n01"] == 1 and c["mcnemar_copy_vs_jlens"]["n10"] == 0


def test_summarize_tolerates_a_near_tie_flip_of_the_unpatched_answer():
    flipped = _grid([{}, {}, {(2, 0): 1}, {}])  # q0's no-patch answer flipped to wrong in this arm only
    s = summarize({"copy": GRID, "jlens": flipped}, QID, lrsd_pair=(2, 1), n_boot=50, seed=0)
    assert s["arms"]["jlens"]["no_patch_mismatch_q"] == 1  # reported, not fatal
    assert s["contrasts"]["jlens_minus_copy"]["n"] == 2  # contrasts use the first arm's failed set
