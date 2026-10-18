"""GPU tests: the batched grid kernels (jtransport/grid.py) reproduce the paper's per-cell kernels, and the coordinates
used by the J-lens fit (HF decoder-layer outputs) are the TransformerLens hook_resid_post the grid patches.

Local: one real model (Qwen/Qwen2.5-0.5B-Instruct base weights from the HF cache), one two-hop question from the
injected sample (data/inject_lrsd/chaining_qwen_seed848, written by distill/make_inject_data.py).
Global context: jtransport/grid.py replaces patch/multi_fact_experiment_nsamples.py's L^2 single-row forwards with
batched rows + a KV cache; these tests are the evidence that the copy arm is the paper's measurement.
Run: uv run pytest -m gpu tests/test_gpu_kernels.py   (needs ~3 GB free GPU memory)
"""
import os

import pytest
import torch

pytestmark = pytest.mark.gpu
BASE = "Qwen/Qwen2.5-0.5B-Instruct"
DATA = os.path.join(os.path.dirname(__file__), "..", "data", "inject_lrsd", "chaining_qwen_seed848")
CELLS = [(18, 12), (5, 20), (23, 0), (10, 10), (0, 23)]  # back-patch, forward patch, extremes, diagonal, last column

if not torch.cuda.is_available() or not os.path.exists(os.path.join(DATA, "experiment_data.json")):
    pytest.skip("needs CUDA and python distill/make_inject_data.py", allow_module_level=True)

from jtransport import grid  # noqa: E402  (imports the paper's patch/ modules via jtransport.paths)
from jtransport.transport import Transport  # noqa: E402
from layer_patching import cross_layer_self_patch, cross_layer_self_patch_generate  # noqa: E402  paper kernels
from utils import get_prediction_metrics, pin_chat_date  # noqa: E402


@pytest.fixture(scope="module")
def model():
    m = grid.load_tl(BASE, None, "cuda")
    yield m
    del m
    torch.cuda.empty_cache()


@pytest.fixture(scope="module")
def question(model):
    qs = grid.load_questions(DATA, model, pin_chat_date(model.tokenizer), max_questions=3)
    q = next(q for q in qs if q["positions"])
    return q


def _paper_cache(model, q):
    """The paper's source cache: run_with_cache on prompt + answer (tgt['tokens'] in the original script)."""
    tokens = model.to_tokens(q["prompt"] + q["answers"][0], prepend_bos=False)
    _, cache = model.run_with_cache(tokens)
    return tokens, cache


def test_tl_resid_post_equals_hf_decoder_layer_output(model, question):
    """jlens records model.model.layers[l] outputs; the grid patches blocks.l.hook_resid_post. Same coordinates?"""
    from transformers import AutoModelForCausalLM
    hf = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float32, device_map="cuda")
    outs = {}
    hooks = [layer.register_forward_hook(lambda m, i, o, l=l: outs.__setitem__(l, o[0] if isinstance(o, tuple) else o))
             for l, layer in enumerate(hf.model.layers)]
    with torch.no_grad():
        hf(input_ids=question["tokens"])
    for h in hooks:
        h.remove()
    _, cache = model.run_with_cache(question["tokens"])
    for l in range(model.cfg.n_layers):
        a, b = cache[f"blocks.{l}.hook_resid_post"][0], outs[l][0]
        rel = ((a - b).norm(dim=-1) / b.norm(dim=-1)).max().item()
        assert rel < 1e-3, f"layer {l}: relative difference {rel:.2e}"
    del hf
    torch.cuda.empty_cache()


def test_batched_ranks_match_paper_mrr(model, question):
    q = question
    tokens_full, cache = _paper_cache(model, q)
    prompt_len = q["tokens"].shape[1]
    H, _ = grid.clean_states(model, q["tokens"], q["positions"])
    T = Transport("copy")
    for s, t in CELLS:
        logits = cross_layer_self_patch(cache, model, tokens_full, s, t, q["positions"])  # paper: one cell
        mrr_paper = get_prediction_metrics(logits[:, :prompt_len + 1, :], q["first_ids"][0], "mrr")
        X = T.apply(H, t)[s:s + 1]
        ours = grid.patched_last_logits(model, q["tokens"], t, q["positions"], X)
        rank = int(grid.ranks_of(ours, q["first_ids"][:1])[0, 0])
        assert abs(1.0 / rank - mrr_paper) < 1e-6, (s, t, rank, mrr_paper)
        torch.testing.assert_close(ours[0], logits[0, prompt_len - 1], atol=2e-3, rtol=1e-4)


def test_restart_at_layer_equals_hooked_full_pass(model, question):
    """patched_last_logits_from (restart at block t+1 from the clean residual) == patching with a hook."""
    q = question
    H, _, R = grid.clean_states(model, q["tokens"], q["positions"], full=True)
    for t in (0, 12, 22, 23):
        X = Transport("copy").apply(H, t)
        a = grid.patched_last_logits(model, q["tokens"], t, q["positions"], X)
        b = grid.patched_last_logits_from(model, R[t], t, q["positions"], X)
        torch.testing.assert_close(a, b, atol=2e-3, rtol=1e-4)
        assert torch.equal(grid.ranks_of(a, q["first_ids"]), grid.ranks_of(b, q["first_ids"]))


def test_batched_greedy_matches_paper_generator(model, question):
    q = question
    tokens_full, cache = _paper_cache(model, q)
    H, _ = grid.clean_states(model, q["tokens"], q["positions"])
    cells = CELLS[:4]
    X = torch.stack([Transport("copy").apply(H, t)[s] for s, t in cells])  # one row per cell, mixed target layers
    ours = grid.patched_greedy(model, q["tokens"], [t for _, t in cells], q["positions"], X,
                               model.tokenizer.eos_token_id, max_new_tokens=24)  # all cells in ONE batch
    for (s, t), text in zip(cells, ours):
        gen = cross_layer_self_patch_generate(cache, model, q["tokens"], s, t, q["positions"], max_new_tokens=24)
        paper_text = model.to_string(gen[0][:-1])[len(q["prompt"]):]  # the paper's decode + slice
        assert text.strip() == paper_text.strip(), (s, t, text, paper_text)


def test_last_column_cannot_change_the_answer(model, question):
    """Patching blocks.{L-1}.hook_resid_post at entity tokens cannot reach the last position (no later layer)."""
    q = question
    L = model.cfg.n_layers
    H, base = grid.clean_states(model, q["tokens"], q["positions"])
    X = Transport("copy").apply(H, L - 1)
    ours = grid.patched_last_logits(model, q["tokens"], L - 1, q["positions"], X)
    torch.testing.assert_close(ours, base.expand_as(ours), atol=1e-4, rtol=1e-5)


def test_jlens_fit_runs_and_has_d_by_d_jacobians():
    import jlens
    from transformers import AutoModelForCausalLM, AutoTokenizer
    hf = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float32, device_map="cuda")
    tok = AutoTokenizer.from_pretrained(BASE)
    m = jlens.from_hf(hf, tok)
    text = "The quick brown fox jumps over the lazy dog. " * 6  # > 17 tokens (16 skipped + 1 target-less)
    lens = jlens.fit(m, [text], source_layers=[0, 12], max_seq_len=32, dim_batch=32)
    assert sorted(lens.jacobians) == [0, 12]
    for J in lens.jacobians.values():
        assert J.shape == (m.d_model, m.d_model) and torch.isfinite(J).all()
    del hf, m
    torch.cuda.empty_cache()
