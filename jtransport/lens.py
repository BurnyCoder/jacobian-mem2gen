"""Fit the Jacobian lens (J-lens) of a base or fine-tuned model with Anthropic's reference implementation.

Local: `fit_lens()` loads the model in fp32 (LoRA merged if given, via distill/common.load_merged), draws prompts from a
pretraining-like corpus, runs `jlens.fit` on two disjoint halves (A, B), merges them with `JacobianLens.merge`, and
saves lens_A.pt / lens_B.pt / lens.pt in fp32 plus per-layer singular-value spectra.
Global context: J_l = E[sum_{p'>=p} d h_final[p'] / d h_l[p]] (jlens/fitting.py module docstring;
https://github.com/anthropics/jacobian-lens), averaged over prompts and source positions; the paper fits it on 1000
pretraining-like sequences of 128 tokens and reports that ~100 prompts already work. The halves feed the split-half
ridge selection in jtransport/transport.select_ridge; the merged lens drives the `jlens` transport and the J-space
LRSD loss. Saved in fp32 (jlens saves fp16 by default) because the transport inverts J_t.
"""
import json
import logging
import math
import os
import random
import time

import torch

from jtransport.paths import use_repo_modules

use_repo_modules()
log = logging.getLogger(__name__)


def corpus_prompts(name, n, seed=0, min_chars=1000):
    """`n` documents of at least `min_chars` characters (comfortably > 128 tokens) from a HF dataset, shuffled.

    Default corpus NeelNanda/pile-10k: the first 10k documents of the Pile (https://huggingface.co/datasets/NeelNanda/pile-10k),
    the TransformerLens standard sample of pretraining-like text.
    """
    from datasets import load_dataset  # https://huggingface.co/docs/datasets/loading
    ds = load_dataset(name, split="train")
    order = list(range(len(ds)))
    random.Random(seed).shuffle(order)  # seeded => the same prompts for every fit
    out = []
    for i in order:
        text = ds[i]["text"]
        if len(text) >= min_chars:
            out.append(text)
            if len(out) == n:
                break
    if len(out) < n:
        raise ValueError(f"{name}: only {len(out)} documents with >= {min_chars} characters")
    return out


def spectra(lens):
    """Per-layer sigma_max, sigma_min and condition number of J_l (float64 singular values)."""
    out = {}
    for l, J in lens.jacobians.items():
        S = torch.linalg.svdvals(J.double())  # descending singular values
        out[int(l)] = {"sigma_max": float(S[0]), "sigma_min": float(S[-1]), "cond": float(S[0] / S[-1])}
    return out


def per_position_jacobian_for_prompt(model, prompt, source_layers, *, positions, dim_batch=16, max_seq_len=128,
                                     skip_first=16, seed=0):
    """Same-position Jacobian of one prompt: J_l = mean over K sampled positions p of d h_final[p] / d h_l[p].

    Local: adapted from jlens.fitting.jacobian_for_prompt (Apache-2.0, https://github.com/anthropics/jacobian-lens):
    one forward on the prompt replicated dim_batch times, then backward passes against the retained graph. Each batch
    row carries a one-hot cotangent at (output dim i, position p) ONLY, and the gradient is read at the same position
    p: by causality nothing at positions > p feeds h_final[p], so this is exactly row i of d h_final[p] / d h_l[p].
    Global context: jlens instead sets the cotangent at every valid position at once, so its gradient at p is
    sum_{p' >= p} d h_final[p'] / d h_l[p] (module docstring of jlens/fitting.py): it adds how the state at p moves the
    LATER tokens through attention. That is the right object for a lens over the whole continuation, but the
    self-patching transport keeps the state at its own token, so this per-position variant (which the jlens docstring
    notes "also works as a lens") is the ablation. Cost: K x the jlens estimator per prompt.
    Args:
        positions: K, number of distinct valid positions sampled per prompt (seeded).
    Returns:
        {layer: [d_model, d_model] fp32 CPU tensor}
    """
    from jlens.fitting import _check_layer_indices, valid_position_mask
    from jlens.hooks import ActivationRecorder

    d = model.d_model
    source_layers, target_layer = _check_layer_indices(source_layers, None, model.n_layers)  # target = final layer
    input_ids = model.encode(prompt, max_length=max_seq_len)
    valid = valid_position_mask(input_ids.shape[1], skip_first=skip_first).nonzero(as_tuple=True)[0]
    g = torch.Generator().manual_seed(seed)
    pos = valid[torch.randperm(len(valid), generator=g)[:positions]].tolist()  # K distinct positions
    pairs = [(i, p) for p in pos for i in range(d)]  # every (output dim, position) row to compute
    J = {l: torch.zeros(d, d) for l in source_layers}
    n_passes = math.ceil(len(pairs) / dim_batch)
    with ActivationRecorder(model.layers, at=[*source_layers, target_layer], start_graph_at=min(source_layers)) as rec, \
            torch.enable_grad():
        model.forward(input_ids.expand(dim_batch, -1))  # one forward, graph retained for every backward below
        tgt = rec.activations[target_layer]  # [dim_batch, T, d]
        srcs = [rec.activations[l] for l in source_layers]
        cot = torch.zeros_like(tgt)
        for k, start in enumerate(range(0, len(pairs), dim_batch)):
            chunk = pairs[start:start + dim_batch]
            b = torch.arange(len(chunk), device=tgt.device)
            dims = torch.tensor([i for i, _ in chunk], device=tgt.device)
            ps = torch.tensor([p for _, p in chunk], device=tgt.device)
            cot.zero_()
            cot[b, ps, dims] = 1.0  # one cotangent per row: output dim i at position p
            grads = torch.autograd.grad(tgt, srcs, grad_outputs=cot, retain_graph=k < n_passes - 1)
            for l, gr in zip(source_layers, grads):
                J[l].index_add_(0, dims.cpu(), gr[b, ps.to(gr.device), :].float().cpu())  # gradient at the SAME p
    return {l: J[l] / len(pos) for l in source_layers}  # mean over the sampled positions


def fit_per_position(model, prompts, positions, dim_batch, max_seq_len, seed=0):
    """Running mean of per_position_jacobian_for_prompt over prompts, as a jlens.JacobianLens (saved/loaded the same)."""
    import jlens
    total, n = None, 0
    for k, prompt in enumerate(prompts):
        t0 = time.time()
        J = per_position_jacobian_for_prompt(model, prompt, None, positions=positions, dim_batch=dim_batch,
                                             max_seq_len=max_seq_len, seed=seed * 100_003 + k)
        total = J if total is None else {l: total[l] + J[l] for l in J}
        n += 1
        log.info("  per-position prompt %d/%d  %.0fs", k + 1, len(prompts), time.time() - t0)
    return jlens.JacobianLens(jacobians={l: v / n for l, v in total.items()}, n_prompts=n, d_model=model.d_model)


def fit_lens(base_model, adapter, out_dir, corpus, n_per_half, max_seq_len, dim_batch, device="cuda", seed=0, tf32=True,
             estimator="jlens", positions=4):
    """Phase fit_lens_*: writes <out_dir>/{lens_A,lens_B,lens}.pt and spectra.json (skipped if lens.pt exists).

    tf32: TF32 matmuls in the backward passes (2.4x faster here; changed J by 0.24% relative, far below the
    split-half difference between two prompt halves).
    estimator: "jlens" (reference jlens.fit, sum over later target positions) or "per_position" (same-position
    Jacobian, `positions` sampled positions per prompt; see per_position_jacobian_for_prompt).
    """
    import jlens
    from common import load_merged, load_tokenizer  # distill/common.py: LoRA merge + tokenizer with pad token
    from transformers import AutoModelForCausalLM

    final = os.path.join(out_dir, "lens.pt")
    if os.path.exists(final):
        log.info("lens exists, skipping fit: %s", final)
        return final
    os.makedirs(out_dir, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = tf32
    if adapter:
        hf = load_merged(base_model, adapter, dtype=torch.float32, device=device)  # fine-tuned weights
    else:
        hf = AutoModelForCausalLM.from_pretrained(base_model, dtype=torch.float32, device_map=device)
    tok = load_tokenizer(base_model)
    model = jlens.from_hf(hf, tok)  # freezes params; records model.model.layers outputs (= TL hook_resid_post)
    prompts = corpus_prompts(corpus, 2 * n_per_half, seed=seed)
    halves = []
    for name, ps in (("A", prompts[:n_per_half]), ("B", prompts[n_per_half:])):
        path = os.path.join(out_dir, f"lens_{name}.pt")
        if os.path.exists(path):
            lens = jlens.JacobianLens.load(path)
        else:
            t0 = time.time()
            if estimator == "per_position":
                lens = fit_per_position(model, ps, positions, dim_batch, max_seq_len, seed=seed + (name == "B"))
            else:
                # resumable running mean over prompts (jlens.fit checkpoint), target = final layer, sources = all below
                lens = jlens.fit(model, ps, max_seq_len=max_seq_len, dim_batch=dim_batch,
                                 checkpoint_path=os.path.join(out_dir, f"fit_{name}.ckpt"), checkpoint_every=5)
            lens.save(path, dtype=torch.float32)
            log.info("half %s: %d prompts in %.1f min -> %s", name, lens.n_prompts, (time.time() - t0) / 60, path)
        halves.append(lens)
    merged = jlens.JacobianLens.merge(halves)  # n_prompts-weighted mean of the halves = the full-corpus estimate
    merged.save(final, dtype=torch.float32)
    sp = spectra(merged)
    json.dump(sp, open(os.path.join(out_dir, "spectra.json"), "w"), indent=1)
    for l in sorted(sp):
        log.info("J_%-2d sigma_max=%.3g sigma_min=%.3g cond=%.3g", l, sp[l]["sigma_max"], sp[l]["sigma_min"],
                 sp[l]["cond"])
    for p in ("A", "B"):  # the running-sum checkpoints are no longer needed
        ck = os.path.join(out_dir, f"fit_{p}.ckpt")
        if os.path.exists(ck):
            os.remove(ck)
    del model, hf
    torch.cuda.empty_cache()
    return final
