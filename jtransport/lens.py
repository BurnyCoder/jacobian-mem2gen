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


def fit_lens(base_model, adapter, out_dir, corpus, n_per_half, max_seq_len, dim_batch, device="cuda", seed=0, tf32=True):
    """Phase fit_lens_*: writes <out_dir>/{lens_A,lens_B,lens}.pt and spectra.json (skipped if lens.pt exists).

    tf32: TF32 matmuls in the backward passes (2.4x faster here; changed J by 0.24% relative, far below the
    split-half difference between two prompt halves).
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
