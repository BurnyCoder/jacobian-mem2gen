"""Batched self-patching grid for several transport arms (copy / jlens / controls) on one fine-tuned checkpoint.

Local: for every unique two-hop question and every target layer t, all L source rows are patched in ONE batched
forward pass: blocks.t.hook_resid_post[E] := Transport.apply(H, t)[s] (E = head-entity tokens). Channel 0 is the
reciprocal rank of the first answer token at the last prompt position, channel 1 the greedy exact match (only decoded
where that rank is 1, since greedy decoding must start with the answer's first token).
Global context: a faster, multi-arm re-implementation of the paper's patch/multi_fact_experiment_nsamples.py
(which runs L^2 single-row forwards per question and re-runs the whole sequence per generated token). It reuses that
script's data indexing and the paper's helpers (layer_patching.load_checkpoint, layer_patching.find_entity_positions,
utils.pin_chat_date) and writes the same (N, 2, L, L) arrays, so patch/compute_oracle.py reads them unchanged.
Equivalence with the original kernels is tested in tests/test_gpu_kernels.py. Differences, all deliberate:
  * baseline-correct questions get their real MRR (the original fills ones) so harms can be measured; EM is gated on
    rank 1 for every row (the original gates only baseline-wrong rows; EM requires rank 1 under greedy anyway);
  * decoding uses a TransformerLens KV cache with the patch applied on the prefill pass, which is identical to
    re-patching every step because the patched positions are in the prompt and attention is causal.
"""
import json
import logging
import os
import time

import numpy as np
import torch

from jtransport.paths import use_repo_modules
from jtransport.transport import Transport, select_ridge, stack_jacobians

use_repo_modules()
from layer_patching import find_entity_positions, load_checkpoint  # noqa: E402  paper's TL loader / entity matcher
from transformer_lens.cache.key_value_cache import TransformerLensKeyValueCache  # noqa: E402
from utils import pin_chat_date  # noqa: E402  paper's chat-template date pin (no-op for Qwen)

log = logging.getLogger(__name__)
MAX_NEW_TOKENS = 64  # same budget as the paper's cross_layer_self_patch_generate


# ----------------------------------------------------------------------------------------------------------- model
def load_tl(base_model, adapter, device):
    """HookedTransformer of base + merged LoRA (fp32, no weight processing => HF residual coordinates).

    Local: the paper's load_checkpoint is called on CPU and the result moved to the GPU, so the HF copy it creates is
    freed instead of staying on the (8 GB) GPU next to the TransformerLens copy.
    """
    model = load_checkpoint(adapter if adapter else "base", base_model, device="cpu")  # merge + convert on CPU
    model = model.to(device)  # HookedTransformer.to also updates cfg.device
    model.eval()
    return model


# ----------------------------------------------------------------------------------------------------------- data
def load_questions(data_dir, model, tokenizer, max_questions=None):
    """Unique two-hop questions of a run directory, with their rows (gold answers) and head-entity positions.

    Mirrors run_patching_experiment in patch/multi_fact_experiment_nsamples.py for task 'chaining' (rows of
    eval_data['chaining'], entity = facts[0]['head'] of original_fact.json, positions found in prompt+answer).
    """
    ed = json.load(open(os.path.join(data_dir, "experiment_data.json")))
    of = json.load(open(os.path.join(data_dir, "original_fact.json")))
    rows = ed["eval_data"]["chaining"]
    assert len(rows) == len(of), "experiment_data and original_fact disagree"
    by_prompt = {}  # chat-formatted prompt -> question record (insertion order = first appearance)
    for idx, row in enumerate(rows):
        p = tokenizer.apply_chat_template(row["prompt"], tokenize=False, add_generation_prompt=True)
        ans = row["completion"][0]["content"]
        q = by_prompt.setdefault(p, {"qid": len(by_prompt), "prompt": p, "question": row["prompt"][1]["content"],
                                     "entity": of[idx]["facts"][0]["head"], "bridge": of[idx]["facts"][0]["tail"],
                                     "rows": [], "answers": []})
        q["rows"].append(idx)
        q["answers"].append(ans)
    qs = list(by_prompt.values())[: max_questions or None]
    for q in qs:
        q["tokens"] = model.to_tokens(q["prompt"], prepend_bos=False)  # [1, T] prompt only (causal => same logits)
        # first answer token per row, tokenized alone as in the original (tgt['answer_tokens'][0])
        q["first_ids"] = [int(model.to_tokens(a, prepend_bos=False)[0, 0]) for a in q["answers"]]
        q["positions"] = find_entity_positions(model, prompt=q["prompt"] + q["answers"][0], entity=q["entity"])
        if any(p >= q["tokens"].shape[1] for p in q["positions"]):  # entity matched inside the answer: unusable
            q["positions"] = []
    n_empty = sum(not q["positions"] for q in qs)
    log.info("%d unique questions (%d rows); %d without an entity span (their grids are all no-patch)",
             len(qs), sum(len(q["rows"]) for q in qs), n_empty)
    return qs


# ----------------------------------------------------------------------------------------------------------- kernels
@torch.no_grad()
def clean_states(model, tokens, positions, full=False):
    """H[l] = blocks.l.hook_resid_post at `positions` for all layers ([L, P, d]) and the clean last-position logits;
    with full=True also the whole residual stream R[l] ([L, T, d]) used to restart the forward pass at layer t+1."""
    logits, cache = model.run_with_cache(tokens, names_filter=lambda n: n.endswith("hook_resid_post"))
    R = torch.stack([cache[f"blocks.{l}.hook_resid_post"][0] for l in range(model.cfg.n_layers)])  # [L, T, d]
    H = R[:, positions]  # [L, P, d]
    return (H, logits[0, -1], R) if full else (H, logits[0, -1])


def _patch_hook(X, positions):
    """TransformerLens hook writing row b's states X[b] at `positions` (advanced indexing over the position axis)."""
    def hook(act, hook):  # act: [B, T, d]; returning it replaces the activation (TL hook_points semantics)
        if act.shape[1] > max(positions):  # prefill pass only (single-token decode steps never contain them)
            act[:, positions, :] = X.to(act.dtype)
        return act
    return hook


@torch.no_grad()
def patched_last_logits(model, tokens, t, positions, X, past_kv_cache=None):
    """Last-position logits [B, V] with blocks.t.hook_resid_post[positions] := X[b] for each row b.

    stop_at_layer = n_layers returns the residual after the last block (HookedTransformer.forward), so only the last
    position goes through ln_final + unembed: avoids a [B, T, V] logits tensor (~1 GB for 24 rows).
    """
    B = X.shape[0]
    hooks = [(f"blocks.{t}.hook_resid_post", _patch_hook(X, positions))] if positions else []
    resid = model.run_with_hooks(tokens.expand(B, -1), fwd_hooks=hooks, stop_at_layer=model.cfg.n_layers,
                                 past_kv_cache=past_kv_cache)
    return model.unembed(model.ln_final(resid[:, -1:, :]))[:, 0]  # same path as forward() for Qwen2 (no softcap)


@torch.no_grad()
def patched_last_logits_from(model, resid_t, t, positions, X):
    """Same as patched_last_logits, but restarts from the clean residual after block t ([T, d]) instead of re-running
    blocks 0..t: the patch only changes layer t's output, so blocks 0..t are identical to the clean pass.
    HookedTransformer.forward(start_at_layer=t+1) takes the residual stream as input (TL docs: 'start_at_layer ...
    input is assumed to be the residual stream'); roughly halves the grid's compute on average over t."""
    r = resid_t.expand(X.shape[0], -1, -1).clone()  # [B, T, d] one copy per row
    if positions:
        r[:, positions, :] = X.to(r.dtype)  # write the transported states at the entity tokens
    if t + 1 < model.cfg.n_layers:
        r = model(r, start_at_layer=t + 1, stop_at_layer=model.cfg.n_layers)  # blocks t+1..L-1 only
    return model.unembed(model.ln_final(r[:, -1:, :]))[:, 0]


def ranks_of(logits, ids):
    """1-based rank of each token id in each row's logits ([B, V] x [K] -> [B, K]); rank 1 = argmax."""
    tgt = logits[:, ids]  # [B, K] logit of each answer's first token
    return (logits[:, None, :] > tgt[:, :, None]).sum(-1) + 1  # number of strictly larger logits + 1


def _rows_hook(X, rows, positions):
    """Hook writing X[i] into batch row rows[i] at `positions` (one target layer, a subset of the batch)."""
    r = torch.tensor(rows)[:, None]  # [R, 1] batch indices
    p = torch.tensor(positions)[None, :]  # [1, P] token indices

    def hook(act, hook):
        if act.shape[1] > max(positions):  # prefill pass only
            act[r.to(act.device), p.to(act.device), :] = X.to(act.dtype)
        return act
    return hook


@torch.no_grad()
def patched_greedy(model, tokens, t, positions, X, eos_id, max_new_tokens=MAX_NEW_TOKENS):
    """Greedy continuation per row with the patch, as text (cut at the first EOS, like the paper's generator).

    `t` is one target layer for every row, or a list with each row's own target layer: then one hook per distinct
    target layer writes only its rows, so cells of many target layers are decoded in one batch.
    """
    B = X.shape[0]
    targets = list(t) if isinstance(t, (list, tuple)) else [t] * B
    by_t = {}
    for b, tt in enumerate(targets):
        by_t.setdefault(tt, []).append(b)
    hooks = [(f"blocks.{tt}.hook_resid_post", _rows_hook(X[rows], rows, positions)) for tt, rows in by_t.items()] \
        if positions else []
    cache = TransformerLensKeyValueCache.init_cache(model.cfg, model.cfg.device, B)  # per-layer K/V of B rows
    resid = model.run_with_hooks(tokens.expand(B, -1), fwd_hooks=hooks, stop_at_layer=model.cfg.n_layers,
                                 past_kv_cache=cache)  # patched prefill fills the KV cache of every layer
    nxt = model.unembed(model.ln_final(resid[:, -1:, :]))[:, 0].argmax(-1)
    steps, done = [nxt], nxt == eos_id
    for _ in range(max_new_tokens - 1):
        if bool(done.all()):
            break
        nxt = model(nxt[:, None], past_kv_cache=cache)[:, -1].argmax(-1)  # one token per step, cached keys/values
        steps.append(nxt)
        done |= nxt == eos_id
    seqs = torch.stack(steps, 1).tolist()  # [B, n_new]
    out = []
    for seq in seqs:
        # the paper's loop stops after appending EOS and then drops the last token ([:-1]), EOS or not
        seq = seq[: seq.index(eos_id)] if eos_id in seq else seq[:-1]
        out.append(model.to_string(torch.tensor(seq)) if seq else "")
    return out


def question_grid(model, q, arms, row_batch, eos_id, gen_batch=96):
    """(rows, 2, L, L) grid of one question for every arm.

    Per arm: one rank pass per target layer t (all L source rows, restarted at block t+1), then ONE batched greedy
    decode over every cell whose first answer token ranks 1 for some gold answer, across all target layers.
    """
    L = model.cfg.n_layers
    H, base_logits, R = clean_states(model, q["tokens"], q["positions"] or [0], full=True)  # [0]: valid shapes
    out = {arm: np.zeros((len(q["rows"]), 2, L, L), dtype=np.float32) for arm in arms}
    for arm, T in arms.items():
        Xs, ranks = {}, {}
        for t in range(L):
            Xs[t] = T.apply(H, t, key=q["qid"])  # [L(src), P, d]
            ranks[t] = torch.cat([ranks_of(patched_last_logits_from(model, R[t], t, q["positions"],
                                                                    Xs[t][i:i + row_batch]), q["first_ids"])
                                  for i in range(0, L, row_batch)])  # [L(src), K rows]
            out[arm][:, 0, :, t] = (1.0 / ranks[t].float()).T.cpu().numpy()
        cells = [(s, t) for t in range(L) for s in range(L) if bool((ranks[t][s] == 1).any())]  # greedy needs rank 1
        for i in range(0, len(cells), gen_batch):
            chunk = cells[i:i + gen_batch]
            X = torch.stack([Xs[t][s] for s, t in chunk])  # [B, P, d] each cell's transported state
            texts = patched_greedy(model, q["tokens"], [t for _, t in chunk], q["positions"], X, eos_id)
            for (s, t), text in zip(chunk, texts):
                for r, ans in enumerate(q["answers"]):
                    if int(ranks[t][s, r]) == 1 and text.strip() == ans.strip():  # paper's exact-match rule
                        out[arm][r, 1, s, t] = 1.0
    return out, H, base_logits


# ----------------------------------------------------------------------------------------------------------- lens diag
@torch.no_grad()
def bridge_readout(model, H, J, bridge):
    """Rank of the bridge entity's first token at the last entity token, per layer: [2, L] (logit lens, J-lens).

    Local: logit lens = unembed(ln_final(h_l)); J-lens = unembed(ln_final(J_l h_l)) (jlens.HFLensModel.unembed
    applies the final norm then the head, same as TransformerLens ln_final + unembed for Qwen2). Best of the
    with/without-leading-space spellings of the bridge.
    Global context: shows at which layer the first hop's answer becomes readable at the entity position, i.e.
    where a back-patch could take it from (Biran et al. 2024, "Hopping Too Late", arXiv 2406.12775).
    """
    h = H[:, -1, :]  # [L, d] last entity token
    ids = sorted({int(model.to_tokens(b, prepend_bos=False)[0, 0]) for b in (bridge, " " + bridge)})
    out = []
    for z in (h, torch.einsum("ld,led->le", h, J.to(h.device, h.dtype))):  # logit lens, J-lens
        logits = model.unembed(model.ln_final(z[:, None, :]))[:, 0]  # [L, V]
        out.append(ranks_of(logits, ids).min(1).values)  # [L]
    return torch.stack(out).cpu().numpy()


# ----------------------------------------------------------------------------------------------------------- runner
def arm_spec(arm):
    """Arm name -> (Transport method, lens suffix): 'jlens_pp' -> ('jlens', '_pp') uses the per-position lens."""
    return (arm[:-3], "_pp") if arm.endswith("_pp") else (arm, "")


def _select_c(model, qs, cfg, J, suffix, out_dir):
    """Split-half ridge selection for one lens (cached in ridge<suffix>.json); answers are never used."""
    path = os.path.join(out_dir, f"ridge{suffix}.json")
    if os.path.exists(path):
        return json.load(open(path))["c"]
    states = [clean_states(model, q["tokens"], q["positions"])[0] for q in qs[: cfg.ridge_questions] if q["positions"]]
    c, table = select_ridge(J[suffix + "_A"], J[suffix + "_B"], states, cfg.ridge_grid, cfg.ridge_tau)
    for r in table:
        log.info("ridge%s c=%-6g instability=%.3f rel_perturbation=%.3f", suffix, r["c"], r["instability"],
                 r["rel_perturbation"])
    json.dump({"c": c, "tau": cfg.ridge_tau, "table": table}, open(path, "w"), indent=1)
    log.info("ridge%s c = %g", suffix, c)
    return c


def run_grids(cfg, data_dir, adapter, lens_dirs, out_dir, arm_names):
    """Phases `patch_grids` / `patch_grids_pp`: ridge selection, then the grids of `arm_names`.

    Writes <out_dir>/<arm>.npy and merges meta.json / diag.json / bridge_rank<suffix>.npy with arms written before,
    so arms can be added later without recomputing the others. lens_dirs: {suffix: dir} ("" = J-lens, "_pp" =
    per-position). Resumable: per-question results are checkpointed every 10 questions to <out_dir>/partial_<arms>.npz.
    """
    os.makedirs(out_dir, exist_ok=True)
    # TF32 (cfg.tf32, default on) makes the grid ~1.4x faster on this GPU; measured on 4 questions x 4 arms it changed
    # the exact-match result of <= 0.02% of cells (same numerics for every arm). TF32=0 gives strict fp32 like the paper.
    # https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-and-later-devices
    torch.backends.cuda.matmul.allow_tf32 = cfg.tf32
    model = load_tl(cfg.base_model, adapter, cfg.device)
    L = model.cfg.n_layers
    tokenizer = pin_chat_date(model.tokenizer)
    eos_id = model.tokenizer.eos_token_id  # <|im_end|> for Qwen2.5-Instruct; the paper's generator stops only here
    qs = load_questions(data_dir, model, tokenizer, cfg.max_questions)

    from jlens import JacobianLens  # reference J-lens container (fp32 matrices saved by jtransport/lens.py)
    suffixes = sorted({arm_spec(a)[1] for a in arm_names if "jlens" in a})  # lenses this run needs
    J = {sfx + k: stack_jacobians(JacobianLens.load(os.path.join(lens_dirs[sfx], f"lens{k}.pt")).jacobians, L)
         .to(cfg.device) for sfx in suffixes for k in ("", "_A", "_B")}  # merged lens and the two split halves
    c = {sfx: _select_c(model, qs, cfg, J, sfx, out_dir) for sfx in suffixes}
    arms = {}
    for a in arm_names:
        method, sfx = arm_spec(a)
        arms[a] = Transport(method, J=J.get(sfx), c=c.get(sfx, 0.0), seed=cfg.seed)
    geo_arms = [a for a in arms if arm_spec(a)[0] in ("copy", "jlens")]  # arms whose geometry is logged

    part_path = os.path.join(out_dir, f"partial_{'-'.join(arm_names)}.npz")
    done = {}
    if os.path.exists(part_path):  # resume
        done = np.load(part_path, allow_pickle=True)["done"].item()
        log.info("resuming: %d/%d questions already done", len(done), len(qs))
    t0 = time.time()
    for q in qs:
        if q["qid"] in done:
            continue
        grids, H, _ = question_grid(model, q, arms, cfg.row_batch, eos_id)
        diag = {"bridge_rank": {sfx: bridge_readout(model, H, J[sfx], q["bridge"]) for sfx in suffixes}
                if q["positions"] else None,
                "geometry": {a: [arms[a].diagnostics(H, t) for t in range(L)] for a in geo_arms}
                if q["positions"] else None}
        done[q["qid"]] = {"grids": grids, "diag": diag}
        n = len(done)
        log.info("question %d/%d  entity=%r P=%d  no-patch EM=%s  oracle EM: %s  (%.1f min)",
                 n, len(qs), q["entity"], len(q["positions"]),
                 [int(g) for g in grids[next(iter(arms))][:, 1, 0, 0]],
                 {a: int((g[:, 1] >= 1).any()) for a, g in grids.items()}, (time.time() - t0) / 60)
        if n % 10 == 0 or n == len(qs):
            np.savez(part_path, done=np.array(done, dtype=object))  # checkpoint for resume
    # --------------------------------------------------------------- assemble (row order = original row index)
    order = sorted((r, q["qid"], j) for q in qs for j, r in enumerate(q["rows"]))
    qmap = {q["qid"]: q for q in qs}
    for arm in arms:
        arr = np.stack([done[qid]["grids"][arm][j] for _, qid, j in order])  # (N_rows, 2, L, L)
        np.save(os.path.join(out_dir, f"{arm}.npy"), arr)
    meta_path, diag_path = os.path.join(out_dir, "meta.json"), os.path.join(out_dir, "diag.json")
    old = json.load(open(meta_path)) if os.path.exists(meta_path) else {"arms": [], "ridge_c": {}}
    rows = [{"row": r, "qid": qid, "answer": qmap[qid]["answers"][j]} for r, qid, j in order]
    if old.get("rows") not in (None, rows):
        raise ValueError(f"{meta_path}: rows differ from this run (different questions?); use a new OUT_ROOT")
    ridge_old = old["ridge_c"] if isinstance(old["ridge_c"], dict) else {"": old["ridge_c"]}
    meta = {"rows": rows,
            "questions": [{k: q[k] for k in ("qid", "question", "entity", "bridge", "positions", "answers")}
                          for q in qs],
            "n_layers": L, "ridge_c": {**ridge_old, **c}, "arms": old["arms"] + [a for a in arms if a not in old["arms"]],
            "adapter": adapter, "base_model": cfg.base_model}
    json.dump(meta, open(meta_path, "w"), indent=1)
    for sfx in suffixes:
        br = [done[q["qid"]]["diag"]["bridge_rank"][sfx] for q in qs if done[q["qid"]]["diag"]["bridge_rank"]]
        np.save(os.path.join(out_dir, f"bridge_rank{sfx}.npy"), np.stack(br) if br else np.zeros((0, 2, L)))
    geo = json.load(open(diag_path))["geometry"] if os.path.exists(diag_path) else {}
    for a in geo_arms:
        keys = ("rel_perturbation", "readout_error") if "jlens" in a else ("rel_perturbation",)
        geo[a] = {k: float(np.mean([d[k] for q in qs if done[q["qid"]]["diag"]["geometry"]
                                    for d in done[q["qid"]]["diag"]["geometry"][a]])) for k in keys}
    json.dump({"geometry": geo}, open(diag_path, "w"), indent=1)
    log.info("grids written to %s (%.1f min)", out_dir, (time.time() - t0) / 60)
    del model
    torch.cuda.empty_cache()
