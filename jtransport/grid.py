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
def clean_states(model, tokens, positions):
    """H[l] = blocks.l.hook_resid_post at `positions` for all layers ([L, P, d]) and the clean last-position logits."""
    logits, cache = model.run_with_cache(tokens, names_filter=lambda n: n.endswith("hook_resid_post"))
    H = torch.stack([cache[f"blocks.{l}.hook_resid_post"][0, positions] for l in range(model.cfg.n_layers)])
    return H, logits[0, -1]


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


def ranks_of(logits, ids):
    """1-based rank of each token id in each row's logits ([B, V] x [K] -> [B, K]); rank 1 = argmax."""
    tgt = logits[:, ids]  # [B, K] logit of each answer's first token
    return (logits[:, None, :] > tgt[:, :, None]).sum(-1) + 1  # number of strictly larger logits + 1


@torch.no_grad()
def patched_greedy(model, tokens, t, positions, X, eos_id, max_new_tokens=MAX_NEW_TOKENS):
    """Greedy continuation per row with the patch, as text (cut at the first EOS, like the paper's generator)."""
    B = X.shape[0]
    cache = TransformerLensKeyValueCache.init_cache(model.cfg, model.cfg.device, B)  # per-layer K/V of B rows
    nxt = patched_last_logits(model, tokens, t, positions, X, past_kv_cache=cache).argmax(-1)  # patched prefill
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


def question_grid(model, q, arms, row_batch, eos_id):
    """(rows, 2, L, L) grid of one question for every arm."""
    L = model.cfg.n_layers
    H, base_logits = clean_states(model, q["tokens"], q["positions"] or [0])  # [0] keeps shapes valid if no span
    out = {arm: np.zeros((len(q["rows"]), 2, L, L), dtype=np.float32) for arm in arms}
    for arm, T in arms.items():
        for t in range(L):
            X = T.apply(H, t, key=q["qid"])  # [L(src), P, d]
            ranks = torch.cat([ranks_of(patched_last_logits(model, q["tokens"], t, q["positions"], X[i:i + row_batch]),
                                        q["first_ids"]) for i in range(0, L, row_batch)])  # [L(src), K rows]
            out[arm][:, 0, :, t] = (1.0 / ranks.float()).T.cpu().numpy()
            todo = [s for s in range(L) if bool((ranks[s] == 1).any())]  # greedy can only match from rank 1
            for i in range(0, len(todo), row_batch):
                chunk = todo[i:i + row_batch]
                texts = patched_greedy(model, q["tokens"], t, q["positions"], X[chunk], eos_id)
                for s, text in zip(chunk, texts):
                    for r, ans in enumerate(q["answers"]):
                        if int(ranks[s, r]) == 1 and text.strip() == ans.strip():  # paper's exact-match rule
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
def run_grids(cfg, data_dir, adapter, lens_dir, out_dir):
    """Phase `patch_grids`: ridge selection, then all arms' grids; writes <out_dir>/<arm>.npy + meta/diag/ridge json.

    Resumable: per-question results are checkpointed every 10 questions to <out_dir>/partial.npz.
    """
    os.makedirs(out_dir, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = False  # exact fp32 matmuls, as in the paper's scripts (PyTorch default)
    model = load_tl(cfg.base_model, adapter, cfg.device)
    L = model.cfg.n_layers
    tokenizer = pin_chat_date(model.tokenizer)
    eos_id = model.tokenizer.eos_token_id  # <|im_end|> for Qwen2.5-Instruct; the paper's generator stops only here
    qs = load_questions(data_dir, model, tokenizer, cfg.max_questions)

    from jlens import JacobianLens  # reference J-lens container (fp32 matrices saved by jtransport/lens.py)
    J = {k: stack_jacobians(JacobianLens.load(os.path.join(lens_dir, f"lens{k}.pt")).jacobians, L).to(cfg.device)
         for k in ("", "_A", "_B")}  # merged lens and the two split halves

    ridge_path = os.path.join(out_dir, "ridge.json")
    if os.path.exists(ridge_path):
        c = json.load(open(ridge_path))["c"]
    else:  # split-half stability on clean entity states of the first questions (answers unused)
        states = [clean_states(model, q["tokens"], q["positions"])[0] for q in qs[: cfg.ridge_questions]
                  if q["positions"]]
        c, table = select_ridge(J["_A"], J["_B"], states, cfg.ridge_grid, cfg.ridge_tau)
        for r in table:
            log.info("ridge c=%-6g instability=%.3f rel_perturbation=%.3f", r["c"], r["instability"],
                     r["rel_perturbation"])
        json.dump({"c": c, "tau": cfg.ridge_tau, "table": table}, open(ridge_path, "w"), indent=1)
    log.info("ridge c = %g", c)
    arms = {a: Transport(a, J=J[""], c=c, seed=cfg.seed) for a in cfg.transports}

    part_path = os.path.join(out_dir, "partial.npz")
    done = {}
    if os.path.exists(part_path):  # resume
        z = np.load(part_path, allow_pickle=True)
        done = z["done"].item()
        log.info("resuming: %d/%d questions already done", len(done), len(qs))
    t0 = time.time()
    for i, q in enumerate(qs):
        if q["qid"] in done:
            continue
        grids, H, _ = question_grid(model, q, arms, cfg.row_batch, eos_id)
        diag = {"bridge_rank": bridge_readout(model, H, J[""], q["bridge"]) if q["positions"] else None,
                "geometry": {a: [arms[a].diagnostics(H, t) for t in range(L)] for a in ("copy", "jlens") if a in arms}
                if q["positions"] else None}
        done[q["qid"]] = {"grids": grids, "diag": diag}
        n = len(done)
        el = (time.time() - t0) / 60
        log.info("question %d/%d  entity=%r P=%d  no-patch EM=%s  oracle EM: %s  (%.1f min)",
                 n, len(qs), q["entity"], len(q["positions"]),
                 [int(g) for g in grids[next(iter(arms))][:, 1, 0, 0]],
                 {a: int((g[:, 1] >= 1).any()) for a, g in grids.items()}, el)
        if n % 10 == 0 or n == len(qs):
            np.savez(part_path, done=np.array(done, dtype=object))  # checkpoint for resume
    # --------------------------------------------------------------- assemble (row order = original row index)
    order = sorted((r, q["qid"], j) for q in qs for j, r in enumerate(q["rows"]))
    qmap = {q["qid"]: q for q in qs}
    for arm in arms:
        arr = np.stack([done[qid]["grids"][arm][j] for _, qid, j in order])  # (N_rows, 2, L, L)
        np.save(os.path.join(out_dir, f"{arm}.npy"), arr)
    meta = {"rows": [{"row": r, "qid": qid, "answer": qmap[qid]["answers"][j]} for r, qid, j in order],
            "questions": [{k: q[k] for k in ("qid", "question", "entity", "bridge", "positions", "answers")}
                          for q in qs],
            "n_layers": L, "ridge_c": c, "arms": list(arms), "adapter": adapter, "base_model": cfg.base_model}
    json.dump(meta, open(os.path.join(out_dir, "meta.json"), "w"), indent=1)
    br = [done[q["qid"]]["diag"]["bridge_rank"] for q in qs if done[q["qid"]]["diag"]["bridge_rank"] is not None]
    geo = {a: {k: float(np.mean([d[k] for q in qs if done[q["qid"]]["diag"]["geometry"]
                                 for d in done[q["qid"]]["diag"]["geometry"][a]]))
               for k in ("rel_perturbation", "readout_error") if k != "readout_error" or a == "jlens"}
           for a in ("copy", "jlens") if a in arms}
    np.save(os.path.join(out_dir, "bridge_rank.npy"), np.stack(br) if br else np.zeros((0, 2, L)))
    json.dump({"geometry": geo}, open(os.path.join(out_dir, "diag.json"), "w"), indent=1)
    log.info("grids written to %s (%.1f min)", out_dir, (time.time() - t0) / 60)
    del model
    torch.cuda.empty_cache()
