"""Cross-layer transport operators: what gets written into layer t when the grid patches from layer s.

Local: pure torch, no model. `Transport(method).apply(H, t)` maps the clean states of ALL layers at the patched
positions, H[l, p, :] = blocks.l.hook_resid_post[E_p], to the states written into layer t: row s = transport of h_s.
Global context (why): the paper's self-patching oracle (patch/multi_fact_experiment_nsamples.py) writes h_s into layer
t unchanged ("copy"), which assumes all layers share one coordinate system. The J-lens (Gurnee et al. 2026,
https://transformer-circuits.pub/2026/workspace/index.html; code https://github.com/anthropics/jacobian-lens) fits
J_l = E[d h_final / d h_l], which expresses a layer-l state in the shared final-layer basis (lens_l(h) = unembed(J_l h)).
The `jlens` arm moves h_s so that its J-lens readout at layer t equals its readout at layer s:
    x = h_t + R_t (J_s h_s - J_t h_t),   R_t = ridge inverse of J_t
which is exactly pinv(J_t) J_s h_s when R_t = J_t^{-1} (delta form: the ridge shrinks toward the no-patch state h_t).
Controls: `norm` (scale only), `noise_copy` / `noise_jlens` (same perturbation size per cell, random direction), which
calibrate how much of an oracle gain any perturbation of that size would buy.
"""
import torch

# Arms understood by Transport; noise_* take their perturbation magnitude from the arm after the underscore
METHODS = ("copy", "jlens", "norm", "noise_copy", "noise_jlens")


def stack_jacobians(fitted, n_layers):
    """Stack per-layer J-lens matrices into one [L, d, d] tensor, with J_{L-1} = I.

    Local: `fitted` is {layer: J_l} as in jlens.JacobianLens.jacobians (source layers 0..L-2).
    Global context: jlens.fit only fits source layers strictly below the target (the final layer, see
    jlens/fitting.py::_check_layer_indices); the final layer maps to itself, so its Jacobian is the identity.
    """
    any_J = next(iter(fitted.values()))  # reference tensor for dtype / device / width
    d = any_J.shape[-1]  # residual-stream width d_model
    eye = torch.eye(d, dtype=any_J.dtype, device=any_J.device)  # d h_final / d h_final
    missing = [l for l in range(n_layers - 1) if l not in fitted]  # only the last layer may be absent
    if missing:
        raise ValueError(f"lens has no Jacobian for layers {missing}")
    return torch.stack([fitted[l] if l < n_layers - 1 else eye for l in range(n_layers)])  # [L, d, d]


def ridge_inverse(J, c):
    """Ridge (Tikhonov) right-inverse R = J^T (J J^T + lambda^2 I)^{-1}, lambda = c * sigma_max(J).

    Local: computed from the SVD J = U S V^T as R = V diag(S / (S^2 + lambda^2)) U^T in float64 on CPU
    (consumer GPUs run float64 at a small fraction of fp32 speed), returned in J's dtype and device.
    Global context: early-layer J-lens Jacobians can be ill-conditioned; the ridge keeps the correction from
    amplifying directions the lens barely sees. c = 0 gives the exact inverse (pinv for full-rank J).
    Source: Tikhonov regularisation via SVD filter factors, e.g. Hansen, "Rank-Deficient and Discrete Ill-Posed
    Problems" (SIAM 1998), ch. 4; torch.linalg.svd docs https://pytorch.org/docs/stable/generated/torch.linalg.svd.html
    """
    U, S, Vh = torch.linalg.svd(J.detach().to("cpu", torch.float64))  # J = U diag(S) Vh, S descending
    lam = c * S[0]  # lambda relative to the largest singular value => scale-free (J's overall scale cancels)
    filt = S / (S ** 2 + lam ** 2)  # Tikhonov filter factors; 1/S when lam = 0
    R = Vh.T @ torch.diag(filt) @ U.T  # V diag(filt) U^T
    return R.to(J.device, J.dtype)


class Transport:
    """One arm of the self-patching grid (see module docstring for the formulas).

    Local: `apply(H, t, key)` returns X[s] = state written into layer t when patching from layer s; X[t] = H[t]
    exactly, so the diagonal of every grid is the unpatched model (a sanity check used by patch/compute_oracle.py).
    Global context: one Transport per arm; jtransport/grid.py calls apply once per (question, target layer) and
    patches all L rows in one batched forward pass.
    """

    def __init__(self, method, J=None, c=0.01, seed=0):
        if method not in METHODS:
            raise ValueError(f"unknown transport {method!r}; expected one of {METHODS}")
        self.method, self.J, self.c, self.seed = method, J, c, seed  # J: [L, d, d] stacked J-lens matrices
        if "jlens" in method and J is None:
            raise ValueError(f"{method} needs the stacked J-lens matrices J")
        self._R = {}  # lazily computed ridge inverses R_t, keyed by target layer
        # noise_<ref>: the reference arm whose per-cell perturbation magnitude the noise control matches
        self._ref = Transport(method.split("_", 1)[1], J=J, c=c) if method.startswith("noise_") else None

    def ridge(self, t):
        """Cached R_t (ridge inverse of J_t)."""
        if t not in self._R:
            self._R[t] = ridge_inverse(self.J[t], self.c)  # one float64 SVD per target layer, reused for all questions
        return self._R[t]

    def apply(self, H, t, key=0):
        """Transported states for every source row s into target layer t.

        Args:
            H: [L, P, d] clean residual states of all layers at the P patched positions (same forward pass).
            t: target layer index.
            key: integer making the noise arms' random directions reproducible per (question, cell).
        Returns:
            X: [L, P, d], X[s] written into blocks.t.hook_resid_post at the patched positions.
        """
        h_t = H[t]  # [P, d] the clean target-layer state (the no-patch value of the cell)
        if self.method == "copy":
            X = H.clone()  # paper: h_t := h_s
        elif self.method == "norm":
            scale = h_t.norm(dim=-1) / H.norm(dim=-1).clamp_min(1e-12)  # [L, P] ||h_t|| / ||h_s||
            X = H * scale[..., None]  # keep h_s's direction, take h_t's norm
        elif self.method == "jlens":
            J = self.J.to(H.device, H.dtype)  # [L, d, d]
            Z = torch.einsum("lpd,led->lpe", H, J)  # Z[l] = J_l h_l: every state in the final-layer (J-lens) basis
            X = h_t + (Z - Z[t]) @ self.ridge(t).to(H.device, H.dtype).T  # h_t + R_t (J_s h_s - J_t h_t)
        else:  # noise_copy / noise_jlens
            ref = self._ref.apply(H, t)  # the reference arm's transported states
            mag = (ref - h_t).norm(dim=-1, keepdim=True)  # [L, P, 1] per-cell perturbation size to match
            g = torch.Generator().manual_seed(self.seed * 1_000_003 + key * 1_009 + t)  # CPU RNG: same on any device
            u = torch.randn(H.shape, generator=g, dtype=torch.float64).to(H.device, H.dtype)  # isotropic direction
            X = h_t + mag * u / u.norm(dim=-1, keepdim=True)  # same size, random direction
        X[t] = h_t  # exact no-op on the diagonal for every arm
        return X

    def diagnostics(self, H, t):
        """Per-call geometry diagnostics (off-diagonal rows): relative perturbation ||x - h_t|| / ||h_t|| and, for
        J-lens arms, the readout-preservation error ||J_t x - J_s h_s|| / ||J_s h_s|| (0 for an exact inverse)."""
        X = self.apply(H, t)
        off = [s for s in range(H.shape[0]) if s != t]  # diagonal rows are no-ops by construction
        h_t = H[t]
        out = {"rel_perturbation": float(((X[off] - h_t).norm(dim=-1) / h_t.norm(dim=-1)).mean())}
        if self.J is not None:
            J = self.J.to(H.device, H.dtype)
            read_x = X[off] @ J[t].T  # J_t x: readout of the transported state at layer t
            read_s = torch.einsum("lpd,led->lpe", H[off], J[off])  # J_s h_s: readout of the source state
            out["readout_error"] = float(((read_x - read_s).norm(dim=-1) / read_s.norm(dim=-1)).mean())
        return out


def select_ridge(J_A, J_B, states, grid, tau):
    """Pick the ridge strength c by split-half stability (never by exact-match accuracy).

    Local: for each c, transport the same states with J-lens matrices fitted on two disjoint prompt halves (A, B) and
    measure instability = sum ||delta_A - delta_B|| / sum ||(delta_A + delta_B)/2|| over off-diagonal cells, where
    delta = x - h_t. Returns the smallest c (least shrinkage toward no-patch) with instability <= tau, else max(grid).
    Global context: the J-lens estimate is an average over prompts (jlens.fit); a c whose transport changes a lot
    between halves is fitting estimation noise. Split-half reliability is the standard check for an averaged
    estimator (Spearman-Brown / split-half reliability, https://en.wikipedia.org/wiki/Spearman%E2%80%93Brown_prediction_formula).
    Args:
        states: list of [L, P, d] clean entity states (from two-hop prompts; no answers used).
    Returns:
        (c, table) with table = [{c, instability, rel_perturbation}] in grid order.
    """
    table = []
    for c in grid:
        TA, TB = Transport("jlens", J=J_A, c=c), Transport("jlens", J=J_B, c=c)  # same c, two independent fits
        diff = mean = rel = 0.0
        n = 0
        for H in states:
            L = H.shape[0]
            for t in range(L):
                off = [s for s in range(L) if s != t]
                dA, dB = TA.apply(H, t)[off] - H[t], TB.apply(H, t)[off] - H[t]  # perturbations of the two halves
                diff += float((dA - dB).norm(dim=-1).sum())  # disagreement between halves
                mean += float(((dA + dB) / 2).norm(dim=-1).sum())  # size of the (average) correction
                rel += float((((dA + dB) / 2).norm(dim=-1) / H[t].norm(dim=-1)).sum())
                n += dA.shape[0] * dA.shape[1]
        table.append({"c": c, "instability": diff / max(mean, 1e-12), "rel_perturbation": rel / max(n, 1)})
    stable = [r["c"] for r in table if r["instability"] <= tau]  # candidates whose correction is mostly signal
    return (min(stable) if stable else max(grid)), table
