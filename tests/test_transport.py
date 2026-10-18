"""Unit tests of jtransport/transport.py: the cross-layer transport operators (CPU only, random matrices).

Local: each test pins one algebraic property of an arm of the self-patching grid.
Global context: the grid (jtransport/grid.py) writes Transport.apply(H, t)[s] into layer t at the head-entity tokens;
these properties guarantee that the diagonal stays a no-op patch, that `copy` reproduces the paper, and that `jlens`
is the J-lens map pinv(J_t) J_s of https://github.com/anthropics/jacobian-lens in its exact-inverse limit.
Run: uv run pytest tests/test_transport.py
"""
import pytest
import torch

from jtransport.transport import Transport, ridge_inverse, select_ridge, stack_jacobians

# Small sizes keep the tests instant: L layers, P patched positions, d residual width
L, P, D = 5, 3, 8


def _states(seed=0):
    """Random clean states H[l, p, :] of all L layers at P positions (stand-in for blocks.l.hook_resid_post[E])."""
    g = torch.Generator().manual_seed(seed)  # local generator: deterministic and isolated from global RNG state
    return torch.randn(L, P, D, generator=g, dtype=torch.float64)  # float64 so exactness tests are tight


def _jacobians(seed=1):
    """Random well-conditioned J_l (identity + small noise) for l < L-1, and J_{L-1} = I as in the J-lens setup."""
    g = torch.Generator().manual_seed(seed)
    fitted = {l: torch.eye(D, dtype=torch.float64) + 0.3 * torch.randn(D, D, generator=g, dtype=torch.float64)
              for l in range(L - 1)}  # jlens fits only source layers below the final layer
    return stack_jacobians(fitted, L)  # appends J_{L-1} = I


def test_stack_jacobians_appends_identity_for_last_layer():
    J = _jacobians()
    assert J.shape == (L, D, D)  # one d x d matrix per layer
    assert torch.equal(J[L - 1], torch.eye(D, dtype=J.dtype))  # d h_final / d h_final = I


def test_copy_is_the_papers_raw_copy():
    H = _states()
    X = Transport("copy").apply(H, t=2)
    assert torch.equal(X, H)  # row s is h_s itself, whatever the target layer


@pytest.mark.parametrize("method", ["copy", "jlens", "norm", "noise_copy", "noise_jlens"])
def test_diagonal_is_exact_no_op(method):
    H, J = _states(), _jacobians()
    for t in range(L):
        X = Transport(method, J=J, c=0.01, seed=3).apply(H, t=t, key=7)
        assert torch.equal(X[t], H[t])  # patching layer t onto itself must not change anything


def test_jlens_with_identity_jacobians_and_no_ridge_equals_copy():
    H = _states()
    J = torch.eye(D, dtype=torch.float64).expand(L, D, D).clone()  # logit-lens special case J_l = I
    X = Transport("jlens", J=J, c=0.0).apply(H, t=1)
    assert torch.allclose(X, H, atol=1e-10)  # shared coordinates => transport degenerates to the raw copy


def test_jlens_exact_inverse_limit_is_pinv_jt_js():
    H, J = _states(), _jacobians()
    t = 1
    X = Transport("jlens", J=J, c=0.0).apply(H, t=t)
    for s in range(L):
        expected = H[s] @ (torch.linalg.pinv(J[t]) @ J[s]).T  # (pinv(J_t) J_s h_s) for row vectors h_s
        assert torch.allclose(X[s], expected, atol=1e-8)


def test_jlens_preserves_the_j_lens_readout():
    H, J = _states(), _jacobians()
    t = 3
    X = Transport("jlens", J=J, c=0.0).apply(H, t=t)
    for s in range(L):
        # J_t x = J_s h_s: the transported state is disposed to make the model say what h_s was disposed to say
        assert torch.allclose(X[s] @ J[t].T, H[s] @ J[s].T, atol=1e-8)


def test_ridge_inverse_matches_formula_and_shrinks():
    J = _jacobians()[0]
    c = 0.1
    R = ridge_inverse(J, c)
    lam = c * torch.linalg.matrix_norm(J, ord=2)  # lambda = c * sigma_max(J)
    expected = J.T @ torch.linalg.inv(J @ J.T + lam ** 2 * torch.eye(D, dtype=J.dtype))
    assert torch.allclose(R, expected, atol=1e-10)
    assert torch.allclose(ridge_inverse(J, 0.0), torch.linalg.inv(J), atol=1e-8)  # c = 0 is the exact inverse
    # more ridge => smaller correction (shrinks toward the no-patch state h_t, not toward noise)
    assert torch.linalg.matrix_norm(ridge_inverse(J, 1.0)) < torch.linalg.matrix_norm(ridge_inverse(J, 0.01))


def test_norm_control_rescales_to_target_norm_and_keeps_direction():
    H = _states()
    t = 0
    X = Transport("norm").apply(H, t=t)
    assert torch.allclose(X.norm(dim=-1), H[t].norm(dim=-1).expand(L, P))  # ||x|| = ||h_t|| per position
    cos = torch.nn.functional.cosine_similarity(X, H, dim=-1)
    assert torch.allclose(cos, torch.ones_like(cos))  # same direction as h_s


@pytest.mark.parametrize("ref", ["copy", "jlens"])
def test_noise_matches_perturbation_size_of_reference_but_not_direction(ref):
    H, J = _states(), _jacobians()
    t = 2
    X_ref = Transport(ref, J=J, c=0.01).apply(H, t=t)
    X_noise = Transport(f"noise_{ref}", J=J, c=0.01, seed=5).apply(H, t=t, key=11)
    d_ref, d_noise = X_ref - H[t], X_noise - H[t]  # perturbations relative to the clean target state
    assert torch.allclose(d_noise.norm(dim=-1), d_ref.norm(dim=-1), atol=1e-10)  # matched magnitude per cell
    off = [s for s in range(L) if s != t]
    cos = torch.nn.functional.cosine_similarity(d_noise[off], d_ref[off], dim=-1)
    assert cos.abs().max() < 0.99  # random direction, not the reference one
    X_again = Transport(f"noise_{ref}", J=J, c=0.01, seed=5).apply(H, t=t, key=11)
    assert torch.equal(X_noise, X_again)  # deterministic given (seed, key)


def test_select_ridge_prefers_smallest_stable_c_and_falls_back_to_largest():
    J = _jacobians()
    states = [_states(i) for i in range(3)]
    c, table = select_ridge(J, J.clone(), states, grid=[0.1, 0.01, 0.001], tau=0.3)
    assert c == 0.001 and len(table) == 3  # identical halves => zero disagreement => least-biased (smallest) c
    c_fb, _ = select_ridge(J, J.clone(), states, grid=[0.1, 0.01], tau=-1.0)
    assert c_fb == 0.1  # nothing passes tau => strongest ridge


def test_unknown_method_raises():
    with pytest.raises(ValueError):
        Transport("rotate")
