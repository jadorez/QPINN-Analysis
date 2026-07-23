"""
ntk.py — empirical Neural Tangent Kernel / spectral-bias analysis.

Implements the Canatar-Bordelon-Pehlevan "task-model alignment" diagnostic
(and its PINN-specific reading via Wang-Yu-Perdikaris): given a model's NTK
at initialization, how much of a target function's power lands in the
kernel's top eigendirections predicts how well/fast the (linearized) model
can learn it. Used here to compare quantum vs. classical models on genuinely
different grounds than entanglement entropy or Fisher information (see
notebook section 6) — this is about representational/spectral alignment with
the target, not "quantumness".

Usage example:
    from src.ntk import empirical_ntk, spectral_power_curve, project_low_eigenvalue_subspace

    eigvals, eigvecs = empirical_ntk(model, xt_grid)          # at initialization
    C_k = spectral_power_curve(eigvecs, eigvals, u_true_flat)  # cumulative target power
    error_low = project_low_eigenvalue_subspace(eigvecs, eigvals, error_flat, frac=0.5)
"""

from typing import Tuple
import torch
import torch.nn as nn


def output_jacobian(model: nn.Module, xt: torch.Tensor, output_index: int = 0) -> torch.Tensor:
    """
    Per-example gradient of a model's scalar output w.r.t. every trainable
    parameter, i.e. the NTK feature map: J[i, :] = d output[i] / d theta.

    One forward pass on the full batch, then one backward pass per grid
    point (can't use the grad-of-batch-sum trick from utils.py's QFI/CFI
    machinery here — NTK needs each row of the Jacobian separately, not
    their sum). Validated against the analytic case of a linear model
    f(x) = theta^T phi(x), where this must equal phi(x) exactly (see tests).

    Cost is dominated by the per-point backward passes: ~12s for N=900 on
    MerlinTunableSimple (P=310 params) on this M3's MPS backend, ~1.4s for a
    parameter-matched ClassicalTunableSimple. Intended for evaluation-time
    diagnostics on modest grids, not per-step training use.

    Args:
        model: nn.Module; forward(xt) -> (output, *extras). Must NOT be
            called under torch.no_grad().
        xt: (N, d) input tensor.
        output_index: which element of forward()'s return tuple is the
            scalar-per-sample output to differentiate (default 0, i.e. `u`).

    Returns:
        J: (N, P) real tensor, detached, where P = total trainable param count.
    """
    params = [p for p in model.parameters() if p.requires_grad]
    out = model(xt)
    u = out[output_index].reshape(-1)
    n = u.shape[0]
    p_dim = sum(p.numel() for p in params)

    J = torch.empty(n, p_dim, dtype=xt.dtype, device="cpu")
    for i in range(n):
        grads = torch.autograd.grad(u[i], params, retain_graph=True, allow_unused=True)
        J[i] = torch.cat([
            g.reshape(-1).cpu() if g is not None else torch.zeros(p.numel())
            for g, p in zip(grads, params)
        ])
    return J.detach()


def empirical_ntk(model: nn.Module, xt: torch.Tensor, output_index: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Empirical NTK Gram matrix on a fixed set of collocation points, eigendecomposed.

    Gram = J @ J.T where J is `output_jacobian`; this is the standard
    finite-width empirical NTK (Jacot et al. 2018), evaluated at whatever
    parameter values `model` currently holds — pass a freshly-initialized
    model for the usual "at initialization" NTK theory (Canatar et al.,
    Wang-Yu-Perdikaris), which is what the linearized-training predictions
    this module is built around actually assume.

    Args:
        model, xt, output_index: see `output_jacobian`.

    Returns:
        eigvals: (N,) real tensor, descending order.
        eigvecs: (N, N) real tensor, eigvals[i] <-> eigvecs[:, i], orthonormal
            (so a target vector's power decomposes exactly across them).
    """
    J = output_jacobian(model, xt, output_index=output_index)
    gram = J @ J.T
    gram = 0.5 * (gram + gram.T)  # symmetrize away any fp asymmetry before eigh
    eigvals, eigvecs = torch.linalg.eigh(gram.double())
    eigvals = eigvals.flip(0).float()
    eigvecs = eigvecs.flip(1).float()
    return eigvals, eigvecs


def spectral_power_curve(eigvecs: torch.Tensor, eigvals: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Cumulative fraction of a target function's power captured by the top-k
    NTK eigendirections, for every k. C(k) = sum_{i<=k} <v_i, target>^2 / ||target||^2.

    Since `eigvecs` is a complete orthonormal basis for R^N, C(N) = 1 exactly
    (trivial). The informative part is how fast C(k) rises with k relative to
    the model's effective rank (~min(N, param count)): faster saturation ==
    the target is concentrated in the kernel's dominant (fast-learning under
    linearized training) directions == better task-model alignment.

    Args:
        eigvecs: (N, N) from `empirical_ntk`.
        eigvals: (N,) unused directly (kept for a consistent call signature
            alongside `project_low_eigenvalue_subspace`), included so callers
            don't need to track which order eigvecs' columns are in separately.
        target: (N,) real tensor, e.g. the exact PDE solution flattened over
            the same grid `xt` used for `empirical_ntk`.

    Returns:
        C: (N,) real tensor, cumulative power curve, C[k-1] = C(k).
    """
    coeffs = eigvecs.T @ target.to(eigvecs.dtype)  # (N,), <v_i, target>
    power = coeffs ** 2
    total = power.sum().clamp_min(1e-12)
    return torch.cumsum(power, dim=0) / total


def project_low_eigenvalue_subspace(eigvecs: torch.Tensor, eigvals: torch.Tensor, field: torch.Tensor, frac: float = 0.5) -> torch.Tensor:
    """
    Reconstructs `field` using only its components along the bottom `frac`
    fraction of NTK eigendirections (smallest eigenvalues = slowest-learning
    directions under linearized/lazy training). Applying this to a *trained*
    model's pointwise error field localizes where persistent error is
    plausibly explained by slow NTK convergence, rather than by e.g.
    insufficient training time or collocation density elsewhere.

    Args:
        eigvecs, eigvals: from `empirical_ntk` (eigvals descending, unused
            for value — only the resulting column order of eigvecs matters —
            but accepted explicitly so the "low" cutoff is unambiguous).
        field: (N,) real tensor over the same grid, e.g. |u_pred - u_true|.
        frac: fraction (by count) of the smallest-eigenvalue directions to keep.

    Returns:
        field_low: (N,) real tensor, the projection of `field` onto that subspace.
    """
    n = eigvecs.shape[0]
    k_low = max(1, int(round(frac * n)))
    v_low = eigvecs[:, -k_low:]  # smallest eigenvalues are last (eigvals descending)
    coeffs = v_low.T @ field.to(eigvecs.dtype)
    return v_low @ coeffs
