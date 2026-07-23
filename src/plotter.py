"""
plotter.py — plotting utilities for QPINN / classical PINN results.

Functions:
    plot_training_history(history, ...)
    plot_solution_comparison(U_pred, U_true, x_grid, t_grid, ...)
    --- Fourier analysis ---
    plot_power_spectrum(U_pred, U_true, ...)
    plot_mode_l2_errors(U_pred, U_true, ...)
    plot_cumulative_energy(U_pred, U_true, ...)
    plot_frequency_sweep(sweep_results, ...)
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

# ── helpers ───────────────────────────────────────────────────────────────────

def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _save(fig: plt.Figure, save_path):
    if save_path:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved: {path}")


# ── standard training / solution plots ────────────────────────────────────────

def plot_training_history(
    history: np.ndarray,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Semilogy plot of every loss component stored in `history`.

    Args:
        history:   (epochs, 6) array — [total, pde, consistency, initial, boundary, entropy]
        save_path: full file path to save to (None = skip)
        show:      call plt.show()
    Returns:
        fig
    """
    _HISTORY_LABELS = ["total", "PDE", "consistency", "initial", "boundary", "entropy"]
    labels = _HISTORY_LABELS[:history.shape[1]]
    fig, ax = plt.subplots(figsize=(10,5))
    for i, label in enumerate(labels):
        ax.semilogy(history[:, i], label=label, linewidth=2, alpha=0.85)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title("Training Hisotry", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, save_path)
    if show:
        plt.show()
    return fig


def plot_solution_comparison(
    U_pred: Union[torch.Tensor, np.ndarray],
    U_true: Union[torch.Tensor, np.ndarray],
    x_grid: Optional[Union[torch.Tensor, np.ndarray]] = None,
    t_grid: Optional[Union[torch.Tensor, np.ndarray]] = None,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
) -> Tuple[plt.Figure, float]:
    """
    Three-panel plot: exact | prediction | pointwise error.

    Returns:
        fig, rel_l2
    """
    U_pred_np = _to_numpy(U_pred)
    U_true_np = _to_numpy(U_true)
    rel_l2 = float(np.linalg.norm(U_pred_np - U_true_np) / np.linalg.norm(U_true_np))

    if x_grid is not None and t_grid is not None:
        xv, tv = _to_numpy(x_grid).squeeze(), _to_numpy(t_grid).squeeze()
        extent = [float(tv[0]), float(tv[-1]), float(xv[0]), float(xv[-1])]
    else:
        extent = [0, 1, 0, 1]

    fig, axes = plt.subplots(1, 3, figsize=(15,4))
    title = "Solution"
    panels = [
        (U_true_np,             f"{title} — Exact"),
        (U_pred_np,             f"{title} — Prediction"),
        (U_pred_np - U_true_np, f"Error  (rel L2 = {rel_l2:.2e})"),
    ]
    for ax, (data, ptitle) in zip(axes, panels):
        im = ax.imshow(data, origin="lower", extent=extent, aspect="auto", cmap="RdBu_r")
        ax.set_xlabel("t", fontsize=11)
        ax.set_ylabel("x", fontsize=11)
        ax.set_title(ptitle, fontsize=12)
        fig.colorbar(im, ax=ax, label="u" if "Error" not in ptitle else "Δu")
    fig.tight_layout()
    _save(fig, save_path)
    if show:
        plt.show()
    return fig, rel_l2


def plot_entropy_field(
    entropy_grid: Union[torch.Tensor, np.ndarray],
    x_grid: Optional[Union[torch.Tensor, np.ndarray]] = None,
    t_grid: Optional[Union[torch.Tensor, np.ndarray]] = None,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Single-panel heatmap of Von Neumann entanglement entropy as a function of
    (x, t), e.g. at the end of training. Get `entropy_grid` from
    `eval.evaluate()`'s `metrics["entropy_grid"]` (only populated for models
    whose forward() returns a per-sample entropy tensor rather than an
    already-reduced scalar).

    Args:
        entropy_grid: (nx, nt) array of per-point entropy
        x_grid:       spatial coordinates, shape (nx, 1) or (nx,)
        t_grid:       temporal coordinates, shape (nt, 1) or (nt,)
        save_path:    full file path to save to (None = skip)
        show:         call plt.show()
    Returns:
        fig
    """
    entropy_np = _to_numpy(entropy_grid)

    if x_grid is not None and t_grid is not None:
        xv, tv = _to_numpy(x_grid).squeeze(), _to_numpy(t_grid).squeeze()
        extent = (float(tv[0]), float(tv[-1]), float(xv[0]), float(xv[-1]))
    else:
        extent = (0.0, 1.0, 0.0, 1.0)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(entropy_np, origin="lower", extent=extent, aspect="auto", cmap="viridis")
    ax.set_xlabel("t", fontsize=11)
    ax.set_ylabel("x", fontsize=11)
    ax.set_title("Entanglement entropy — S(x, t)", fontsize=12)
    fig.colorbar(im, ax=ax, label="Von Neumann entropy")
    fig.tight_layout()
    _save(fig, save_path)
    if show:
        plt.show()
    return fig


def plot_qfi_field(
    qfi_grid: Union[torch.Tensor, np.ndarray],
    x_grid: Optional[Union[torch.Tensor, np.ndarray]] = None,
    t_grid: Optional[Union[torch.Tensor, np.ndarray]] = None,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Four-panel heatmap of the Quantum Fisher Information Matrix (QFIM) as a
    function of (x, t): F_xx, F_tt, F_xt (off-diagonal), and Tr(F) = F_xx+F_tt.
    Get `qfi_grid` from `eval.evaluate(..., compute_qfi=True)`'s
    `metrics["qfi_grid"]`. Higher values mean the quantum-encoded state is
    more sensitive to — i.e. more easily distinguishes — small shifts in that
    input near that point.

    Args:
        qfi_grid: (nx, nt, 2, 2) array, [[F_xx, F_xt], [F_xt, F_tt]] per point
        x_grid:   spatial coordinates, shape (nx, 1) or (nx,)
        t_grid:   temporal coordinates, shape (nt, 1) or (nt,)
        save_path: full file path to save to (None = skip)
        show:      call plt.show()
    Returns:
        fig
    """
    qfi_np = _to_numpy(qfi_grid)

    if x_grid is not None and t_grid is not None:
        xv, tv = _to_numpy(x_grid).squeeze(), _to_numpy(t_grid).squeeze()
        extent = (float(tv[0]), float(tv[-1]), float(xv[0]), float(xv[-1]))
    else:
        extent = (0.0, 1.0, 0.0, 1.0)

    panels = [
        (qfi_np[..., 0, 0], "F_xx"),
        (qfi_np[..., 1, 1], "F_tt"),
        (qfi_np[..., 0, 1], "F_xt"),
        (qfi_np[..., 0, 0] + qfi_np[..., 1, 1], "Tr(F) = F_xx + F_tt"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    for ax, (data, title) in zip(axes.flat, panels):
        im = ax.imshow(data, origin="lower", extent=extent, aspect="auto", cmap="magma")
        ax.set_xlabel("t", fontsize=11)
        ax.set_ylabel("x", fontsize=11)
        ax.set_title(title, fontsize=12)
        fig.colorbar(im, ax=ax)
    fig.suptitle("Quantum Fisher Information vs. (x, t)", fontsize=14)
    fig.tight_layout()
    _save(fig, save_path)
    if show:
        plt.show()
    return fig


def plot_fisher_efficiency_field(
    qfi_grid: Union[torch.Tensor, np.ndarray],
    cfi_grid: Union[torch.Tensor, np.ndarray],
    x_grid: Optional[Union[torch.Tensor, np.ndarray]] = None,
    t_grid: Optional[Union[torch.Tensor, np.ndarray]] = None,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Three-panel comparison of Tr(QFI) — the encoded state's theoretical
    resolving power under an ideal measurement — against Tr(CFI) — what the
    model's actual Fock-basis probability readout realizes — and their ratio
    Tr(CFI)/Tr(QFI), the fraction of theoretical sensitivity the readout
    actually captures at each (x, t). Get qfi_grid/cfi_grid from
    `eval.evaluate(..., compute_qfi=True)`'s `metrics["qfi_grid"]` /
    `metrics["cfi_grid"]`.

    A low efficiency somewhere means the quantum encoding carries sensitivity
    to (x, t) there that the readout can't access — headroom for a better
    measurement/readout design, not evidence the encoding itself is weak.

    Args:
        qfi_grid, cfi_grid: (nx, nt, 2, 2) arrays, same layout as plot_qfi_field
        x_grid, t_grid, save_path, show: see plot_qfi_field
    Returns:
        fig
    """
    qfi_trace = _to_numpy(qfi_grid)[..., 0, 0] + _to_numpy(qfi_grid)[..., 1, 1]
    cfi_trace = _to_numpy(cfi_grid)[..., 0, 0] + _to_numpy(cfi_grid)[..., 1, 1]
    efficiency = cfi_trace / np.clip(qfi_trace, 1e-12, None)

    if x_grid is not None and t_grid is not None:
        xv, tv = _to_numpy(x_grid).squeeze(), _to_numpy(t_grid).squeeze()
        extent = (float(tv[0]), float(tv[-1]), float(xv[0]), float(xv[-1]))
    else:
        extent = (0.0, 1.0, 0.0, 1.0)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    panels = [
        (qfi_trace, "Tr(QFI) — theoretical resolving power", "magma", None),
        (cfi_trace, "Tr(CFI) — realized by actual readout", "magma", None),
        (efficiency, "Tr(CFI) / Tr(QFI) — extraction efficiency", "viridis", (0.0, 1.0)),
    ]
    for ax, (data, title, cmap, clim) in zip(axes, panels):
        im = ax.imshow(data, origin="lower", extent=extent, aspect="auto", cmap=cmap,
                        vmin=clim[0] if clim else None, vmax=clim[1] if clim else None)
        ax.set_xlabel("t", fontsize=11)
        ax.set_ylabel("x", fontsize=11)
        ax.set_title(title, fontsize=11)
        fig.colorbar(im, ax=ax)
    fig.tight_layout()
    _save(fig, save_path)
    if show:
        plt.show()
    return fig


def plot_ntk_spectrum(
    eigvals_by_model: Dict[str, Union[torch.Tensor, np.ndarray]],
    param_counts: Optional[Dict[str, int]] = None,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Log-scale empirical-NTK eigenvalue decay for one or more models, overlaid.
    Get eigvals from `ntk.empirical_ntk(model, xt)`.

    Faster decay = the kernel's "effective dimension" is smaller, i.e. fewer
    directions carry most of the representational capacity — the spectral-bias
    counterpart to a model's raw parameter count. If `param_counts` is given, a
    vertical dashed line marks each model's parameter count (P), since a
    finite-width empirical NTK Gram on N>P points has rank <= P, so eigenvalues
    should fall off sharply near index P.

    Args:
        eigvals_by_model: {model_label: eigvals}, each (N,) descending
        param_counts:      optional {model_label: P} to mark rank cutoffs
        save_path:         full file path to save to (None = skip)
        show:               call plt.show()
    Returns:
        fig
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    for label, eigvals in eigvals_by_model.items():
        ev = _to_numpy(eigvals)
        ev_clipped = np.clip(ev, 1e-12, None)
        ax.semilogy(np.arange(1, len(ev) + 1), ev_clipped, lw=2, alpha=0.85, label=label)
        if param_counts and label in param_counts:
            ax.axvline(param_counts[label], linestyle="--", alpha=0.5,
                        color=ax.lines[-1].get_color())
    ax.set_xlabel("Eigenvalue index", fontsize=12)
    ax.set_ylabel("Eigenvalue (log scale)", fontsize=12)
    ax.set_title("Empirical NTK spectrum", fontsize=13)
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=10)
    fig.tight_layout()
    _save(fig, save_path)
    if show:
        plt.show()
    return fig


def plot_spectral_alignment(
    C_by_model: Dict[str, Union[torch.Tensor, np.ndarray]],
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Cumulative target-power curves C(k) for one or more models, overlaid —
    the fraction of the target function's power captured by each model's
    top-k NTK eigendirections. Get C from `ntk.spectral_power_curve(...)`.

    C(k) rising faster (saturating near 1 at smaller k) means the target is
    more concentrated in that model's dominant, fast-learning-under-linearized
    -training directions — i.e. better task-model spectral alignment. All
    curves reach exactly 1.0 at k=N (trivial, eigvecs are a complete basis);
    the informative comparison is the shape/rise, not the endpoint.

    Args:
        C_by_model: {model_label: C}, each (N,) non-decreasing, C[-1]==1
        save_path:  full file path to save to (None = skip)
        show:       call plt.show()
    Returns:
        fig
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    for label, C in C_by_model.items():
        c = _to_numpy(C)
        ax.plot(np.arange(1, len(c) + 1), c, lw=2, alpha=0.85, label=label)
    ax.set_xlabel("k (top-k NTK eigendirections)", fontsize=12)
    ax.set_ylabel("C(k) — cumulative target power captured", fontsize=12)
    ax.set_title("Task-model spectral alignment", fontsize=13)
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    fig.tight_layout()
    _save(fig, save_path)
    if show:
        plt.show()
    return fig


def plot_low_eigenvalue_error_field(
    actual_error_grid: Union[torch.Tensor, np.ndarray],
    low_eig_error_grid: Union[torch.Tensor, np.ndarray],
    x_grid: Optional[Union[torch.Tensor, np.ndarray]] = None,
    t_grid: Optional[Union[torch.Tensor, np.ndarray]] = None,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Two-panel comparison: a trained model's actual pointwise error |u_pred -
    u_true|, vs. that same error field projected onto the bottom (slowest-
    learning) fraction of the model's at-initialization NTK eigendirections
    (`ntk.project_low_eigenvalue_subspace`). Visual/quantitative agreement
    between the two panels is evidence that persistent error is explained by
    NTK spectral bias rather than e.g. insufficient training or collocation
    density.

    Args:
        actual_error_grid, low_eig_error_grid: (nx, nt) arrays
        x_grid, t_grid: spatial/temporal coordinates, shape (nx,1)/(nt,1) or (nx,)/(nt,)
        save_path:      full file path to save to (None = skip)
        show:            call plt.show()
    Returns:
        fig
    """
    actual_np = _to_numpy(actual_error_grid)
    low_np = _to_numpy(low_eig_error_grid)

    if x_grid is not None and t_grid is not None:
        xv, tv = _to_numpy(x_grid).squeeze(), _to_numpy(t_grid).squeeze()
        extent = (float(tv[0]), float(tv[-1]), float(xv[0]), float(xv[-1]))
    else:
        extent = (0.0, 1.0, 0.0, 1.0)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    panels = [
        (actual_np, "Actual |u_pred - u_true|"),
        (low_np, "Error projected onto low-eigenvalue NTK subspace"),
    ]
    for ax, (data, title) in zip(axes, panels):
        im = ax.imshow(data, origin="lower", extent=extent, aspect="auto", cmap="inferno")
        ax.set_xlabel("t", fontsize=11)
        ax.set_ylabel("x", fontsize=11)
        ax.set_title(title, fontsize=11)
        fig.colorbar(im, ax=ax)
    fig.tight_layout()
    _save(fig, save_path)
    if show:
        plt.show()
    return fig


def plot_entropy_accuracy_trajectory(
    epochs: Union[np.ndarray, list],
    entropy_mean: Union[np.ndarray, list],
    entropy_std: Union[np.ndarray, list],
    rel_l2_mean: Union[np.ndarray, list],
    rel_l2_std: Union[np.ndarray, list],
    errorevery: Optional[int] = None,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Two-panel trajectory plot: entanglement entropy and rel L2 accuracy vs. epoch,
    each averaged over seeds with std shown as error bars.

    Args:
        epochs:       epoch indices, shape (n_points,)
        entropy_mean: mean entropy across seeds, shape (n_points,)
        entropy_std:  std of entropy across seeds, shape (n_points,)
        rel_l2_mean:  mean rel L2 error across seeds, shape (n_points,)
        rel_l2_std:   std of rel L2 error across seeds, shape (n_points,)
        errorevery:   spacing between drawn error bars (avoids overplotting).
                      Defaults to ~15 bars spread across the trajectory, so
                      short runs (e.g. smoke tests) don't end up with a single
                      bar at epoch 1 and nothing else.
        save_path:    full file path to save to (None = skip)
        show:         call plt.show()
    Returns:
        fig
    """
    if errorevery is None:
        errorevery = max(1, len(epochs) // 15)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax1.errorbar(epochs, entropy_mean, yerr=entropy_std, errorevery=errorevery,
                 color='tab:purple', linewidth=2, ecolor='tab:purple', alpha=0.85,
                 capsize=3, label='Entanglement entropy')
    ax1.set_ylabel("Von Neumann entropy", fontsize=12)
    ax1.set_title("Entanglement entropy vs. training epoch", fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)

    ax2.errorbar(epochs, rel_l2_mean, yerr=rel_l2_std, errorevery=errorevery,
                 color='tab:green', linewidth=2, ecolor='tab:green', alpha=0.85,
                 capsize=3, label='Relative L2 error')
    ax2.set_yscale("log")
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Relative L2 error", fontsize=12)
    ax2.set_title("Model accuracy vs. training epoch", fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)

    fig.tight_layout()
    _save(fig, save_path)
    if show:
        plt.show()
    return fig


def plot_model_comparison_trajectory(
    epochs: Union[np.ndarray, list],
    rel_l2_by_model: Dict[str, Tuple[np.ndarray, np.ndarray]],
    entropy_by_model: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]] = None,
    errorevery: Optional[int] = None,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Overlays rel L2 (and optionally entanglement entropy) trajectories for
    several models on shared axes, each averaged over seeds with std as error
    bars. Use this to compare e.g. quantum models with different
    n_encoding_layers against a parameter-matched classical baseline.

    Args:
        epochs:           epoch indices, shape (n_points,)
        rel_l2_by_model:  {model_label: (mean, std)} arrays, each (n_points,)
        entropy_by_model: optional {model_label: (mean, std)}; entropy panel is
                           omitted entirely if None (e.g. an all-classical run)
        errorevery:       spacing between drawn error bars. Defaults to ~15
                           bars spread across the trajectory.
        save_path:        full file path to save to (None = skip)
        show:             call plt.show()
    Returns:
        fig
    """
    if errorevery is None:
        errorevery = max(1, len(epochs) // 15)

    n_panels = 2 if entropy_by_model else 1
    fig, axes = plt.subplots(n_panels, 1, figsize=(10, 4 * n_panels), sharex=True, squeeze=False)
    axes = axes[:, 0]

    if entropy_by_model:
        ax_ent, ax_l2 = axes
        for label, (mean, std) in entropy_by_model.items():
            ax_ent.errorbar(epochs, mean, yerr=std, errorevery=errorevery, label=label,
                             linewidth=2, alpha=0.85, capsize=3)
        ax_ent.set_ylabel("Von Neumann entropy", fontsize=12)
        ax_ent.set_title("Entanglement entropy vs. training epoch", fontsize=14)
        ax_ent.grid(True, alpha=0.3)
        ax_ent.legend(fontsize=10)
    else:
        (ax_l2,) = axes

    for label, (mean, std) in rel_l2_by_model.items():
        ax_l2.errorbar(epochs, mean, yerr=std, errorevery=errorevery, label=label,
                        linewidth=2, alpha=0.85, capsize=3)
    ax_l2.set_yscale("log")
    ax_l2.set_xlabel("Epoch", fontsize=12)
    ax_l2.set_ylabel("Relative L2 error", fontsize=12)
    ax_l2.set_title("Model accuracy vs. training epoch", fontsize=14)
    ax_l2.grid(True, alpha=0.3)
    ax_l2.legend(fontsize=10)

    fig.tight_layout()
    _save(fig, save_path)
    if show:
        plt.show()
    return fig


def seed_l2_comparison(l2_array_1, l2_array_2, name_1 = "Model 1", name_2 = "Model 2", save_path = None, show = True):
    """"
    Given L2 errors of two models with same settings but varying seed
    Plot errors comparison of two models for each seed as well as mean error

    Args:
        l2_array_1:     L2 error of model 1 for each seed
        l2_array_2:     L2 error of model 2 for each seed
        name_1,2:       names of models
        save_path:      optional save path for figure
        show:           set to False to not show figure
    """
    # x-coordinates (indices)
    x = np.arange(len(l2_array_1))

    # Plot points
    plt.scatter(x, l2_array_1, color='blue', label=name_1)
    plt.scatter(x, l2_array_2, color='red', label=name_2)

    # means
    mean1 = np.mean(l2_array_1)
    mean2 = np.mean(l2_array_2)

    # Connect corresponding points
    for i in range(len(l2_array_1)):
        plt.plot(
            [x[i], x[i]],
            [l2_array_1[i], l2_array_2[i]],
            linestyle='--',
            color='gray',
            linewidth=1
        )

    # Mean lines
    plt.axhline(mean1, color='blue', linewidth=2,
                label=name_1 + f' mean = {mean1:.2f}', alpha=0.35)
    plt.axhline(mean2, color='red', linewidth=2,
                label=name_2 + f' mean = {mean2:.2f}', alpha=0.35)

    plt.xlabel('Seed index')
    plt.ylabel('L2 Error')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path is not None:
        _save(plt.gcf(), save_path)

    if show : plt.show()

# ── Fourier analysis plots ─────────────────────────────────────────────────────
def plot_power_spectrum(
    U_pred: Union[torch.Tensor, np.ndarray],
    U_true: Union[torch.Tensor, np.ndarray],
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
    exclude_0_mode: bool = True,
) -> plt.Figure:
    """
    Power spectrum of predicted vs true solution (averaged over time),
    shown only on semilogy axes.

    Args:
        U_pred / U_true: solution arrays of shape (nx, nt)
    Returns:
        fig
    """
    U_pred_t = torch.as_tensor(_to_numpy(U_pred))
    U_true_t = torch.as_tensor(_to_numpy(U_true))

    pred_hat = torch.fft.fft(U_pred_t, dim=0)
    true_hat = torch.fft.fft(U_true_t, dim=0)
    power_pred = (pred_hat.abs() ** 2).mean(dim=1)
    power_true = (true_hat.abs() ** 2).mean(dim=1)

    nx = U_pred_t.shape[0]
    freqs = np.fft.fftfreq(nx)[:nx // 2]
    pp = power_pred[:nx // 2].numpy()
    pt = power_true[:nx // 2].numpy()

    start = 1 if exclude_0_mode else 0


    fig, ax = plt.subplots(1, 1, figsize=(6, 5))

    ax.semilogy(freqs[start:], pt[start:], "o-", label="True", lw=2, ms=4)
    ax.semilogy(freqs[start:], pp[start:], "s--", label="Predicted", lw=2, ms=4, alpha=0.75)

    ax.set_xlabel("Normalised frequency", fontsize=11)
    ax.set_ylabel("Power", fontsize=11)
    ax.set_title("Power spectrum (semilogy)", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    _save(fig, save_path)

    if show:
        plt.show()

    return fig

def plot_mode_l2_errors(
    U_pred: Union[torch.Tensor, np.ndarray],
    U_true: Union[torch.Tensor, np.ndarray],
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
    exclude_0_mode: bool = True

) -> plt.Figure:
    """
    Two-panel analysis of per-mode L2 errors in Fourier space:
      • semilogy absolute error vs k
      • semilogy relative error vs k

    Args:
        U_pred / U_true: solution arrays of shape (nx, nt)
    Returns:
        fig
    """
    U_pred_t = torch.as_tensor(_to_numpy(U_pred))
    U_true_t = torch.as_tensor(_to_numpy(U_true))

    pred_hat = torch.fft.fft(U_pred_t, dim=0)
    true_hat = torch.fft.fft(U_true_t, dim=0)

    mode_err  = torch.norm(pred_hat - true_hat, dim=1)
    true_norm = torch.norm(true_hat, dim=1)
    rel_err   = mode_err / (true_norm + 1e-8)

    nx = U_pred_t.shape[0]
    half = nx // 2

    start = 1 if exclude_0_mode else 0


    modes = np.arange(start,half)

    me = mode_err[start:half].numpy()
    re = rel_err[start:half].numpy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Absolute error
    ax1.semilogy(modes, me, "o-", lw=2.5, ms=5, color="C0")
    ax1.set_xlabel("Mode k", fontsize=12)
    ax1.set_ylabel("L2 error (log)", fontsize=12)
    ax1.set_title("Absolute L2 error vs mode k", fontsize=13, fontweight="bold")
    ax1.grid(True, alpha=0.3)

    # Relative error
    ax2.semilogy(modes, re, "s-", lw=2.5, ms=5, color="C2")
    ax2.set_xlabel("Mode k", fontsize=12)
    ax2.set_ylabel("Relative L2 error (log)", fontsize=12)
    ax2.set_title("Relative error vs mode k", fontsize=13, fontweight="bold")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    _save(fig, save_path)

    if show:
        plt.show()

    return fig


def plot_multi_model_l2_error(
    U_true: Union[torch.Tensor, np.ndarray],
    U_preds: List[Union[torch.Tensor, np.ndarray]],
    model_names: Optional[List[str]] = None,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
    exclude_0_mode: bool = True,
) -> plt.Figure:
    """
    Plot relative Fourier mode L2 error for multiple models on one semilogy plot.

    Args:
        U_true: shape (nx, nt)
        U_preds: list of model predictions, each shape (nx, nt)
        model_names: optional list of names (default: "model i")
    Returns:
        fig
    """

    def _to_tensor(x):
        if isinstance(x, torch.Tensor):
            return x
        return torch.as_tensor(np.asarray(x))

    U_true_t = _to_tensor(U_true)

    nx = U_true_t.shape[0]
    half = nx // 2

    start = 1 if exclude_0_mode else 0

    modes = np.arange(start, half)

    # true spectrum
    true_hat = torch.fft.fft(U_true_t, dim=0)
    true_norm = torch.norm(true_hat, dim=1)[start:half]

    # names
    if model_names is None:
        model_names = [f"model {i}" for i in range(len(U_preds))]

    if len(model_names) != len(U_preds):
        model_names = [f"model {i}" for i in range(len(U_preds))]

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    for i, (U_pred, name) in enumerate(zip(U_preds, model_names)):
        U_pred_t = _to_tensor(U_pred)

        pred_hat = torch.fft.fft(U_pred_t, dim=0)

        mode_err = torch.norm(pred_hat - true_hat, dim=1)[start:half]
        rel_err = mode_err / (true_norm + 1e-16)

        ax.semilogy(modes, rel_err.numpy(), lw=2, label=name, alpha=0.9)

    ax.set_xlabel("Mode k")
    ax.set_ylabel("Relative L2 error (log scale)")
    ax.set_title("Relative Fourier mode error across models")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend()

    fig.tight_layout()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")

    if show:
        plt.show()

    plt.close(fig)

    return fig


def plot_frequency_sweep(
    sweep_results: Dict[str, List[Dict]],
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
    label: str = 'Frequency',
) -> plt.Figure:
    """
    Final relative-L2 error vs PDE frequency for each model type.

    Args:
        sweep_results: dict mapping model label -> list of dicts with keys
                       'frequency' and 'rel_l2'
                       e.g. {"qpinn": [{"frequency": 0.5, "rel_l2": 0.03}, ...],
                              "classical": [...]}
        save_path: full file path to save to (None = skip)
        show:      call plt.show()
    Returns:
        fig
    """
    fig, ax = plt.subplots(figsize=(9,5))

    for model_label, rows in sweep_results.items():
        rows_sorted = sorted(rows, key=lambda r: r["frequency"])
        freqs  = [r["frequency"] for r in rows_sorted]
        errors = [r["rel_l2"]    for r in rows_sorted]
        errors_mean = [np.mean(r) for r in errors]
        errors_std = [np.std(r) for r in errors]
        #ax.semilogy(freqs, errors, marker="o", lw=2, label=model_label)
        ax.errorbar(freqs, errors_mean, yerr=errors_std / np.sqrt(len(errors[0])), marker="o", lw=2, label=model_label, capsize=3)

    ax.set_xlabel("PDE {label}", fontsize=12)
    ax.set_ylabel("Final relative L2 error", fontsize=12)
    ax.set_title(f"{label} sweep: final error vs {label}", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, save_path)
    if show:
        plt.show()
    return fig


# ── spectral diagnostics (src/spectral.py) ──────────────────────────────────

def plot_omega_eff_trajectory(
    epochs: Union[np.ndarray, list],
    omega_eff_by_feature: Dict[int, Union[np.ndarray, list]],
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
) -> plt.Figure:
    """
    omega_eff (spectral second moment, see spectral.spectral_second_moment)
    vs. training epoch, one line per encoded feature. omega_eff in [0, 1]
    measures how much high-frequency content the quantum layer supplies for
    that feature at that point in training; nan values (layer constant in
    that feature) are simply not plotted for that epoch (matplotlib skips
    NaNs in a line plot automatically).

    Args:
        epochs:               epoch indices at which diagnostics were evaluated
        omega_eff_by_feature: {feature_index: omega_eff values}, each same
                               length as epochs (nan where undefined)
        save_path:             full file path to save to (None = skip)
        show:                  call plt.show()
    Returns:
        fig
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    for feature_index, values in omega_eff_by_feature.items():
        ax.plot(epochs, values, marker="o", ms=4, lw=2, alpha=0.85, label=f"feature {feature_index}")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel(r"$\omega_{eff}$", fontsize=12)
    ax.set_title("Spectral second moment vs. training epoch", fontsize=13)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    fig.tight_layout()
    _save(fig, save_path)
    if show:
        plt.show()
    return fig


def plot_eta_lambda_comparison(
    eta_init_by_mode: Dict[int, Union[np.ndarray, "torch.Tensor"]],
    eta_final_by_mode: Dict[int, Union[np.ndarray, "torch.Tensor"]],
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
) -> plt.Figure:
    """
    eta_lambda (Haar-relative harmonic weight, see
    spectral.haar_relative_harmonic_weight) vs. lambda, at initialization vs.
    after training, one panel per encoded mode. The horizontal line at
    eta=1.0 is the Haar-random reference (Schur's lemma expectation); values
    above/below it mean training concentrated/depleted amplitude in that
    lambda-photons-in-mode-k sector relative to a generic (Haar-random) W1.

    Args:
        eta_init_by_mode, eta_final_by_mode: {mode_index: eta_lambda array},
            each of length n_photons+1
        save_path: full file path to save to (None = skip)
        show:      call plt.show()
    Returns:
        fig
    """
    modes = sorted(eta_init_by_mode.keys())
    fig, axes = plt.subplots(1, len(modes), figsize=(4.5 * len(modes), 4.5), squeeze=False)
    axes = axes[0]

    for ax, mode in zip(axes, modes):
        eta_init = _to_numpy(eta_init_by_mode[mode])
        eta_final = _to_numpy(eta_final_by_mode[mode])
        lam = np.arange(len(eta_init))

        ax.axhline(1.0, color="gray", linestyle="--", alpha=0.6, label="Haar reference")
        ax.plot(lam, eta_init, marker="o", ms=5, lw=2, alpha=0.85, label="init")
        ax.plot(lam, eta_final, marker="s", ms=5, lw=2, alpha=0.85, label="after training")
        ax.set_xlabel(r"$\lambda$ (photons in mode)", fontsize=11)
        ax.set_ylabel(r"$\eta_\lambda$", fontsize=11)
        ax.set_title(f"mode {mode}", fontsize=12)
        ax.set_xticks(lam)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

    fig.suptitle("Haar-relative harmonic weight: init vs. after training", fontsize=13)
    fig.tight_layout()
    _save(fig, save_path)
    if show:
        plt.show()
    return fig


def plot_gauge_stability(
    omega_eff_labeled_by_seed: Dict[int, Union[np.ndarray, "torch.Tensor"]],
    omega_eff_sorted_by_seed: Dict[int, Union[np.ndarray, "torch.Tensor"]],
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Feature index is not a physical label — a mode permutation combined with
    the matching W1/W2/feature_map row permutation is an exact symmetry of
    the model (see spectral.py's gauge-invariant-summaries section, verified
    in tests/test_spectral.py::test_mode_permutation_gauge_symmetry), so a
    claim like "feature k has low omega_eff" only means something about one
    specific trained instance, not the architecture. This plots both views
    across seeds to make that concrete: left panel is the labeled vector
    (x = feature index) — if lines cross a lot across seeds, the per-index
    ranking is set by initialization, not architecture; right panel is the
    same data sorted (x = rank) — if it's markedly more stable across seeds
    than the left panel, that confirms the sorted profile is the
    architectural signal and the labeled one isn't.

    Args:
        omega_eff_labeled_by_seed: {seed: omega_eff}, native feature order
        omega_eff_sorted_by_seed:  {seed: omega_eff}, sorted descending
        save_path: full file path to save to (None = skip)
        show:      call plt.show()
    Returns:
        fig
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)

    for seed, values in omega_eff_labeled_by_seed.items():
        v = _to_numpy(values)
        ax1.plot(np.arange(len(v)), v, marker="o", ms=5, lw=1.5, alpha=0.75, label=f"seed {seed}")
    ax1.set_xlabel("Feature index (labeled)", fontsize=11)
    ax1.set_ylabel(r"$\omega_{eff}$", fontsize=11)
    ax1.set_title("Labeled — gauge-dependent", fontsize=12)
    ax1.set_xticks(np.arange(len(next(iter(omega_eff_labeled_by_seed.values())))))
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8)

    for seed, values in omega_eff_sorted_by_seed.items():
        v = _to_numpy(values)
        ax2.plot(np.arange(len(v)), v, marker="s", ms=5, lw=1.5, alpha=0.75, label=f"seed {seed}")
    ax2.set_xlabel("Rank (sorted descending)", fontsize=11)
    ax2.set_title("Sorted — gauge-invariant", fontsize=12)
    ax2.set_xticks(np.arange(len(next(iter(omega_eff_sorted_by_seed.values())))))
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8)

    fig.suptitle("omega_eff across seeds: labeled vs. sorted", fontsize=13)
    fig.tight_layout()
    _save(fig, save_path)
    if show:
        plt.show()
    return fig


def plot_physical_realized_frequency(
    freq_by_feature: Dict[int, Union[np.ndarray, "torch.Tensor"]],
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Local physical (k_x, k_t) frequency each feature's theta-space harmonic
    induces at every reference point (see spectral.physical_realized_frequency),
    scattered rather than reduced to a single number because the pushforward
    Jacobian d(feature_map)/d(x,t) varies across the domain for this model's
    nonlinear (Tanh) feature_map — a theta-space lattice point becomes a
    chirp, not a fixed tone, in physical space. A tight cluster for a given
    feature means that harmonic behaves like a near-constant physical
    frequency there; a spread-out cloud means it doesn't, and reporting a
    single "the frequency" number for it would be misleading.

    Args:
        freq_by_feature: {feature_index: (B, 2) array of local (k_x, k_t) pairs}
        save_path: full file path to save to (None = skip)
        show:      call plt.show()
    Returns:
        fig
    """
    fig, ax = plt.subplots(figsize=(7, 6))
    for feature_index, freq in freq_by_feature.items():
        f = _to_numpy(freq)
        ax.scatter(f[:, 0], f[:, 1], s=18, alpha=0.6, label=f"feature {feature_index}")
    ax.set_xlabel(r"local $k_x$", fontsize=12)
    ax.set_ylabel(r"local $k_t$", fontsize=12)
    ax.set_title("Physical realized frequency (pushforward through feature_map)", fontsize=12)
    ax.axhline(0, color="gray", lw=0.8, alpha=0.5)
    ax.axvline(0, color="gray", lw=0.8, alpha=0.5)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    _save(fig, save_path)
    if show:
        plt.show()
    return fig