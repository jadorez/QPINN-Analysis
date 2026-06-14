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