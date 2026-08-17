"""
plotter.py — plotting utilities for QPINN / classical PINN results.

Functions:
    plot_training_history(history, ...)
    plot_solution_comparison(U_pred, U_true, x_grid, t_grid, ...)
    --- Fourier analysis (joint (k_x, omega) spectra — 2-D FFT over x AND t) ---
    plot_power_spectrum(U_pred, U_true, ...)
    plot_mode_l2_errors(U_pred, U_true, ...)
    plot_multi_model_l2_error(U_true, U_preds, ...)
    plot_frequency_sweep(sweep_results, ...)
    plot_2d_sweep(result, ...)          # src.sweep.run_2d_sweep results
    plot_depth_sweep(result, ...)       # src.sweep.run_depth_sweep results
    plot_depth_sweep_training_curves(result, ...)  # per-run loss curves
    plot_depth_sweep_solution(result, ...)         # per-run learnt vs. exact solution
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
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
#
# All three functions below FFT over BOTH axes (torch.fft.fft2), so the
# temporal content of every spatial mode is visible — e.g. the Lorentzian
# broadening a heat-equation mode picks up along omega (width ~ alpha*k^2),
# vs. the sharp, undamped line a wave-equation mode keeps there. The
# previous versions FFT'd over x only and averaged over t first, which
# throws exactly that information away before it's ever plotted.
#
# k_x : integer spatial mode index (cycles over the sampled x-domain),
#       same "Mode k" convention as before.
# omega : angular temporal frequency (rad / unit time), built from the
#       actual time step so it's directly comparable to a PDE's analytic
#       omega_n (e.g. n*pi*c for the wave equation) or decay/broadening
#       rate (alpha*n^2*pi^2 for the heat equation).

def _dt_from_grid(t_grid, nt: int) -> float:
    """Physical time step: from `t_grid` if given, else assume the same
    endpoint-inclusive linspace `eval.evaluate` builds (dt = 1/(nt-1))."""
    if t_grid is not None:
        tv = _to_numpy(t_grid).squeeze()
        return float(tv[1] - tv[0])
    return 1.0 / max(nt - 1, 1)


def _fft2_axes(nx: int, nt: int, dt: float) -> Tuple[np.ndarray, np.ndarray]:
    kx = np.fft.fftshift(np.fft.fftfreq(nx)) * nx
    omega = np.fft.fftshift(np.fft.fftfreq(nt, d=dt)) * 2 * np.pi
    return kx, omega


def _lognorm_bounds(*arrays: np.ndarray, floor: float = 1e-12, dynamic_range: float = 1e6) -> Tuple[float, float]:
    """vmin/vmax for LogNorm, robust to all-zero arrays (e.g. a perfect fit)."""
    vmax = max(float(a.max()) for a in arrays)
    vmax = max(vmax, floor)
    vmin = max(vmax / dynamic_range, floor)
    if vmin >= vmax:
        vmin = vmax / 10 if vmax > 0 else floor
    return vmin, vmax


def plot_power_spectrum(
    U_pred: Union[torch.Tensor, np.ndarray],
    U_true: Union[torch.Tensor, np.ndarray],
    t_grid: Optional[Union[torch.Tensor, np.ndarray]] = None,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
    exclude_0_mode: bool = True,
) -> plt.Figure:
    """
    Joint (k_x, omega) power spectrum of predicted vs true solution: a full
    2-D FFT over x AND t, plotted as heatmaps (log color scale) instead of
    the old 1-D, time-averaged line plot.

    Args:
        U_pred / U_true: solution arrays of shape (nx, nt)
        t_grid: optional physical time coordinates (shape (nt,) or (nt,1)),
                e.g. the `t_grid` returned by `eval.evaluate`. Used to scale
                the omega axis to real angular frequency; falls back to
                assuming the same grid `evaluate` builds.
        exclude_0_mode: zero out the (k_x=0, omega=0) bin before plotting so
                it doesn't dominate the color scale.
    Returns:
        fig
    """
    U_pred_t = torch.as_tensor(_to_numpy(U_pred)).to(torch.float64)
    U_true_t = torch.as_tensor(_to_numpy(U_true)).to(torch.float64)
    nx, nt = U_true_t.shape

    dt = _dt_from_grid(t_grid, nt)
    kx, omega = _fft2_axes(nx, nt, dt)

    pred_hat = torch.fft.fftshift(torch.fft.fft2(U_pred_t))
    true_hat = torch.fft.fftshift(torch.fft.fft2(U_true_t))

    power_pred = (pred_hat.abs() ** 2).numpy()
    power_true = (true_hat.abs() ** 2).numpy()
    power_diff = np.abs(power_pred - power_true)

    if exclude_0_mode:
        power_true[nx // 2, nt // 2] = 0.0
        power_pred[nx // 2, nt // 2] = 0.0
        power_diff[nx // 2, nt // 2] = 0.0

    vmin, vmax = _lognorm_bounds(power_true, power_pred)
    vmin_d, vmax_d = _lognorm_bounds(power_diff)

    extent = [omega[0], omega[-1], kx[0], kx[-1]]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    panels = [
        (power_true, r"True  $|\hat u(k_x,\omega)|^2$", vmin, vmax),
        (power_pred, r"Predicted  $|\hat u(k_x,\omega)|^2$", vmin, vmax),
        (power_diff, r"$|\Delta \mathrm{power}|$", vmin_d, vmax_d),
    ]
    for ax, (data, title, lo, hi) in zip(axes, panels):
        im = ax.imshow(
            np.clip(data, lo, None), origin="lower", extent=extent, aspect="auto",
            cmap="magma", norm=LogNorm(vmin=lo, vmax=hi),
        )
        ax.set_xlabel(r"$\omega$  (angular temporal frequency)", fontsize=10)
        ax.set_ylabel(r"$k_x$  (spatial mode)", fontsize=11)
        ax.set_title(title, fontsize=12)
        fig.colorbar(im, ax=ax, label="power")

    fig.suptitle("Joint (x, t) power spectrum", fontsize=13)
    fig.tight_layout()
    _save(fig, save_path)

    if show:
        plt.show()

    return fig


def plot_mode_l2_errors(
    U_pred: Union[torch.Tensor, np.ndarray],
    U_true: Union[torch.Tensor, np.ndarray],
    t_grid: Optional[Union[torch.Tensor, np.ndarray]] = None,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
    exclude_0_mode: bool = True,
    eps: float = 1e-8,
) -> plt.Figure:
    """
    Two-panel analysis of per-(k_x, omega) error in the joint (x, t)
    Fourier domain — no time-averaging, so errors that only show up at
    certain temporal frequencies (e.g. slow drift vs. fast oscillation
    mismatches) aren't washed out:
      • |û_pred - û_true|                       (absolute error)
      • |û_pred - û_true| / (|û_true| + eps)     (relative error)

    Args:
        U_pred / U_true: solution arrays of shape (nx, nt)
        t_grid: optional physical time coordinates, see `plot_power_spectrum`.
    Returns:
        fig
    """
    U_pred_t = torch.as_tensor(_to_numpy(U_pred)).to(torch.float64)
    U_true_t = torch.as_tensor(_to_numpy(U_true)).to(torch.float64)
    nx, nt = U_true_t.shape

    dt = _dt_from_grid(t_grid, nt)
    kx, omega = _fft2_axes(nx, nt, dt)

    pred_hat = torch.fft.fftshift(torch.fft.fft2(U_pred_t))
    true_hat = torch.fft.fftshift(torch.fft.fft2(U_true_t))

    abs_err = (pred_hat - true_hat).abs().numpy()
    rel_err = abs_err / (true_hat.abs().numpy() + eps)

    if exclude_0_mode:
        abs_err[nx // 2, nt // 2] = 0.0
        rel_err[nx // 2, nt // 2] = 0.0

    extent = [omega[0], omega[-1], kx[0], kx[-1]]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    vmin1, vmax1 = _lognorm_bounds(abs_err)
    im1 = ax1.imshow(np.clip(abs_err, vmin1, None), origin="lower", extent=extent,
                      aspect="auto", cmap="viridis", norm=LogNorm(vmin=vmin1, vmax=vmax1))
    ax1.set_xlabel(r"$\omega$", fontsize=12)
    ax1.set_ylabel(r"$k_x$", fontsize=12)
    ax1.set_title(r"Absolute error  $|\hat u_{pred} - \hat u_{true}|$", fontsize=12, fontweight="bold")
    fig.colorbar(im1, ax=ax1, label="abs. error")

    vmin2, vmax2 = _lognorm_bounds(rel_err)
    im2 = ax2.imshow(np.clip(rel_err, vmin2, None), origin="lower", extent=extent,
                      aspect="auto", cmap="viridis", norm=LogNorm(vmin=vmin2, vmax=vmax2))
    ax2.set_xlabel(r"$\omega$", fontsize=12)
    ax2.set_ylabel(r"$k_x$", fontsize=12)
    ax2.set_title("Relative error", fontsize=12, fontweight="bold")
    fig.colorbar(im2, ax=ax2, label="rel. error")

    fig.tight_layout()
    _save(fig, save_path)

    if show:
        plt.show()

    return fig


def plot_multi_model_l2_error(
    U_true: Union[torch.Tensor, np.ndarray],
    U_preds: List[Union[torch.Tensor, np.ndarray]],
    model_names: Optional[List[str]] = None,
    t_grid: Optional[Union[torch.Tensor, np.ndarray]] = None,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
    exclude_0_mode: bool = True,
    eps: float = 1e-16,
) -> plt.Figure:
    """
    Relative error in the joint (k_x, omega) Fourier domain for multiple
    models, one heatmap panel per model on a shared color scale (instead of
    the old single semilogy-vs-k line plot, which had already averaged
    over t).

    Args:
        U_true: shape (nx, nt)
        U_preds: list of model predictions, each shape (nx, nt)
        model_names: optional list of names (default: "model i")
        t_grid: optional physical time coordinates, see `plot_power_spectrum`.
    Returns:
        fig
    """
    def _to_tensor(x):
        if isinstance(x, torch.Tensor):
            return x.to(torch.float64)
        return torch.as_tensor(np.asarray(x), dtype=torch.float64)

    U_true_t = _to_tensor(U_true)
    nx, nt = U_true_t.shape

    dt = _dt_from_grid(t_grid, nt)
    kx, omega = _fft2_axes(nx, nt, dt)
    extent = [omega[0], omega[-1], kx[0], kx[-1]]

    true_hat = torch.fft.fftshift(torch.fft.fft2(U_true_t))

    if model_names is None or len(model_names) != len(U_preds):
        model_names = [f"model {i}" for i in range(len(U_preds))]

    rel_errs = []
    for U_pred in U_preds:
        pred_hat = torch.fft.fftshift(torch.fft.fft2(_to_tensor(U_pred)))
        rel_err = (pred_hat - true_hat).abs().numpy() / (true_hat.abs().numpy() + eps)
        if exclude_0_mode:
            rel_err[nx // 2, nt // 2] = 0.0
        rel_errs.append(rel_err)

    vmin, vmax = _lognorm_bounds(*rel_errs)

    n = len(U_preds)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 5), squeeze=False)
    axes = axes[0]

    im = None
    for ax, rel_err, name in zip(axes, rel_errs, model_names):
        im = ax.imshow(np.clip(rel_err, vmin, None), origin="lower", extent=extent,
                        aspect="auto", cmap="viridis", norm=LogNorm(vmin=vmin, vmax=vmax))
        ax.set_xlabel(r"$\omega$", fontsize=11)
        ax.set_ylabel(r"$k_x$", fontsize=11)
        ax.set_title(name, fontsize=12)

    fig.colorbar(im, ax=axes.tolist(), label="relative error", shrink=0.85)
    fig.suptitle(r"Relative error in joint $(k_x, \omega)$ Fourier domain", fontsize=13)

    _save(fig, save_path)

    if show:
        plt.show()

    plt.close(fig)

    return fig


def _choose_annotation_layout(
    cell_w_pt: float, cell_h_pt: float, min_fs: float = 5.5, max_fs: float = 10.0,
) -> Tuple[str, float]:
    """
    Pick the richest annotation style that fits inside a cell of size
    (cell_w_pt, cell_h_pt) points, degrading gracefully instead of ever
    overlapping: "two_line" ("$v$\\n$\\pm s$") -> "one_line"
    ("$v \\pm s$") -> "value_only" ("$v$") -> "off" if even a bare value
    wouldn't fit legibly. Uses a rough ~0.6*fontsize average glyph width
    and ~1.3x line-height, which is conservative enough in practice for
    matplotlib's default mathtext font.
    """
    # (mode, widest-line char count, total height in line-heights)
    candidates = [
        ("two_line", 5, 2.6),
        ("one_line", 9, 1.3),
        ("value_only", 4, 1.3),
    ]
    for mode, max_chars, h_factor in candidates:
        fs = min(cell_w_pt / (0.6 * max_chars), cell_h_pt / h_factor)
        if fs >= min_fs:
            return mode, float(np.clip(fs, min_fs, max_fs))
    return "off", 0.0


def plot_2d_sweep(
    result,  # Sweep2DResult from src.sweep.run_2d_sweep
    metric: str = "rel_l2",
    diff_mode: str = "signed",
    log_scale: bool = False,
    cmap: str = "viridis",
    annotate: bool = True,
    annotate_fmt: str = "{:.2f}",
    annotate_mode: Optional[str] = None,
    annotate_fontsize: Optional[float] = None,
    cell_size_in: float = 0.8,
    max_panel_size_in: float = 7.0,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Three-panel heatmap over a 2-D (param1, param2) sweep from
    `src.sweep.run_2d_sweep`: classical | quantum ("qpinn_weights") |
    their difference, for one fixed metric (`rel_l2`, `mae`,
    `max_abs_error`, or `rmse`).

    Every cell is annotated "mean +/- std" across seeds. The diff panel
    uses PAIRED per-seed statistics (`run_2d_sweep` trains classical and
    quantum with the same seed at a given grid point), so its std reflects
    the actual seed-to-seed spread of the paired difference/ratio rather
    than a naive combination of two independent stds.

    Layout scales with grid size instead of shrinking text indefinitely:
    each panel grows by ~`cell_size_in` inches per grid cell (capped at
    `max_panel_size_in` per panel), and once the grid is dense enough that
    growing the figure further stops being practical, the annotation
    itself degrades — two-line "$v$\\n$\\pm s$" -> one-line "$v \\pm s$"
    -> value-only "$v$" -> off — so cells never overlap; a note is
    printed if annotations get dropped entirely.

    Args:
        result: `Sweep2DResult` (`.param1_values`, `.param2_values`, `.raw`,
                 `.mean(model_type, metric)`, `.std(model_type, metric)`).
        metric: which metric to plot; one of `sweep.METRIC_KEYS`.
        diff_mode: "signed" -> classical - quantum (paired per seed),
                   diverging colormap around 0, positive = quantum better;
                   "ratio" -> quantum / classical (paired per seed),
                   diverging around 1, <1 = quantum better.
        log_scale: log color scale on the classical/quantum panels (the
                   metric is >= 0 and can span orders of magnitude
                   across the grid, e.g. near a stiff/high-alpha corner).
        annotate: overlay per-cell text at all (set False to always skip).
        annotate_fmt: format string applied to both mean and std.
        annotate_mode: force "two_line" / "one_line" / "value_only" / "off"
                 instead of auto-selecting from the fitted cell size.
        annotate_fontsize: force a specific font size instead of the
                 auto-fitted one (implies a fixed `annotate_mode` unless
                 you also pass one explicitly — see below).
        cell_size_in: target inches per grid cell before the size cap
                 kicks in; increase for a roomier layout.
        max_panel_size_in: cap on a single panel's width/height, so very
                 large grids degrade the annotation instead of producing
                 an unwieldy multi-foot figure.
    Returns:
        fig
    """
    p1, p2 = result.param1_values, result.param2_values
    n1, n2 = len(p1), len(p2)

    Zc, Sc = result.mean("classical", metric), result.std("classical", metric)
    Zq, Sq = result.mean("qpinn_weights", metric), result.std("qpinn_weights", metric)

    raw_c = result.raw["classical"][metric]
    raw_q = result.raw["qpinn_weights"][metric]

    if diff_mode == "signed":
        diff_raw = raw_c - raw_q  # paired per-seed difference
        diff_title = "Classical − Quantum  (>0 ⇒ quantum better)"
        diff_cmap = "RdBu_r"
        Zd = np.nanmean(diff_raw, axis=-1)
        Sd = np.nanstd(diff_raw, axis=-1)
        dmax = max(np.nanmax(np.abs(Zd)), 1e-12)
        diff_vmin, diff_vmax = -dmax, dmax
    elif diff_mode == "ratio":
        diff_raw = raw_q / (raw_c + 1e-16)  # paired per-seed ratio
        diff_title = "Quantum / Classical  (<1 ⇒ quantum better)"
        diff_cmap = "RdBu_r"
        Zd = np.nanmean(diff_raw, axis=-1)
        Sd = np.nanstd(diff_raw, axis=-1)
        dmax = max(np.nanmax(Zd), 1.0 / max(np.nanmin(Zd), 1e-12), 1.0 + 1e-6)
        diff_vmin, diff_vmax = 1.0 / dmax, dmax
    else:
        raise ValueError(f'diff_mode must be "signed" or "ratio", got {diff_mode!r}')

    field_vmin = float(min(np.nanmin(Zc), np.nanmin(Zq)))
    field_vmax = float(max(np.nanmax(Zc), np.nanmax(Zq)))
    field_norm = (
        LogNorm(vmin=max(field_vmin, 1e-12), vmax=max(field_vmax, 1e-12))
        if log_scale else Normalize(vmin=field_vmin, vmax=field_vmax)
    )
    diff_norm = Normalize(vmin=diff_vmin, vmax=diff_vmax)

    # ---- layout: panel size grows with grid resolution (capped) so cells
    # keep roughly the same physical footprint as more freqs/alphas are
    # added, instead of shrinking indefinitely -- this is what avoids
    # overlap for dense sweeps.
    margin_in = 1.6  # rough allowance for axis labels/ticks/colorbar per panel
    panel_w = float(np.clip(n2 * cell_size_in + margin_in, 4.0, max_panel_size_in))
    panel_h = float(np.clip(n1 * cell_size_in + margin_in, 3.5, max_panel_size_in))
    cell_w_pt = (panel_w - margin_in) / max(n2, 1) * 72.0
    cell_h_pt = (panel_h - margin_in) / max(n1, 1) * 72.0

    if annotate_mode is not None:
        mode = annotate_mode
        fs = annotate_fontsize if annotate_fontsize is not None else 8.0
    elif annotate_fontsize is not None:
        mode, fs = "two_line", annotate_fontsize
    else:
        mode, fs = _choose_annotation_layout(cell_w_pt, cell_h_pt)

    if not annotate:
        mode = "off"
    elif mode == "off":
        print(
            f"[plot_2d_sweep] {n1}x{n2} grid is too dense for legible per-cell "
            f"annotations at panel size {panel_w:.1f}x{panel_h:.1f}in -- disabling "
            f"them. Increase max_panel_size_in/cell_size_in, force annotate_mode="
            f"\"value_only\", or set annotate=False to silence this message."
        )

    def _text_for(val: float, std: float) -> str:
        v = annotate_fmt.format(val)
        s = annotate_fmt.format(std)
        if mode == "two_line":
            return f"${v}$\n$\\pm{s}$"
        if mode == "one_line":
            return f"${v}\\pm{s}$"
        return f"${v}$"  # value_only

    def _annotate(ax, values, stds, cmap_name, norm):
        cmap_obj = plt.get_cmap(cmap_name)
        for i, y in enumerate(p1):
            for j, x in enumerate(p2):
                val, std = values[i, j], stds[i, j]
                if np.isnan(val):
                    continue
                rgba = cmap_obj(norm(val))
                luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                text_color = "white" if luminance < 0.5 else "black"
                ax.text(
                    x, y, _text_for(val, std),
                    ha="center", va="center", fontsize=fs, color=text_color,
                    linespacing=1.3,
                )

    figsize = (3 * panel_w + 1.2, panel_h)
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    for ax, Z, S, title in [
        (axes[0], Zc, Sc, f"Classical — {metric}"),
        (axes[1], Zq, Sq, f"Quantum (qpinn_weights) — {metric}"),
    ]:
        im = ax.pcolormesh(p2, p1, Z, shading="auto", cmap=cmap, norm=field_norm)
        ax.set_xlabel(result.param2_name, fontsize=11)
        ax.set_ylabel(result.param1_name, fontsize=11)
        ax.set_title(title, fontsize=12)
        fig.colorbar(im, ax=ax, label=metric)
        if mode != "off":
            _annotate(ax, Z, S, cmap, field_norm)

    im3 = axes[2].pcolormesh(p2, p1, Zd, shading="auto", cmap=diff_cmap, norm=diff_norm)
    axes[2].set_xlabel(result.param2_name, fontsize=11)
    axes[2].set_ylabel(result.param1_name, fontsize=11)
    axes[2].set_title(diff_title, fontsize=12)
    fig.colorbar(im3, ax=axes[2], label=diff_mode)
    if mode != "off":
        _annotate(axes[2], Zd, Sd, diff_cmap, diff_norm)

    fig.suptitle(
        f"2-D sweep ({result.param1_name} × {result.param2_name}) — "
        f"feature_size={result.feature_size}, hidden_readout={result.hidden_readout}, "
        f"{len(result.seeds)} seed(s)",
        fontsize=13,
    )
    fig.tight_layout()
    _save(fig, save_path)

    if show:
        plt.show()

    return fig


def plot_depth_sweep(
    result,  # DepthSweepResult from src.sweep.run_depth_sweep
    metric: str = "rel_l2",
    target_frequency: Optional[float] = None,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
) -> plt.Figure:
    """
    rel_l2 (or another metric) vs. encoding depth L, quantum
    (`MerlinTunableEntropy`) against a parameter-matched classical
    baseline — the figure for the depth-sweep tuning experiment: does
    accuracy track the analytic frequency ceiling `L * n_photons`?

    Args:
        result: `DepthSweepResult` from `src.sweep.run_depth_sweep`.
        metric: one of `sweep.METRIC_KEYS`.
        target_frequency: if given, draws a vertical line at the smallest
                 swept L where `result.analytic_bound >= target_frequency`
                 — the theory's predicted plateau point. E.g. pass `2*freq`
                 for the heat-equation convention used elsewhere in this
                 repo (`k = 2*pi*freq`, integer mode `n = 2*freq`).
    Returns:
        fig
    """
    L = result.depths
    Zc, Sc = result.mean("classical", metric), result.std("classical", metric)
    Zq, Sq = result.mean("quantum", metric), result.std("quantum", metric)

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.errorbar(L, Zc, yerr=Sc, marker="o", lw=2, capsize=3,
                label="classical (param-matched)")
    ax.errorbar(L, Zq, yerr=Sq, marker="s", lw=2, capsize=3,
                label="quantum (MerlinTunableEntropy)")

    if target_frequency is not None:
        crossing = next(
            (l for l, bound in zip(L, result.analytic_bound) if bound >= target_frequency),
            None,
        )
        if crossing is not None:
            ax.axvline(
                crossing, color="gray", ls="--", lw=1.5,
                label=rf"predicted plateau ($L{{\cdot}}N \geq$ target, $L={crossing}$)",
            )
        else:
            print(
                f"[plot_depth_sweep] target_frequency={target_frequency} exceeds "
                f"L*n_photons for every swept depth (max {result.analytic_bound.max()}) "
                f"-- no crossing to mark; sweep larger L to see the predicted plateau."
            )

    ax.set_xlabel("encoding depth $L$ (n\\_encoding\\_layers)", fontsize=11)
    ax.set_ylabel(metric, fontsize=11)
    ax.set_yscale("log")
    ax.set_xticks(L)
    ax.set_title("Depth sweep — does spectral coverage explain the gap?", fontsize=12)
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=9)

    # secondary x-axis: matched parameter count, so the reader can see the
    # comparison stays fair (both curves) even as it grows with L
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(L)
    ax2.set_xticklabels([f"{p:,}" for p in result.quantum_params], fontsize=8, rotation=45)
    ax2.set_xlabel("quantum trainable parameters", fontsize=9)

    fig.tight_layout()
    _save(fig, save_path)

    if show:
        plt.show()

    return fig


def plot_depth_sweep_training_curves(
    result,  # DepthSweepResult from src.sweep.run_depth_sweep (needs .histories)
    component: str = "total",
    depths: Optional[List] = None,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
) -> plt.Figure:
    """
    One training-loss subplot per swept depth L, each showing every seed's
    run for both model types overlaid (thin, semi-transparent lines so
    run-to-run variance is visible directly, not just summarized by an error
    bar) -- lets you actually look at the training curves behind a
    `run_depth_sweep` result instead of only the final rel_l2 numbers.

    Args:
        result:    `DepthSweepResult` from `src.sweep.run_depth_sweep`,
                   called with default settings (`.histories` is always
                   populated; no special sweep-time flag needed).
        component: which logged loss column to plot -- one of
                   "total", "pde", "consistency", "initial", "boundary".
        depths:    subset of L values to plot (default: all swept depths).
                   Useful to avoid a huge/slow figure on wide sweeps.
        save_path: full file path to save to (None = skip)
        show:      call plt.show()

    Marks, on each curve, the epoch whose weights the reported metric
    (`raw`/`mean`/`std`) actually corresponds to -- `result.eval_epochs`
    (populated by `run_depth_sweep`): the best-checkpoint epoch if that
    sweep used `track_best=True`, otherwise just the final epoch. This is
    what you're looking at when you ask "which point on this curve did we
    evaluate?" -- with track_best off it's always the right-hand end; with
    it on it can sit anywhere the fixed-grid metric was best.

    Returns:
        fig
    """
    _COLS = {"total": 0, "pde": 1, "consistency": 2, "initial": 3, "boundary": 4}
    if component not in _COLS:
        raise ValueError(f"component must be one of {list(_COLS)}, got {component!r}")
    col = _COLS[component]

    if result.histories is None:
        raise ValueError(
            "result.histories is empty -- this DepthSweepResult was produced "
            "before history-tracking was added to run_depth_sweep; rerun the sweep."
        )

    all_depths = list(result.depths)
    plot_depths = list(depths) if depths is not None else all_depths
    idxs = [all_depths.index(d) for d in plot_depths]

    n = len(idxs)
    fig, axes = plt.subplots(n, 1, figsize=(8, 2.4 * n), squeeze=False)
    axes = axes[:, 0]

    colors = {"classical": "C0", "quantum": "C1"}
    has_eval_epochs = getattr(result, "eval_epochs", None) is not None
    marker_plotted = False

    for row, i in enumerate(idxs):
        ax = axes[row]
        L = all_depths[i]
        for model_type in ("classical", "quantum"):
            runs = result.histories[model_type][i]
            eval_ep_row = result.eval_epochs[model_type][i] if has_eval_epochs else None
            for s, h in enumerate(runs):
                if h is None:
                    continue
                h = _to_numpy(h)
                if col >= h.shape[1]:
                    continue
                # only label once per model_type per subplot to avoid a
                # legend entry per seed
                label = model_type if s == 0 else None
                ax.semilogy(h[:, col], color=colors[model_type], alpha=0.55,
                            lw=1.3, label=label)

                # mark the epoch the reported metric actually came from
                if eval_ep_row is not None and eval_ep_row[s] is not None:
                    ep = eval_ep_row[s]
                    x = min(max(ep - 1, 0), len(h) - 1)  # epochs are 1-indexed
                    marker_label = "eval checkpoint" if not marker_plotted else None
                    ax.scatter([x], [h[x, col]], color=colors[model_type], s=32,
                               zorder=5, edgecolor="black", linewidth=0.6,
                               label=marker_label)
                    marker_plotted = True
        ax.set_title(f"L = {L}", fontsize=10, loc="left")
        ax.set_ylabel(f"{component} loss", fontsize=9)
        ax.grid(True, alpha=0.3, which="both")
        if row == 0:
            ax.legend(fontsize=8, loc="upper right")
        if row == n - 1:
            ax.set_xlabel("epoch", fontsize=10)

    if not has_eval_epochs:
        print(
            "[plot_depth_sweep_training_curves] result.eval_epochs not found "
            "(sweep run before this was added) -- curves shown without "
            "eval-checkpoint markers; rerun the sweep to get them."
        )

    fig.suptitle("Depth sweep — per-run training curves", fontsize=12, y=1.0)
    fig.tight_layout()
    _save(fig, save_path)

    if show:
        plt.show()

    return fig


def plot_depth_sweep_solution(
    result,  # DepthSweepResult from src.sweep.run_depth_sweep (needs .solutions)
    model_type: str,
    depth,
    seed: Optional[int] = None,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
) -> Tuple[plt.Figure, float]:
    """
    The learnt solution for one specific run in a depth sweep: exact |
    prediction | pointwise error, with rel_L2 in the title -- a thin
    wrapper around `plot_solution_comparison` that looks the grids up from
    a `DepthSweepResult` by depth/seed/model_type instead of requiring you
    to have kept the raw tensors around yourself.

    Args:
        result:     `DepthSweepResult` from `src.sweep.run_depth_sweep`
                    (`.solutions` is always populated; no special sweep-time
                    flag needed).
        model_type: "classical" or "quantum".
        depth:      the actual L VALUE (not an array index), e.g. `3`.
        seed:       the actual seed VALUE (not an array index), e.g. `42`.
                    Defaults to the first seed in `result.seeds` if omitted.
        save_path:  full file path to save to (None = skip)
        show:       call plt.show()

    Returns:
        fig, rel_l2  (rel_l2 computed fresh from the stored grids, so it
        matches exactly what's drawn -- it may differ in the last couple of
        digits from `result.raw[model_type]["rel_l2"]` if that run used
        `track_best`, since evaluate()'s internal float ops aren't bitwise
        reordered identically on every call, but should agree to float32
        precision).
    """
    if seed is None:
        seed = result.seeds[0]

    sol = result.solution(model_type, depth, seed)
    fig, rel_l2 = plot_solution_comparison(
        U_pred=sol["U_pred"], U_true=sol["U_true"],
        x_grid=sol["x_grid"], t_grid=sol["t_grid"],
        save_path=None,  # save after adding the suptitle below
        show=False,
    )

    eval_epoch = None
    if getattr(result, "eval_epochs", None) is not None:
        i, s = result._index(depth, seed)
        eval_epoch = result.eval_epochs[model_type][i][s]

    subtitle = f"{model_type}  L={depth}  seed={seed}"
    if eval_epoch is not None:
        subtitle += f"  (epoch {eval_epoch})"
    fig.suptitle(subtitle, fontsize=11, y=1.04)
    fig.tight_layout()
    _save(fig, save_path)

    if show:
        plt.show()

    return fig, rel_l2

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
