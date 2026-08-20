"""
plot_freq_sweep_by_alpha.py — classical vs. quantum performance vs.
frequency, one set of figures per alpha, built straight from
`checkpoints/param_sweep/` checkpoint files (see `src.checkpoints`). Never
retrains anything.

Two views of the same sweep:
    plot_rel_l2_vs_freq / plot_all_alphas
        The scalar `rel_l2` (a single L2-norm ratio over the whole (x,t)
        evaluation grid -- see `src.eval.evaluate`) vs. frequency, mean ± std
        over seeds -- a companion to
        `src.checkpoint_plots.plot_param_sweep_heatmap` (same (freq, alpha)
        grid, line plots instead of a heatmap).
    plot_error_field_grid / plot_all_alphas_error_fields
        The pointwise error field U_pred - U_true over the actual (x,t) grid
        (like `src.plotter.plot_solution_comparison`'s error panel), laid out
        as a classical/quantum x frequency grid of small heatmaps -- shows
        WHERE in (x,t) each run's error sits, not just the single rel_l2
        number it collapses to.

Usage (from the repo root):

    python -m src.plot_freq_sweep_by_alpha

or from a notebook:

    from src.plot_freq_sweep_by_alpha import plot_rel_l2_vs_freq
    from src.checkpoints import load_checkpoints

    ckpts = load_checkpoints("checkpoints/param_sweep", pde_type="heat", L=1, alpha=0.001)
    plot_rel_l2_vs_freq(ckpts, alpha_value=0.001)
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Union

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from src.plotter import _save, _to_numpy
from src.checkpoints import checkpoints_to_dataframe, load_checkpoints
from src.checkpoint_plots import plot_param_sweep_scatter


def plot_rel_l2_vs_freq(
    checkpoints: List[dict],
    alpha_value: float,
    freq_name: str = "freq",
    alpha_name: str = "alpha",
    metric: str = "rel_l2",
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
) -> plt.Figure:
    """
    `metric` vs. frequency at one fixed alpha, classical vs. quantum, mean ±
    std over however many seeds are present at each frequency -- built
    directly from a list of checkpoints (e.g. from
    `src.checkpoints.load_checkpoints`), all at the same L and alpha (this
    function doesn't filter for you -- pass checkpoints already restricted
    to one `alpha_name` value, e.g. via
    `load_checkpoints(..., L=..., alpha=alpha_value)`).
    """
    df = checkpoints_to_dataframe(checkpoints)
    if len(df) == 0:
        raise ValueError("no checkpoints given.")
    df = df[np.isfinite(df[metric])]
    freqs = sorted(df[freq_name].unique())

    def _series(model_type):
        sub = df[df.model_type == model_type]
        mean, std, n = [], [], []
        for f in freqs:
            cell = sub[np.isclose(sub[freq_name], f)]
            mean.append(cell[metric].mean() if len(cell) else np.nan)
            std.append(cell[metric].std(ddof=0) if len(cell) else np.nan)
            n.append(len(cell))
        return np.array(mean), np.array(std), n

    Zc, Sc, Nc = _series("classical")
    Zq, Sq, Nq = _series("quantum")

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.errorbar(freqs, Zc, yerr=Sc, marker="o", lw=2, capsize=3, label="classical (param-matched)")
    ax.errorbar(freqs, Zq, yerr=Sq, marker="s", lw=2, capsize=3, label="quantum")

    ax.set_xlabel(freq_name, fontsize=11)
    ax.set_ylabel(metric, fontsize=11)
    ax.set_yscale("log")
    ax.set_xticks(freqs)
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=9)

    L_vals = df["L"].unique()
    L_str = str(int(L_vals[0])) if len(L_vals) == 1 else f"{sorted(L_vals)} (mixed!)"
    n_seeds = df.groupby(freq_name)["seed"].nunique().max()
    fig.suptitle(
        f"{metric} vs. {freq_name} at {alpha_name}={alpha_value:g}, L={L_str} — "
        f"up to {n_seeds} seed(s) per point",
        fontsize=12,
    )
    fig.tight_layout()
    _save(fig, save_path)
    if show:
        plt.show()
    return fig


def plot_all_alphas(
    ckpt_dir: Union[str, Path],
    alpha_values: Sequence[float],
    pde_type: str = "heat",
    L: int = 1,
    freq_name: str = "freq",
    alpha_name: str = "alpha",
    metric: str = "rel_l2",
    save_dir: Optional[Union[str, Path]] = None,
    show: bool = True,
) -> List[plt.Figure]:
    """
    Convenience driver: one `plot_rel_l2_vs_freq` figure per value in
    `alpha_values`, loading straight from `ckpt_dir` and filtering by
    `pde_type`/`L`/`alpha_name` each time. Saves each figure to
    `save_dir / f"freq_sweep_{alpha_name}={alpha_value:g}.png"` if `save_dir`
    is given.
    """
    figs = []
    for alpha_value in alpha_values:
        ckpts = load_checkpoints(
            ckpt_dir, pde_type=pde_type, L=L, **{alpha_name: alpha_value},
        )
        if not ckpts:
            print(f"[plot_all_alphas] no checkpoints found for {alpha_name}={alpha_value:g} -- skipping.")
            continue
        save_path = None
        if save_dir is not None:
            save_path = Path(save_dir) / f"freq_sweep_{alpha_name}={alpha_value:g}.png"
        fig = plot_rel_l2_vs_freq(
            ckpts, alpha_value=alpha_value, freq_name=freq_name, alpha_name=alpha_name,
            metric=metric, save_path=save_path, show=show,
        )
        figs.append(fig)
    return figs


def _pick_checkpoint(ckpt_dir, seed, verbose_label, **filters):
    """One checkpoint matching `filters` -- prefer `seed`, fall back to
    whatever seed is on disk (with a warning) so a grid still plots even if
    that exact seed is missing at some (freq, alpha) point."""
    exact = load_checkpoints(ckpt_dir, seed=seed, **filters)
    if exact:
        return exact[0]
    fallback = load_checkpoints(ckpt_dir, **filters)
    if fallback:
        used_seed = fallback[0]["config"]["seed"]
        print(f"[plot_error_field_grid] seed={seed} missing for {verbose_label} -- "
              f"using seed={used_seed} instead.")
        return fallback[0]
    print(f"[plot_error_field_grid] no checkpoint found for {verbose_label} -- leaving blank.")
    return None


def plot_error_field_grid(
    ckpt_dir: Union[str, Path],
    alpha_value: float,
    freq_values: Sequence[float],
    seed: int = 42,
    pde_type: str = "heat",
    L: int = 1,
    freq_name: str = "freq",
    alpha_name: str = "alpha",
    shared_scale: bool = True,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Pointwise error field U_pred - U_true over the (x, t) evaluation grid, at
    one fixed alpha, laid out as a (classical, quantum) x `freq_values` grid
    of small heatmaps -- one specific `seed`'s checkpoint per cell (falls
    back to whatever seed is on disk if that exact one is missing).

    `shared_scale=True` (default): every panel in the figure shares ONE
    colormap normalisation (symmetric around 0, diverging colormap) set by
    the largest |error| anywhere in the grid, with a single colorbar --
    panels are then directly comparable by eye, at the cost of the lowest-
    frequency panels looking nearly blank next to the highest-frequency
    ones. `shared_scale=False` normalises each panel on its own instead, so
    the error PATTERN is visible at every frequency regardless of magnitude
    (the panel's rel_l2 is annotated in its title either way, so magnitudes
    stay readable even when the color scale doesn't distinguish them).
    """
    freq_values = list(freq_values)
    model_types = ["classical", "quantum"]

    # Pass 1: load every cell's checkpoint once (also needed for the
    # shared-scale global vmax) and cache it for pass 2.
    cells = {}
    for model_type in model_types:
        for freq_value in freq_values:
            label = f"{model_type} {freq_name}={freq_value:g} {alpha_name}={alpha_value:g}"
            cells[(model_type, freq_value)] = _pick_checkpoint(
                ckpt_dir, seed, label,
                pde_type=pde_type, L=L, model_type=model_type,
                **{freq_name: freq_value, alpha_name: alpha_value},
            )

    global_vmax = max(
        (float(np.abs(_to_numpy(c["U_pred"]) - _to_numpy(c["U_true"])).max())
         for c in cells.values() if c is not None),
        default=1e-12,
    )
    global_vmax = global_vmax if global_vmax > 0 else 1e-12

    fig, axes = plt.subplots(
        len(model_types), len(freq_values),
        figsize=(2.6 * len(freq_values) + 1.0, 2.6 * len(model_types) + 0.8),
        squeeze=False,
    )

    last_im = None
    for i, model_type in enumerate(model_types):
        for j, freq_value in enumerate(freq_values):
            ax = axes[i, j]
            ckpt = cells[(model_type, freq_value)]
            if ckpt is None:
                ax.axis("off")
                continue

            U_pred = _to_numpy(ckpt["U_pred"])
            U_true = _to_numpy(ckpt["U_true"])
            xv = _to_numpy(ckpt["x_grid"]).squeeze()
            tv = _to_numpy(ckpt["t_grid"]).squeeze()
            error = U_pred - U_true
            rel_l2 = ckpt["metrics"]["rel_l2"]

            if shared_scale:
                vmax = global_vmax
            else:
                vmax = float(np.abs(error).max())
                vmax = vmax if vmax > 0 else 1e-12
            im = ax.imshow(
                error, origin="lower", extent=[tv[0], tv[-1], xv[0], xv[-1]],
                aspect="auto", cmap="RdBu_r", norm=Normalize(vmin=-vmax, vmax=vmax),
            )
            last_im = im
            ax.set_title(f"{freq_name}={freq_value:g}\nrel_l2={rel_l2:.2e}", fontsize=9)
            if j == 0:
                ax.set_ylabel(f"{model_type}\nx", fontsize=9)
            else:
                ax.set_yticklabels([])
            if i == len(model_types) - 1:
                ax.set_xlabel("t", fontsize=9)
            else:
                ax.set_xticklabels([])
            if not shared_scale:
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(labelsize=7)

    if shared_scale and last_im is not None:
        fig.colorbar(last_im, ax=axes, fraction=0.02, pad=0.02, label="Δu").ax.tick_params(labelsize=8)

    scale_str = "shared color scale" if shared_scale else "per-panel color scale"
    fig.suptitle(
        f"Error field U_pred − U_true vs. {freq_name}, {alpha_name}={alpha_value:g}, "
        f"L={L}, seed={seed} ({scale_str})",
        fontsize=12,
    )
    if not shared_scale:
        fig.tight_layout()
    _save(fig, save_path)
    if show:
        plt.show()
    return fig


def plot_all_alphas_error_fields(
    ckpt_dir: Union[str, Path],
    alpha_values: Sequence[float],
    freq_values: Sequence[float],
    seed: int = 42,
    pde_type: str = "heat",
    L: int = 1,
    freq_name: str = "freq",
    alpha_name: str = "alpha",
    shared_scale: bool = True,
    save_dir: Optional[Union[str, Path]] = None,
    show: bool = True,
) -> List[plt.Figure]:
    """
    Convenience driver: one `plot_error_field_grid` figure per value in
    `alpha_values`. Saves each figure to
    `save_dir / f"error_field_{alpha_name}={alpha_value:g}.png"` if
    `save_dir` is given.
    """
    figs = []
    for alpha_value in alpha_values:
        save_path = None
        if save_dir is not None:
            save_path = Path(save_dir) / f"error_field_{alpha_name}={alpha_value:g}.png"
        fig = plot_error_field_grid(
            ckpt_dir, alpha_value=alpha_value, freq_values=freq_values, seed=seed,
            pde_type=pde_type, L=L, freq_name=freq_name, alpha_name=alpha_name,
            shared_scale=shared_scale, save_path=save_path, show=show,
        )
        figs.append(fig)
    return figs


if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parent.parent

    PDE_TYPE = "heat"
    L_PARAM_SWEEP = 1
    ALPHA_VALUES = [0.001, 0.005, 0.01]  # the three fully-swept alphas in checkpoints/param_sweep
    FREQ_VALUES = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0]
    SEED = 42

    FIG_DIR = ROOT / "figures" / "checkpoint_pipeline"
    CKPT_DIR = ROOT / "checkpoints" / "param_sweep"

    plot_all_alphas(
        ckpt_dir=CKPT_DIR, alpha_values=ALPHA_VALUES, pde_type=PDE_TYPE, L=L_PARAM_SWEEP,
        save_dir=FIG_DIR, show=False,
    )
    plot_all_alphas_error_fields(
        ckpt_dir=CKPT_DIR, alpha_values=ALPHA_VALUES, freq_values=FREQ_VALUES, seed=SEED,
        pde_type=PDE_TYPE, L=L_PARAM_SWEEP, save_dir=FIG_DIR, show=False,
    )

    # Same (freq x alpha) grid as plot_param_sweep_heatmap, all three alphas
    # combined into one scatter figure (classical | quantum | diff panels,
    # colour-coded by alpha) instead of one heatmap per alpha.
    all_alpha_ckpts = load_checkpoints(
        CKPT_DIR, pde_type=PDE_TYPE, L=L_PARAM_SWEEP, alpha=ALPHA_VALUES,
    )
    plot_param_sweep_scatter(
        all_alpha_ckpts, param1_name="freq", param2_name="alpha", metric="rel_l2",
        save_path=FIG_DIR / "param_sweep_scatter.png", show=False,
    )
