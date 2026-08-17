"""
checkpoint_plots.py — plotting utilities that read directly from checkpoint
files (see `src.checkpoints`), so every plot below can be regenerated at any
time without retraining anything.

Per-run plots (one checkpoint in, one figure out):
    plot_checkpoint_training_curve(ckpt, ...)
    plot_checkpoint_solution(ckpt, ...)
    plot_checkpoint_spectrum(ckpt, ...)          # quantum checkpoints only

Aggregate plots (a list of checkpoints in, combining however many seeds are
present per grid point -- the "combine checkpoints to increase statistics"
mechanism):
    plot_param_sweep_heatmap(checkpoints, ...)   # experiment 1
    plot_L_sweep(checkpoints, ...)               # experiment 2
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from src.plotter import plot_solution_comparison, _save, _to_numpy
from src.checkpoints import checkpoints_to_dataframe, build_pde_from_config, build_model


# ─────────────────────────────────────────────────────────────────────────
# per-run plots
# ─────────────────────────────────────────────────────────────────────────

_HISTORY_COLS = {"total": 0, "pde": 1, "consistency": 2, "initial": 3, "boundary": 4}


def plot_checkpoint_training_curve(
    ckpt: dict,
    component: str = "total",
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Loss curve for a single checkpoint, marking the epoch its saved
    weights/metrics actually correspond to (`ckpt["eval_epoch"]`) -- the
    best-checkpoint epoch if that run used `track_best=True`, otherwise the
    final epoch.
    """
    if component not in _HISTORY_COLS:
        raise ValueError(f"component must be one of {list(_HISTORY_COLS)}, got {component!r}")
    col = _HISTORY_COLS[component]

    h = np.asarray(ckpt["history"])
    cfg = ckpt["config"]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.semilogy(h[:, col], color="C0", lw=1.4, alpha=0.9)

    ep = ckpt.get("eval_epoch")
    if ep is not None and 1 <= ep <= len(h):
        ax.scatter([ep - 1], [h[ep - 1, col]], color="crimson", s=45, zorder=5,
                   edgecolor="black", linewidth=0.6, label=f"eval checkpoint (epoch {ep})")
        ax.legend(fontsize=9)

    ax.set_xlabel("epoch", fontsize=11)
    ax.set_ylabel(f"{component} loss", fontsize=11)
    ax.grid(True, alpha=0.3, which="both")
    ax.set_title(
        f"{cfg['model_type']}  L={cfg['L']}  seed={cfg['seed']}  "
        f"{cfg['pde_type']} {cfg['pde_params']}",
        fontsize=11,
    )
    fig.tight_layout()
    _save(fig, save_path)
    if show:
        plt.show()
    return fig


def plot_checkpoint_solution(
    ckpt: dict,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
):
    """
    Learnt solution vs. exact solution for a single checkpoint: exact |
    prediction | pointwise error, with rel_L2 in the title. Thin wrapper
    around `src.plotter.plot_solution_comparison` reading straight from the
    stored evaluation grid, so it never needs the model rebuilt.

    Returns (fig, rel_l2).
    """
    cfg = ckpt["config"]
    fig, rel_l2 = plot_solution_comparison(
        U_pred=ckpt["U_pred"], U_true=ckpt["U_true"],
        x_grid=ckpt["x_grid"], t_grid=ckpt["t_grid"],
        save_path=None, show=False,
    )
    fig.suptitle(
        f"{cfg['model_type']}  L={cfg['L']}  seed={cfg['seed']}  "
        f"epoch={ckpt.get('eval_epoch')}  {cfg['pde_type']} {cfg['pde_params']}",
        fontsize=11, y=1.04,
    )
    fig.tight_layout()
    _save(fig, save_path)
    if show:
        plt.show()
    return fig, rel_l2


def _import_spectral_bandwidth():
    """Import notebooks/spectral_bandwidth.py by file path (it isn't a
    package member of `notebooks/`), resolved relative to this file so it
    works regardless of the notebook's working directory."""
    path = Path(__file__).resolve().parent.parent / "notebooks" / "spectral_bandwidth.py"
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find notebooks/spectral_bandwidth.py at {path} -- "
            "plot_checkpoint_spectrum needs it for quantum_layer_spectrum/"
            "plot_spectrum_by_layers."
        )
    if "spectral_bandwidth" in sys.modules:
        return sys.modules["spectral_bandwidth"]
    spec = importlib.util.spec_from_file_location("spectral_bandwidth", path)
    mod = importlib.util.module_from_spec(spec)
    # Register in sys.modules BEFORE exec_module: spectral_bandwidth.py
    # defines @dataclass classes (e.g. SpectrumResult), and dataclass
    # processing looks its own module up via sys.modules[cls.__module__] --
    # if that lookup fails (module not registered yet), it raises
    # AttributeError: 'NoneType' object has no attribute '__dict__' deep
    # inside exec_module, before this function ever gets to return.
    sys.modules["spectral_bandwidth"] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception:
        del sys.modules["spectral_bandwidth"]
        raise
    return mod


def plot_checkpoint_spectrum(
    ckpt: dict,
    feature_index: int = 0,
    threshold: float = 1e-3,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
):
    """
    Spectral content of a single QUANTUM checkpoint's quantum layer:
    rebuilds the live model from `state_dict` + `config`, then calls
    `notebooks/spectral_bandwidth.py`'s `quantum_layer_spectrum` /
    `plot_spectrum_by_layers` on it (`trained=True`) -- energy vs. integer
    frequency k, with the analytic bound `L * n_photons` marked.

    Classical checkpoints have no `.quantum` submodule; raises ValueError.

    Returns (fig, SpectrumResult).
    """
    cfg = ckpt["config"]
    if cfg["model_type"] != "quantum":
        raise ValueError(
            "plot_checkpoint_spectrum only applies to quantum checkpoints "
            f"(got model_type={cfg['model_type']!r}) -- classical models have no "
            "quantum layer to measure the spectrum of."
        )

    pde = build_pde_from_config(cfg)
    model = build_model("quantum", cfg, pde)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    spec_mod = _import_spectral_bandwidth()
    result = spec_mod.quantum_layer_spectrum(
        model, feature_size=cfg["feature_size"], n_encoding_layers=cfg["L"],
        feature_index=feature_index, trained=True,
    )
    fig = spec_mod.plot_spectrum_by_layers(
        {cfg["L"]: result}, threshold=threshold, save_path=save_path, show=show,
    )
    return fig, result


# ─────────────────────────────────────────────────────────────────────────
# aggregate plots (combine however many checkpoints/seeds are present)
# ─────────────────────────────────────────────────────────────────────────

def _edges_from_centers(centers: List[float]) -> np.ndarray:
    """
    Cell edges for `pcolormesh(..., shading="flat")` from a list of cell-
    center coordinates, handling the single-value case explicitly.

    `pcolormesh(X, Y, C, shading="auto")` infers edges from centers itself
    when X/Y have the same shape as C -- but when an axis has only ONE
    unique value (e.g. a smoke test that sweeps only one PDE parameter),
    that inference degenerates to a ZERO-WIDTH cell on that axis: the mesh
    face has no area and is invisible, even though the axes autoscale to a
    small nonzero range around the point (from matplotlib's default margin,
    not real cell geometry) -- so the plot LOOKS like it has axis room for
    data that was actually never drawn. Computing real edges explicitly
    avoids relying on "auto"'s heuristic altogether.
    """
    c = np.asarray(centers, dtype=float)
    if len(c) == 1:
        half = max(abs(c[0]) * 0.05, 1e-6)
        return np.array([c[0] - half, c[0] + half])
    mids = (c[:-1] + c[1:]) / 2
    first = c[0] - (mids[0] - c[0])
    last = c[-1] + (c[-1] - mids[-1])
    return np.concatenate([[first], mids, [last]])


def plot_param_sweep_heatmap(
    checkpoints: List[dict],
    param1_name: str,
    param2_name: str,
    metric: str = "rel_l2",
    diff_mode: str = "signed",
    log_scale: bool = False,
    annotate: bool = True,
    annotate_fmt: str = "{:.2f}",
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Experiment 1's headline figure: three-panel heatmap over the
    (param1, param2) grid -- classical | quantum | difference -- built
    directly from a list of checkpoints (e.g. from
    `src.checkpoints.load_checkpoints`), grouping by however many seeds
    happen to be present at each grid point (ragged counts are fine -- no
    fixed-shape array to keep in sync as you add seeds incrementally).

    Every cell is annotated "mean ± std (n=k)" across the seeds found at
    that grid point.

    Args:
        checkpoints: list of checkpoint dicts, ALL at the same L and PDE
                     type (filter with `load_checkpoints(..., L=..., pde_type=...)`
                     first -- this function doesn't do that filtering itself,
                     since mixing L values into one heatmap wouldn't be a
                     fair comparison).
        param1_name, param2_name: PDE parameter names swept, e.g.
                     ("freq", "alpha") or ("freq", "c").
        metric:      one of "rel_l2", "mae", "max_abs_error", "rmse".
        diff_mode:   "signed" -> classical - quantum (>0 = quantum better);
                     "ratio"  -> quantum / classical (<1 = quantum better).
                     NOTE: unlike a fixed-array sweep, this is an UNPAIRED
                     mean difference (classical/quantum seed sets at a grid
                     point aren't required to line up 1:1 when collected
                     incrementally); its uncertainty band combines the two
                     stds in quadrature rather than reflecting a true paired
                     seed-by-seed difference.
    Returns:
        fig
    """
    df = checkpoints_to_dataframe(checkpoints)
    if len(df) == 0:
        raise ValueError("no checkpoints given.")
    df = df[np.isfinite(df[metric])]

    p1_vals = sorted(df[param1_name].unique())
    p2_vals = sorted(df[param2_name].unique())
    n1, n2 = len(p1_vals), len(p2_vals)

    def _grid(model_type):
        sub = df[df.model_type == model_type]
        mean = np.full((n1, n2), np.nan)
        std = np.full((n1, n2), np.nan)
        n = np.zeros((n1, n2), dtype=int)
        for i, p1 in enumerate(p1_vals):
            for j, p2 in enumerate(p2_vals):
                cell = sub[np.isclose(sub[param1_name], p1) & np.isclose(sub[param2_name], p2)]
                if len(cell):
                    mean[i, j] = cell[metric].mean()
                    std[i, j] = cell[metric].std(ddof=0)
                    n[i, j] = len(cell)
        return mean, std, n

    Zc, Sc, Nc = _grid("classical")
    Zq, Sq, Nq = _grid("quantum")

    if diff_mode == "signed":
        Zd = Zc - Zq
        Sd = np.sqrt(Sc ** 2 + Sq ** 2)
        diff_title = "Classical − Quantum  (>0 ⇒ quantum better)"
        dmax = np.nanmax(np.abs(Zd)) if np.any(np.isfinite(Zd)) else 1.0
        diff_norm = Normalize(vmin=-dmax, vmax=dmax)
    elif diff_mode == "ratio":
        Zd = Zq / np.where(Zc == 0, np.nan, Zc)
        Sd = np.abs(Zd) * np.sqrt((Sc / np.where(Zc == 0, np.nan, Zc)) ** 2
                                   + (Sq / np.where(Zq == 0, np.nan, Zq)) ** 2)
        diff_title = "Quantum / Classical  (<1 ⇒ quantum better)"
        finite = Zd[np.isfinite(Zd)]
        dmax = max(np.max(finite) if finite.size else 1.0,
                    1.0 / np.min(finite) if finite.size and np.min(finite) > 0 else 1.0, 1.0 + 1e-6)
        diff_norm = Normalize(vmin=1.0 / dmax, vmax=dmax)
    else:
        raise ValueError(f'diff_mode must be "signed" or "ratio", got {diff_mode!r}')

    field_vals = np.concatenate([Zc[np.isfinite(Zc)], Zq[np.isfinite(Zq)]])
    field_vmin = float(field_vals.min()) if field_vals.size else 0.0
    field_vmax = float(field_vals.max()) if field_vals.size else 1.0
    from matplotlib.colors import LogNorm
    field_norm = (LogNorm(vmin=max(field_vmin, 1e-12), vmax=max(field_vmax, 1e-12))
                  if log_scale else Normalize(vmin=field_vmin, vmax=field_vmax))

    # degrade annotation automatically on dense grids instead of overlapping
    max_annot_cells = 100
    do_annotate = annotate and (n1 * n2 <= max_annot_cells)
    if annotate and not do_annotate:
        print(f"[plot_param_sweep_heatmap] {n1}x{n2} grid too dense for legible "
              f"annotations -- disabling them (set annotate=False to silence this).")

    def _annotate(ax, values, stds, ns, cmap_name, norm):
        cmap_obj = plt.get_cmap(cmap_name)
        for i, y in enumerate(p1_vals):
            for j, x in enumerate(p2_vals):
                val = values[i, j]
                if np.isnan(val):
                    continue
                rgba = cmap_obj(norm(val))
                luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                color = "white" if luminance < 0.5 else "black"
                text = f"{annotate_fmt.format(val)}\n±{annotate_fmt.format(stds[i, j])}\n(n={ns[i, j]})"
                ax.text(x, y, text, ha="center", va="center", fontsize=7.5, color=color, linespacing=1.2)

    panel_w = float(np.clip(n2 * 1.1 + 1.6, 4.0, 7.0))
    panel_h = float(np.clip(n1 * 1.1 + 1.6, 3.5, 7.0))
    fig, axes = plt.subplots(1, 3, figsize=(3 * panel_w + 1.2, panel_h))

    p1_edges = _edges_from_centers(p1_vals)
    p2_edges = _edges_from_centers(p2_vals)

    for ax, Z, S, N, title in [
        (axes[0], Zc, Sc, Nc, f"Classical — {metric}"),
        (axes[1], Zq, Sq, Nq, f"Quantum — {metric}"),
    ]:
        im = ax.pcolormesh(p2_edges, p1_edges, Z, shading="flat", cmap="viridis", norm=field_norm)
        ax.set_xlabel(param2_name, fontsize=11)
        ax.set_ylabel(param1_name, fontsize=11)
        ax.set_title(title, fontsize=12)
        fig.colorbar(im, ax=ax, label=metric)
        if do_annotate:
            _annotate(ax, Z, S, N, "viridis", field_norm)

    im3 = axes[2].pcolormesh(p2_edges, p1_edges, Zd, shading="flat", cmap="RdBu_r", norm=diff_norm)
    axes[2].set_xlabel(param2_name, fontsize=11)
    axes[2].set_ylabel(param1_name, fontsize=11)
    axes[2].set_title(diff_title, fontsize=12)
    fig.colorbar(im3, ax=axes[2], label=diff_mode)
    if do_annotate:
        _annotate(axes[2], Zd, Sd, Nc, "RdBu_r", diff_norm)

    L_vals = df["L"].unique()
    L_str = str(int(L_vals[0])) if len(L_vals) == 1 else f"{sorted(L_vals)} (mixed!)"
    fig.suptitle(
        f"Parameter sweep ({param1_name} × {param2_name}) at L={L_str} — "
        f"{int(df['seed'].nunique())} seed(s) max per cell",
        fontsize=13,
    )
    fig.tight_layout()
    _save(fig, save_path)
    if show:
        plt.show()
    return fig


def plot_L_sweep(
    checkpoints: List[dict],
    metric: str = "rel_l2",
    target_frequency: Optional[float] = None,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Experiment 2's headline figure: `metric` vs. architecture size L,
    classical (param-matched) vs. quantum, mean ± std over however many
    seeds are present at each L -- built directly from a list of
    checkpoints (e.g. from `src.checkpoints.load_checkpoints`), all at the
    same fixed PDE parameters (mixed-parameter input will still plot but
    the title will warn you).

    Args:
        target_frequency: if given, draws a vertical line at the smallest
                 swept L whose stored `analytic_bound` (quantum checkpoints)
                 exceeds it -- the theory's predicted plateau point.
    Returns:
        fig
    """
    df = checkpoints_to_dataframe(checkpoints)
    if len(df) == 0:
        raise ValueError("no checkpoints given.")
    df = df[np.isfinite(df[metric])]
    Ls = sorted(df["L"].unique())

    def _series(model_type):
        sub = df[df.model_type == model_type]
        mean, std, n, params = [], [], [], []
        for L in Ls:
            cell = sub[sub.L == L]
            mean.append(cell[metric].mean() if len(cell) else np.nan)
            std.append(cell[metric].std(ddof=0) if len(cell) else np.nan)
            n.append(len(cell))
            params.append(cell["param_count"].iloc[0] if len(cell) else np.nan)
        return np.array(mean), np.array(std), n, params

    Zc, Sc, Nc, _ = _series("classical")
    Zq, Sq, Nq, Pq = _series("quantum")

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.errorbar(Ls, Zc, yerr=Sc, marker="o", lw=2, capsize=3, label="classical (param-matched)")
    ax.errorbar(Ls, Zq, yerr=Sq, marker="s", lw=2, capsize=3, label="quantum")

    if target_frequency is not None:
        bounds_by_L = df[df.model_type == "quantum"].groupby("L")["analytic_bound"].first()
        crossing = next((L for L in Ls if L in bounds_by_L.index
                          and bounds_by_L[L] >= target_frequency), None)
        if crossing is not None:
            ax.axvline(crossing, color="gray", ls="--", lw=1.5,
                       label=rf"predicted plateau ($L={crossing}$)")
        else:
            print(f"[plot_L_sweep] target_frequency={target_frequency} exceeds the analytic "
                  f"bound at every swept L -- no crossing to mark.")

    ax.set_xlabel("encoding depth $L$", fontsize=11)
    ax.set_ylabel(metric, fontsize=11)
    ax.set_yscale("log")
    ax.set_xticks(Ls)
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=9)

    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(Ls)
    ax2.set_xticklabels([f"{p:,.0f}" if np.isfinite(p) else "?" for p in Pq], fontsize=8, rotation=45)
    ax2.set_xlabel("quantum trainable parameters", fontsize=9)

    param_vals = df[["freq"]].drop_duplicates() if "freq" in df.columns else None
    fixed_params = {c: df[c].unique() for c in ("freq", "alpha", "c") if c in df.columns}
    fixed_str = ", ".join(
        f"{k}={v[0]:g}" if len(v) == 1 else f"{k}={list(v)} (mixed!)"
        for k, v in fixed_params.items()
    )
    n_seeds = df.groupby("L")["seed"].nunique().max()
    fig.suptitle(f"L sweep at {fixed_str} — up to {n_seeds} seed(s) per L", fontsize=12)
    fig.tight_layout()
    _save(fig, save_path)
    if show:
        plt.show()
    return fig
