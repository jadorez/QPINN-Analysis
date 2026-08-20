"""
plot_feature_size_results.py — Experiment 3's two headline figures, built
straight from `checkpoints/feature_size_sweep/` checkpoint files (see
`src.checkpoints`). Never retrains anything, so it's safe to re-run at any
point mid-sweep (e.g. while the largest feature_size is still short a seed
or two -- each run gets slower as feature_size grows, ~7h for feature_size=10
at the settings this repo uses).

Two views of the same sweep, mirroring `src.plot_freq_sweep_by_alpha`:
    1D — plot_feature_size_sweep (src.checkpoint_plots)
        The scalar `rel_l2` (an L2-norm ratio over the WHOLE (x,t)
        evaluation grid -- see `src.eval.evaluate`) vs. feature_size, mean ±
        std over seeds.
    2D — plot_feature_size_error_field_grid (src.checkpoint_plots)
        The pointwise error field U_pred - U_true over the actual (x,t)
        grid, laid out as a classical/quantum x feature_size grid of small
        heatmaps -- shows WHERE in (x,t) each run's error sits, not just
        the single rel_l2 number it collapses to. One seed's checkpoint per
        cell (falls back to whatever seed is on disk where the preferred
        one is still missing).

Usage (from the repo root):

    python -m src.plot_feature_size_results
"""

from __future__ import annotations

from pathlib import Path

from src.checkpoints import load_checkpoints
from src.checkpoint_plots import plot_feature_size_sweep, plot_feature_size_error_field_grid


if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parent.parent

    PDE_TYPE = "heat"
    L_FEATURE_SIZE_SWEEP = 1
    FIXED_FREQ_FS = 7.0     # matches FIXED_FREQ_FS in Checkpoint_pipeline.ipynb's Experiment 3
    FIXED_ALPHA_OR_C_FS = 0.0025
    SEED = 42                # present at every feature_size trained so far

    # feature_size=10 runs are excluded below: each one takes ~7h and the
    # sweep hasn't been carried through to full seed coverage there yet
    # (classical: 2/3 seeds, quantum: 1/3) -- drop it to keep every plotted
    # point backed by the same seed count. Add it back to FEATURE_SIZES_EXCLUDE
    # (or just remove the filter) once it's caught up.
    FEATURE_SIZES_EXCLUDE = [10]

    FIG_DIR = ROOT / "figures" / "checkpoint_pipeline"
    CKPT_DIR = ROOT / "checkpoints" / "feature_size_sweep"

    ckpts = load_checkpoints(
        CKPT_DIR, pde_type=PDE_TYPE, L=L_FEATURE_SIZE_SWEEP,
        freq=FIXED_FREQ_FS, alpha=FIXED_ALPHA_OR_C_FS,
    )
    ckpts = [c for c in ckpts if c["config"]["feature_size"] not in FEATURE_SIZES_EXCLUDE]
    print(f"{len(ckpts)} checkpoints loaded")

    plot_feature_size_sweep(
        ckpts, metric="rel_l2", target_frequency=2 * FIXED_FREQ_FS,
        save_path=FIG_DIR / "feature_size_sweep.png", show=False,
    )
    plot_feature_size_error_field_grid(
        ckpts, seed=SEED, shared_scale=True,
        save_path=FIG_DIR / "feature_size_error_field.png", show=False,
    )
