"""
diagnostics.py — per-run training sanity checks for classical/quantum PINNs.

Catches the failure modes already hit by hand in this project (collapsed
constant predictions, a broken hard-BC envelope, a decoupled PDE residual,
rel_l2 blown up by a tiny denominator) automatically, right after each
train()+evaluate() call, instead of only noticing them when a sweep plot
looks "fishy".

Typical usage — drop-in replacement for a train()+evaluate() pair:

    from src.diagnostics import train_and_diagnose

    metrics, diag, extras = train_and_diagnose(
        model=model, pde=pde, loss_fn=loss_fn, seed=seed,
        epochs=epochs, device=device, dtype=dtype, model_type="quantum",
    )
    diagnostics_log.append(diag)
    ...
    from src.diagnostics import summarize_diagnostics
    summarize_diagnostics(diagnostics_log)

Or call the checks standalone on an existing (history, U_pred, U_true):

    from src.diagnostics import check_training_sanity
    diag = check_training_sanity(model, pde, history, U_pred, U_true,
                                  x_grid=x_grid, t_grid=t_grid, seed=42,
                                  model_type="classical")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from src.train import train
from src.eval import evaluate


# ─────────────────────────────────────────────────────────────────────────
# result container
# ─────────────────────────────────────────────────────────────────────────

@dataclass
class TrainingDiagnostics:
    seed: Optional[int]
    model_type: str
    flags: Dict[str, bool] = field(default_factory=dict)      # True == problem found
    values: Dict[str, float] = field(default_factory=dict)
    messages: List[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not any(self.flags.values())

    def warn(self, key: str, condition: bool, message: str, value: Optional[float] = None) -> None:
        condition = bool(condition)
        self.flags[key] = condition
        if value is not None:
            self.values[key] = value
        if condition:
            self.messages.append(message)

    def report(self, verbose: bool = True) -> bool:
        if verbose:
            status = "OK" if self.passed else "ISSUES FOUND"
            print(f"[diagnostics] {self.model_type} seed={self.seed}: {status}")
            for m in self.messages:
                print(f"    - {m}")
        return self.passed


def _to_numpy(t) -> np.ndarray:
    if torch.is_tensor(t):
        return t.detach().cpu().numpy()
    return np.asarray(t)


# which .values entries to show alongside each flag when it fires, so the
# printed summary says *how bad*, not just *how often*.
FLAG_VALUE_KEYS: Dict[str, List[str]] = {
    "nan_loss": [],
    "loss_not_decreasing": ["loss_reduction_factor"],
    "oscillating": ["frac_increasing_smoothed"],
    "hard_constraint_not_zero": ["final_ic_loss", "final_bc_loss"],
    "nan_pred": [],
    "collapsed_prediction": ["pred_std", "true_std"],
    "tiny_denominator": ["true_norm", "rel_l2"],
    "worse_than_baseline": ["rel_l2", "baseline_rel_l2"],
    "no_trainable_params": [],
}


# ─────────────────────────────────────────────────────────────────────────
# core checks
# ─────────────────────────────────────────────────────────────────────────

def check_training_sanity(
    model,
    pde,
    history: np.ndarray,          # (epochs, 5): total, pde, consistency, initial, boundary
    U_pred,                       # (nx, nt) tensor/array
    U_true,                       # (nx, nt) tensor/array
    x_grid=None,                  # (nx, 1), needed for the trivial-baseline check
    t_grid=None,                  # unused directly, accepted for symmetry/future checks
    seed: Optional[int] = None,
    model_type: str = "model",
    hard_ic_bc: bool = True,      # set False if this model doesn't hard-constrain IC/BC
    loss_reduction_min_factor: float = 2.0,
    oscillation_frac_tol: float = 0.4,
    tiny_norm_tol: float = 1e-3,
    collapse_std_frac_tol: float = 0.05,
    soft_ic_bc_tol: float = 1e-3,
    verbose: bool = True,
) -> TrainingDiagnostics:
    """
    Run a battery of cheap sanity checks on one finished train()+evaluate()
    run and return a TrainingDiagnostics with a boolean flag + human message
    per failure mode. Nothing here requires re-running training or extra
    forward passes beyond what evaluate() already computed, except the
    optional trivial-baseline comparison.
    """
    diag = TrainingDiagnostics(seed=seed, model_type=model_type)

    history = np.asarray(history)
    U_pred_np = _to_numpy(U_pred)
    U_true_np = _to_numpy(U_true)

    # 1 — NaN/Inf anywhere in the loss history -> training diverged outright.
    diag.warn(
        "nan_loss",
        not np.all(np.isfinite(history)),
        "loss history contains NaN/Inf — training diverged.",
    )

    # 2 — total loss actually went down, not just wobbled near its init value.
    total = history[:, 0]
    if np.all(np.isfinite(total)) and len(total) >= 10:
        k = max(1, len(total) // 20)
        early, late = float(np.mean(total[:k])), float(np.mean(total[-k:]))
        factor = early / max(late, 1e-12)
        diag.values["loss_reduction_factor"] = factor
        diag.warn(
            "loss_not_decreasing",
            factor < loss_reduction_min_factor,
            f"loss only shrank {factor:.2f}x over training ({early:.3e} -> {late:.3e}); "
            f"expected >= {loss_reduction_min_factor:.1f}x. Model likely undertrained "
            "or stuck.",
        )

        # 3 — smoothed loss trending up more than trending down = instability,
        # not just healthy noisy-but-decreasing optimisation.
        window = max(5, len(total) // 50)
        if len(total) > 2 * window:
            smoothed = np.convolve(total, np.ones(window) / window, mode="valid")
            frac_increasing = float(np.mean(np.diff(smoothed) > 0))
            diag.values["frac_increasing_smoothed"] = frac_increasing
            diag.warn(
                "oscillating",
                frac_increasing > oscillation_frac_tol,
                f"smoothed total loss increases over {frac_increasing:.0%} of training "
                f"(tol {oscillation_frac_tol:.0%}) — looks unstable rather than converging.",
            )

    # 4 — IC/BC loss terms should already be ~0 throughout for a
    # hard-constrained architecture (the envelope makes the constraint exact
    # by construction). If they aren't, the envelope wiring is probably
    # broken -- this is exactly the class of bug hit earlier when the hard
    # BC term got accidentally dropped from forward().
    if hard_ic_bc and history.shape[1] >= 5:
        final_ic, final_bc = float(history[-1, 3]), float(history[-1, 4])
        diag.values["final_ic_loss"] = final_ic
        diag.values["final_bc_loss"] = final_bc
        diag.warn(
            "hard_constraint_not_zero",
            max(final_ic, final_bc) > soft_ic_bc_tol,
            f"model is supposed to hard-constrain IC/BC but the logged IC/BC loss "
            f"terms are not ~0 at the end of training (IC={final_ic:.2e}, "
            f"BC={final_bc:.2e}, tol={soft_ic_bc_tol:.0e}) — check the envelope "
            "wiring in forward() (u = f_x + bc_envelope * ic_envelope * u_raw).",
        )

    # 5 — NaN/Inf in the final prediction grid.
    diag.warn(
        "nan_pred",
        not np.all(np.isfinite(U_pred_np)),
        "predicted solution grid contains NaN/Inf.",
    )

    # 6 — prediction collapsed to a near-constant surface (the "flat solution"
    # failure mode diagnosed earlier for large f / large alpha).
    pred_std, true_std = float(np.std(U_pred_np)), float(np.std(U_true_np))
    diag.values["pred_std"], diag.values["true_std"] = pred_std, true_std
    diag.warn(
        "collapsed_prediction",
        true_std > 1e-6 and pred_std < collapse_std_frac_tol * true_std,
        f"prediction std ({pred_std:.2e}) is under {collapse_std_frac_tol:.0%} of the "
        f"true solution's std ({true_std:.2e}) — model likely collapsed to a "
        "near-constant output.",
    )

    # 7 — rel_l2's denominator isn't degenerate. A tiny ||U_true|| inflates
    # rel_l2 for even a modest absolute error, which can look like a much
    # bigger failure than it is (this is the >100% rel_l2 case).
    true_norm = float(np.linalg.norm(U_true_np))
    err_norm = float(np.linalg.norm(U_pred_np - U_true_np))
    rel_l2 = err_norm / max(true_norm, 1e-12)
    diag.values["true_norm"], diag.values["rel_l2"] = true_norm, rel_l2
    diag.warn(
        "tiny_denominator",
        true_norm < tiny_norm_tol,
        f"||U_true||={true_norm:.2e} is tiny — rel_l2 (={rel_l2:.2f}) is likely "
        "dominated by a small denominator rather than reflecting a large absolute "
        "error. Report absolute L2/RMSE alongside rel_l2 at this operating point.",
    )

    # 8 — beats the trivial "freeze the initial condition for all t" baseline.
    # If it doesn't, the model hasn't learned any time evolution at all.
    if x_grid is not None and hasattr(pde, "initial_condition"):
        x_np = _to_numpy(x_grid).reshape(-1, 1)
        x_t = torch.as_tensor(x_np, dtype=torch.float32)
        with torch.no_grad():
            f_x = _to_numpy(pde.initial_condition(x_t)).reshape(-1)
        baseline = np.repeat(f_x[:, None], U_true_np.shape[1], axis=1)
        baseline_err = float(np.linalg.norm(baseline - U_true_np) / max(true_norm, 1e-12))
        diag.values["baseline_rel_l2"] = baseline_err
        diag.warn(
            "worse_than_baseline",
            rel_l2 >= baseline_err,
            f"trained model (rel_l2={rel_l2:.2f}) does not beat the trivial "
            f"'frozen initial condition' baseline (rel_l2={baseline_err:.2f}) — "
            "no meaningful time evolution was learned.",
        )

    # 9 — model actually has trainable parameters (catches a mis-constructed
    # model, e.g. accidentally frozen weights during a param-matching search).
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    diag.values["n_trainable_params"] = float(n_params)
    diag.warn(
        "no_trainable_params",
        n_params == 0,
        "model has zero trainable parameters — nothing could have been learned.",
    )

    if verbose:
        diag.report()
    return diag


# ─────────────────────────────────────────────────────────────────────────
# convenience wrapper: train + evaluate + diagnose in one call
# ─────────────────────────────────────────────────────────────────────────

def train_and_diagnose(
    model,
    pde,
    loss_fn=None,
    seed: int = 42,
    epochs: int = 1000,
    log_every: Optional[int] = None,
    device=None,
    dtype: torch.dtype = torch.float32,
    nx: int = 60,
    nt: int = 60,
    model_type: str = "model",
    hard_ic_bc: bool = True,
    diag_verbose: bool = True,
    track_best: bool = False,
    eval_every: int = 50,
    **train_kwargs: Any,
):
    """
    Drop-in replacement for the `train(...); metrics, *_ = evaluate(...)`
    pair used in src/sweep.py's inner loops. Returns
    `(metrics, diag, (U_pred, U_true, x_grid, t_grid, history))` so existing
    call sites only need to add one line to collect `diag`.

    `track_best=False` (default) reproduces prior behaviour exactly: the
    returned `metrics`/`U_pred`/`U_true` are evaluated on the model's final-
    epoch weights. `track_best=True` instead evaluates the checkpoint with
    the lowest rel_l2 on a fixed (eval_every-spaced, non-resampled) grid
    seen during training -- see `train`'s docstring for why this differs
    from picking the epoch with the lowest *training* loss. When enabled,
    `diag.values` additionally gets "best_epoch" / "best_rel_l2_tracked" /
    "final_epoch_rel_l2_tracked" so you can see, per run, how much this
    mattered (e.g. best_epoch far from the final epoch signals late-training
    instability worth cross-checking against the oscillation flag).
    """
    if loss_fn is None:
        from src.loss import total_loss
        loss_fn = total_loss
    if log_every is None:
        log_every = epochs + 1

    best_info: Optional[dict] = {} if track_best else None
    history = train(
        model=model, pde=pde, loss_fn=loss_fn, seed=seed,
        epochs=epochs, log_every=log_every, device=device, dtype=dtype,
        track_best=track_best, eval_every=eval_every, eval_nx=nx, eval_nt=nt,
        best_info=best_info,
        **train_kwargs,
    )
    metrics, U_pred, U_true, x_grid, t_grid = evaluate(
        model=model, pde=pde, nx=nx, nt=nt, device=device, dtype=dtype,
    )
    diag = check_training_sanity(
        model=model, pde=pde, history=history, U_pred=U_pred, U_true=U_true,
        x_grid=x_grid, t_grid=t_grid, seed=seed, model_type=model_type,
        hard_ic_bc=hard_ic_bc, verbose=diag_verbose,
    )
    if best_info:
        diag.values["best_epoch"] = float(best_info.get("best_epoch") or -1)
        if best_info.get("best_rel_l2") is not None:
            diag.values["best_rel_l2_tracked"] = float(best_info["best_rel_l2"])
        if best_info.get("final_epoch_rel_l2") is not None:
            diag.values["final_epoch_rel_l2_tracked"] = float(best_info["final_epoch_rel_l2"])
    return metrics, diag, (U_pred, U_true, x_grid, t_grid, history)


# ─────────────────────────────────────────────────────────────────────────
# aggregate report across many runs (e.g. every run in a sweep)
# ─────────────────────────────────────────────────────────────────────────

def summarize_diagnostics(diagnostics: List[TrainingDiagnostics], verbose: bool = True) -> Dict[str, Dict[str, float]]:
    """
    Aggregate a list of TrainingDiagnostics (e.g. one per run in a sweep)
    into a per-model_type failure-rate table: {model_type: {flag: fraction_failed}}.
    Prints a summary if verbose. Use this after a sweep to spot systemic
    issues (e.g. "hard_constraint_not_zero fails in 100% of quantum runs at
    L=1") rather than reading each run's printout individually.
    """
    by_type: Dict[str, List[TrainingDiagnostics]] = {}
    for d in diagnostics:
        by_type.setdefault(d.model_type, []).append(d)

    summary: Dict[str, Dict[str, float]] = {}
    for model_type, ds in by_type.items():
        all_keys = sorted({k for d in ds for k in d.flags})
        rates = {k: float(np.mean([d.flags.get(k, False) for d in ds])) for k in all_keys}
        n_passed = sum(d.passed for d in ds)
        rates["_all_checks_passed"] = n_passed / len(ds)
        summary[model_type] = rates

    if verbose:
        for model_type, rates in summary.items():
            ds = by_type[model_type]
            n = len(ds)
            print(f"\n[diagnostics summary] {model_type} — {n} runs, "
                  f"{rates['_all_checks_passed']:.0%} fully clean")
            for k, v in rates.items():
                if k == "_all_checks_passed" or v == 0.0:
                    continue
                print(f"    {k}: failed in {v:.0%} of runs")
                # show the actual magnitudes behind the flag, not just the rate
                for vk in FLAG_VALUE_KEYS.get(k, []):
                    vals = [d.values[vk] for d in ds if vk in d.values]
                    if vals:
                        print(f"        {vk}: min={min(vals):.3e}  "
                              f"mean={np.mean(vals):.3e}  max={max(vals):.3e}")

    return summary


def diagnostics_to_dataframe(diagnostics: List[TrainingDiagnostics]):
    """
    Flatten a list of TrainingDiagnostics into one row per run, with every
    flag as a `flag_<name>` bool column and every recorded value as its own
    column (e.g. `final_ic_loss`, `rel_l2`, `baseline_rel_l2`) — so you can
    sort/filter runs yourself instead of only reading the printed summary.

    Usage:
        df = diagnostics_to_dataframe(depth_result.diagnostics)
        df[~df["passed"]]                          # only failing runs
        df.sort_values("rel_l2", ascending=False)   # worst runs first
        df.groupby("model_type")[["final_ic_loss", "final_bc_loss"]].mean()
    """
    import pandas as pd

    rows = []
    for d in diagnostics:
        row = {"model_type": d.model_type, "seed": d.seed, "passed": d.passed}
        row.update(d.values)
        for k, v in d.flags.items():
            row[f"flag_{k}"] = v
        rows.append(row)
    return pd.DataFrame(rows)
