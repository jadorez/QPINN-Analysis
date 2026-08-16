"""
sweep.py — generic 2-D parameter sweeps (freq x a second PDE parameter)
comparing the classical and quantum ("weights") QPINN models on matched
parameter counts.

Typical usage (from a notebook):

    from src.sweep import get_matched_widths, run_2d_sweep
    from src.plotter import plot_2d_sweep
    from src.pde import HeatEquation

    feature_size, hidden_readout, quantum_output_size = get_matched_widths(4)

    result = run_2d_sweep(
        pde_factory=lambda freq, alpha: HeatEquation(freq=freq, alpha=alpha),
        param1_name="freq",  param1_values=[0.5, 1.5, 2.5, 3.5],
        param2_name="alpha", param2_values=[0.01, 0.05, 0.1, 0.25],
        feature_size=feature_size, hidden_readout=hidden_readout,
        quantum_output_size=quantum_output_size,
        seeds=(42, 666), epochs=400,
    )
    plot_2d_sweep(result, metric="rel_l2")
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from src.models import ClassicalHeatQPINN, MerlinTunableSimple, MerlinTunableEntropy
from src.train import train
from src.eval import evaluate
from src.loss import total_loss
from src.utils import find_matching_hidden_readout, quantum_outputsize_for_param_match
from src.diagnostics import train_and_diagnose, summarize_diagnostics


METRIC_KEYS = ("rel_l2", "mae", "max_abs_error", "rmse")


# ── analytic frequency ceiling (mirrors notebooks/spectral_bandwidth.py) ────
# Small, torch-free re-implementation so `run_depth_sweep` doesn't need a
# fragile cross-directory import from `notebooks/`. Keep in sync with
# `notebooks/spectral_bandwidth.py::n_photons_for_feature_size` /
# `analytic_frequency_bound` if either changes.

def n_photons_for_feature_size(feature_size: int) -> int:
    """Photon count in the fixed input Fock state built by
    `MerlinTunableSimple`/`MerlinTunableEntropy`: n_modes = feature_size+1,
    one photon per even-indexed mode."""
    n_modes = feature_size + 1
    return int(math.ceil(n_modes / 2))


def analytic_frequency_bound(feature_size: int, n_encoding_layers: int) -> int:
    """Symmetric ceiling |k| <= L * n_photons on the accessible integer
    frequency after L data re-uploads (see notebooks/spectral_bandwidth.py
    for the derivation)."""
    return n_encoding_layers * n_photons_for_feature_size(feature_size)


def get_matched_widths(
    feature_size_guess: int,
    search_range: Tuple[int, int] = (1, 100),
) -> Tuple[int, int, int]:
    """
    Find (feature_size, hidden_readout, quantum_output_size) such that
    `ClassicalHeatQPINN` and `MerlinTunableSimple` built with these values
    have matched parameter counts — same weight-matching logic already
    used in section 1 of the notebook (`find_matching_hidden_readout` /
    `quantum_outputsize_for_param_match`), wrapped so changing network
    size is a one-line change instead of copy-pasting that search loop.

    Args:
        feature_size_guess: desired `feature_size` (number of quantum
            input modes minus 1). Bigger -> more weights, but the matched
            `quantum_output_size` grows fast (combinatorial Hilbert-space
            dimension), so parameter count grows steeply with this.
        search_range: range searched for a compatible hidden_readout
            (passed straight to `find_matching_hidden_readout`).

    Returns:
        (feature_size, hidden_readout, quantum_output_size)

    Raises:
        ValueError if no integer-valued match exists for
        `feature_size_guess` within `search_range` — widen the range.
    """
    feature_sizes, hidden_readouts = find_matching_hidden_readout(
        feature_sizes=[feature_size_guess], ranges=list(search_range),
    )
    if not feature_sizes:
        raise ValueError(
            f"No matching hidden_readout found for feature_size={feature_size_guess} "
            f"within search_range={search_range}. Try widening search_range."
        )
    feature_size = feature_sizes[0]
    hidden_readout = hidden_readouts[0]
    quantum_output_size = int(quantum_outputsize_for_param_match(feature_size, hidden_readout))
    return feature_size, hidden_readout, quantum_output_size


@dataclass
class Sweep2DResult:
    param1_name: str
    param2_name: str
    param1_values: np.ndarray
    param2_values: np.ndarray
    seeds: Tuple[int, ...]
    # raw[model_type][metric_key] -> array (n_param1, n_param2, n_seeds)
    raw: Dict[str, Dict[str, np.ndarray]]
    feature_size: int
    hidden_readout: int
    quantum_output_size: int
    diagnostics: list = None                 # flat list of TrainingDiagnostics, one per run

    def mean(self, model_type: str, metric: str) -> np.ndarray:
        return np.nanmean(self.raw[model_type][metric], axis=-1)

    def std(self, model_type: str, metric: str) -> np.ndarray:
        return np.nanstd(self.raw[model_type][metric], axis=-1)


def run_2d_sweep(
    pde_factory: Callable[[float, float], object],
    param1_name: str,
    param1_values: Sequence[float],
    param2_name: str,
    param2_values: Sequence[float],
    feature_size: int,
    hidden_readout: int,
    quantum_output_size: int,
    seeds: Sequence[int] = (42, 666, 69),
    epochs: int = 600,
    classical_cls=ClassicalHeatQPINN,
    quantum_cls=MerlinTunableSimple,
    loss_fn=total_loss,
    log_every: Optional[int] = None,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
    verbose: bool = True,
    track_best: bool = False,
    eval_every: int = 50,
) -> Sweep2DResult:
    """
    Train + evaluate `classical_cls` and `quantum_cls` (matched parameter
    count via `feature_size`/`hidden_readout`/`quantum_output_size` — see
    `get_matched_widths`) over every (param1, param2) grid point, for
    every seed in `seeds`.

    `pde_factory(p1, p2) -> PDE` builds the PDE instance for one grid
    point, e.g. `lambda freq, alpha: HeatEquation(freq=freq, alpha=alpha)`
    or `lambda freq, c: WaveEquation(freq=freq, c=c)`.

    Cost: `len(param1_values) * len(param2_values) * len(seeds) * 2`
    full training runs (classical + quantum), same per-run cost as the
    1-D sweep in section 1 — this grows fast, start with a small grid /
    few epochs to sanity check before scaling up. Prints a running
    elapsed/ETA estimate if `verbose`.

    track_best=True switches from reporting each run's final-epoch metrics
    to its best-fixed-grid-checkpoint metrics (evaluated every `eval_every`
    epochs on a fixed, non-resampled grid) -- see `src.train.train`'s
    docstring for why this is preferable to picking by minimum *training*
    loss. Default False reproduces prior behaviour exactly.

    Returns a `Sweep2DResult` with per-seed raw metrics
    (`result.raw[model_type][metric]`, shape
    `(len(param1_values), len(param2_values), len(seeds))`) and
    `.mean(model_type, metric)` / `.std(model_type, metric)` accessors.
    """
    if log_every is None:
        log_every = epochs + 1  # silence per-epoch training logs; sweep itself logs progress

    param1_values = np.asarray(list(param1_values), dtype=float)
    param2_values = np.asarray(list(param2_values), dtype=float)
    n1, n2, ns = len(param1_values), len(param2_values), len(seeds)

    raw = {
        "classical": {k: np.full((n1, n2, ns), np.nan) for k in METRIC_KEYS},
        "qpinn_weights": {k: np.full((n1, n2, ns), np.nan) for k in METRIC_KEYS},
    }

    total_runs = n1 * n2 * ns * 2
    run_idx = 0
    t_start = time.time()
    diagnostics_log = []

    for i, p1 in enumerate(param1_values):
        for j, p2 in enumerate(param2_values):
            pde = pde_factory(float(p1), float(p2))
            for s, seed in enumerate(seeds):
                for model_type, cls, kwargs in (
                    ("classical", classical_cls,
                     dict(feature_size=feature_size, hidden_readout=hidden_readout,
                          quantum_output_size=quantum_output_size, pde=pde)),
                    ("qpinn_weights", quantum_cls,
                     dict(feature_size=feature_size, hidden_readout=hidden_readout, pde=pde)),
                ):
                    # reseed right before construction -- see the matching
                    # comment in run_depth_sweep for why this matters
                    # (nn.Module init happens before train()'s internal seed).
                    torch.manual_seed(seed)
                    np.random.seed(seed)
                    model = cls(**kwargs)
                    metrics, diag, _ = train_and_diagnose(
                        model=model, pde=pde, loss_fn=loss_fn, seed=seed,
                        epochs=epochs, log_every=log_every, device=device, dtype=dtype,
                        model_type=model_type, diag_verbose=False,
                        track_best=track_best, eval_every=eval_every,
                    )
                    diagnostics_log.append(diag)
                    for k in METRIC_KEYS:
                        raw[model_type][k][i, j, s] = metrics[k]

                    run_idx += 1
                    if verbose:
                        elapsed = time.time() - t_start
                        eta = elapsed / run_idx * (total_runs - run_idx)
                        flag = "" if diag.passed else "  [!] " + "; ".join(diag.messages[:1])
                        print(
                            f"[{run_idx:4d}/{total_runs}] {param1_name}={p1:g} "
                            f"{param2_name}={p2:g} seed={seed} {model_type:14s} "
                            f"rel_l2={metrics['rel_l2']:.3e}  "
                            f"elapsed={elapsed/60:.1f}m  eta={eta/60:.1f}m{flag}"
                        )

    if verbose:
        summarize_diagnostics(diagnostics_log)

    return Sweep2DResult(
        param1_name=param1_name, param2_name=param2_name,
        param1_values=param1_values, param2_values=param2_values,
        seeds=tuple(seeds), raw=raw,
        feature_size=feature_size, hidden_readout=hidden_readout,
        quantum_output_size=quantum_output_size,
        diagnostics=diagnostics_log,
    )


# ── depth (encoding-layer) sweep: tests the spectral-coverage /
# trainability hypotheses by varying `MerlinTunableEntropy`'s
# `n_encoding_layers` (L) at fixed feature_size, instead of varying a PDE
# parameter. Theory (analytic_frequency_bound) predicts accuracy should
# improve with L while L*n_photons is the binding constraint on the
# target's spatial frequency, then plateau. ─────────────────────────────

def _param_count(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def match_classical_params(
    target_params: int,
    feature_size: int,
    hidden_feature: int,
    pde,
    classical_cls=ClassicalHeatQPINN,
    hidden_readout_range: Sequence[int] = range(1, 64),
    quantum_output_size_range: Sequence[int] = range(2, 240),
) -> Tuple[int, int, int]:
    """
    Find the `classical_cls` config (`quantum_output_size`, `hidden_readout`)
    whose REAL, measured parameter count is closest to `target_params`, by
    brute-force search over instantiated models.

    Used instead of a closed-form matching formula (like
    `utils.quantum_outputsize_for_param_match`, which assumes a single
    encoding layer) because `MerlinTunableEntropy`'s parameter count
    depends on `n_encoding_layers` in a way that formula doesn't cover —
    counting real, instantiated models sidesteps re-deriving (and
    possibly getting wrong) a new formula for the re-uploading case.
    Cheap: building an `nn.Module` and counting its parameters costs
    nothing near what training it does.

    Returns:
        (quantum_output_size, hidden_readout, achieved_param_count)
    """
    best: Optional[Tuple[int, int, int, int]] = None  # (diff, qos, hr, n)
    for qos in quantum_output_size_range:
        for hr in hidden_readout_range:
            model = classical_cls(
                feature_size=feature_size, hidden_feature=hidden_feature,
                quantum_output_size=qos, hidden_readout=hr, pde=pde,
            )
            n = _param_count(model)
            diff = abs(n - target_params)
            if best is None or diff < best[0]:
                best = (diff, qos, hr, n)
            if diff == 0:
                return qos, hr, n
    assert best is not None
    _, qos, hr, n = best
    return qos, hr, n


@dataclass
class DepthSweepResult:
    depths: np.ndarray                       # L values swept
    seeds: Tuple[int, ...]
    # raw[model_type][metric_key] -> array (n_depths, n_seeds)
    raw: Dict[str, Dict[str, np.ndarray]]
    quantum_params: np.ndarray               # measured param count per depth
    classical_params: np.ndarray             # matched classical param count per depth
    analytic_bound: np.ndarray               # L * n_photons per depth
    feature_size: int
    diagnostics: list = None                 # flat list of TrainingDiagnostics, one per run
    # histories[model_type][i][s] -> (epochs, 5) loss-history array for the
    # run at depths[i], seeds[s]. None entries mean "not collected"
    # (shouldn't happen for a completed sweep, but keeps indexing safe).
    histories: Dict[str, List[List[Optional[np.ndarray]]]] = None
    # eval_epochs[model_type][i][s] -> the 1-indexed epoch whose weights
    # `metrics`/`raw` actually reflect: the best-checkpoint epoch if
    # track_best=True, otherwise the final epoch (epochs). Lets plots mark
    # which point on the training curve corresponds to the reported metric.
    eval_epochs: Dict[str, List[List[Optional[int]]]] = None

    def mean(self, model_type: str, metric: str) -> np.ndarray:
        return np.nanmean(self.raw[model_type][metric], axis=-1)

    def std(self, model_type: str, metric: str) -> np.ndarray:
        return np.nanstd(self.raw[model_type][metric], axis=-1)

    def history(self, model_type: str, depth, seed) -> np.ndarray:
        """Look up one run's loss-history array by depth VALUE and seed
        VALUE (not array index) -- e.g. `result.history("quantum", 3, 42)`."""
        i = int(np.where(self.depths == depth)[0][0])
        s = self.seeds.index(seed)
        return self.histories[model_type][i][s]


def run_depth_sweep(
    pde,
    depths: Sequence[int],
    feature_size: int,
    hidden_feature: int = 16,
    hidden_readout: int = 4,
    seeds: Sequence[int] = (42, 666, 69),
    epochs: int = 1000,
    quantum_cls=MerlinTunableEntropy,
    classical_cls=ClassicalHeatQPINN,
    loss_fn=total_loss,
    log_every: Optional[int] = None,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
    verbose: bool = True,
    track_best: bool = False,
    eval_every: int = 50,
) -> DepthSweepResult:
    """
    Sweep the quantum layer's re-upload depth L (`n_encoding_layers`) at a
    FIXED `feature_size` and PDE, comparing `quantum_cls`
    (`MerlinTunableEntropy`) against a `classical_cls` baseline whose
    parameter count is matched at every L via `match_classical_params`
    (a closed-form match, like section 1's, hasn't been derived for the
    re-uploading case).

    This is the experiment that turns the spectral-coverage hypothesis
    (`analytic_frequency_bound = L * n_photons`) into a tuning knob: it
    predicts rel_l2 should improve with L while the bound is the binding
    constraint on the target's spatial frequency, then plateau once it
    comfortably exceeds it. See `plot_depth_sweep` to visualise against
    that prediction.

    Cost: `len(depths) * len(seeds) * 2` full training runs, plus a cheap
    (untrained-model) parameter search per depth.

    track_best=True switches from reporting each run's final-epoch metrics
    to its best-fixed-grid-checkpoint metrics (see `src.train.train`'s
    docstring). Default False reproduces prior behaviour exactly.

    Returns a `DepthSweepResult` with per-seed raw metrics
    (`result.raw[model_type][metric]`, shape `(len(depths), len(seeds))`)
    and `.mean(model_type, metric)` / `.std(model_type, metric)`.
    """
    depths = list(depths)
    n_depths, n_seeds = len(depths), len(seeds)
    if log_every is None:
        log_every = epochs + 1

    raw = {
        "classical": {k: np.full((n_depths, n_seeds), np.nan) for k in METRIC_KEYS},
        "quantum": {k: np.full((n_depths, n_seeds), np.nan) for k in METRIC_KEYS},
    }
    quantum_params = np.zeros(n_depths, dtype=int)
    classical_params = np.zeros(n_depths, dtype=int)
    analytic_bound = np.array([analytic_frequency_bound(feature_size, L) for L in depths])
    diagnostics_log = []
    histories: Dict[str, List[List[Optional[np.ndarray]]]] = {
        "classical": [[None] * n_seeds for _ in range(n_depths)],
        "quantum": [[None] * n_seeds for _ in range(n_depths)],
    }
    eval_epochs: Dict[str, List[List[Optional[int]]]] = {
        "classical": [[None] * n_seeds for _ in range(n_depths)],
        "quantum": [[None] * n_seeds for _ in range(n_depths)],
    }

    total_runs = n_depths * n_seeds * 2
    run_idx = 0
    t_start = time.time()

    for i, L in enumerate(depths):
        # measure the quantum model's real parameter count at this depth,
        # then find a classical config matching it (structure only -- no
        # training needed for either step here).
        probe = quantum_cls(feature_size=feature_size, hidden_feature=hidden_feature,
                             hidden_readout=hidden_readout, n_encoding_layers=L, pde=pde)
        target_params = _param_count(probe)
        quantum_params[i] = target_params
        qos, hr_c, achieved = match_classical_params(
            target_params, feature_size, hidden_feature, pde, classical_cls=classical_cls,
        )
        classical_params[i] = achieved
        if verbose:
            print(f"L={L}: quantum_params={target_params}  bound=±{analytic_bound[i]}  "
                  f"classical match -> quantum_output_size={qos} hidden_readout={hr_c} "
                  f"params={achieved} (diff={achieved - target_params:+d})")

        for s, seed in enumerate(seeds):
            model_builders = (
                ("quantum", lambda: quantum_cls(feature_size=feature_size, hidden_feature=hidden_feature,
                                                 hidden_readout=hidden_readout, n_encoding_layers=L, pde=pde)),
                ("classical", lambda: classical_cls(feature_size=feature_size, hidden_feature=hidden_feature,
                                                     quantum_output_size=qos, hidden_readout=hr_c, pde=pde)),
            )
            for model_type, build_model in model_builders:
                # Reseed right before construction (not just inside train()):
                # nn.Module init draws from the global RNG at construction
                # time, which happens before train()'s internal
                # torch.manual_seed(seed) call. Without this, initial
                # weights are an uncontrolled function of how much RNG
                # state prior constructions (the probe model, the
                # match_classical_params brute-force search, the other
                # model_type built just before this one) happened to
                # consume -- not reproducible from `seed` at all, and for
                # this sweep specifically, correlated with L (deeper
                # models change the search cost that precedes them).
                torch.manual_seed(seed)
                np.random.seed(seed)
                model = build_model()
                metrics, diag, extras = train_and_diagnose(
                    model=model, pde=pde, loss_fn=loss_fn, seed=seed,
                    epochs=epochs, log_every=log_every, device=device, dtype=dtype,
                    model_type=model_type, diag_verbose=False,
                    track_best=track_best, eval_every=eval_every,
                )
                diagnostics_log.append(diag)
                histories[model_type][i][s] = extras[-1]  # (U_pred,U_true,x,t,history)
                # epoch that metrics/raw actually reflect: best checkpoint
                # if track_best, else the final epoch.
                eval_epochs[model_type][i][s] = (
                    int(diag.values["best_epoch"]) if "best_epoch" in diag.values else epochs
                )
                for k in METRIC_KEYS:
                    raw[model_type][k][i, s] = metrics[k]
                run_idx += 1
                if verbose:
                    elapsed = time.time() - t_start
                    eta = elapsed / run_idx * (total_runs - run_idx)
                    flag = "" if diag.passed else "  [!] " + "; ".join(diag.messages[:1])
                    print(f"[{run_idx:4d}/{total_runs}] L={L} seed={seed} {model_type:10s} "
                          f"rel_l2={metrics['rel_l2']:.3e}  elapsed={elapsed/60:.1f}m "
                          f"eta={eta/60:.1f}m{flag}")

    if verbose:
        summarize_diagnostics(diagnostics_log)

    return DepthSweepResult(
        depths=np.array(depths), seeds=tuple(seeds), raw=raw,
        quantum_params=quantum_params, classical_params=classical_params,
        analytic_bound=analytic_bound, feature_size=feature_size,
        diagnostics=diagnostics_log, histories=histories, eval_epochs=eval_epochs,
    )
