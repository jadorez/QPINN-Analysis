"""
checkpoints.py — checkpoint-based experiment runner for the two headline
experiments (parameter sweep, depth/L sweep), built so that:

  1) Training is never repeated just to change a plot. Every single
     (pde params, L, model_type, seed) run is trained ONCE and saved to
     its own file (`run_checkpoint`), containing everything a plot could
     need: weights, loss history, evaluation grids, metrics, diagnostics.
  2) Runs can be added incrementally. `run_checkpoint` skips training and
     just returns the existing path if a checkpoint for that exact config
     is already on disk (unless `overwrite=True`). So running more seeds
     later, or resuming after an interrupted sweep, means calling the same
     driver function again with a bigger `seeds=(...)` tuple -- already-
     done runs are skipped, only the new ones train.
  3) Combining runs to increase statistics is just "load every checkpoint
     file in a directory matching some filters" (`load_checkpoints`) plus
     a groupby/aggregate over however many seeds happen to be present per
     grid point -- no fixed-shape (n1, n2, n_seeds) array to keep in sync.

Both experiments share ONE notion of "model size L": `MerlinTunableEntropy`
with `n_encoding_layers=L` for the quantum model, matched at every L to a
`ClassicalHeatQPINN` baseline with (as close as brute-force search gets)
the same trainable parameter count, via `src.sweep.match_classical_params`.

    Experiment 1 (parameter sweep): fix L, sweep (freq, alpha) [heat] or
    (freq, c) [wave], compare classical vs. quantum at every grid point,
    for several seeds -> `run_param_sweep_checkpoints` +
    `src.checkpoint_plots.plot_param_sweep_heatmap`.

    Experiment 2 (L sweep): fix (freq, alpha)/(freq, c), sweep L, compare
    classical vs. quantum at every L, for several seeds ->
    `run_L_sweep_checkpoints` + `src.checkpoint_plots.plot_L_sweep`.

Typical usage (see notebooks/Checkpoint_pipeline.ipynb for the full thing):

    from src.pde import HeatEquation
    from src.checkpoints import run_param_sweep_checkpoints, load_checkpoints

    run_param_sweep_checkpoints(
        ckpt_dir="checkpoints/param_sweep",
        pde_type="heat",
        param1_name="freq", param1_values=[0.5, 1.5, 2.5],
        param2_name="alpha", param2_values=[0.01, 0.05, 0.1],
        L=3, feature_size=4, hidden_feature=16, hidden_readout_quantum=4,
        seeds=(42, 666), epochs=2000, track_best=True,
    )
    ckpts = load_checkpoints("checkpoints/param_sweep", pde_type="heat")
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from src.pde import HeatEquation, WaveEquation
from src.models import ClassicalHeatQPINN, MerlinTunableEntropy
from src.loss import total_loss
from src.diagnostics import train_and_diagnose, TrainingDiagnostics
from src.sweep import match_classical_params, analytic_frequency_bound


SCHEMA_VERSION = 1


# ─────────────────────────────────────────────────────────────────────────
# PDE <-> config dispatch
# ─────────────────────────────────────────────────────────────────────────

def pde_to_config(pde) -> Tuple[str, Dict[str, float]]:
    """(pde_type, pde_params) for the two swept-parameter conventions used
    by the two experiments: (freq, alpha) for HeatEquation, (freq, c) for
    WaveEquation. Domain bounds are left at their dataclass defaults and
    not stored -- this repo always uses [0,1]x[0,1]."""
    if isinstance(pde, HeatEquation):
        return "heat", {"freq": float(pde.freq), "alpha": float(pde.alpha)}
    if isinstance(pde, WaveEquation):
        return "wave", {"freq": float(pde.freq), "c": float(pde.c)}
    raise TypeError(f"Unsupported pde type: {type(pde)!r}")


def build_pde(pde_type: str, pde_params: Dict[str, float]):
    if pde_type == "heat":
        return HeatEquation(**pde_params)
    if pde_type == "wave":
        return WaveEquation(**pde_params)
    raise ValueError(f"Unknown pde_type: {pde_type!r}")


def build_pde_from_config(config: Dict) -> object:
    return build_pde(config["pde_type"], config["pde_params"])


def build_model(model_type: str, config: Dict, pde,
                 quantum_cls=MerlinTunableEntropy, classical_cls=ClassicalHeatQPINN):
    if model_type == "quantum":
        return quantum_cls(
            feature_size=config["feature_size"],
            hidden_feature=config["hidden_feature"],
            hidden_readout=config["hidden_readout"],
            n_encoding_layers=config["L"],
            pde=pde,
        )
    if model_type == "classical":
        if config.get("quantum_output_size") is None:
            raise ValueError(
                "config['quantum_output_size'] is required to rebuild a classical "
                "checkpoint's model -- it should always be set by run_checkpoint/"
                "resolve_matched_config."
            )
        return classical_cls(
            feature_size=config["feature_size"],
            hidden_feature=config["hidden_feature"],
            quantum_output_size=config["quantum_output_size"],
            hidden_readout=config["hidden_readout"],
            pde=pde,
        )
    raise ValueError(f"model_type must be 'classical' or 'quantum', got {model_type!r}")


# ─────────────────────────────────────────────────────────────────────────
# parameter matching (shared "model size L" concept across both experiments)
# ─────────────────────────────────────────────────────────────────────────

def _param_count(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def resolve_matched_config(
    L: int,
    feature_size: int,
    hidden_feature: int,
    hidden_readout_quantum: int,
    pde,
    quantum_cls=MerlinTunableEntropy,
    classical_cls=ClassicalHeatQPINN,
    match_cache: Optional[dict] = None,
    verbose: bool = True,
) -> Dict:
    """
    For a given architecture size L, build an (untrained, cheap) quantum
    probe model, measure its real parameter count, and find the matching
    classical config via `src.sweep.match_classical_params`. Matching only
    depends on (L, feature_size, hidden_feature, hidden_readout_quantum,
    the two model classes) -- NOT on the specific PDE parameter values --
    so this is safe to compute once and reuse across an entire parameter
    sweep at fixed L, or once per L in a depth sweep. Pass a shared dict as
    `match_cache` (e.g. across notebook cells) to skip recomputing it
    entirely for an L visited before.

    Returns a dict with keys: quantum_hidden_readout, quantum_params,
    classical_quantum_output_size, classical_hidden_readout,
    classical_params, analytic_bound.
    """
    key = (int(L), int(feature_size), int(hidden_feature), int(hidden_readout_quantum),
           quantum_cls.__name__, classical_cls.__name__)
    if match_cache is not None and key in match_cache:
        return match_cache[key]

    probe = quantum_cls(
        feature_size=feature_size, hidden_feature=hidden_feature,
        hidden_readout=hidden_readout_quantum, n_encoding_layers=L, pde=pde,
    )
    target_params = _param_count(probe)
    qos, hr_c, achieved = match_classical_params(
        target_params, feature_size, hidden_feature, pde, classical_cls=classical_cls,
    )
    result = {
        "quantum_hidden_readout": int(hidden_readout_quantum),
        "quantum_params": int(target_params),
        "classical_quantum_output_size": int(qos),
        "classical_hidden_readout": int(hr_c),
        "classical_params": int(achieved),
        "analytic_bound": int(analytic_frequency_bound(feature_size, L)),
    }
    if verbose:
        print(f"[checkpoints] L={L}: quantum_params={target_params}  "
              f"bound=±{result['analytic_bound']}  classical match -> "
              f"quantum_output_size={qos} hidden_readout={hr_c} params={achieved} "
              f"(diff={achieved - target_params:+d})")
    if match_cache is not None:
        match_cache[key] = result
    return result


# ─────────────────────────────────────────────────────────────────────────
# checkpoint filename / (de)serialisation
# ─────────────────────────────────────────────────────────────────────────

def _fmt(v) -> str:
    s = f"{v:g}" if isinstance(v, float) else str(v)
    return s.replace(".", "p").replace("-", "m")


def checkpoint_filename(config: Dict) -> str:
    """Deterministic filename encoding every field that makes a checkpoint
    unique. The `config` dict saved INSIDE the file is the actual source of
    truth for filtering (`load_checkpoints`) -- this filename only needs to
    avoid collisions and be human-scannable in a directory listing."""
    pp = config["pde_params"]
    param_part = "_".join(f"{k}{_fmt(pp[k])}" for k in sorted(pp))
    qos = config.get("quantum_output_size")
    qos_part = f"_qos{qos}" if qos is not None else ""
    return (
        f"{config['pde_type']}__{param_part}__L{config['L']}"
        f"__fs{config['feature_size']}_hf{config['hidden_feature']}"
        f"_hr{config['hidden_readout']}{qos_part}"
        f"__{config['model_type']}__seed{config['seed']}__ep{config['epochs']}.pt"
    )


def _diag_to_dict(diag: TrainingDiagnostics) -> Dict:
    return {
        "seed": diag.seed,
        "model_type": diag.model_type,
        "flags": dict(diag.flags),
        "values": dict(diag.values),
        "messages": list(diag.messages),
        "passed": diag.passed,
    }


# ─────────────────────────────────────────────────────────────────────────
# single-run checkpoint: train + evaluate + diagnose + save (or skip)
# ─────────────────────────────────────────────────────────────────────────

def run_checkpoint(
    ckpt_dir: Union[str, Path],
    pde,
    model_type: str,                 # "classical" | "quantum"
    L: int,
    feature_size: int,
    hidden_feature: int,
    hidden_readout: int,             # ACTUAL hidden_readout for this model_type
    quantum_output_size: Optional[int] = None,   # required if model_type == "classical"
    seed: int = 42,
    epochs: int = 1000,
    quantum_cls=MerlinTunableEntropy,
    classical_cls=ClassicalHeatQPINN,
    loss_fn=total_loss,
    log_every: Optional[int] = None,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
    track_best: bool = True,
    eval_every: int = 50,
    eval_nx: int = 60,
    eval_nt: int = 60,
    overwrite: bool = False,
    verbose: bool = True,
) -> Path:
    """
    Train (or skip, if already on disk) exactly ONE run and persist a
    self-contained checkpoint file with everything needed to replot it
    later without retraining: weights, loss history, the evaluation grid
    (U_pred/U_true/x_grid/t_grid), metrics, and diagnostics.

    Skip-if-exists (unless `overwrite=True`) is the whole mechanism behind
    "add more seeds later without redoing everything" -- the filename is a
    deterministic function of every field in `config`, so re-running this
    with the same arguments is a no-op, and changing just `seed` (or a PDE
    parameter, or L) always produces a new file alongside the old ones.

    Returns the checkpoint's path (whether freshly trained or pre-existing).
    """
    if model_type not in ("classical", "quantum"):
        raise ValueError(f"model_type must be 'classical' or 'quantum', got {model_type!r}")
    if model_type == "classical" and quantum_output_size is None:
        raise ValueError(
            "quantum_output_size is required for model_type='classical' -- pass the "
            "matched value from resolve_matched_config()['classical_quantum_output_size']."
        )

    pde_type, pde_params = pde_to_config(pde)
    config = {
        "schema_version": SCHEMA_VERSION,
        "pde_type": pde_type, "pde_params": pde_params,
        "model_type": model_type, "L": int(L),
        "feature_size": int(feature_size), "hidden_feature": int(hidden_feature),
        "hidden_readout": int(hidden_readout),
        "quantum_output_size": int(quantum_output_size) if quantum_output_size is not None else None,
        "seed": int(seed), "epochs": int(epochs),
        "track_best": bool(track_best), "eval_every": int(eval_every),
        "eval_nx": int(eval_nx), "eval_nt": int(eval_nt),
    }

    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / checkpoint_filename(config)

    if path.exists() and not overwrite:
        if verbose:
            print(f"[checkpoints] skip (already exists): {path.name}")
        return path

    if log_every is None:
        log_every = epochs + 1  # silence per-epoch logs; driver loops log progress themselves

    # Reseed immediately before construction -- nn.Module init consumes the
    # global RNG at construction time, before train()'s internal reseed.
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = build_model(model_type, config, pde, quantum_cls=quantum_cls, classical_cls=classical_cls)

    metrics, diag, extras = train_and_diagnose(
        model=model, pde=pde, loss_fn=loss_fn, seed=seed, epochs=epochs,
        log_every=log_every, device=device, dtype=dtype, model_type=model_type,
        diag_verbose=False, track_best=track_best, eval_every=eval_every,
        nx=eval_nx, nt=eval_nt,
    )
    U_pred, U_true, x_grid, t_grid, history = extras

    eval_epoch = int(diag.values["best_epoch"]) if "best_epoch" in diag.values else int(epochs)
    param_count = _param_count(model)
    analytic_bound = analytic_frequency_bound(feature_size, L) if model_type == "quantum" else None

    ckpt = {
        "config": config,
        "state_dict": model.state_dict(),
        "history": history,
        "metrics": metrics,
        "eval_epoch": eval_epoch,
        "param_count": int(param_count),
        "U_pred": U_pred, "U_true": U_true, "x_grid": x_grid, "t_grid": t_grid,
        "diagnostics": _diag_to_dict(diag),
        "analytic_bound": analytic_bound,
    }
    torch.save(ckpt, path)

    if verbose:
        flag = "OK" if diag.passed else "ISSUES: " + "; ".join(diag.messages[:2])
        print(f"[checkpoints] saved {path.name}  rel_l2={metrics['rel_l2']:.3e}  {flag}")
    return path


# ─────────────────────────────────────────────────────────────────────────
# loading / filtering / flattening
# ─────────────────────────────────────────────────────────────────────────

def load_checkpoint(path: Union[str, Path]) -> Dict:
    return torch.load(Path(path), map_location="cpu", weights_only=False)


def _matches(config: Dict, filters: Dict) -> bool:
    for k, v in filters.items():
        if v is None:
            continue
        if k in config:
            cv = config[k]
        elif k in config.get("pde_params", {}):
            cv = config["pde_params"][k]
        else:
            return False
        candidates = v if isinstance(v, (list, tuple, set)) else [v]
        ok = False
        for cand in candidates:
            if isinstance(cv, float) or isinstance(cand, float):
                ok = ok or math.isclose(float(cv), float(cand), rel_tol=1e-9, abs_tol=1e-9)
            else:
                ok = ok or (cv == cand)
        if not ok:
            return False
    return True


def load_checkpoints(ckpt_dir: Union[str, Path], **filters) -> List[Dict]:
    """
    Load every `*.pt` checkpoint in `ckpt_dir` whose config matches all
    given filters (keyword = config field or pde_params field; a list/tuple
    value matches any of its elements). E.g.:

        load_checkpoints("checkpoints/param_sweep", pde_type="heat", L=3)
        load_checkpoints("checkpoints/L_sweep", model_type="quantum", seed=[42, 666])

    This is the "combine checkpoints to increase statistics" mechanism:
    call it again after adding more seed runs and every matching file on
    disk (old and new) comes back in the result.
    """
    ckpt_dir = Path(ckpt_dir)
    out = []
    for p in sorted(ckpt_dir.glob("*.pt")):
        ckpt = load_checkpoint(p)
        if _matches(ckpt["config"], filters):
            ckpt["_path"] = str(p)
            out.append(ckpt)
    return out


def checkpoints_to_dataframe(checkpoints: List[Dict]):
    """
    Flatten a list of checkpoint dicts (as returned by `load_checkpoints`)
    into one row per run -- config fields, pde params, metrics, eval_epoch,
    param_count, analytic_bound, diagnostics pass/fail -- WITHOUT the large
    arrays (state_dict, history, U_pred/U_true/x_grid/t_grid), so it's cheap
    to inspect/filter/sort a whole checkpoint directory at a glance:

        df = checkpoints_to_dataframe(load_checkpoints("checkpoints/L_sweep"))
        df[~df.diag_passed]
        df.sort_values("rel_l2", ascending=False).head()
    """
    import pandas as pd

    rows = []
    for c in checkpoints:
        cfg = c["config"]
        row = {k: v for k, v in cfg.items() if k != "pde_params"}
        row.update(cfg.get("pde_params", {}))
        row.update(c["metrics"])
        row["eval_epoch"] = c["eval_epoch"]
        row["param_count"] = c["param_count"]
        row["analytic_bound"] = c.get("analytic_bound")
        row["diag_passed"] = c["diagnostics"]["passed"]
        row["path"] = c.get("_path")
        rows.append(row)
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────
# experiment 1 — parameter sweep at fixed L
# ─────────────────────────────────────────────────────────────────────────

def run_param_sweep_checkpoints(
    ckpt_dir: Union[str, Path],
    pde_type: str,                       # "heat" | "wave"
    param1_name: str,                    # "freq"
    param1_values: Sequence[float],
    param2_name: str,                    # "alpha" (heat) or "c" (wave)
    param2_values: Sequence[float],
    L: int,
    feature_size: int,
    hidden_feature: int,
    hidden_readout_quantum: int,
    seeds: Sequence[int] = (42, 666, 69),
    epochs: int = 1000,
    quantum_cls=MerlinTunableEntropy,
    classical_cls=ClassicalHeatQPINN,
    loss_fn=total_loss,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
    track_best: bool = True,
    eval_every: int = 50,
    eval_nx: int = 60,
    eval_nt: int = 60,
    overwrite: bool = False,
    verbose: bool = True,
    match_cache: Optional[dict] = None,
) -> List[Path]:
    """
    Experiment 1: fixed architecture size L, sweep (param1, param2) over
    the PDE's grid, classical vs. quantum, for every seed. Safe to call
    repeatedly with a bigger grid/seeds tuple -- already-trained
    (param1, param2, seed, model_type) combinations are skipped (see
    `run_checkpoint`). Matching (classical config for this L) is computed
    ONCE up front since it doesn't depend on the PDE parameter values.

    Returns the list of checkpoint paths touched (trained or skipped).
    """
    param1_values = list(param1_values)
    param2_values = list(param2_values)
    probe_pde = build_pde(pde_type, {param1_name: param1_values[0], param2_name: param2_values[0]})
    matched = resolve_matched_config(
        L, feature_size, hidden_feature, hidden_readout_quantum, probe_pde,
        quantum_cls=quantum_cls, classical_cls=classical_cls,
        match_cache=match_cache, verbose=verbose,
    )

    total_runs = len(param1_values) * len(param2_values) * len(seeds) * 2
    run_idx = 0
    t_start = time.time()
    paths = []

    for p1 in param1_values:
        for p2 in param2_values:
            pde = build_pde(pde_type, {param1_name: p1, param2_name: p2})
            for seed in seeds:
                for model_type in ("classical", "quantum"):
                    if model_type == "quantum":
                        kwargs = dict(hidden_readout=hidden_readout_quantum, quantum_output_size=None)
                    else:
                        kwargs = dict(hidden_readout=matched["classical_hidden_readout"],
                                      quantum_output_size=matched["classical_quantum_output_size"])
                    path = run_checkpoint(
                        ckpt_dir, pde, model_type, L, feature_size, hidden_feature,
                        seed=seed, epochs=epochs, quantum_cls=quantum_cls, classical_cls=classical_cls,
                        loss_fn=loss_fn, device=device, dtype=dtype, track_best=track_best,
                        eval_every=eval_every, eval_nx=eval_nx, eval_nt=eval_nt,
                        overwrite=overwrite, verbose=verbose, **kwargs,
                    )
                    paths.append(path)
                    run_idx += 1
                    if verbose:
                        elapsed = time.time() - t_start
                        eta = elapsed / run_idx * (total_runs - run_idx)
                        print(f"  [{run_idx:4d}/{total_runs}] {param1_name}={p1:g} "
                              f"{param2_name}={p2:g} seed={seed} {model_type:10s} "
                              f"elapsed={elapsed/60:.1f}m eta={eta/60:.1f}m")
    return paths


# ─────────────────────────────────────────────────────────────────────────
# experiment 2 — depth (L) sweep at fixed PDE parameters
# ─────────────────────────────────────────────────────────────────────────

def run_L_sweep_checkpoints(
    ckpt_dir: Union[str, Path],
    pde,
    depths: Sequence[int],
    feature_size: int,
    hidden_feature: int,
    hidden_readout_quantum: int,
    seeds: Sequence[int] = (42, 666, 69),
    epochs: int = 1000,
    quantum_cls=MerlinTunableEntropy,
    classical_cls=ClassicalHeatQPINN,
    loss_fn=total_loss,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
    track_best: bool = True,
    eval_every: int = 50,
    eval_nx: int = 60,
    eval_nt: int = 60,
    overwrite: bool = False,
    verbose: bool = True,
    match_cache: Optional[dict] = None,
) -> List[Path]:
    """
    Experiment 2: fixed PDE (param1, param2), sweep architecture size L,
    classical vs. quantum, for every seed. Safe to call repeatedly with a
    bigger `depths`/`seeds` -- already-trained combinations are skipped.
    Matching is computed once per L (shared across all seeds at that L).

    Returns the list of checkpoint paths touched (trained or skipped).
    """
    depths = list(depths)
    total_runs = len(depths) * len(seeds) * 2
    run_idx = 0
    t_start = time.time()
    paths = []

    for L in depths:
        matched = resolve_matched_config(
            L, feature_size, hidden_feature, hidden_readout_quantum, pde,
            quantum_cls=quantum_cls, classical_cls=classical_cls,
            match_cache=match_cache, verbose=verbose,
        )
        for seed in seeds:
            for model_type in ("classical", "quantum"):
                if model_type == "quantum":
                    kwargs = dict(hidden_readout=hidden_readout_quantum, quantum_output_size=None)
                else:
                    kwargs = dict(hidden_readout=matched["classical_hidden_readout"],
                                  quantum_output_size=matched["classical_quantum_output_size"])
                path = run_checkpoint(
                    ckpt_dir, pde, model_type, L, feature_size, hidden_feature,
                    seed=seed, epochs=epochs, quantum_cls=quantum_cls, classical_cls=classical_cls,
                    loss_fn=loss_fn, device=device, dtype=dtype, track_best=track_best,
                    eval_every=eval_every, eval_nx=eval_nx, eval_nt=eval_nt,
                    overwrite=overwrite, verbose=verbose, **kwargs,
                )
                paths.append(path)
                run_idx += 1
                if verbose:
                    elapsed = time.time() - t_start
                    eta = elapsed / run_idx * (total_runs - run_idx)
                    print(f"  [{run_idx:4d}/{total_runs}] L={L} seed={seed} {model_type:10s} "
                          f"elapsed={elapsed/60:.1f}m eta={eta/60:.1f}m")
    return paths
