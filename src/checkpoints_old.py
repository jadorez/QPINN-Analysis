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

import functools
import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from src.pde import HeatEquation, WaveEquation
from src.models import (
    ClassicalHeatQPINN, MerlinTunableEntropy, FourierFeaturesHeatQPINN,
    ModulusFourierHeatQPINN,
)
from src.loss import total_loss
from src.diagnostics import train_and_diagnose, check_training_sanity, TrainingDiagnostics
from src.sweep import (
    match_classical_params, analytic_frequency_bound, n_photons_for_feature_size,
    measurement_dim_for_feature_size,
)


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
                 quantum_cls=MerlinTunableEntropy, classical_cls=ClassicalHeatQPINN,
                 fourier_features_cls=FourierFeaturesHeatQPINN,
                 modulus_fourier_cls=ModulusFourierHeatQPINN):
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
    if model_type in ("fourier_features", "fourier_full", "fourier_modulus"):
        if config.get("quantum_output_size") is None:
            raise ValueError(
                f"config['quantum_output_size'] is required to rebuild a "
                f"{model_type} checkpoint's model -- it should always be set "
                "by run_checkpoint/resolve_matched_ff_config."
            )
        # n_photons_per_pass = the SAME per-mode photon-number bound the
        # quantum arm's single mesh layer has (src.sweep.n_photons_for_
        # feature_size), and n_encoding_layers=L mirrors the quantum arm's
        # own re-upload depth -- feature_map outputs feature_size*L values,
        # split into L independent chunks, each given the fixed per-pass
        # encoding (see FourierFeaturesHeatQPINN's docstring for why this
        # re-upload structure -- not just a bigger harmonic count on one
        # fixed feature vector -- is what makes L comparable to the
        # quantum arm's L).
        n_photons_per_pass = n_photons_for_feature_size(config["feature_size"])
        common = dict(
            feature_size=config["feature_size"],
            hidden_feature=config["hidden_feature"],
            quantum_output_size=config["quantum_output_size"],
            hidden_readout=config["hidden_readout"],
            n_photons_per_pass=n_photons_per_pass,
            n_encoding_layers=config["L"],
            pde=pde,
        )
        if model_type == "fourier_modulus":
            # Parameter-matched control: sparse Fock comb + modulus squared,
            # which reproduces all of Omega from ~D trainable amplitudes.
            return modulus_fourier_cls(**common)
        # "fourier_features" keeps the original axis-aligned frequency set
        # (retained only for comparison -- see FIX #3 in the model docstring);
        # "fourier_full" uses the real difference set Omega.
        frequency_mode = "axis" if model_type == "fourier_features" else "full"
        return fourier_features_cls(frequency_mode=frequency_mode, **common)
    raise ValueError(
        f"model_type must be one of 'classical', 'quantum', 'fourier_features', "
        f"'fourier_full', 'fourier_modulus', got {model_type!r}"
    )


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


def resolve_matched_ff_config(
    L: int,
    feature_size: int,
    hidden_feature: int,
    hidden_readout_quantum: int,
    pde,
    quantum_cls=MerlinTunableEntropy,
    fourier_features_cls=FourierFeaturesHeatQPINN,
    modulus_fourier_cls=ModulusFourierHeatQPINN,
    model_type: str = "fourier_full",
    match_cache: Optional[dict] = None,
    verbose: bool = True,
) -> Dict:
    """
    Architecture-matched (NOT brute-force parameter-matched) config for the
    `fourier_features_cls` spectral-coverage control (see
    `FourierFeaturesHeatQPINN`'s docstring) against the quantum arm at this
    `L`.

    HISTORY: an earlier version of this function called
    `match_classical_params` to brute-force search (`ff_quantum_output_size`,
    `ff_hidden_readout`) for an EXACT total-param match against the quantum
    arm. That search reliably found a match, but only by collapsing
    `combine`'s output width as `L` grew (observed in practice: 4 -> 4 -> 3
    -> 2 across L=1..4, with `hidden_readout` exploding 13 -> 12 -> 25 -> 52
    to compensate) -- `combine`'s input width grows by a fixed amount every
    `L` step (`2 * n_photons_per_pass * feature_size`), eating its weight
    budget far faster than quantum's own total params grow with `L`, and
    shrinking `qos` was the ONLY way the search could still hit an exact
    target at high `L`. That made "matched parameter count" true but
    architecturally misleading: everything except `combine` was being
    silently re-shaped by the search, not held fixed, so the FF arm was
    never truly comparable to quantum across `L`.

    Fixed by matching ARCHITECTURE instead of total params, mirroring
    quantum's own structure exactly everywhere except the middle block:
      - `feature_map`: same `hidden_feature`, same output width
        (`feature_size * L`) as quantum's own feature map -- unchanged,
        already identical between the two arms.
      - `hidden_readout` is set to `hidden_readout_quantum` directly -- the
        SAME constant passed to the quantum arm, not searched.
      - Only the middle block itself (fixed Fourier encoding + linear
        `combine`, in place of the photonic circuit + measurement) is
        genuinely different between the two arms, and only ITS input width
        (`encoding_dim`) still grows with `L`, giving `combine`'s own
        parameter count natural L-dependent growth without any search.
    Total parameter counts are consequently NOT forced to match exactly --
    `ff_params` and `quantum_params` may differ by a nontrivial amount,
    reported transparently via `diff`. That's an accepted, honest trade-off
    for genuine architectural comparability, per explicit direction: exact
    total-param matching and a non-degenerate, unsearched middle block are
    structurally incompatible at these `L`, and architectural comparability
    is what's required for the two arms' results to mean anything side by
    side.

    NOTE on `ff_quantum_output_size` (still computed here as
    `src.sweep.measurement_dim_for_feature_size(feature_size)`, the SAME
    fixed, L-independent width quantum's own circuit measures into): this
    field is now INFORMATIONAL ONLY. `FourierFeaturesHeatQPINN` fix #2
    (see its docstring) collapsed `combine` and what used to be `readout`'s
    own first linear layer into a single `Linear(encoding_dim,
    hidden_readout)`, because two consecutive un-activated linear layers
    are always mathematically equivalent to one, just harder to optimize
    -- so `quantum_output_size` is accepted by the model for interface
    compatibility but no longer sizes any real layer. It's kept in the
    returned dict (and threaded through the checkpoint filename/config
    schema, same as the classical arm) purely so nothing downstream needs
    to change, not because it still means anything architecturally.

    Returns a dict with keys: quantum_hidden_readout, quantum_params,
    ff_quantum_output_size, ff_hidden_readout, ff_params, n_freqs,
    analytic_bound. (Field names deliberately mirror `resolve_matched_config`'s
    "classical_*" keys as "ff_*" so the two are easy to use side by side.)
    """
    key = (int(L), int(feature_size), int(hidden_feature), int(hidden_readout_quantum),
           quantum_cls.__name__, fourier_features_cls.__name__, str(model_type), "ff")
    if match_cache is not None and key in match_cache:
        return match_cache[key]

    probe = quantum_cls(
        feature_size=feature_size, hidden_feature=hidden_feature,
        hidden_readout=hidden_readout_quantum, n_encoding_layers=L, pde=pde,
    )
    target_params = _param_count(probe)
    n_photons_per_pass = n_photons_for_feature_size(feature_size)

    # No search: combine's output width and the readout width are BOTH
    # held fixed, mirroring quantum's own fixed (L-independent) bottleneck
    # and its own fixed hidden_readout_quantum constant. Only combine's
    # INPUT width (encoding_dim, inside fourier_features_cls itself) still
    # grows with L, exactly like quantum's circuit growing with L -- that's
    # the one thing this control is meant to test, not an artifact of a
    # parameter-count search.
    qos = measurement_dim_for_feature_size(feature_size)
    hr_ff = int(hidden_readout_quantum)
    common = dict(
        feature_size=feature_size, hidden_feature=hidden_feature,
        quantum_output_size=qos, hidden_readout=hr_ff,
        n_photons_per_pass=n_photons_per_pass, n_encoding_layers=L, pde=pde,
    )
    # Build whichever arm this config is for, so `ff_params` reflects the
    # model that will actually be trained (the three arms differ a lot here:
    # ~1.9k params for the full-Omega linear encoding versus ~300-400 for the
    # modulus-squared one -- see each model's docstring).
    if model_type == "fourier_modulus":
        ff_model = modulus_fourier_cls(**common)
    else:
        ff_model = fourier_features_cls(
            frequency_mode="axis" if model_type == "fourier_features" else "full",
            **common,
        )
    achieved = _param_count(ff_model)

    n_freqs = L * n_photons_per_pass  # informational only: nominal max single-pass-equivalent bound
    diff = achieved - target_params
    result = {
        "quantum_hidden_readout": int(hidden_readout_quantum),
        "quantum_params": int(target_params),
        "ff_quantum_output_size": int(qos),
        "ff_hidden_readout": int(hr_ff),
        "ff_params": int(achieved),
        "n_freqs": int(n_freqs),
        "analytic_bound": int(analytic_frequency_bound(feature_size, L)),
    }
    if verbose:
        print(f"[checkpoints] L={L}: quantum_params={target_params}  "
              f"bound=±{result['analytic_bound']} (n_freqs={n_freqs})  ff (architecture-matched) -> "
              f"quantum_output_size={qos} hidden_readout={hr_ff} params={achieved} "
              f"(diff={diff:+d})")
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
    model_type: str,                 # "classical" | "quantum" | "fourier_features"
    L: int,
    feature_size: int,
    hidden_feature: int,
    hidden_readout: int,             # ACTUAL hidden_readout for this model_type
    quantum_output_size: Optional[int] = None,   # required if model_type in ("classical", "fourier_features")
    seed: int = 42,
    epochs: int = 1000,
    quantum_cls=MerlinTunableEntropy,
    classical_cls=ClassicalHeatQPINN,
    fourier_features_cls=FourierFeaturesHeatQPINN,
    modulus_fourier_cls=ModulusFourierHeatQPINN,
    loss_fn=total_loss,
    log_every: Optional[int] = 50,
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
    if model_type not in ("classical", "quantum", "fourier_features",
                          "fourier_full", "fourier_modulus"):
        raise ValueError(
            f"model_type must be one of 'classical', 'quantum', 'fourier_features', "
            f"'fourier_full', 'fourier_modulus', got {model_type!r}"
        )
    if model_type != "quantum" and quantum_output_size is None:
        raise ValueError(
            f"quantum_output_size is required for model_type={model_type!r} -- pass the "
            "matched value from resolve_matched_config()['classical_quantum_output_size'] "
            "(classical) or resolve_matched_ff_config()['ff_quantum_output_size'] "
            "(the fourier_* arms)."
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
    model = build_model(model_type, config, pde, quantum_cls=quantum_cls, classical_cls=classical_cls,
                         fourier_features_cls=fourier_features_cls,
                         modulus_fourier_cls=modulus_fourier_cls)

    metrics, diag, extras = train_and_diagnose(
        model=model, pde=pde, loss_fn=loss_fn, seed=seed, epochs=epochs,
        log_every=log_every, device=device, dtype=dtype, model_type=model_type,
        diag_verbose=False, track_best=track_best, eval_every=eval_every,
        nx=eval_nx, nt=eval_nt,
    )
    U_pred, U_true, x_grid, t_grid, history = extras

    eval_epoch = int(diag.values["best_epoch"]) if "best_epoch" in diag.values else int(epochs)
    param_count = _param_count(model)
    # "fourier_features" is deliberately given the SAME bound formula as
    # "quantum" -- n_freqs is set (in build_model) to exactly L*n_photons,
    # i.e. the same analytic ceiling, by construction (see
    # FourierFeaturesHeatQPINN's docstring). That's what makes the two
    # arms' bounds directly overlay-able on the same plot.
    analytic_bound = (
        analytic_frequency_bound(feature_size, L)
        if model_type in ("quantum", "fourier_features", "fourier_full", "fourier_modulus")
        else None
    )

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


# ─────────────────────────────────────────────────────────────────────────
# fourier-features control arm -- spectral-coverage-hypothesis ablation
#
# Adds a THIRD arm, "fourier_features" (see FourierFeaturesHeatQPINN), to
# the SAME checkpoint directories experiments 1 and 2 already write to.
# Deliberately does not retrain "classical"/"quantum" -- run
# run_param_sweep_checkpoints / run_L_sweep_checkpoints first (or already
# have those checkpoints on disk); these two functions only add the
# control arm alongside them, with the usual skip-if-exists behaviour.
# ─────────────────────────────────────────────────────────────────────────

def run_param_sweep_ff_checkpoints(
    ckpt_dir: Union[str, Path],
    pde_type: str,
    param1_name: str,
    param1_values: Sequence[float],
    param2_name: str,
    param2_values: Sequence[float],
    L: int,
    feature_size: int,
    hidden_feature: int,
    hidden_readout_quantum: int,
    seeds: Sequence[int] = (42, 666, 69),
    epochs: int = 1000,
    model_type: str = "fourier_full",
    quantum_cls=MerlinTunableEntropy,
    fourier_features_cls=FourierFeaturesHeatQPINN,
    modulus_fourier_cls=ModulusFourierHeatQPINN,
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
    Fourier-features control for Experiment 1 (the (f,alpha)/(f,c) sweep at
    fixed L): trains only the "fourier_features" arm, over the SAME grid,
    into the SAME `ckpt_dir` experiment 1 already used -- so
    `load_checkpoints(ckpt_dir, ...)` afterward returns all three arms
    together. Config (fixed per-pass frequency reach, combine/readout width
    ARCHITECTURE-matched -- not brute-force parameter-matched -- to the
    quantum arm, see `resolve_matched_ff_config`'s docstring) is resolved
    ONCE up front, same pattern as `run_param_sweep_checkpoints`.

    Returns the list of checkpoint paths touched (trained or skipped).
    """
    param1_values = list(param1_values)
    param2_values = list(param2_values)
    probe_pde = build_pde(pde_type, {param1_name: param1_values[0], param2_name: param2_values[0]})
    matched = resolve_matched_ff_config(
        L, feature_size, hidden_feature, hidden_readout_quantum, probe_pde,
        quantum_cls=quantum_cls, fourier_features_cls=fourier_features_cls,
        modulus_fourier_cls=modulus_fourier_cls, model_type=model_type,
        match_cache=match_cache, verbose=verbose,
    )

    total_runs = len(param1_values) * len(param2_values) * len(seeds)
    run_idx = 0
    t_start = time.time()
    paths = []

    for p1 in param1_values:
        for p2 in param2_values:
            pde = build_pde(pde_type, {param1_name: p1, param2_name: p2})
            for seed in seeds:
                path = run_checkpoint(
                    ckpt_dir, pde, model_type, L, feature_size, hidden_feature,
                    hidden_readout=matched["ff_hidden_readout"],
                    quantum_output_size=matched["ff_quantum_output_size"],
                    seed=seed, epochs=epochs, quantum_cls=quantum_cls,
                    fourier_features_cls=fourier_features_cls,
                    modulus_fourier_cls=modulus_fourier_cls,
                    loss_fn=loss_fn, device=device, dtype=dtype, track_best=track_best,
                    eval_every=eval_every, eval_nx=eval_nx, eval_nt=eval_nt,
                    overwrite=overwrite, verbose=verbose,
                )
                paths.append(path)
                run_idx += 1
                if verbose:
                    elapsed = time.time() - t_start
                    eta = elapsed / run_idx * (total_runs - run_idx)
                    print(f"  [{run_idx:4d}/{total_runs}] {param1_name}={p1:g} "
                          f"{param2_name}={p2:g} seed={seed} fourier_features "
                          f"elapsed={elapsed/60:.1f}m eta={eta/60:.1f}m")
    return paths


def run_L_sweep_ff_checkpoints(
    ckpt_dir: Union[str, Path],
    pde,
    depths: Sequence[int],
    feature_size: int,
    hidden_feature: int,
    hidden_readout_quantum: int,
    seeds: Sequence[int] = (42, 666, 69),
    epochs: int = 1000,
    model_type: str = "fourier_full",
    quantum_cls=MerlinTunableEntropy,
    fourier_features_cls=FourierFeaturesHeatQPINN,
    modulus_fourier_cls=ModulusFourierHeatQPINN,
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
    Fourier-features control for Experiment 2 (the L sweep at fixed PDE
    params): trains only the "fourier_features" arm, at every L in
    `depths`, into the SAME `ckpt_dir` experiment 2 already used. At each
    L, n_freqs = L*n_photons -- exactly the quantum arm's own analytic
    ceiling at that L -- so this directly tests whether the WIDENING
    behaviour of `plot_L_sweep` also shows up in a purely classical model
    given the same growing frequency reach, with no quantum circuit at all.
    Matching is computed once per L, same pattern as `run_L_sweep_checkpoints`.

    Returns the list of checkpoint paths touched (trained or skipped).
    """
    depths = list(depths)
    total_runs = len(depths) * len(seeds)
    run_idx = 0
    t_start = time.time()
    paths = []

    for L in depths:
        matched = resolve_matched_ff_config(
            L, feature_size, hidden_feature, hidden_readout_quantum, pde,
            quantum_cls=quantum_cls, fourier_features_cls=fourier_features_cls,
            modulus_fourier_cls=modulus_fourier_cls, model_type=model_type,
            match_cache=match_cache, verbose=verbose,
        )
        for seed in seeds:
            path = run_checkpoint(
                ckpt_dir, pde, model_type, L, feature_size, hidden_feature,
                hidden_readout=matched["ff_hidden_readout"],
                quantum_output_size=matched["ff_quantum_output_size"],
                seed=seed, epochs=epochs, quantum_cls=quantum_cls,
                fourier_features_cls=fourier_features_cls,
                modulus_fourier_cls=modulus_fourier_cls,
                loss_fn=loss_fn, device=device, dtype=dtype, track_best=track_best,
                eval_every=eval_every, eval_nx=eval_nx, eval_nt=eval_nt,
                overwrite=overwrite, verbose=verbose,
            )
            paths.append(path)
            run_idx += 1
            if verbose:
                elapsed = time.time() - t_start
                eta = elapsed / run_idx * (total_runs - run_idx)
                print(f"  [{run_idx:4d}/{total_runs}] L={L} seed={seed} fourier_features "
                      f"elapsed={elapsed/60:.1f}m eta={eta/60:.1f}m")
    return paths


# ─────────────────────────────────────────────────────────────────────────
# resuming — continue training an existing checkpoint for more epochs
# ─────────────────────────────────────────────────────────────────────────

def resume_checkpoint(
    ckpt_dir: Union[str, Path],
    source: Union[dict, str, Path],
    additional_epochs: int,
    quantum_cls=MerlinTunableEntropy,
    classical_cls=ClassicalHeatQPINN,
    fourier_features_cls=FourierFeaturesHeatQPINN,
    modulus_fourier_cls=ModulusFourierHeatQPINN,
    loss_fn=total_loss,
    log_every: Optional[int] = 50,
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
    Continue training from an already-saved checkpoint for `additional_epochs`
    more epochs, and save the result as a NEW checkpoint file -- same config
    as `source` except `epochs = source_epochs + additional_epochs` (and a
    `resumed_from_epochs` field recording where it picked up). The source
    file is never modified, so both the shorter and longer runs stay around
    to compare, and the usual skip-if-exists behaviour applies to the new
    file too (calling this twice for the same `additional_epochs` is a
    no-op the second time).

    LIGHTWEIGHT resume, not bit-exact: only the model weights are restored
    (from `source["state_dict"]` -- the best-checkpoint weights if that run
    used `track_best=True`, otherwise its final-epoch weights). The
    optimizer is reinitialised from scratch (Adam's momentum/variance
    buffers restart at zero -- a mild, usually unnoticeable transient, not a
    correctness issue) and training is reseeded from the run's own `seed`
    rather than continuing its exact prior RNG state, so the collocation
    points sampled during these extra epochs are NOT a continuation of the
    same pseudo-random sequence a single uninterrupted
    `epochs=source_epochs+additional_epochs` run would have produced.
    `sampler.py` resamples fresh random points every epoch regardless of
    where in training you are, so this doesn't change what the model can
    learn -- it only means this function isn't bit-for-bit reproducible
    against an unbroken run of the combined length.

    `track_best`/`eval_every` apply to the CONTINUATION only: if enabled,
    the first fixed-grid eval (epoch 1 of the continuation) closely
    approximates "keep the starting point" as a floor, but isn't exactly it
    -- a few gradient steps happen first, so it's possible (if unlikely) for
    the continuation to end up slightly worse than `source` if it never
    recovers past that point. Diagnostics on the new checkpoint are
    recomputed over the FULL (concatenated source + continuation) loss
    history, so trend/oscillation checks reflect the whole run, not just
    the epochs just added.

    Args:
        source: a checkpoint dict (e.g. from `load_checkpoint`/
                `load_checkpoints`) or a path to one on disk.
    Returns:
        path to the new, longer checkpoint (or the existing one, if a
        checkpoint for the resulting epoch count is already on disk and
        `overwrite=False`).
    """
    if isinstance(source, (str, Path)):
        source = load_checkpoint(source)

    cfg = dict(source["config"])
    original_epochs = cfg["epochs"]
    new_epochs = original_epochs + additional_epochs
    new_config = dict(cfg, epochs=new_epochs, resumed_from_epochs=original_epochs)

    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / checkpoint_filename(new_config)
    if path.exists() and not overwrite:
        if verbose:
            print(f"[checkpoints] skip (already exists): {path.name}")
        return path

    if log_every is None:
        log_every = additional_epochs + 1

    pde = build_pde_from_config(cfg)
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    model = build_model(cfg["model_type"], cfg, pde, quantum_cls=quantum_cls, classical_cls=classical_cls,
                         fourier_features_cls=fourier_features_cls,
                         modulus_fourier_cls=modulus_fourier_cls)
    model.load_state_dict(source["state_dict"])

    metrics, _continuation_diag, extras = train_and_diagnose(
        model=model, pde=pde, loss_fn=loss_fn, seed=cfg["seed"], epochs=additional_epochs,
        log_every=log_every, device=device, dtype=dtype, model_type=cfg["model_type"],
        diag_verbose=False, track_best=track_best, eval_every=eval_every,
        nx=eval_nx, nt=eval_nt,
    )
    U_pred, U_true, x_grid, t_grid, continuation_history = extras
    full_history = np.concatenate(
        [np.asarray(source["history"]), np.asarray(continuation_history)], axis=0,
    )

    local_eval_epoch = (
        int(_continuation_diag.values["best_epoch"])
        if "best_epoch" in _continuation_diag.values else int(additional_epochs)
    )
    eval_epoch = original_epochs + local_eval_epoch

    # recompute diagnostics over the WHOLE run (not just the epochs just
    # added), so trend/oscillation checks see the full picture.
    diag = check_training_sanity(
        model=model, pde=pde, history=full_history, U_pred=U_pred, U_true=U_true,
        x_grid=x_grid, t_grid=t_grid, seed=cfg["seed"], model_type=cfg["model_type"], verbose=False,
    )

    param_count = _param_count(model)
    analytic_bound = (
        analytic_frequency_bound(cfg["feature_size"], cfg["L"])
        if cfg["model_type"] in ("quantum", "fourier_features") else None
    )

    ckpt = {
        "config": new_config,
        "state_dict": model.state_dict(),
        "history": full_history,
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
        src_name = source.get("_path", "<in-memory checkpoint>")
        print(f"[checkpoints] resumed {src_name} ({original_epochs} -> {new_epochs} epochs) "
              f"-> saved {path.name}  rel_l2={metrics['rel_l2']:.3e}  {flag}")
    return path
