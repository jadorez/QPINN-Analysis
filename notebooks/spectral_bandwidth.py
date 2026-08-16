"""
spectral_bandwidth.py — analytic vs. effective frequency spectrum of the
MerLin quantum layer (`W1 -> angle encoding -> W2`) used in
`MerlinTunableEntropy` / `MerlinTunableSimple`, plus two follow-on
diagnostics: whether `hidden_readout` bottlenecks task performance, and
whether x and t are kept spectrally separate anywhere in the pipeline.

Contents
--------
1. Analytic bound          — `n_photons_for_feature_size`, `analytic_frequency_bound`
                              (see PROOF below for the derivation).
2. Spectrum extraction      — `build_swept_input`, `verify_input_mapping`,
                              `quantum_layer_spectrum`, `effective_bandwidth`,
                              `spectral_second_moment`, `fraction_of_ceiling_used`.
3. Spectrum plots           — `plot_spectrum_by_layers`, `plot_bandwidth_vs_layers`,
                              `plot_analytic_vs_effective_gap`.
4. Readout bottleneck sweep — `run_readout_bottleneck_sweep`,
                              `plot_bottleneck_heatmap`, `plot_bottleneck_lines`.
5. x/t mode separation       — `feature_map_jacobian`, `x_t_mixing_summary`,
                              `plot_x_t_mode_mixing`.
6. Physical spectrum vs. L   — `estimate_physical_frequency_ceiling`,
                              `plot_power_spectrum_by_layers`.

Every function in sections 1-3 calls `model.quantum(...)` directly with a
synthetic, hand-built input tensor, bypassing `feature_map` entirely.
`feature_map` is a `Tanh`-activated MLP: it mixes x and t nonlinearly on
its own and would confound any reading of the quantum layer's spectral
content — see section 5 for that question specifically.


PROOF of the analytic bound
----------------------------
Fix one encoded feature index k and consider `L = n_encoding_layers`
re-uploads of the same scalar theta into mode k (what `build_swept_input`
constructs). Between re-uploads, `LI_simple`, `MI_simple_1, ...,
MI_simple_{L-1}`, `RI_simple` are trainable *passive linear-optical*
unitaries U_0, ..., U_L acting on the full `n_modes` register (they may
freely entangle mode k with every other mode — nothing below assumes
otherwise). The state after the whole circuit is

    |psi(theta)> = U_L S(theta) U_{L-1} S(theta) ... U_1 S(theta) U_0 |psi_0>,
    S(theta) = exp(-i * theta * n_hat_k)

where n_hat_k is the photon-number operator on mode k and |psi_0> is the
fixed n_photons-photon Fock input state.

Step 1 — eigenvalues of n_hat_k. Every U_i is passive linear-optical, so
the circuit conserves total photon number: the accessible space is the
n_photons-particle Fock sector, spanned by occupation tuples s with
sum(s) = n_photons, s_j >= 0 integers. On this sector n_hat_k's eigenvalues
are exactly the integers s_k can take: {0, 1, ..., n_photons} — every
value in between is reachable by some Fock state (e.g.
(n_photons - m, m, 0, ..., 0)), so this set has no gaps. Call it
Lambda = {0, ..., n_photons}, with eigenprojectors Pi_m onto {n_hat_k = m}.

Step 2 — expand each gate. exp(-i*theta*n_hat_k) = sum_{m in Lambda}
exp(-i*theta*m) Pi_m. Substituting all L copies and multiplying out:

    |psi(theta)> = sum_{m_1,...,m_L in Lambda} exp(-i*theta*(m_1+...+m_L))
                    * U_L Pi_{m_L} U_{L-1} Pi_{m_{L-1}} ... U_1 Pi_{m_1} U_0 |psi_0>

Grouping by omega = m_1 + ... + m_L:

    |psi(theta)> = sum_{omega=0}^{L*n_photons} exp(-i*theta*omega) |v_omega>

omega ranges over the FULL contiguous interval {0, ..., L*n_photons}: each
m_i independently ranges over the contiguous block {0,...,n_photons}, and
the Minkowski sum of L contiguous integer intervals is again contiguous
(you can always move any single m_i up or down by 1 to shift omega by
exactly 1 without leaving Lambda, so no integer in [0, L*n_photons] is
skipped) — this is the "L-fold sumset" step.

Amplitude bound (one-sided). Any Fock-basis component psi_f(theta) =
<f|psi(theta)> = sum_omega exp(-i*theta*omega) <f|v_omega> is a
trigonometric polynomial in theta with frequency support restricted to
{0, ..., L*n_photons} — a ONE-SIDED comb (or its mirror image
{-L*n_photons, ..., 0}, depending on the encoding gate's sign convention,
which this module doesn't assume). This is `model.quantum(z)`'s raw
complex output. Its Fourier *phases* depend on the arbitrary global-phase
convention of |psi_0> and the U_i's; its frequency *support* does not.

Probability / population bound (symmetric — what actually reaches the
readout). `MerlinTunableSimple` / `MerlinTunableEntropy` feed
`q_probs = |q_amplitudes|**2` into `readout`, not the raw amplitude. For
output f,

    P_f(theta) = |psi_f(theta)|^2 = psi_f(theta) * conj(psi_f(theta))
               = sum_{omega, omega'=0}^{L*n_photons}
                     exp(-i*theta*(omega - omega')) <f|v_omega><v_omega'|f>

Collecting by kappa = omega - omega', which ranges over the full
contiguous interval {-L*n_photons, ..., +L*n_photons} (same sumset
argument, now on a difference), gives

    P_f(theta) = sum_{kappa=-L*n_photons}^{L*n_photons} exp(-i*theta*kappa) c_{f,kappa}

So the probability vector — and everything the readout MLP's *first*
(linear) layer builds from it, before its own Tanh — is a trigonometric
polynomial in theta with frequency support exactly
{-L*n_photons, ..., +L*n_photons}: this is `analytic_frequency_bound`
below. Unlike the raw amplitude, it's invariant to the encoding gate's
global-phase convention (it's built from |.|^2), which is the second
reason `quantum_layer_spectrum(..., quantity="probability")` — the
default — is the more meaningful of the two "quantum-layer-only" spectra.
`quantity="amplitude"` measures the one-sided version for reference/debugging.

Why this is a ceiling, not a guarantee. Nothing above required U_0,...,U_L
to be any *specific* unitaries — the derivation holds for literally any
choice, which is exactly why it's an upper bound: it says these are the
only frequencies that CAN appear, for any setting of
`LI_simple`/`MI_simple_i`/`RI_simple`'s trainable parameters, not that
every frequency in range does appear for a generic or trained setting.
Whether the coefficients at the band edges are non-negligible depends on
what those trainable unitaries actually do — that's an empirical question,
answered in Sections 4/5 below.

References: Schuld, Sweke & Meyer, "The effect of data encoding on the
expressive power of variational quantum machine learning models",
Phys. Rev. A 103, 032430 (2021); Schuld & Killoran, "Quantum machine
learning in feature Hilbert spaces".
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────
# 1. analytic bound
# ─────────────────────────────────────────────────────────────────────────

def n_photons_for_feature_size(feature_size: int) -> int:
    """
    Number of photons in the fixed input Fock state, as constructed in
    `MerlinTunableSimple.__init__` / `MerlinTunableEntropy.__init__`:
    `n_modes = feature_size + 1` modes, one photon placed on every
    even-indexed mode -> n_photons = ceil(n_modes / 2). Matches
    `src.utils.hilbert_space_dimension`.
    """
    n_modes = feature_size + 1
    return int(math.ceil(n_modes / 2))


def analytic_frequency_bound(feature_size: int, n_encoding_layers: int) -> int:
    """
    Symmetric ceiling |kappa| <= L * n_photons on the accessible integer
    frequency for ONE encoded feature's *probability/population* signal
    (`quantity="probability"`, the default in `quantum_layer_spectrum`),
    after `n_encoding_layers` re-uploads of that feature onto the same
    mode. See the module-level PROOF. For the raw complex amplitude
    (`quantity="amplitude"`), the same integer is still a valid magnitude
    ceiling, but the true support is one-sided, not symmetric — see the
    proof's "Amplitude bound" paragraph.
    """
    if n_encoding_layers < 1:
        raise ValueError(f"n_encoding_layers must be >= 1, got {n_encoding_layers}")
    return n_encoding_layers * n_photons_for_feature_size(feature_size)


def analytic_spectrum_set(feature_size: int, n_encoding_layers: int) -> np.ndarray:
    """All integer frequencies the probability spectrum could in principle realize."""
    bound = analytic_frequency_bound(feature_size, n_encoding_layers)
    return np.arange(-bound, bound + 1)


# ─────────────────────────────────────────────────────────────────────────
# 2. spectrum extraction — talks to model.quantum directly
# ─────────────────────────────────────────────────────────────────────────

def build_swept_input(
    feature_size: int,
    n_encoding_layers: int,
    feature_index: int,
    theta: np.ndarray,
    other_value: Optional[np.ndarray] = None,
    seed: int = 0,
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, np.ndarray]:
    """
    Build the (n_points, feature_size * n_encoding_layers) tensor fed
    straight to `model.quantum`, sweeping ONE physical feature
    (`feature_index`) while re-uploading the SAME swept value into every
    one of its `n_encoding_layers` copies — i.e. genuine data
    re-uploading of a single scalar, which is what the proof above and
    `analytic_frequency_bound` describe.

    Every other input slot is held fixed at `other_value` (a length
    `feature_size * n_encoding_layers` vector). If `other_value` is None,
    a fixed pseudo-random vector in [-pi, pi) is drawn from `seed` — off
    the high-symmetry point 0, where several gates would coincide.

    Feature -> mode/slot ordering assumption (see `verify_input_mapping`
    to check this empirically against your build of MerLin/Perceval):
    `add_angle_encoding` is called once per encoding layer, in order,
    each consuming the next `feature_size`-wide chunk of the input
    vector, so layer `i`'s copy of `feature_index` sits at flat index
    `i * feature_size + feature_index`.
    """
    total_input_size = feature_size * n_encoding_layers
    n_points = len(theta)

    if other_value is None:
        rng = np.random.default_rng(seed)
        other_value = rng.uniform(-math.pi, math.pi, size=total_input_size)
    else:
        other_value = np.broadcast_to(np.asarray(other_value, dtype=np.float64), (total_input_size,)).copy()

    z = np.tile(other_value, (n_points, 1))
    for layer in range(n_encoding_layers):
        idx = layer * feature_size + feature_index
        z[:, idx] = theta

    z_t = torch.as_tensor(z, dtype=dtype, device=device)
    return z_t, other_value


def verify_input_mapping(model, feature_size: int, n_encoding_layers: int) -> Dict[Tuple[int, int], float]:
    """
    Empirical sanity check for the slot-index assumption in
    `build_swept_input`: for every (layer, feature_index) pair, perturb
    only that flat input slot and confirm the quantum output actually
    changes. Returns {(layer, feature_index): ||q(e_slot) - q(0)||}; any
    entry close to 0 means that slot is dead (either the mapping
    assumption above is wrong for your MerLin version, or that
    feature/layer combination truly isn't wired to the circuit) and the
    bandwidth numbers below should not be trusted until that's resolved.
    """
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    total_input_size = feature_size * n_encoding_layers

    model.eval()
    with torch.no_grad():
        z0 = torch.zeros((1, total_input_size), dtype=dtype, device=device)
        q0 = model.quantum(z0)

        deltas = {}
        for layer in range(n_encoding_layers):
            for k in range(feature_size):
                idx = layer * feature_size + k
                z1 = z0.clone()
                z1[0, idx] = 1.0
                q1 = model.quantum(z1)
                deltas[(layer, k)] = float(torch.norm(q1 - q0).item())
    return deltas


@dataclass
class SpectrumResult:
    k: np.ndarray                # integer frequency bins, ascending
    energy: np.ndarray            # aggregated energy per bin, same order as k, normalised to peak = 1
    signal_fft: np.ndarray        # raw (n_points, n_fock_states) complex FFT, same order as k
    feature_size: int
    n_encoding_layers: int
    feature_index: int
    analytic_bound: int
    other_value: np.ndarray
    quantity: str                 # "probability" (symmetric bound) or "amplitude" (one-sided)
    trained: bool = False


def _next_pow_two(n: int) -> int:
    return 1 << max(1, (n - 1)).bit_length()


def quantum_layer_spectrum(
    model,
    feature_size: int,
    n_encoding_layers: int,
    feature_index: int = 0,
    n_points: Optional[int] = None,
    other_value: Optional[np.ndarray] = None,
    seed: int = 0,
    oversample: int = 8,
    quantity: str = "probability",
    trained: bool = False,
) -> SpectrumResult:
    """
    Sweep `feature_index` through model.quantum (bypassing feature_map),
    re-uploaded identically across all `n_encoding_layers` copies (see
    `build_swept_input`), and FFT the resulting signal.

    `quantity`:
      "probability" (default) — FFT of `|amplitude|**2`, i.e. what
          `MerlinTunableSimple`/`MerlinTunableEntropy` actually feed into
          `readout`. Symmetric support in [-bound, bound], invariant to
          the encoding gate's global-phase convention. Use this one.
      "amplitude" — FFT of the raw complex output of `model.quantum`.
          One-sided support (magnitude still <= bound, but concentrated
          on one side of k=0 — see the module PROOF). Provided for
          debugging/comparison against `SpectralDiagnostics.ipynb`'s
          amplitude-based sweeps, not because it's what the model uses.

    `n_points` defaults to `oversample * (2*bound+1)` rounded to a power
    of two, comfortably above the Nyquist rate needed to resolve
    frequencies up to `analytic_frequency_bound` without aliasing — if
    the bound is a genuine ceiling, energy at |k| > bound should be
    numerically zero, and that's only a meaningful check if the sweep
    isn't itself band-limiting the measurement.
    """
    if quantity not in ("probability", "amplitude"):
        raise ValueError(f'quantity must be "probability" or "amplitude", got {quantity!r}')

    bound = analytic_frequency_bound(feature_size, n_encoding_layers)
    if n_points is None:
        n_points = max(256, _next_pow_two(oversample * (2 * bound + 1)))

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    theta = np.linspace(-math.pi, math.pi, n_points, endpoint=False)
    z, other_value = build_swept_input(
        feature_size, n_encoding_layers, feature_index, theta,
        other_value=other_value, seed=seed, dtype=dtype, device=device,
    )

    model.eval()
    with torch.no_grad():
        amps = model.quantum(z)  # (n_points, n_fock_states) complex
    amps_np = amps.detach().cpu().numpy()

    signal = (np.abs(amps_np) ** 2).astype(complex) if quantity == "probability" else amps_np

    fft = np.fft.fft(signal, axis=0) / n_points
    k = np.fft.fftfreq(n_points) * n_points
    k = np.round(k).astype(int)

    order = np.argsort(k)
    k = k[order]
    fft = fft[order]

    energy = np.sum(np.abs(fft) ** 2, axis=1)
    peak = energy.max()
    if peak > 0:
        energy = energy / peak

    return SpectrumResult(
        k=k, energy=energy, signal_fft=fft,
        feature_size=feature_size, n_encoding_layers=n_encoding_layers,
        feature_index=feature_index, analytic_bound=bound,
        other_value=other_value, quantity=quantity, trained=trained,
    )


def effective_bandwidth(result: SpectrumResult, threshold: float = 1e-3) -> float:
    """
    Largest |k| whose normalised energy clears `threshold` * peak. This is
    a hard-cutoff summary; see `spectral_second_moment` for an
    energy-weighted alternative that doesn't depend on picking a cutoff.
    """
    mask = result.energy >= threshold
    if not mask.any():
        return 0.0
    return float(np.abs(result.k[mask]).max())


def spectral_second_moment(result: SpectrumResult) -> float:
    """
    Energy-weighted RMS frequency: sqrt(sum(w_k * k^2)), w_k = energy_k /
    sum(energy). Same summary as `omega_eff` elsewhere in this repo's
    spectral diagnostics — how much high-frequency content this feature
    carries, without a hard threshold.
    """
    total = result.energy.sum()
    if total <= 0:
        return 0.0
    weights = result.energy / total
    return float(np.sqrt(np.sum(weights * result.k.astype(np.float64) ** 2)))


def fraction_of_ceiling_used(result: SpectrumResult, threshold: float = 1e-3) -> float:
    """effective_bandwidth / analytic_bound, clipped to [0, 1]."""
    if result.analytic_bound == 0:
        return 0.0
    return float(min(1.0, effective_bandwidth(result, threshold) / result.analytic_bound))


# ─────────────────────────────────────────────────────────────────────────
# 3. spectrum plots
# ─────────────────────────────────────────────────────────────────────────

def _save(fig: plt.Figure, save_path):
    if save_path:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved: {path}")


def plot_spectrum_by_layers(
    results: Dict[int, SpectrumResult],
    threshold: float = 1e-3,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
) -> plt.Figure:
    """
    One panel per `n_encoding_layers` value in `results`, energy vs.
    integer frequency k, with a dashed vertical pair marking
    +/- analytic_frequency_bound for that L. Shared x-axis across panels
    (set by the largest L present) so growth is visually comparable.
    """
    layers = sorted(results.keys())
    max_bound = max(r.analytic_bound for r in results.values())
    quantity = next(iter(results.values())).quantity

    fig, axes = plt.subplots(1, len(layers), figsize=(4 * len(layers), 4), sharey=True)
    if len(layers) == 1:
        axes = [axes]

    for ax, L in zip(axes, layers):
        r = results[L]
        ax.stem(r.k, r.energy, basefmt=" ")
        ax.axvline(r.analytic_bound, color="crimson", ls="--", lw=1.5, label="analytic bound")
        ax.axvline(-r.analytic_bound, color="crimson", ls="--", lw=1.5)
        ax.axhline(threshold, color="gray", ls=":", lw=1, label=f"threshold={threshold:g}")
        ax.set_xlim(-max_bound - 1, max_bound + 1)
        ax.set_title(f"L = {L}  (bound = ±{r.analytic_bound})" + ("  [trained]" if r.trained else "  [init]"))
        ax.set_xlabel("frequency k")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel(f"normalised energy ({quantity})")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    _save(fig, save_path)
    if show:
        plt.show()
    return fig


def plot_bandwidth_vs_layers(
    layers_list: Sequence[int],
    series: Dict[str, Sequence[float]],
    analytic_bounds: Sequence[float],
    ylabel: str = "bandwidth (|k|)",
    title: str = "Effective vs. analytically allowed bandwidth",
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Core figure: effective bandwidth/second-moment (one or more named
    series, e.g. "init" and "trained") vs. n_encoding_layers, against the
    analytic ceiling L * n_photons. If a series tracks the ceiling
    linearly, more layers are genuinely buying spectral bandwidth; if it
    plateaus below it, the trainable-unitary family or the readout is the
    bottleneck, not the encoding.
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(layers_list, analytic_bounds, "k--", lw=2, marker="d", label="analytic bound (L·n_photons)")
    for name, values in series.items():
        ax.plot(layers_list, values, marker="o", lw=2, label=name)
    ax.set_xlabel("n_encoding_layers (L)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(list(layers_list))
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    _save(fig, save_path)
    if show:
        plt.show()
    return fig


def plot_analytic_vs_effective_gap(
    layers_list: Sequence[int],
    series: Dict[str, Sequence[float]],
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Fraction of the analytic ceiling actually used (effective_bandwidth /
    analytic_bound) per L, one bar group per named series (e.g. "init"
    vs "trained"). 1.0 = the circuit saturates the bound for that L;
    well below 1.0 = most of the analytically allowed spectrum is unused
    by the learned function.
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    n_series = len(series)
    width = 0.8 / max(1, n_series)
    x = np.arange(len(layers_list))

    for i, (name, values) in enumerate(series.items()):
        ax.bar(x + i * width, values, width=width, label=name)

    ax.axhline(1.0, color="crimson", ls="--", lw=1, label="ceiling saturated")
    ax.set_xticks(x + width * (n_series - 1) / 2)
    ax.set_xticklabels([str(L) for L in layers_list])
    ax.set_xlabel("n_encoding_layers (L)")
    ax.set_ylabel("fraction of analytic ceiling used")
    ax.set_title("How much of the allowed spectrum does the circuit use?")
    ax.set_ylim(0, max(1.15, ax.get_ylim()[1]))
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend()
    fig.tight_layout()
    _save(fig, save_path)
    if show:
        plt.show()
    return fig


# ─────────────────────────────────────────────────────────────────────────
# 4. readout bottleneck sweep
# ─────────────────────────────────────────────────────────────────────────

@dataclass
class BottleneckSweepResult:
    layers_list: List[int]
    hidden_readout_list: List[int]
    rel_l2: Dict[Tuple[int, int], List[float]]  # (L, hidden_readout) -> rel_l2 per seed


def run_readout_bottleneck_sweep(
    model_cls,
    feature_size: int,
    pde,
    layers_list: Sequence[int],
    hidden_readout_list: Sequence[int],
    train_fn: Callable,
    evaluate_fn: Callable,
    epochs: int = 1000,
    n_seeds: int = 1,
    seed_base: int = 0,
    device=None,
    dtype: torch.dtype = torch.float32,
    log_every: Optional[int] = None,
) -> BottleneckSweepResult:
    """
    Joint sweep over (n_encoding_layers, hidden_readout), holding
    feature_size and the PDE fixed. Tests whether `hidden_readout` — the
    width of the classical MLP between `q_probs` and (u, ux_hat) — is
    capping task performance, independent of how much analytic/effective
    spectral bandwidth the quantum layer itself has available at a given
    L (Sections 2-3 above).

    Read it against Sections 2-3's results, not in isolation:
      - If rel_l2 is flat across `hidden_readout_list` even at the L with
        the widest *trained* effective bandwidth, hidden_readout is NOT
        the bottleneck — look at whether the effective bandwidth itself
        is flat vs. L (encoding/entangling-layer bottleneck) or whether
        something upstream/downstream of the quantum layer (loss
        weighting, optimizer, sampler) is capping performance instead.
      - If rel_l2 improves with hidden_readout specifically at large L
        (where there's more allowed spectrum to work with) but not at
        L=1, the readout genuinely couldn't exploit the extra bandwidth
        at its old width — that's the bottleneck signature.

    `train_fn` / `evaluate_fn` are passed in explicitly (e.g.
    `src.train.train`, `src.eval.evaluate`) rather than imported here, so
    this module has no hard dependency on where the training pipeline
    lives.

    Returns rel_l2 for every (L, hidden_readout, seed) combination; plot
    with `plot_bottleneck_heatmap` / `plot_bottleneck_lines`.
    """
    results: Dict[Tuple[int, int], List[float]] = {}
    for L in layers_list:
        for hr in hidden_readout_list:
            rel_l2s = []
            for seed_offset in range(n_seeds):
                seed = seed_base + seed_offset
                torch.manual_seed(seed)
                model = model_cls(
                    feature_size=feature_size, hidden_readout=hr,
                    n_encoding_layers=L, pde=pde,
                ).to(device)
                train_fn(
                    model=model, pde=pde, epochs=epochs, seed=seed,
                    log_every=log_every or epochs, device=device, dtype=dtype,
                )
                metrics, *_ = evaluate_fn(model=model, pde=pde, device=device, dtype=dtype)
                rel_l2s.append(metrics["rel_l2"])
                print(f"L={L:2d}  hidden_readout={hr:3d}  seed={seed}  rel_l2={metrics['rel_l2']:.4e}")
            results[(L, hr)] = rel_l2s

    return BottleneckSweepResult(
        layers_list=list(layers_list), hidden_readout_list=list(hidden_readout_list), rel_l2=results,
    )


def plot_bottleneck_heatmap(
    result: BottleneckSweepResult,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
) -> plt.Figure:
    layers_list, hr_list = result.layers_list, result.hidden_readout_list
    grid = np.array([[np.mean(result.rel_l2[(L, hr)]) for hr in hr_list] for L in layers_list])

    fig, ax = plt.subplots(figsize=(1.3 * len(hr_list) + 2, 1.0 * len(layers_list) + 2))
    im = ax.imshow(np.log10(grid), aspect="auto", cmap="viridis_r")
    ax.set_xticks(range(len(hr_list)))
    ax.set_xticklabels(hr_list)
    ax.set_yticks(range(len(layers_list)))
    ax.set_yticklabels(layers_list)
    ax.set_xlabel("hidden_readout")
    ax.set_ylabel("n_encoding_layers (L)")
    ax.set_title("mean rel_l2 (log10 color scale) over (L, hidden_readout)")
    for i in range(len(layers_list)):
        for j in range(len(hr_list)):
            ax.text(j, i, f"{grid[i, j]:.2e}", ha="center", va="center", color="white", fontsize=8)
    fig.colorbar(im, ax=ax, label="log10(mean rel_l2)")
    fig.tight_layout()
    _save(fig, save_path)
    if show:
        plt.show()
    return fig


def plot_bottleneck_lines(
    result: BottleneckSweepResult,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 5))
    for L in result.layers_list:
        means = [np.mean(result.rel_l2[(L, hr)]) for hr in result.hidden_readout_list]
        stds = [np.std(result.rel_l2[(L, hr)]) for hr in result.hidden_readout_list]
        ax.errorbar(result.hidden_readout_list, means, yerr=stds, marker="o", lw=2, capsize=3, label=f"L={L}")
    ax.set_xlabel("hidden_readout")
    ax.set_yscale("log")
    ax.set_ylabel("rel_l2 (log)")
    ax.set_title("rel_l2 vs. hidden_readout, by n_encoding_layers")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    _save(fig, save_path)
    if show:
        plt.show()
    return fig


# ─────────────────────────────────────────────────────────────────────────
# 5. x/t mode separation
# ─────────────────────────────────────────────────────────────────────────
#
# Short answer: no, x and t are not kept spectrally separate anywhere in
# this pipeline. `feature_map` is a single `Linear(2, H) -> Tanh ->
# Linear(H, total_input_size)` MLP: the very first layer already maps
# (x, t) through a SHARED hidden layer, so every one of its
# `total_input_size` outputs is generically a nonlinear function of BOTH
# x and t — there is no output slot that depends on x alone or t alone.
# That mixed scalar is then what gets angle-encoded onto a given mode; the
# quantum layer (and the analytic bound above) only ever sees that mixed
# scalar and has no notion of "x-frequency" vs. "t-frequency" to keep
# apart. The functions below make this concrete rather than asserted.

def feature_map_jacobian(model, xt: torch.Tensor) -> np.ndarray:
    """
    d(feature_map_k)/d(x, t) at each row of `xt` (shape (batch, 2)), via
    autograd. Returns an array of shape (batch, total_input_size, 2): for
    each of the quantum layer's `feature_size * n_encoding_layers` scalar
    inputs and each sample point, the partial derivative w.r.t. x
    (column 0) and t (column 1).

    If input slot j were purely a function of x (or purely of t), column
    1 (or column 0) would be ~0 across the whole domain. In this
    architecture that generically does not happen — see
    `plot_x_t_mode_mixing` / `x_t_mixing_summary`.
    """
    device = next(model.parameters()).device
    xt = xt.detach().to(device).requires_grad_(True)
    z = model.feature_map(xt)  # (batch, total_input_size)
    batch, total_input_size = z.shape
    jac = torch.zeros(batch, total_input_size, 2, device=device)
    for j in range(total_input_size):
        grad_j = torch.autograd.grad(
            z[:, j], xt, grad_outputs=torch.ones(batch, device=device),
            retain_graph=True, create_graph=False,
        )[0]
        jac[:, j, :] = grad_j
    return jac.detach().cpu().numpy()


def x_t_mixing_summary(jacobian: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Per-input-slot summary of `feature_map_jacobian`'s output: mean
    |d/dx|, mean |d/dt| (averaged over the sampled domain points), and
    their ratio. A slot with ratio >> 1 or << 1 leans x- or t-dominated
    *on average*, but `plot_x_t_mode_mixing` should be checked too — the
    Tanh nonlinearity means this ratio is not constant across the domain.
    """
    mean_abs_dx = np.abs(jacobian[..., 0]).mean(axis=0)
    mean_abs_dt = np.abs(jacobian[..., 1]).mean(axis=0)
    ratio = mean_abs_dx / (mean_abs_dt + 1e-12)
    return {"mean_abs_dx": mean_abs_dx, "mean_abs_dt": mean_abs_dt, "dx_over_dt_ratio": ratio}


def plot_x_t_mode_mixing(
    jacobian: np.ndarray,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
) -> plt.Figure:
    """
    For each of the quantum layer's scalar input slots, scatter |d/dx|
    vs. |d/dt| across the sampled domain points (one color per slot). A
    slot that were purely x- or purely t-dependent would sit exactly on
    one axis; expect every slot's cloud to sit off both axes (non-zero
    sensitivity to both) and to be spread out, not a single point —
    because of `feature_map`'s Tanh, the degree of x/t mixing is itself
    position-dependent, not just its presence.
    """
    batch, total_input_size, _ = jacobian.shape
    fig, ax = plt.subplots(figsize=(6, 6))
    for j in range(total_input_size):
        dx = np.abs(jacobian[:, j, 0])
        dt = np.abs(jacobian[:, j, 1])
        ax.scatter(dx, dt, s=10, alpha=0.5, label=f"slot {j}")
    lim = max(np.abs(jacobian[..., 0]).max(), np.abs(jacobian[..., 1]).max()) * 1.05
    ax.plot([0, lim], [0, lim], "k--", lw=1, alpha=0.5, label="|d/dx| = |d/dt|")
    ax.set_xlabel(r"|$\partial$(feature_map slot)/$\partial$x|")
    ax.set_ylabel(r"|$\partial$(feature_map slot)/$\partial$t|")
    ax.set_title("x/t sensitivity of each quantum-layer input slot")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    _save(fig, save_path)
    if show:
        plt.show()
    return fig


# ─────────────────────────────────────────────────────────────────────────
# 6. physical solution spectrum vs. L, against the analytical solution
# ─────────────────────────────────────────────────────────────────────────
#
# Sections 2-5 measured the quantum layer's own theta-space spectrum,
# bypassing feature_map. This section is the complementary, physical-space
# question: for each trained L, how does the model's actual output
# u(x, t)'s Fourier spectrum compare to the analytical PDE solution's
# spectrum (pde.exact_solution)?

def estimate_physical_frequency_ceiling(
    model,
    xt_sample: torch.Tensor,
    feature_size: int,
    n_encoding_layers: int,
) -> float:
    """
    Rough, NON-RIGOROUS estimate of the largest physical spatial
    frequency (in the same "cycles over the domain" units as the x-axis
    of `plot_power_spectrum_by_layers` / `plotter.plot_mode_l2_errors`)
    the trained circuit's theta-space bound could plausibly reach —
    pushes `analytic_frequency_bound(feature_size, n_encoding_layers)`
    through the LARGEST observed `|d(feature_map slot)/dx|` over
    `xt_sample` (via `feature_map_jacobian`), divided by 2*pi.

    This is NOT a bound in the sense `analytic_frequency_bound` is: it
    collapses every Fock-state coefficient and every one of the L*feature_size
    encoded slots into a single number via its single largest Jacobian
    entry, and `feature_map`'s Tanh makes the true push-forward
    position-dependent (a chirp), not this one constant slope — see
    `physical_bandwidth_note`. Treat the result as an order-of-magnitude
    sanity-check reference line on `plot_power_spectrum_by_layers`, not a
    proven ceiling on the physical spectrum.
    """
    jac = feature_map_jacobian(model, xt_sample)  # (batch, total_input_size, 2)
    max_dzdx = float(np.abs(jac[..., 0]).max())
    bound = analytic_frequency_bound(feature_size, n_encoding_layers)
    return bound * max_dzdx / (2 * math.pi)


def plot_power_spectrum_by_layers(
    U_true: Union[torch.Tensor, np.ndarray],
    U_pred_by_layers: Dict[int, Union[torch.Tensor, np.ndarray]],
    layers_list: Sequence[int],
    physical_ceiling_by_L: Optional[Dict[int, float]] = None,
    exclude_0_mode: bool = True,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Semilogy power spectrum (FFT over x, averaged over t) of the
    analytical PDE solution vs. every trained L's prediction, overlaid on
    one plot — the physical (x, t)-space complement of Section 5's
    theta-space quantum-layer spectrum plots. x-axis is the integer FFT
    mode index k (cycles over the sampled domain), matching
    `plotter.plot_mode_l2_errors`'s convention — NOT
    `plotter.plot_power_spectrum`'s fractional cycles-per-sample axis, so
    don't compare the two x-axes directly.

    `physical_ceiling_by_L`, if given (see
    `estimate_physical_frequency_ceiling`), overlays one dotted vertical
    line per L, colour-matched to that L's curve — a rough
    order-of-magnitude reference, not a proven bound.
    """
    def _to_t(x):
        return x if isinstance(x, torch.Tensor) else torch.as_tensor(np.asarray(x))

    U_true_t = _to_t(U_true)
    nx = U_true_t.shape[0]
    half = nx // 2
    start = 1 if exclude_0_mode else 0
    k = np.arange(start, half)

    true_hat = torch.fft.fft(U_true_t, dim=0)
    power_true = (true_hat.abs() ** 2).mean(dim=1)[start:half].numpy()

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.semilogy(k, power_true, "k-", lw=2.5, label="analytical solution", zorder=10)

    colors = plt.cm.viridis(np.linspace(0.15, 0.9, max(1, len(layers_list))))
    for color, L in zip(colors, layers_list):
        U_pred_t = _to_t(U_pred_by_layers[L])
        pred_hat = torch.fft.fft(U_pred_t, dim=0)
        power_pred = (pred_hat.abs() ** 2).mean(dim=1)[start:half].numpy()
        ax.semilogy(k, power_pred, "--", color=color, lw=1.8, alpha=0.9, label=f"L={L}")
        if physical_ceiling_by_L is not None and L in physical_ceiling_by_L:
            ax.axvline(physical_ceiling_by_L[L], color=color, ls=":", lw=1.2, alpha=0.7)

    ax.set_xlabel("mode k (cycles over domain)")
    ax.set_ylabel("power")
    ax.set_title("Physical-space power spectrum: analytical solution vs. trained QPINN, by L")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=9)
    fig.tight_layout()
    _save(fig, save_path)
    if show:
        plt.show()
    return fig


physical_bandwidth_note = """
Even fixing the x/t-mixing question above, a single theta-space integer
frequency k does not correspond to a single physical frequency in
u(x, t): the local push-forward through feature_map is
k * d(feature_map_j)/d(x, t) (a 2-vector, one component per input slot j
that carries that k), which varies across the domain because of the Tanh
— i.e. each lattice point is a chirp, not a fixed tone, in physical
(kx, kt) space. If you need a genuine, analytically-boundable comb in kx
and kt separately, the architecture would need to change: e.g. route x
and t through disjoint quantum modes (so each has its own L*n_photons
bound and the achievable (kx, kt) lattice is their Cartesian product)
and/or replace feature_map's Tanh with a linear (or no) map so the
push-forward Jacobian is constant across the domain instead of a chirp.
That's a design question for a follow-up, not something this module
computes.
"""
