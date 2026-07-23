"""
spectral.py — read-only spectral diagnostics for MerlinTunableSimple's quantum layer.

Two diagnostics, both currently restricted to n_encoding_layers=1 (a single
W1 -> angle encoding -> W2 block):

1. omega_eff (spectral second moment): how much high-frequency content the
   quantum layer supplies per encoded feature, via an FFT sweep of that
   feature's phase with everything else held fixed. Threshold-free.
2. eta_lambda (Haar-relative harmonic weight): whether the state W1 prepares
   (before any encoding) concentrates or depletes amplitude in each
   "lambda-photons-in-mode-k" sector relative to a Haar-random U(m), via
   Schur's lemma on the bosonic representation Sym^n(C^m).

Everything here is read-only: functions run under torch.no_grad() and never
touch a model's parameters (only read them, or copy their *values* into a
freshly-constructed, separate sub-circuit for W1-only evaluation). No
training behaviour is affected by importing or calling this module.

n_encoding_layers=1 restriction: both deliverables' core definitions
(Deliverable 2's "state after W1, before any encoding"; Deliverable 1's
per-feature FFT sweep) remain individually well-defined for
n_encoding_layers>1 (see conversation notes), but `_check_single_encoding_layer`
raises rather than silently generalizing, pending a decision on how to report
per-encoding-pass results for the multi-layer case.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import logging
import math

import torch
import merlin as ML
from scipy.special import comb

from src.models import MerlinTunableSimple

logger = logging.getLogger(__name__)


def _check_single_encoding_layer(model: MerlinTunableSimple) -> None:
    n_layers = getattr(model, "n_encoding_layers", 1)
    if n_layers != 1:
        raise ValueError(
            f"spectral diagnostics are only implemented for n_encoding_layers=1 "
            f"(got {n_layers}). Deliverable 2's W1-only extraction and "
            f"Deliverable 1's per-feature FFT sweep are each individually "
            f"well-defined for a re-uploading circuit, but reporting needs an "
            f"explicit per-pass convention first rather than silently picking one."
        )


# ── Deliverable 2: Haar-relative harmonic weight ────────────────────────────

def extract_post_w1_state(model: MerlinTunableSimple) -> torch.Tensor:
    """
    Returns |psi_1> = W1 |psi_in>, the state immediately after the first
    trainable entangling layer ("LI_simple" / W1) and before any angle
    encoding, as a (D,) complex tensor (D = model.quantum.output_size),
    ordered exactly as model.quantum.output_keys.

    Builds a fresh QuantumLayer containing *only* a freshly-constructed
    LI_simple-named entangling layer (same mesh structure as the trained
    model's, via trainable_outer), copies the trained LI_simple parameter
    values into it, and runs it with the same input_state.

    Validated (not just assumed) against the full model: composing this with
    a matching W2-only extraction (same technique, RI_simple, fed this
    function's output as a custom StateVector input) reproduces
    model.quantum(zeros) to float32 precision — see the notebook's
    verification cell.
    """
    _check_single_encoding_layer(model)
    q = model.quantum
    n_modes = len(q.output_keys[0])

    builder = ML.CircuitBuilder(n_modes=n_modes)
    builder.add_entangling_layer(trainable=True, name="LI_simple", trainable_outer=model.trainable_outer)
    w1_only = ML.QuantumLayer(
        input_size=0,
        builder=builder,
        input_state=q.input_state,
        measurement_strategy=ML.MeasurementStrategy.amplitudes(computation_space='fock'),
    )
    with torch.no_grad():
        w1_only.LI_simple.copy_(q.LI_simple)
        psi_1 = w1_only()

    return psi_1.squeeze(0).detach().cpu()


@dataclass
class HarmonicWeightResult:
    mode: int
    n_photons: int
    n_modes: int
    hilbert_dim: int
    w_lambda: torch.Tensor    # (n_photons+1,) realised weight per lambda
    d_over_D: torch.Tensor    # (n_photons+1,) Haar expectation of w_lambda
    eta_lambda: torch.Tensor  # (n_photons+1,) w_lambda / d_over_D


def harmonic_weight_from_state(
    psi_1: torch.Tensor,
    output_keys: List[Tuple[int, ...]],
    mode_k: int,
    n_photons: int,
) -> HarmonicWeightResult:
    """
    Pure-math core of Deliverable 2: given a post-W1 state and the Fock basis
    it's expressed in (as (mode-occupation tuples), library-provided order —
    e.g. model.quantum.output_keys), computes w_lambda, the Haar expectation
    d_lambda/D, and eta_lambda for encoded mode `mode_k`.

    Split out from `haar_relative_harmonic_weight` so the Haar-calibration
    test can call this directly on states prepared by literal Haar-random
    unitaries (bypassing MerlinTunableSimple and its trained mesh entirely),
    without needing a fitted model at all.

    Args:
        psi_1:       (D,) complex state.
        output_keys: length-D list of length-n_modes occupation tuples, the
                     SAME ordering psi_1's entries are expressed in.
        mode_k:      which physical mode's occupation number to project on.
        n_photons:   total (conserved) photon number of the Fock sector.

    Returns:
        HarmonicWeightResult

    Raises:
        AssertionError if sum_lambda w_lambda != 1 or sum_lambda d_lambda != D
        (a projector/combinatorics bug, not a physically possible outcome).
    """
    n_modes = len(output_keys[0])
    hilbert_dim = len(output_keys)
    expected_dim = int(comb(n_photons + n_modes - 1, n_photons, exact=True))
    if hilbert_dim != expected_dim:
        raise ValueError(
            f"len(output_keys)={hilbert_dim} != C(n+m-1,n)={expected_dim} "
            f"for n_photons={n_photons}, n_modes={n_modes}"
        )

    occ_k = torch.tensor([key[mode_k] for key in output_keys], dtype=torch.long)
    probs = (psi_1.abs() ** 2).double()

    w = torch.zeros(n_photons + 1, dtype=torch.float64)
    d = torch.zeros(n_photons + 1, dtype=torch.float64)
    for lam in range(n_photons + 1):
        mask = occ_k == lam
        w[lam] = probs[mask].sum()
        # d_lambda: ways to distribute (n_photons - lambda) photons among the
        # remaining (n_modes - 1) modes ("stars and bars"). Handles m=2
        # correctly without special-casing: comb(n-lambda, 0) == 1 for all lambda.
        d[lam] = comb(n_photons - lam + n_modes - 2, n_modes - 2, exact=True)

    if abs(w.sum().item() - 1.0) > 1e-4:
        raise AssertionError(
            f"sum_lambda w_lambda = {w.sum().item():.6f} != 1 "
            f"(state not normalized, or projector/ordering bug)"
        )
    if int(round(d.sum().item())) != expected_dim:
        raise AssertionError(
            f"sum_lambda d_lambda = {int(d.sum().item())} != D={expected_dim} "
            f"(degeneracy formula or n_modes/n_photons mismatch)"
        )

    d_over_D = d / expected_dim
    eta = w / d_over_D

    return HarmonicWeightResult(
        mode=mode_k, n_photons=n_photons, n_modes=n_modes, hilbert_dim=expected_dim,
        w_lambda=w.float(), d_over_D=d_over_D.float(), eta_lambda=eta.float(),
    )


def haar_relative_harmonic_weight(model: MerlinTunableSimple, mode_k: int) -> HarmonicWeightResult:
    """
    Deliverable 2, model-facing: extracts |psi_1>=W1|psi_in> from `model` and
    computes its Haar-relative harmonic weight for encoded mode `mode_k`.
    See `harmonic_weight_from_state` for the underlying computation.
    """
    _check_single_encoding_layer(model)
    psi_1 = extract_post_w1_state(model)
    q = model.quantum
    return harmonic_weight_from_state(psi_1, q.output_keys, mode_k, q.n_photons)


# ── Deliverable 1: spectral second moment ───────────────────────────────────

@dataclass
class SpectralMomentResult:
    feature_index: int
    n_photons: int
    G: int
    omega_eff: float          # nan if the layer is constant in this feature
    S: Dict[int, float] = field(repr=False)  # un-normalised power per integer omega
    dc_fraction: float = float("nan")


def spectral_power_from_probs(probs: torch.Tensor, G: int) -> Dict[int, float]:
    """
    Pure-math core of Deliverable 1 (steps 3-4 + the batch averaging): FFTs a
    batch of probability-vs-swept-phase sweeps along the grid axis, sums
    power across output channels, and averages across any leading
    (reference-configuration) batch dimensions — in that order, matching the
    spec's requirement to average post-hoc power S(omega), not raw complex
    coefficients or the final normalised/omega_eff quantities.

    Args:
        probs: (..., G, D) real tensor. Leading dims (if any) are averaged
            over; G is the phase-sweep grid axis; D is the number of output
            (Fock-basis) channels.
        G: sweep grid size (must match probs.shape[-2]).

    Returns:
        S: {omega: value} for every integer omega representable at this G
           (i.e. G values, from torch.fft.fftfreq(G)*G). Normalised
           (norm="forward", i.e. divided by G) so that S(omega) for a given
           true frequency is comparable across different choices of G, not
           just within a single G — plain torch.fft.fft is unnormalised, so
           its raw magnitudes scale with G itself and are not comparable
           across sweep resolutions (caught by the band-limit test).
    """
    # Preserve probs' own precision (e.g. complex128 if probs is float64) rather
    # than forcing complex64 — a prior version of this function hardcoded
    # complex64 here, silently discarding any higher-precision input.
    complex_dtype = torch.complex128 if probs.dtype == torch.float64 else torch.complex64
    fft_coeffs = torch.fft.fft(probs.to(complex_dtype), dim=-2, norm="forward")  # (..., G, D)
    power = (fft_coeffs.abs() ** 2).sum(dim=-1)  # (..., G) -- sum over channels c
    while power.dim() > 1:
        power = power.mean(dim=0)  # average S(omega) over leading batch dims
    omega_idx = torch.round(torch.fft.fftfreq(G) * G).to(torch.int64)
    return {int(w): float(power[i]) for i, w in enumerate(omega_idx.tolist())}


def _omega_eff_from_S(S: Dict[int, float], n_photons: int) -> float:
    """Steps 5-6: exclude DC, normalise, second moment. See spectral_second_moment."""
    nonzero_sum = sum(v for w, v in S.items() if w != 0)
    if nonzero_sum < 1e-12:
        logger.warning(
            "spectral.spectral_second_moment: no non-DC spectral power found "
            "(layer is constant in this feature); returning nan"
        )
        return float("nan")
    second_moment = sum((w ** 2) * (v / nonzero_sum) for w, v in S.items() if w != 0)
    return math.sqrt(second_moment) / n_photons


def spectral_second_moment(
    model: MerlinTunableSimple,
    reference_xt: torch.Tensor,
    feature_index: int,
    G: Optional[int] = None,
) -> SpectralMomentResult:
    """
    Deliverable 1, model-facing: sweeps encoded feature `feature_index`'s
    phase over a uniform grid on [0, 2*pi/scale) — the *phase* the circuit
    actually sees, i.e. this feature's `scale` from
    model.quantum.angle_encoding_specs is divided out, not assumed to be 1 —
    while every other encoded feature is held at reference values obtained by
    pushing `reference_xt` through the trained feature_map, averages the
    resulting per-channel FFT power across those reference configurations,
    and reduces to a single [0,1] second-moment summary.

    Args:
        model:         trained MerlinTunableSimple (n_encoding_layers=1).
        reference_xt:  (B, 2) real [x, t] points, e.g. sampled from the
            training domain via `DataSampler.sample_interior(B)`. B=64 is the
            spec's suggested default — pass a differently-sized batch to change it.
        feature_index: which of the total_input_size encoded features to sweep.
        G: sweep grid size. Defaults to 2*n_photons+1 (the critical sampling
            rate for a degree-n_photons trig polynomial); pass e.g. 4*n+1 to
            oversample as a band-limit consistency check.

    Returns:
        SpectralMomentResult
    """
    _check_single_encoding_layer(model)
    q = model.quantum
    n = q.n_photons
    if G is None:
        G = 2 * n + 1

    scale = q.angle_encoding_specs["input"]["scales"][feature_index]
    period = 2 * math.pi / scale

    device = next(model.parameters()).device
    with torch.no_grad():
        z_ref = model.feature_map(reference_xt.to(device))  # (B, total_input_size)
    b = z_ref.shape[0]

    grid = torch.linspace(0, period, G + 1, device=device, dtype=z_ref.dtype)[:-1]  # (G,)

    z_batch = z_ref.unsqueeze(1).expand(b, G, -1).clone()
    z_batch[:, :, feature_index] = grid.unsqueeze(0).expand(b, G)
    z_batch = z_batch.reshape(b * G, -1)

    with torch.no_grad():
        amps = q(z_batch)  # (B*G, D)
        probs = (amps.abs() ** 2).reshape(b, G, -1).cpu()

    S = spectral_power_from_probs(probs, G)
    total_all = sum(S.values())
    dc_fraction = S.get(0, 0.0) / total_all if total_all > 0 else float("nan")
    omega_eff = _omega_eff_from_S(S, n)

    return SpectralMomentResult(
        feature_index=feature_index, n_photons=n, G=G,
        omega_eff=omega_eff, S=S, dc_fraction=dc_fraction,
    )


# ── Gauge-invariant summaries ────────────────────────────────────────────────
#
# Mode/feature index k is not a physical label: relabeling modes by any
# permutation sigma while simultaneously permuting the pre-encoding unitary's
# rows (W1 -> P_sigma W1), the post-encoding unitary's columns
# (W2 -> W2 P_sigma^dagger), and the corresponding rows of the classical
# feature_map's output layer leaves every observable identical — same
# function, same loss. Both w_lambda^(k) and omega_eff^(k) transform under
# this the same way (w'^(sigma(k))_lambda = w^(k)_lambda), so "feature k has a
# low score" is not by itself a gauge-invariant statement. What IS invariant
# is a symmetric function of the labeled vector across k: the sorted profile,
# the max, the mean. Use these for any claim about the *architecture*; use
# the labeled vectors only for claims about one specific trained instance.

def all_features_omega_eff(
    model: MerlinTunableSimple,
    reference_xt: torch.Tensor,
    G: Optional[int] = None,
) -> torch.Tensor:
    """omega_eff for every encoded feature, in native (labeled, gauge-dependent) order."""
    _check_single_encoding_layer(model)
    return torch.tensor([
        spectral_second_moment(model, reference_xt, feature_index=k, G=G).omega_eff
        for k in range(model.feature_size)
    ])


def all_modes_harmonic_weight(model: MerlinTunableSimple) -> torch.Tensor:
    """w_lambda for every encoded mode, shape (feature_size, n_photons+1), labeled (gauge-dependent) mode order."""
    _check_single_encoding_layer(model)
    return torch.stack([
        haar_relative_harmonic_weight(model, mode_k=k).w_lambda
        for k in range(model.feature_size)
    ])


@dataclass
class GaugeInvariantSummary:
    omega_eff_labeled: torch.Tensor   # (feature_size,) native order -- NOT gauge-invariant, kept for reference
    omega_eff_sorted: torch.Tensor    # (feature_size,) descending -- gauge-invariant
    w_lambda_labeled: torch.Tensor    # (feature_size, n_photons+1) native order -- NOT gauge-invariant
    w_lambda_sorted: torch.Tensor     # (feature_size, n_photons+1), each column sorted descending across modes
    w_lambda_max: torch.Tensor        # (n_photons+1,) max_k w_lambda^(k) -- gauge-invariant
    w_lambda_mean: torch.Tensor       # (n_photons+1,) mean_k w_lambda^(k) -- gauge-invariant


def gauge_invariant_summary(
    model: MerlinTunableSimple,
    reference_xt: torch.Tensor,
    G: Optional[int] = None,
) -> GaugeInvariantSummary:
    """
    Computes both the labeled (mode/feature-index-dependent, hence
    gauge-*dependent*) vectors and their gauge-*invariant* symmetric-function
    reductions (sorted profile, max, mean) in one call, for both deliverables.
    """
    omega_eff_labeled = all_features_omega_eff(model, reference_xt, G=G)
    w_labeled = all_modes_harmonic_weight(model)  # (feature_size, n_photons+1)

    omega_eff_sorted, _ = torch.sort(omega_eff_labeled, descending=True)
    w_lambda_sorted, _ = torch.sort(w_labeled, dim=0, descending=True)  # sort across modes, per lambda

    return GaugeInvariantSummary(
        omega_eff_labeled=omega_eff_labeled,
        omega_eff_sorted=omega_eff_sorted,
        w_lambda_labeled=w_labeled,
        w_lambda_sorted=w_lambda_sorted,
        w_lambda_max=w_labeled.max(dim=0).values,
        w_lambda_mean=w_labeled.mean(dim=0),
    )


# ── Physical realized frequency (pushforward through the classical feature_map) ──
#
# omega_eff and eta_lambda are statements about theta-space (the encoded
# phases), not about (x, t) directly. Pushing forward through the (possibly
# nonlinear) feature_map Phi, a theta-space harmonic omega on feature k
# contributes a LOCAL spatial frequency omega * d(Phi_k)/d(x,t) to u(x,t) --
# not omega itself. For a linear Phi this is a single fixed vector (a genuine
# harmonic stack with learnable fundamental); for a nonlinear Phi (this
# model's feature_map has a Tanh) it varies over the domain, so a theta-space
# lattice point becomes a chirp rather than a tone in physical space. Report
# the DISTRIBUTION over reference points, not a single number.

def feature_map_jacobian(model: MerlinTunableSimple, reference_xt: torch.Tensor) -> torch.Tensor:
    """
    d(feature_map(xt)_k) / d(x, t) for every encoded feature k, at every row
    of `reference_xt`. Uses the same grad-of-batch-sum trick as
    `utils.gradient` (exact here: feature_map is a plain per-sample MLP with
    no cross-batch mixing), one backward pass per feature (cheap: 4 for the
    default feature_size).

    Args:
        model:        MerlinTunableSimple (n_encoding_layers=1).
        reference_xt: (B, 2) real [x, t] points.

    Returns:
        J: (B, feature_size, 2) real tensor, J[b, k, :] = d z_k/d(x,t) at reference_xt[b].
    """
    _check_single_encoding_layer(model)
    device = next(model.parameters()).device
    xt = reference_xt.to(device).detach().clone().requires_grad_(True)
    z = model.feature_map(xt)  # (B, feature_size)
    b, k_dim = z.shape
    ones = torch.ones(b, device=device, dtype=xt.dtype)

    jac = torch.empty(b, k_dim, 2, device=device, dtype=xt.dtype)
    for k in range(k_dim):
        grad = torch.autograd.grad(z[:, k], xt, grad_outputs=ones, retain_graph=True)[0]
        jac[:, k, :] = grad

    return jac.detach().cpu()


def physical_realized_frequency(jacobian: torch.Tensor, feature_index: int, omega: int) -> torch.Tensor:
    """
    omega * J[:, feature_index, :] -- the local physical (k_x, k_t) frequency
    a theta-space harmonic `omega` on `feature_index` induces at every
    reference point, given `jacobian` from `feature_map_jacobian`. A
    distribution (one (k_x, k_t) pair per reference point), not a scalar,
    since the Jacobian varies over the domain for this model's nonlinear
    feature_map.

    Args:
        jacobian:      (B, feature_size, 2) from `feature_map_jacobian`.
        feature_index: which feature's Jacobian column to use.
        omega:         the theta-space harmonic (typically from
                       `spectral_second_moment`'s `S`, e.g. its argmax|omega!=0|).

    Returns:
        (B, 2) real tensor of local (k_x, k_t) pairs.
    """
    return omega * jacobian[:, feature_index, :]
