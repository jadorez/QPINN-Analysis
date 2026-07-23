"""
Real assertions for src/spectral.py, not smoke tests. Each test's docstring
states the specific numeric claim it checks and why that claim, if false,
indicates a specific bug (wrong sweep variable, wrong ordering, wrong
combinatorics, ...).

Run with: python -m pytest tests/test_spectral.py -v
"""

import math

import pytest
import torch
import perceval as pcvl
import merlin as ML
from merlin.core.state_vector import StateVector
from scipy.stats import unitary_group

from src.models import MerlinTunableSimple
from src.spectral import (
    spectral_second_moment,
    spectral_power_from_probs,
    _omega_eff_from_S,
    haar_relative_harmonic_weight,
    harmonic_weight_from_state,
    extract_post_w1_state,
    all_features_omega_eff,
    all_modes_harmonic_weight,
    gauge_invariant_summary,
    feature_map_jacobian,
    physical_realized_frequency,
)


def _default_model(feature_size: int = 4, seed: int = 0) -> MerlinTunableSimple:
    torch.manual_seed(seed)
    return MerlinTunableSimple(feature_size=feature_size, n_encoding_layers=1)


# ── Test 1: band-limit exactness ────────────────────────────────────────────

def test_band_limit_exactness():
    """
    Deliverable 1's omega_eff assumes the per-feature FFT sweep is a
    trigonometric polynomial of degree exactly n_photons in the swept phase.
    If that's true, sampling at G=2n+1 (critical rate) and G=4n+1
    (oversampled 2x) must recover IDENTICAL Fourier coefficients for
    |omega|<=n, and G=4n+1's extra coefficients (|omega|>n) must be zero.
    If the wrong variable is being swept (e.g. the raw feature instead of
    the phase it multiplies via `scale`), or the basis ordering used to
    aggregate power is wrong, this shows up as either non-vanishing
    high-frequency content or disagreement between the two samplings.

    Uses float64 throughout (both the model and the sweep) to actually reach
    the <1e-10 tolerance: at the model's default float32 precision the
    |omega|>n content is already comfortably below 1e-10, but rounding
    accumulation across a 64-reference-config average pushes the |omega|<=n
    agreement to ~3.7e-9 — a real float32 ceiling, not a bug (verified by
    checking it's bit-for-bit identical whether or not the FFT normalisation
    bug this test caught during development is present; only genuinely
    running the forward pass in float64 closes the gap, confirmed here to
    ~1e-17).
    """
    model = _default_model()
    model.to(torch.float64)  # feature_map AND quantum, via MerlinTunableSimple's .to() override
    n = model.quantum.n_photons

    torch.manual_seed(1)
    ref_xt = torch.rand(64, 2, dtype=torch.float64)

    r_2n = spectral_second_moment(model, ref_xt, feature_index=0, G=2 * n + 1)
    r_4n = spectral_second_moment(model, ref_xt, feature_index=0, G=4 * n + 1)

    for omega in r_2n.S:
        assert abs(r_2n.S[omega] - r_4n.S[omega]) < 1e-10, (
            f"S(omega={omega}) disagrees between G=2n+1 and G=4n+1: "
            f"{r_2n.S[omega]} vs {r_4n.S[omega]}"
        )

    high_freq = {w: v for w, v in r_4n.S.items() if abs(w) > n}
    assert high_freq, "expected some |omega|>n bins to exist at G=4n+1"
    for omega, value in high_freq.items():
        assert abs(value) < 1e-10, f"S(omega={omega}) = {value}, expected ~0 (|omega|>n)"


# ── Test 2: Haar calibration ─────────────────────────────────────────────────

def test_haar_calibration(R: int = 200):
    """
    Replaces W1 with R Haar-random U(m) matrices (bypassing the trainable
    mesh entirely via a raw pcvl.Unitary component fed straight into
    merlin's QuantumLayer) and checks the sample mean of eta_lambda lands
    within 1.0 +/- 3*SEM for every lambda. This is the key correctness check
    on the degeneracy formula and the projector construction: eta_lambda's
    definition is exactly the statement that a Haar-random unitary's
    lambda-photons-in-mode-k weight matches its Schur's-lemma expectation on
    average, so if the combinatorics or the occupation-number masking is
    wrong, this is the test most likely to catch it (Test 1 doesn't exercise
    Deliverable 2 at all, and Deliverable 2's internal sum-to-1 / sum-to-D
    assertions only catch normalization bugs, not e.g. an off-by-one in
    which lambda gets which weight).
    """
    model = _default_model()
    q = model.quantum
    n_modes = len(q.output_keys[0])
    n = q.n_photons
    mode_k = 0

    etas = []
    for r in range(R):
        U = unitary_group.rvs(n_modes, random_state=10_000 + r)
        circ = pcvl.Circuit(n_modes)
        circ.add(0, pcvl.Unitary(pcvl.Matrix(U)))
        haar_w1 = ML.QuantumLayer(
            input_size=0, circuit=circ, input_state=q.input_state,
            measurement_strategy=ML.MeasurementStrategy.amplitudes(computation_space='fock'),
        )
        with torch.no_grad():
            psi = haar_w1().squeeze(0)
        result = harmonic_weight_from_state(psi, q.output_keys, mode_k, n)
        etas.append(result.eta_lambda)

    etas = torch.stack(etas)  # (R, n+1)
    mean = etas.mean(dim=0)
    sem = etas.std(dim=0, unbiased=True) / math.sqrt(R)

    for lam in range(n + 1):
        assert abs(mean[lam].item() - 1.0) <= 3 * sem[lam].item(), (
            f"lambda={lam}: mean eta = {mean[lam].item():.4f}, "
            f"SEM = {sem[lam].item():.4f}, outside 1.0 +/- 3*SEM"
        )


# ── Test 3: single-photon degenerate case ───────────────────────────────────

def test_single_photon_degenerate_case():
    """
    n_photons=1 (feature_size=1 -> n_modes=2, input_state=[1,0]) leaves only
    omega in {-1, 0, 1} representable at the critical G=2n+1=3 grid, so ANY
    non-DC power is necessarily entirely at |omega|=1 — omega_eff must equal
    exactly 1.0 regardless of model details, as a structural consequence of
    the formula, not a property that needs a "close enough" tolerance. This
    is also the m=2 edge case for Deliverable 2 (d_lambda = C(n-lambda,0) = 1
    for all lambda, i.e. d_lambda/D = 1/(n+1) uniformly), checked here too
    since feature_size=1 gives m=2 for free.
    """
    model = _default_model(feature_size=1)
    assert len(model.quantum.output_keys[0]) == 2
    assert model.quantum.n_photons == 1

    ref_xt = torch.rand(64, 2)
    result = spectral_second_moment(model, ref_xt, feature_index=0)
    assert result.omega_eff == pytest.approx(1.0, abs=1e-6)

    harmonic = haar_relative_harmonic_weight(model, mode_k=0)
    assert len(harmonic.eta_lambda) == 2
    assert harmonic.d_over_D.tolist() == pytest.approx([0.5, 0.5])


# ── Test 4: identity W2 collapse ────────────────────────────────────────────

def test_identity_w2_collapse():
    """
    With W2 replaced by an absent (== identity) transformation, Fock
    probabilities are exactly constant in the encoded phases: without a
    mixing unitary after encoding, phase information is invisible to a
    photon-number measurement (this is a real physical property — a
    photon-counting measurement is blind to relative phase, the same fact
    underlying CFI_tt=0 for the QFI/CFI diagnostics in
    EntanglementEntropy.ipynb — not a bug). So S(omega)=0 for every omega!=0
    and omega_eff must be nan (logged as a warning, not raised), per the
    spec's explicit "Sigma_{omega!=0} S(omega) < 1e-12 -> nan" requirement.
    """
    model = _default_model()
    q = model.quantum
    n_modes = len(q.output_keys[0])
    n = q.n_photons

    builder = ML.CircuitBuilder(n_modes=n_modes)
    builder.add_entangling_layer(trainable=True, name="LI_simple", trainable_outer=model.trainable_outer)
    builder.add_angle_encoding(modes=list(range(model.feature_size)), name="input", subset_combinations=False)
    no_ri_layer = ML.QuantumLayer(
        input_size=model.feature_size, builder=builder, input_state=q.input_state,
        measurement_strategy=ML.MeasurementStrategy.amplitudes(computation_space='fock'),
    )
    with torch.no_grad():
        no_ri_layer.LI_simple.copy_(q.LI_simple)

    G = 2 * n + 1
    grid = torch.linspace(0, 2 * math.pi, G + 1)[:-1]
    z = torch.zeros(G, model.feature_size)
    z[:, 0] = grid
    with torch.no_grad():
        amps = no_ri_layer(z)
        probs = (amps.abs() ** 2).unsqueeze(0)  # (1, G, D): one reference config

    S = spectral_power_from_probs(probs, G)
    for omega, value in S.items():
        if omega != 0:
            assert abs(value) < 1e-10, f"S(omega={omega}) = {value}, expected exactly 0"

    omega_eff = _omega_eff_from_S(S, n)
    assert math.isnan(omega_eff)


# ── Test 5: normalisation invariance ────────────────────────────────────────

def test_normalization_invariance():
    """
    omega_eff is defined via a ratio (normalised power p(omega)), so scaling
    every output probability by a positive constant must leave it unchanged
    — the constant cancels between numerator and denominator in step 5's
    normalisation. Pure math, no model needed: synthetic random "probs" is
    enough, scaled by an arbitrary constant.
    """
    torch.manual_seed(0)
    n = 3
    G = 2 * n + 1
    D = 10
    probs = torch.rand(4, G, D)  # (B, G, D) synthetic

    S_unscaled = spectral_power_from_probs(probs, G)
    S_scaled = spectral_power_from_probs(probs * 7.3, G)

    omega_eff_unscaled = _omega_eff_from_S(S_unscaled, n)
    omega_eff_scaled = _omega_eff_from_S(S_scaled, n)

    assert omega_eff_unscaled == pytest.approx(omega_eff_scaled, rel=1e-6)


# ── Test 6: exact mode-permutation gauge symmetry ───────────────────────────

def test_mode_permutation_gauge_symmetry():
    """
    The central claim from the code review: relabeling modes by sigma while
    simultaneously transforming W1 -> P_sigma W1, W2 -> W2 P_sigma^dagger, and
    theta_j -> theta_{sigma^-1(j)} must leave every output channel's
    probability EXACTLY unchanged (same function). Verified here by directly
    reconstructing P_sigma W1 |psi_in> via Fock-tuple relabeling of the
    already-extracted post-W1 state (no new circuit needed — relabeling which
    occupation tuple an amplitude belongs to *is* the permutation action),
    applying the permuted diagonal encoding phase, undoing the relabeling,
    and applying the ORIGINAL W2 via the same StateVector-injection technique
    used to validate `extract_post_w1_state` itself.

    This sidesteps a real gap found while checking the review: LI_simple has
    20 real parameters, but a literal full-depth-Clements U(5) needs m^2=25,
    so "P_sigma W1 is realizable because the mesh is universal" is not quite
    the right justification for THIS mesh (confirmed by fitting: some
    permutations needed multiple random restarts to reach exactly, consistent
    with a non-universal but still-flexible-enough parameter manifold). The
    exact symmetry is verified here anyway via direct Fock-space
    reconstruction, independent of that mesh-universality question — the
    physics holds even though the review's stated proof route doesn't
    straightforwardly apply to this specific circuit.
    """
    torch.manual_seed(0)
    model = MerlinTunableSimple(n_encoding_layers=1)
    q = model.quantum
    output_keys = q.output_keys
    n_modes = len(output_keys[0])
    key_to_idx = {k: i for i, k in enumerate(output_keys)}

    sigma = {0: 1, 1: 0, 2: 2, 3: 3, 4: 4}  # swap encoded modes 0,1; fix the rest
    sigma_inv = {v: k for k, v in sigma.items()}

    def apply_mode_permutation(psi, perm):
        new_psi = torch.zeros_like(psi)
        for i, s in enumerate(output_keys):
            s_prime = tuple(s[perm[j]] for j in range(len(s)))
            new_psi[key_to_idx[s_prime]] = psi[i]
        return new_psi

    def apply_diagonal_phase(psi, theta_map):
        out = torch.zeros_like(psi)
        for i, s in enumerate(output_keys):
            phase = sum(theta_map[j] * s[j] for j in range(len(s)))
            out[i] = psi[i] * torch.exp(1j * torch.tensor(phase, dtype=torch.float64))
        return out

    psi_1 = extract_post_w1_state(model)
    psi_1_perm = apply_mode_permutation(psi_1, sigma)

    torch.manual_seed(1)
    theta = (torch.rand(model.feature_size) * 2 * math.pi)
    theta_full = {k: theta[k].item() for k in range(model.feature_size)}
    theta_full[n_modes - 1] = 0.0  # ancilla mode: never encoded
    theta_prime_full = {j: theta_full[sigma_inv[j]] for j in range(n_modes)}

    with torch.no_grad():
        amps_original = q(theta.unsqueeze(0)).squeeze(0)

    psi_2_prime = apply_diagonal_phase(psi_1_perm.to(torch.complex128), theta_prime_full)
    psi_3_prime = apply_mode_permutation(psi_2_prime, sigma_inv)

    builder_w2 = ML.CircuitBuilder(n_modes=n_modes)
    builder_w2.add_entangling_layer(trainable=True, name="RI_simple", trainable_outer=model.trainable_outer)
    psi_3_sv = StateVector(tensor=psi_3_prime.to(torch.complex64), n_modes=n_modes, n_photons=q.n_photons)
    w2_only = ML.QuantumLayer(
        input_size=0, builder=builder_w2, input_state=psi_3_sv,
        measurement_strategy=ML.MeasurementStrategy.amplitudes(computation_space='fock'),
    )
    with torch.no_grad():
        w2_only.RI_simple.copy_(q.RI_simple)
        amps_permuted = w2_only().squeeze(0)

    prob_diff = ((amps_original.abs() ** 2) - (amps_permuted.abs() ** 2)).abs().max().item()
    assert prob_diff < 1e-5, f"gauge-transformed reconstruction disagrees with original: max |ΔP| = {prob_diff}"


# ── Test 7: gauge-invariant summaries are actually sorted ──────────────────

def test_gauge_invariant_summary_is_sorted():
    """
    `gauge_invariant_summary` claims omega_eff_sorted is descending and each
    column of w_lambda_sorted is descending (sorted across modes, per
    lambda). Direct check that the sort axis/direction is what's documented
    — an easy off-by-axis bug (e.g. sorting along lambda instead of across
    modes) would silently produce a plausible-looking but wrong "invariant".
    """
    torch.manual_seed(0)
    model = MerlinTunableSimple(n_encoding_layers=1)
    torch.manual_seed(1)
    ref_xt = torch.rand(32, 2)

    summary = gauge_invariant_summary(model, ref_xt)

    assert torch.all(summary.omega_eff_sorted[:-1] >= summary.omega_eff_sorted[1:])
    assert torch.all(summary.w_lambda_sorted[:-1] >= summary.w_lambda_sorted[1:])
    assert summary.w_lambda_sorted.shape == (model.feature_size, model.quantum.n_photons + 1)

    # sorted profile must be a genuine permutation of the labeled one, per lambda column
    labeled_sorted, _ = torch.sort(summary.w_lambda_labeled, dim=0, descending=True)
    assert torch.allclose(labeled_sorted, summary.w_lambda_sorted)


# ── Test 8: feature_map Jacobian against finite differences ────────────────

def test_feature_map_jacobian_matches_finite_differences():
    """
    `feature_map_jacobian` is the pushforward used to convert a theta-space
    harmonic into a local physical (k_x, k_t) frequency
    (`physical_realized_frequency`) — if this Jacobian is wrong, every
    downstream physical-frequency number is wrong. Checked against central
    finite differences, since feature_map is just a small MLP with an
    independently-known forward pass.
    """
    torch.manual_seed(0)
    model = MerlinTunableSimple(n_encoding_layers=1)
    torch.manual_seed(1)
    ref_xt = torch.rand(8, 2)

    jac = feature_map_jacobian(model, ref_xt)
    assert jac.shape == (8, model.feature_size, 2)

    eps = 1e-3
    with torch.no_grad():
        xt_dx_p, xt_dx_m = ref_xt.clone(), ref_xt.clone()
        xt_dx_p[:, 0] += eps; xt_dx_m[:, 0] -= eps
        xt_dt_p, xt_dt_m = ref_xt.clone(), ref_xt.clone()
        xt_dt_p[:, 1] += eps; xt_dt_m[:, 1] -= eps
        fd_dx = (model.feature_map(xt_dx_p) - model.feature_map(xt_dx_m)) / (2 * eps)
        fd_dt = (model.feature_map(xt_dt_p) - model.feature_map(xt_dt_m)) / (2 * eps)

    assert (jac[:, :, 0] - fd_dx).abs().max().item() < 1e-4
    assert (jac[:, :, 1] - fd_dt).abs().max().item() < 1e-4

    # physical_realized_frequency is just omega * J -- check the scaling directly
    freq = physical_realized_frequency(jac, feature_index=0, omega=3)
    assert torch.allclose(freq, 3 * jac[:, 0, :])
