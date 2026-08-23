import torch
import torch.nn as nn
from typing import Tuple, Optional, List
from merlin.core.state_vector import StateVector
from src.utils import generate_fock_basis

import perceval as pcvl
import scipy

import merlin as ML
import math


class MerlinHeatQPINN(nn.Module):
    """
    QPINN model using MerLin quantum layer for pde resolution.
    """

    def __init__(self, feature_size : int = 4, quantum_output_size : int = 20, hidden_feature : int = 16, hidden_readout : int = 4, pde = None):
        super().__init__()
        self.feature_size = feature_size
        self.quantum_output_size = quantum_output_size
        self.hidden_feature = hidden_feature
        self.hidden_readout = hidden_readout

        self.pde = pde

        # Feature map: classical neural network to map (x, t) to feature space
        self.feature_map = nn.Sequential(
            nn.Linear(2, self.hidden_feature),
            nn.Tanh(),
            nn.Linear(self.hidden_feature, self.feature_size),
        )

        # Import MerLin quantum layer
        try:
            import merlin as ML
            self.quantum = ML.QuantumLayer.simple(
                input_size=self.feature_size,
                output_size=self.quantum_output_size,
            )
        except ImportError as e:
            raise ImportError(
                "Could not import MerLin. Install with: pip install merlinquantum"
            ) from e

        # Readout: classical neural network to extract u and du/dx
        self.readout = nn.Sequential(
            nn.Linear(self.quantum_output_size, self.hidden_readout),
            nn.Tanh(),
            nn.Linear(self.hidden_readout, 2),  # [q_u, ux_hat]
        )

    def forward(self, xt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            xt: tensor of shape (batch, 2) with columns [x, t]

        Returns:
            u: solution field, shape (batch, 1)
            ux_hat: x-derivative approximation, shape (batch, 1)
        """
        x = xt[:, 0:1]
        t = xt[:, 1:2]
        z = self.feature_map(xt)
        q = self.quantum(z)
        out = self.readout(q)

        u = out[:, 0:1]
        ux_hat = out[:, 1:2]

        f_x = torch.zeros_like(x)

        if hasattr(self.pde, "initial_condition"):
            f_x = self.pde.initial_condition(x)

            envelope = 1 - torch.cos(0.5 * math.pi * t)

            u = (self.pde.x_min + x) * (self.pde.x_max - x) * u #(envelope * u) + f_x
        else:
            u = (self.pde.x_min + x) * (self.pde.x_max - x) * u

        return u, ux_hat


class ClassicalHeatQPINN(nn.Module):
    """
    Classical neural network QPINN (baseline without quantum layer).
    Useful for comparing quantum vs classical performance.
    """

    def __init__(self, feature_size : int = 4, quantum_output_size : int = 20, hidden_feature : int = 16, hidden_readout : int = 4, pde = None):
        super().__init__()
        self.feature_size = feature_size
        self.quantum_output_size = quantum_output_size
        self.hidden_feature = hidden_feature
        self.hidden_readout = hidden_readout
        self.pde = pde

        # Larger classical network to match quantum layer capacity
        self.network = nn.Sequential(
            nn.Linear(2, self.hidden_feature),
            nn.Tanh(),
            nn.Linear(self.hidden_feature, self.feature_size),
            nn.Tanh(),
            nn.Linear(self.feature_size, self.quantum_output_size),
            nn.Tanh(),
            nn.Linear(self.quantum_output_size, self.hidden_readout),
            nn.Tanh(),
            nn.Linear(self.hidden_readout, 2),  # [u, ux_hat]
        )

    def forward(self, xt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            xt: tensor of shape (batch, 2) with columns [x, t]

        Returns:
            u: solution field, shape (batch, 1)
            ux_hat: x-derivative approximation, shape (batch, 1)
        """
        x = xt[:, 0:1]
        t = xt[:, 1:2]
        out = self.network(xt)

        u = out[:, 0:1]
        ux_hat = out[:, 1:2]
        f_x = torch.zeros_like(x)

        if hasattr(self.pde, "initial_condition"):
            f_x = self.pde.initial_condition(x)

            envelope = 1 - torch.cos(0.5 * math.pi * t)

#            u = (self.pde.x_min + x) * (self.pde.x_max - x) * u # (envelope * u) + f_x
#        else:
        u = (self.pde.x_min + x) * (self.pde.x_max - x) * u

        return u, ux_hat

def fock_occupations(feature_size: int, n_photons: int) -> List[Tuple[int, ...]]:
    """The occupation vectors `n_{1:F}` on the F ENCODED modes, for N photons
    in m = F+1 modes: every v in Z_{>=0}^F with sum(v) <= N (the remaining
    photons sit in the unencoded reference mode). These are exactly the
    frequency vectors appearing in the photonic layer's AMPLITUDE
    c_k(z) = sum_n B_{k,n} A_n e^{i n_{1:F} . z} (paper, sec:spectral).
    At the reference config (F=4, N=3) there are 35 of them = D."""
    from itertools import product
    return [v for v in product(range(n_photons + 1), repeat=feature_size)
            if sum(v) <= n_photons]


def omega_difference_set(feature_size: int, n_photons: int) -> List[Tuple[int, ...]]:
    """The paper's frequency set Omega (eq:Omega): the DIFFERENCE SET
    {n_{1:F} - n'_{1:F}} of the Fock lattice, which is what the
    modulus-squared readout p_k = |c_k|^2 actually produces. Returns one
    representative of each {omega, -omega} pair, excluding omega = 0 --
    since cos/sin at omega and -omega span the same real functions, and
    omega = 0 only contributes a constant (absorbed by the linear layer's
    bias). At the reference config (F=4, N=3) the full set has 471 vectors,
    so this returns (471-1)/2 = 235 representatives, with per-mode support
    omega_j in {-3..3} and ||omega||_1 <= 2N = 6 (prop:lattice)."""
    S = fock_occupations(feature_size, n_photons)
    diffs = set()
    for a in S:
        for b in S:
            diffs.add(tuple(ai - bi for ai, bi in zip(a, b)))
    kept = []
    for w in sorted(diffs):
        if all(c == 0 for c in w):
            continue
        first_nonzero = next(c for c in w if c != 0)
        if first_nonzero > 0:          # keep exactly one of {w, -w}
            kept.append(w)
    return kept


def axis_aligned_frequencies(feature_size: int, n_photons: int) -> List[Tuple[int, ...]]:
    """The much smaller AXIS-ALIGNED frequency set {n * e_j : n=1..N, j=1..F}
    -- the original (and, as it turned out, inadequate) approximation used by
    `FourierFeaturesHeatQPINN` before the full-Omega implementation. Kept so
    the two can be compared directly. F*N vectors (12 at the reference
    config) versus Omega's 235 representatives."""
    return [tuple(n if j == axis else 0 for j in range(feature_size))
            for axis in range(feature_size)
            for n in range(1, n_photons + 1)]


class FourierFeaturesHeatQPINN(nn.Module):
    #"""
    #Classical control for the spectral-coverage hypothesis (paper
    #`sec:spectral` / `sec:readout-control`): NO quantum circuit at all,
    #but the middle block is hard-coded with a FIXED (non-trainable)
    #sinusoidal encoding at the same per-mode frequency reach the photonic
    #layer has, `{-n_photons_per_pass, ..., n_photons_per_pass}` on each of
    #`feature_size` axes, REPEATED independently across `n_encoding_layers`
    #re-uploaded chunks -- mirroring `MerlinTunableEntropy`'s own re-upload
    #structure (see `_fourier_features` below).
#
    #This is a literal implementation of `eq:layer-fourier` in the paper:
    #p'_k(z) = sum_{omega in Omega'} gamma_{k,omega} * e^{i omega . z}, with
    #Omega' fixed at initialisation and ONLY the coefficients gamma_{k,.}
    #trainable -- via a strictly LINEAR combination layer, deliberately
    #with no activation.
#
    #IMPORTANT FIX (found from a real L-sweep run): an earlier version of
    #this class computed a single, L-INDEPENDENT `z = feature_map(xt)` and
    #only grew the number of harmonics applied to that one fixed view of
    #`(x,t)` as L grew (`n_freqs = L * n_photons`). That is not what the
    #quantum arm's `L` actually does. `MerlinTunableEntropy.feature_map`
    #outputs `feature_size * L` values, split into `L` INDEPENDENT chunks,
    #each a different learned nonlinear function of `(x,t)`, each re-fed to
    #its own trainable mesh layer. Growing `L` there gives the model more
    #independent nonlinear views of `(x,t)` to combine, not just more
    #harmonics of one fixed view. With the old design, increasing L for
    #this control never gave it any new information about `(x,t)` -- only
    #a bigger, but redundant, harmonic palette on the same scalar features
    #-- which is the most likely reason it flatlined at the classical
    #baseline for L >= 2 instead of improving like the quantum arm. Fixed
    #below by making `feature_map` also output `feature_size * L` values,
    #split into `L` independent chunks, each individually given the fixed
    #per-pass harmonic encoding (frequencies `1..n_photons_per_pass`, the
    #genuine single-layer photon-number bound) before all `L` chunks'
    #encodings are concatenated into the linear `combine` layer. This is
    #still NOT a literal reproduction of the quantum arm's L-fold sumset
    #(that arises from PRODUCTS/interference of unitaries across mesh
    #layers, a nonlinear composition we deliberately avoid here to keep
    #the "fixed frequencies, only linear coefficients trainable" premise
    #intact) -- but it does restore the one capability that was actually
    #missing: L independent, learned nonlinear reparametrisations of
    #`(x,t)` feeding the fixed encoding, rather than one repeated for
    #every L.
#
    #Two things this design choice is still NOT:
#
      #- It is NOT the literal difference-set Omega of Prop. 1 (471
        #frequency vectors at the L=1 reference config): that would make
        #the fixed encoding far too large to matched-parameter-fit without
        #starving the combination layer. Using the smaller, axis-aligned
        #per-pass frequency set keeps the encoding dimension tractable
        #while still giving this arm the same per-pass MAXIMUM frequency
        #reach (`n_photons_per_pass`) the quantum arm's single mesh layer
        #has. It is a faithful test of "same spectral *reach*, same
        #re-upload structure", not "same spectral *set*".
      #- The combination layer is deliberately LINEAR (no tanh): any
        #nonlinearity here could itself manufacture frequencies beyond the
        #fixed encoding's support (the exact confound `sec:readout-control`
        #flags for the readout that follows), which would silently break
        #the "fixed frequency support" premise this control is supposed to
        #isolate.
#
    #If this arm matches the quantum arm's accuracy at matched parameter
    #count, matched per-pass frequency reach, and matched re-upload depth,
    #that's strong support for the spectral-coverage hypothesis
    #(quantumness isn't required). If it doesn't, that's evidence the
    #quantum layer's specific bilinear, unitarity-constrained
    #parameterisation of gamma_{k,omega} (rather than free trainable
    #coefficients on the same lattice) is doing real work.
#
    #IMPORTANT FIX #2 (found after a real re-trained L-sweep, post fix #1,
    #came back with EVERY seed at EVERY L stuck at rel_l2~1.0 -- worse than
    #before, despite this arm now having MANY more trainable parameters
    #than quantum via the architecture-matched `quantum_output_size` =
    #`measurement_dim_for_feature_size(feature_size)`, e.g. 35 at the usual
    #feature_size=4): the old `combine` (Linear, deliberately no activation
    #-- required by the "only linear coefficients trainable" premise above)
    #fed DIRECTLY into `readout`'s own first layer, which was ALSO a plain
    #`nn.Linear`, with nothing nonlinear in between. Composing two
    #consecutive linear layers with no activation between them is ALWAYS
    #mathematically identical to a single linear layer of rank <=
    #min(encoding_dim, quantum_output_size, hidden_readout) -- bottlenecked
    #here by `hidden_readout` (4), the smallest of the three. So the entire
    #`quantum_output_size`-wide intermediate stage added ZERO expressive
    #power over a plain `Linear(encoding_dim, hidden_readout)`, while
    #forcing gradient descent to push everything through an artificially
    #factored, badly-conditioned linear map instead -- a known pathology in
    #deep linear network optimization (redundant un-activated linear
    #factorization doesn't just fail to add capacity, it actively slows or
    #stalls convergence). Quantum's own readout does NOT have this problem:
    #its `Linear(measurement_dim, hidden_readout)` is preceded by a
    #genuinely NONLINEAR representation (measurement probabilities are a
    #nonlinear, bilinear function of the circuit's trainable phases), so
    #there is no redundant linear-on-linear composition there. This bug
    #predates the architecture-match fix too (it was just less consequential
    #when `quantum_output_size` was small, 2-4, under the old brute-force
    #search) -- jumping it to 35 made the wasted, poorly-conditioned
    #factorization much worse, consistent with going from "some seeds
    #converge" to "zero seeds ever converge".
#
    #Fixed by collapsing `combine` and `readout`'s first layer into ONE
    #actual `nn.Linear(encoding_dim, hidden_readout)` -- mathematically the
    #SAME function class the old two-layer version could represent (they
    #were already exactly equivalent), just without the wasted parameters
    #or the conditioning cost of the pointless factorization. `combine` is
    #still strictly linear on the fixed encoding, exactly per
    #`eq:layer-fourier`. `quantum_output_size` is kept as a constructor
    #argument purely so existing call sites (`build_model`,
    #`resolve_matched_ff_config`, the checkpoint config schema) don't need
    #to change -- it is accepted and stored, but no longer used to size any
    #layer.
#
    #IMPORTANT FIX #3 -- the frequency SET, not just its reach. Fixes #1 and
        ##2 both left this arm flat across every L while quantum improved. The
        #remaining (and, we believe, actual) cause was the frequency set itself.
        #This class originally used only AXIS-ALIGNED frequencies
        #{n * e_j : n=1..N, j=1..F} -- 12 vectors at the reference config -- and
        #the docstring above defended that as same spectral *reach*, not same
        #spectral *set*. But the photonic layer's Omega is the DIFFERENCE SET of
        #the Fock lattice: 471 MULTIVARIATE vectors with ||omega||_1 <= 2N = 6
        #(prop:lattice). Two consequences, both fatal to the axis-aligned
        #approximation as a test of the hypothesis:
#    
          #1. A linear combination of {cos(n z_j), sin(n z_j)} is a sum of
             #UNIVARIATE functions -- an additive model in z. It cannot represent
             #any cross-mode product such as cos(z_1 - z_2). Omega is full of
             #such terms; the axis-aligned set contains none of them. That is a
             #different (strictly smaller) function class, not a smaller version
             #of the same one.
          #2. A term e^{i omega . z} oscillates in x at rate |sum_j omega_j s_j|
             #where s_j is the slope of z_j(x). Quantum can use ||omega||_1 = 6
             #(e.g. omega = (3,-3,0,0) -- note sum_j omega_j = 0 always, since
             #Omega is a difference set of fixed-photon-number states, so these
             #cross terms are the ONLY route to large ||omega||_1). Axis-aligned
             #tops out at ||omega||_1 = N = 3. At a target frequency chosen to
             #sit outside the baseline window, a factor-of-2 shortfall in
             #reachable frequency plausibly accounts for the whole failure.
#    
        #`frequency_mode` now selects between the two, defaulting to "full"
        #(the real Omega). "axis" reproduces the old, inadequate behaviour for
        #comparison. Note that "full" makes the encoding ~470-dimensional, so
        #`combine` alone carries ~1.9k parameters against the quantum arm's ~310
        #-- this arm is therefore a deliberate CAPACITY UPPER BOUND, not a
        #parameter-matched test. See `ModulusFourierHeatQPINN` for the
        #parameter-matched version that gets the same 471-frequency support out
        #of ~35 trainable amplitudes by keeping the modulus-squared structure.
    #"""


    

    def __init__(self, feature_size: int = 4, quantum_output_size: int = 20,
                 hidden_feature: int = 16, hidden_readout: int = 4,
                 n_photons_per_pass: int = 3, n_encoding_layers: int = 1,
                 frequency_mode: str = "full", pde=None):
        super().__init__()
        self.feature_size = feature_size
        # Accepted for interface/checkpoint-schema compatibility only -- see
        # "IMPORTANT FIX #2" above for why this no longer sizes any layer.
        self.quantum_output_size = quantum_output_size
        self.hidden_feature = hidden_feature
        self.hidden_readout = hidden_readout
        self.n_photons_per_pass = n_photons_per_pass
        self.n_encoding_layers = n_encoding_layers
        self.frequency_mode = frequency_mode
        self.pde = pde

        # Output feature_size * n_encoding_layers values, one independent
        # feature_size-dim chunk per re-upload -- mirrors
        # `MerlinTunableEntropy.feature_map`'s `total_input_size`.
        total_input_size = self.feature_size * self.n_encoding_layers
        self.feature_map = nn.Sequential(
            nn.Linear(2, self.hidden_feature),
            nn.Tanh(),
            nn.Linear(self.hidden_feature, total_input_size),
        )

        # Fixed (non-trainable) frequency VECTORS, shape (n_omega, feature_size)
        # -- registered as a buffer, not a parameter, so it moves with
        # .to(device) but never trains and never counts toward the trainable
        # parameter budget. See "IMPORTANT FIX #3" for why the default is the
        # real difference set rather than the axis-aligned approximation.
        if frequency_mode == "full":
            omega = omega_difference_set(feature_size, n_photons_per_pass)
        elif frequency_mode == "axis":
            omega = axis_aligned_frequencies(feature_size, n_photons_per_pass)
        else:
            raise ValueError(
                f"frequency_mode must be 'full' or 'axis', got {frequency_mode!r}"
            )
        self.register_buffer("omega", torch.tensor(omega, dtype=torch.float32))
        self.n_omega = len(omega)

        # cos+sin per frequency vector per re-upload chunk.
        encoding_dim = 2 * self.n_omega * self.n_encoding_layers
        # Strictly LINEAR combination over the fixed encoding, mapping
        # DIRECTLY to hidden_readout width -- collapsed with what used to
        # be readout's own first Linear layer (see "IMPORTANT FIX #2").
        self.combine = nn.Linear(encoding_dim, self.hidden_readout)

        self.readout = nn.Sequential(
            nn.Tanh(),
            nn.Linear(self.hidden_readout, 2),  # [q_u, ux_hat]
        )

    def _fourier_features(self, z: torch.Tensor) -> torch.Tensor:
        """z: (batch, feature_size * n_encoding_layers) -> reshaped into
        `n_encoding_layers` independent (batch, feature_size) chunks, each
        given the SAME fixed cos/sin encoding over the frequency VECTORS in
        `self.omega` (a genuine multivariate inner product omega . z, not a
        per-axis scalar product), then concatenated ->
        (batch, 2 * n_omega * n_encoding_layers)."""
        batch = z.shape[0]
        z = z.view(batch, self.n_encoding_layers, self.feature_size)
        # (batch, L, feature_size) x (n_omega, feature_size) -> (batch, L, n_omega)
        angle = torch.einsum("blf,of->blo", z, self.omega)
        feats = torch.cat([torch.cos(angle), torch.sin(angle)], dim=-1)  # (batch, L, 2*n_omega)
        return feats.flatten(start_dim=1)

    def forward(self, xt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            xt: tensor of shape (batch, 2) with columns [x, t]

        Returns:
            u: solution field, shape (batch, 1)
            ux_hat: x-derivative approximation, shape (batch, 1)
        """
        x = xt[:, 0:1]
        t = xt[:, 1:2]
        z = self.feature_map(xt)
        e = self._fourier_features(z)
        p = self.combine(e)
        out = self.readout(p)

        u = out[:, 0:1]
        ux_hat = out[:, 1:2]

        u = (self.pde.x_min + x) * (self.pde.x_max - x) * u

        return u, ux_hat


class ModulusFourierHeatQPINN(nn.Module):
    """
    The PARAMETER-MATCHED spectral-coverage control, and the sharper of the
    two: a purely classical model that reproduces the photonic layer's full
    471-vector frequency support Omega using a comparable number of trainable
    parameters, by keeping the one structural feature that makes the quantum
    layer parameter-efficient -- the MODULUS SQUARED.

    Why this is the right control. The photonic layer is parameter-efficient
    not because it is quantum, but because it squares a SPARSE comb:
        c_k(z) = sum_{n in F} B_{k,n} A_n e^{i n . z}        (|F| = D = 35)
        p_k(z) = |c_k(z)|^2 = sum_{omega in Omega} gamma_{k,omega} e^{i omega . z}
    Squaring a trig polynomial supported on 35 frequency vectors
    automatically produces every pairwise DIFFERENCE n - n', i.e. all 471
    vectors of Omega -- this is just autocorrelation (Wiener-Khinchin), and
    it is entirely classical. The coefficients gamma are likewise not free:
    they are bilinearly tied to the same 35 amplitudes, which is how ~40
    trainable phases can steer ~35*471 coefficients. What the photonics
    contributes is the particular CONSTRAINT set (unitarity, sum_k p_k = 1,
    which combs are physically realisable) -- not the
    frequency-richness-per-parameter.

    This class tests exactly that claim. It keeps the sparse comb over the
    35 Fock occupation vectors and the modulus-squared readout, but makes
    the amplitudes FREE trainable complex numbers with no unitarity
    constraint and no quantum circuit:
        c^{(l)}_k(z) = sum_{n in F} M^{(l)}_{k,n} e^{i n . z^{(l)}},
        p^{(l)}_k    = |c^{(l)}_k|^2
    with one independent M per re-upload chunk l (so parameter count grows
    with L, as the quantum arm's does when it adds mesh layers), and
    `n_channels` output channels per chunk. Parameters: 2 * L * n_channels *
    D, e.g. 2*1*4*35 = 280 at L=1 with n_channels=4 -- the same order as the
    quantum arm's 310, versus ~1.9k for `FourierFeaturesHeatQPINN`'s
    full-Omega LINEAR encoding.

    Reading the result, against `FourierFeaturesHeatQPINN(frequency_mode=
    "full")` (the capacity upper bound) and the quantum arm:
      - If this arm matches quantum at matched parameters, the advantage is
        fully explained by "sparse comb + modulus squared" -- a classical
        structure -- and nothing quantum is required.
      - If the full-Omega LINEAR arm (far more parameters, same frequency
        support, free coefficients) fails while THIS one succeeds, the
        operative ingredient is the bilinear coefficient tying from
        squaring, not spectral support alone -- which would be a sharper
        and more interesting claim than the paper currently makes.
      - If both classical arms fail while quantum succeeds, that is real
        evidence the unitarity-constrained parameterisation of
        gamma_{k,omega} is doing work no free classical parameterisation on
        the same lattice reproduces.
    """

    def __init__(self, feature_size: int = 4, quantum_output_size: int = 20,
                 hidden_feature: int = 16, hidden_readout: int = 4,
                 n_photons_per_pass: int = 3, n_encoding_layers: int = 1,
                 n_channels: Optional[int] = None, normalize: bool = False,
                 pde=None):
        super().__init__()
        self.feature_size = feature_size
        # Interface/checkpoint-schema compatibility only; does not size a layer.
        self.quantum_output_size = quantum_output_size
        self.hidden_feature = hidden_feature
        self.hidden_readout = hidden_readout
        self.n_photons_per_pass = n_photons_per_pass
        self.n_encoding_layers = n_encoding_layers
        self.normalize = normalize
        self.pde = pde

        total_input_size = self.feature_size * self.n_encoding_layers
        self.feature_map = nn.Sequential(
            nn.Linear(2, self.hidden_feature),
            nn.Tanh(),
            nn.Linear(self.hidden_feature, total_input_size),
        )

        # The FIXED sparse comb: the D Fock occupation vectors n_{1:F} that
        # appear in the AMPLITUDE (not the difference set -- squaring
        # generates that for free). Buffer, not a parameter.
        fock = fock_occupations(feature_size, n_photons_per_pass)
        self.register_buffer("fock", torch.tensor(fock, dtype=torch.float32))
        self.n_fock = len(fock)

        # IMPORTANT FIX #4 (found when this control failed at F=5 while the
        # quantum arm succeeded): `n_channels` defaults to D = self.n_fock,
        # NOT to `hidden_readout`. The original default of `hidden_readout`
        # (4) silently capped this control's expressivity far below the
        # quantum arm's, in a way that has nothing to do with spectral
        # support. Each channel here is |sum_n M_{k,n} e^{i n.z}|^2 -- a PSD
        # RANK-1 quadratic form in the comb. The quantum arm's readout
        # computes sum_k W_k p_k(z) with W_k of MIXED SIGN, i.e. a general
        # Hermitian form of rank up to D (35 at F=4, 56 at F=5), including
        # indefinite ones. Capping this arm at 4 channels caps it at rank 4:
        # same frequency support, same modulus structure, but a much smaller
        # cone of representable quadratic forms -- and the gap widens with F,
        # which is exactly where the failure showed up. Defaulting to D makes
        # the two arms rank-comparable.
        #
        # Note the parameter cost, and that it is itself the interesting
        # result: free amplitudes need 2*L*D*D reals (2*56*56 = 6272 at F=5)
        # to span what the quantum arm spans with 2F(F+1) = 60 phases, because
        # ONE unitary supplies all D channels from a shared parameter set,
        # whereas an untied parameterisation pays O(D^2). That efficiency is a
        # property of the UNITARY (equivalently, Givens/MZI-mesh)
        # parameterisation -- classically available, not quantum-mechanical --
        # and is worth stating as such rather than as a quantum advantage.
        self.n_channels = self.n_fock if n_channels is None else n_channels

        # Free trainable complex amplitudes M^{(l)}_{k,n}, one independent
        # set per re-upload chunk. Scaled by 1/sqrt(n_fock) so |c|^2 starts
        # O(1) rather than growing with the comb size.
        scale = 1.0 / math.sqrt(self.n_fock)
        shape = (self.n_encoding_layers, self.n_channels, self.n_fock)
        self.M_real = nn.Parameter(torch.randn(*shape) * scale)
        self.M_imag = nn.Parameter(torch.randn(*shape) * scale)

        self.readout = nn.Sequential(
            nn.Linear(self.n_encoding_layers * self.n_channels, self.hidden_readout),
            nn.Tanh(),
            nn.Linear(self.hidden_readout, 2),  # [q_u, ux_hat]
        )

    def _modulus_features(self, z: torch.Tensor) -> torch.Tensor:
        """z: (batch, feature_size * n_encoding_layers) -> per-chunk
        |sum_n M_{k,n} e^{i n . z}|^2 -> (batch, n_encoding_layers*n_channels).
        Mirrors the photonic layer's amplitude-then-square structure exactly,
        with free (non-unitary) classical amplitudes."""
        batch = z.shape[0]
        z = z.view(batch, self.n_encoding_layers, self.feature_size)
        # (batch, L, feature_size) x (n_fock, feature_size) -> (batch, L, n_fock)
        angle = torch.einsum("blf,sf->bls", z, self.fock)
        cos_a, sin_a = torch.cos(angle), torch.sin(angle)
        # c = sum_n (Mr + i Mi)(cos + i sin)
        re = (torch.einsum("bls,lks->blk", cos_a, self.M_real)
              - torch.einsum("bls,lks->blk", sin_a, self.M_imag))
        im = (torch.einsum("bls,lks->blk", sin_a, self.M_real)
              + torch.einsum("bls,lks->blk", cos_a, self.M_imag))
        p = re ** 2 + im ** 2                      # (batch, L, n_channels)
        if self.normalize:
            # Optional Born-rule-style normalisation: make each chunk's
            # channel vector sum to 1, as the quantum arm's probabilities do
            # by unitarity. Isolates "does the simplex constraint matter?"
            # from "does the frequency support / rank matter?".
            p = p / (p.sum(dim=-1, keepdim=True) + 1e-12)
        return p.flatten(start_dim=1)

    def forward(self, xt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            xt: tensor of shape (batch, 2) with columns [x, t]

        Returns:
            u: solution field, shape (batch, 1)
            ux_hat: x-derivative approximation, shape (batch, 1)
        """
        x = xt[:, 0:1]
        t = xt[:, 1:2]
        z = self.feature_map(xt)
        p = self._modulus_features(z)
        out = self.readout(p)

        u = out[:, 0:1]
        ux_hat = out[:, 1:2]

        u = (self.pde.x_min + x) * (self.pde.x_max - x) * u

        return u, ux_hat


class UnitaryFourierHeatQPINN(nn.Module):
    """
    The GIVENS / UNITARY control: identical to `ModulusFourierHeatQPINN`
    except that the amplitude matrix is CONSTRAINED to factor through a
    unitary, M = B diag(A), instead of being a free complex matrix.

    This is the decisive positive experiment. The transplant diagnostic
    (`modulus_model_from_quantum`) established that the quantum arm's
    trained solution lies INSIDE the free-M model's function class -- same
    rel_l2 to seven significant figures -- yet the free-M model never finds
    it from random init. That rules out spectral support and capacity as
    explanations and leaves optimisation geometry. The natural suspect is
    the unitary parameterisation itself, which:
      - bounds every amplitude (no scale blow-up),
      - makes sum_k p_k = ||A||^2 CONSTANT in z (the Born-rule simplex
        structure, here automatic: |e^{i n.z}| = 1 and B is norm-preserving,
        so the channel vector's total mass cannot depend on the input),
      - charts the parameters on a compact manifold of rotations rather
        than an unbounded linear space.
    This class tests whether that bias alone is enough, with NO quantum
    circuit anywhere.

    What this control does and does NOT reproduce. The photonic layer's
    B_{k,n} = <k|W_R|n> is not a free D x D unitary: it is the symmetric-
    power (permanent) action of an m x m mesh, so m(m-1) = F(F+1) = 20
    phases generate the whole 35 x 35 unitary. Reproducing THAT exactly
    means computing permanents, i.e. reimplementing the photonic simulation
    -- which would not be an independent control. So B here is a FREE
    unitary, parameterised by D(D-1)/2 rotation angles. Consequences:
      - It isolates UNITARITY as the variable, holding spectral support and
        the modulus-squared structure fixed.
      - It does NOT reproduce the photonic layer's parameter TYING. At
        F=4 this arm has ~935 parameters against quantum's 310 and the
        free-M model's 2720 -- a genuine middle point.
    Reading the outcome:
      - Trains successfully => unitarity is the operative inductive bias.
        The advantage is a classical, reproducible architectural effect,
        and the paper gets a constructive positive result rather than only
        falsified hypotheses.
      - Still fails => unitarity alone is insufficient and the specific
        bosonic tying (few phases driving all D channels) is doing the
        work -- a sharper and more surprising claim, worth stating as such.

    Parameterisation. B is built by the Cayley transform of a skew-symmetric
    matrix, B = (I + S)^{-1}(I - S) with S = -S^T, which is exactly the
    manifold an MZI/Givens mesh charts (a product of plane rotations) but
    costs one linear solve instead of a Python loop over D(D-1)/2 rotations
    -- numerically and computationally far better behaved, and differentiable
    end-to-end. Only the strictly-lower-triangular entries are stored, so the
    reported parameter count is the honest D(D-1)/2 rather than D^2. B is
    real orthogonal rather than complex unitary; amplitudes remain complex
    through A and the phase encoding, so p_k = |c_k|^2 still spans the full
    difference set Omega exactly as before.
    """

    def __init__(self, feature_size: int = 4, quantum_output_size: int = 20,
                 hidden_feature: int = 16, hidden_readout: int = 4,
                 n_photons_per_pass: int = 3, n_encoding_layers: int = 1,
                 n_channels: Optional[int] = None, normalize: bool = True,
                 pde=None):
        super().__init__()
        self.feature_size = feature_size
        self.quantum_output_size = quantum_output_size   # schema compat only
        self.hidden_feature = hidden_feature
        self.hidden_readout = hidden_readout
        self.n_photons_per_pass = n_photons_per_pass
        self.n_encoding_layers = n_encoding_layers
        self.normalize = normalize
        self.pde = pde

        total_input_size = self.feature_size * self.n_encoding_layers
        self.feature_map = nn.Sequential(
            nn.Linear(2, self.hidden_feature),
            nn.Tanh(),
            nn.Linear(self.hidden_feature, total_input_size),
        )

        fock = fock_occupations(feature_size, n_photons_per_pass)
        self.register_buffer("fock", torch.tensor(fock, dtype=torch.float32))
        self.n_fock = len(fock)
        # B is square on the comb, mirroring the photonic layer's D x D.
        self.n_channels = self.n_fock if n_channels is None else n_channels
        if self.n_channels != self.n_fock:
            raise ValueError(
                "UnitaryFourierHeatQPINN requires n_channels == n_fock (B is "
                f"square), got {self.n_channels} vs {self.n_fock}."
            )
        D = self.n_channels

        # Trainable complex input amplitudes A (the analogue of
        # A_n = <n|W_L|psi_in>), one independent vector per re-upload chunk.
        scale = 1.0 / math.sqrt(self.n_fock)
        self.A_real = nn.Parameter(torch.randn(self.n_encoding_layers, self.n_fock) * scale)
        self.A_imag = nn.Parameter(torch.randn(self.n_encoding_layers, self.n_fock) * scale)

        # Skew-symmetric generator, stored as its strictly-lower triangle so
        # the parameter count is the true D(D-1)/2 of the rotation manifold.
        idx = torch.tril_indices(D, D, offset=-1)
        self.register_buffer("tril_idx", idx)
        self.skew = nn.Parameter(torch.randn(self.n_encoding_layers, idx.shape[1]) * 0.01)

        self.readout = nn.Sequential(
            nn.Linear(self.n_encoding_layers * self.n_channels, self.hidden_readout),
            nn.Tanh(),
            nn.Linear(self.hidden_readout, 2),  # [q_u, ux_hat]
        )

    def orthogonal(self) -> torch.Tensor:
        """Cayley transform of the skew-symmetric generator -> (L, D, D)
        orthogonal matrices. Same manifold as a Givens/MZI mesh."""
        L, D = self.n_encoding_layers, self.n_channels
        S = torch.zeros(L, D, D, device=self.skew.device, dtype=self.skew.dtype)
        S[:, self.tril_idx[0], self.tril_idx[1]] = self.skew
        S = S - S.transpose(-1, -2)
        eye = torch.eye(D, device=S.device, dtype=S.dtype).expand(L, D, D)
        return torch.linalg.solve(eye + S, eye - S)

    def _unitary_features(self, z: torch.Tensor) -> torch.Tensor:
        """z: (batch, feature_size * n_encoding_layers) -> per-chunk
        p_k = |sum_n B_{k,n} A_n e^{i n . z}|^2 -> (batch, L*n_channels)."""
        batch = z.shape[0]
        z = z.view(batch, self.n_encoding_layers, self.feature_size)
        angle = torch.einsum("blf,sf->bls", z, self.fock)
        cos_a, sin_a = torch.cos(angle), torch.sin(angle)

        A_re, A_im = self.A_real, self.A_imag
        if self.normalize:
            # Unit-norm input "state" => sum_k p_k == 1 exactly, since B is
            # orthogonal and the encoding is a pure phase.
            norm = torch.sqrt((A_re ** 2 + A_im ** 2).sum(dim=-1, keepdim=True)) + 1e-12
            A_re, A_im = A_re / norm, A_im / norm

        # v_n = A_n e^{i n.z}, broadcasting A over the batch.
        v_re = A_re.unsqueeze(0) * cos_a - A_im.unsqueeze(0) * sin_a
        v_im = A_re.unsqueeze(0) * sin_a + A_im.unsqueeze(0) * cos_a

        B = self.orthogonal()                      # (L, D, S) with D == S
        c_re = torch.einsum("lks,bls->blk", B, v_re)
        c_im = torch.einsum("lks,bls->blk", B, v_im)
        p = c_re ** 2 + c_im ** 2                  # (batch, L, n_channels)
        return p.flatten(start_dim=1)

    def forward(self, xt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            xt: tensor of shape (batch, 2) with columns [x, t]

        Returns:
            u: solution field, shape (batch, 1)
            ux_hat: x-derivative approximation, shape (batch, 1)
        """
        x = xt[:, 0:1]
        t = xt[:, 1:2]
        z = self.feature_map(xt)
        p = self._unitary_features(z)
        out = self.readout(p)

        u = out[:, 0:1]
        ux_hat = out[:, 1:2]

        u = (self.pde.x_min + x) * (self.pde.x_max - x) * u

        return u, ux_hat


@torch.no_grad()
def modulus_model_from_quantum(quantum_model, pde=None, dtype=torch.float32,
                                verbose: bool = True):
    """
    TRANSPLANT: build a `ModulusFourierHeatQPINN` that reproduces a TRAINED
    `MerlinTunableEntropy` (at n_encoding_layers=1) EXACTLY, by copying its
    feature map and readout verbatim and recovering its amplitude
    coefficients M_{k,n} = B_{k,n} A_n directly.

    Why this exists. At L=1 and n_channels=D, the modulus model strictly
    CONTAINS the quantum model's function class: quantum computes
    p_k = |sum_n B_{k,n} A_n e^{i n.z}|^2, the modulus model computes
    p_k = |sum_n M_{k,n} e^{i n.z}|^2 with M free, and setting
    M = B diag(A) makes them identical (the quantum arm is exactly the
    subset where M factors through a unitary). So if the modulus model
    trained from random init FAILS where the quantum model SUCCEEDS, the
    difference cannot be representational -- it must be optimisation. This
    function settles that: transplant the trained solution and evaluate.

      - If the transplanted model matches the quantum model's rel_l2, the
        target IS inside the free-M class, reachable by construction but not
        by Adam from random init => the unitary parameterisation is acting
        as a preconditioner / implicit regulariser, and the advantage is a
        TRAINABILITY effect, not a capacity or spectral-support one.
      - If it does NOT match, the correspondence above is wrong somewhere
        and should be debugged before drawing any conclusion.

    How M is recovered. Rather than reaching into MerLin's internals for
    W_L/W_R (fragile, version-dependent), we exploit the fact that the
    layer's own output IS the quantity we want: c_k(z) is a trigonometric
    polynomial supported exactly on the D occupation vectors n_{1:F} with
    sum(n) <= N. Sampling z on a uniform (N+1)-point grid per axis and
    applying an inverse DFT recovers M exactly -- every frequency component
    n_j lies in {0..N} = {0..G-1} with G = N+1, so there is NO aliasing.
    Cost is G^F forward evaluations (256 at F=4, 1024 at F=5): trivial.
    `reconstruction_error` in the returned dict is the residual of that
    inversion and should be ~1e-6 or below; anything larger means the comb
    assumption is violated and the result should not be trusted.

    Returns:
        (model, info) where `info` has keys "reconstruction_error",
        "n_fock", "n_channels", "grid_points".
    """
    if getattr(quantum_model, "n_encoding_layers", 1) != 1:
        raise ValueError(
            "transplant is only exact at n_encoding_layers=1 -- for L>1 the "
            "quantum arm interleaves encodings with separate meshes, which "
            "produces the L-fold sumset rather than a single comb, and the "
            "containment argument above does not apply."
        )

    F = quantum_model.feature_size
    n_modes = F + 1
    N = int(math.ceil(n_modes / 2))
    S = fock_occupations(F, N)
    G = N + 1

    # Uniform grid over [0, 2pi)^F -- exact inversion for frequencies 0..N.
    axis = torch.arange(G, dtype=torch.float64) * (2.0 * math.pi / G)
    mesh = torch.meshgrid(*([axis] * F), indexing="ij")
    Z = torch.stack([m.reshape(-1) for m in mesh], dim=1)      # (G^F, F)
    n_points = Z.shape[0]

    device = next(quantum_model.parameters()).device
    c = quantum_model.quantum(Z.to(device=device, dtype=dtype))  # (G^F, D) complex
    c = c.detach().to(torch.complex128).cpu()

    fock_t = torch.tensor(S, dtype=torch.float64)               # (n_fock, F)
    # e^{-i n.z}: (G^F, n_fock)
    basis = torch.exp(-1j * (Z @ fock_t.T).to(torch.complex128))
    M = (c.T @ basis) / n_points                                # (D, n_fock)

    # Verify the comb assumption by reconstructing c from M.
    c_hat = M @ basis.conj().T                                  # (D, G^F)
    denom = c.abs().max().item() or 1.0
    reconstruction_error = float((c_hat.T - c).abs().max().item() / denom)

    n_channels = M.shape[0]
    if n_channels != len(S):
        raise ValueError(
            f"amplitude dimension D={n_channels} does not match the number of "
            f"occupation vectors {len(S)} -- the comb assumption is violated."
        )

    model = ModulusFourierHeatQPINN(
        feature_size=F,
        hidden_feature=quantum_model.hidden_feature,
        hidden_readout=quantum_model.hidden_readout,
        n_photons_per_pass=N,
        n_encoding_layers=1,
        n_channels=n_channels,
        normalize=False,
        pde=pde if pde is not None else quantum_model.pde,
    )
    model.M_real.copy_(M.real.to(dtype).unsqueeze(0))
    model.M_imag.copy_(M.imag.to(dtype).unsqueeze(0))
    # feature_map and readout are structurally identical between the two
    # classes (same widths, same activations), so they copy verbatim.
    model.feature_map.load_state_dict(quantum_model.feature_map.state_dict())
    model.readout.load_state_dict(quantum_model.readout.state_dict())
    model = model.to(device=device, dtype=dtype)

    info = {
        "reconstruction_error": reconstruction_error,
        "n_fock": len(S),
        "n_channels": n_channels,
        "grid_points": n_points,
    }
    if verbose:
        status = "OK" if reconstruction_error < 1e-4 else "** SUSPECT **"
        print(f"[transplant] F={F} N={N} D={n_channels} grid={n_points} "
              f"amplitude reconstruction rel. error={reconstruction_error:.2e} {status}")
    return model, info


@torch.no_grad()
def verify_transplant(quantum_model, transplanted, n_probe: int = 512,
                       dtype=torch.float32, verbose: bool = True) -> float:
    """Sanity check for `modulus_model_from_quantum`: push the same random
    (x,t) batch through both models and return the max relative discrepancy
    in `u`. Should be ~1e-5 or below. A large value means the transplant is
    wrong and any conclusion drawn from it is invalid."""
    device = next(quantum_model.parameters()).device
    pde = quantum_model.pde
    x = torch.rand(n_probe, 1, device=device, dtype=dtype)
    t = torch.rand(n_probe, 1, device=device, dtype=dtype)
    x = pde.x_min + x * (pde.x_max - pde.x_min)
    t = pde.t_min + t * (pde.t_max - pde.t_min)
    xt = torch.cat([x, t], dim=1)

    u_q = quantum_model(xt)[0]
    u_m = transplanted(xt)[0]
    denom = u_q.abs().max().item() or 1.0
    err = float((u_q - u_m).abs().max().item() / denom)
    if verbose:
        status = "OK" if err < 1e-4 else "** MISMATCH **"
        print(f"[transplant] max relative |u| discrepancy over {n_probe} "
              f"points: {err:.2e} {status}")
    return err


class MerlinTunableSimple(nn.Module):
    """
    QPINN model using MerLin quantum layer for heat equation.
    Enforces homogeneous Dirichlet BCs through hard constraint: u = x(1-x)*q_u
    """
    
    def __init__(self, feature_size : int = 4, hidden_feature : int = 16, hidden_readout : int = 4, trainable_outer : bool = True, pde = None):
        super().__init__()
        self.feature_size = feature_size
        self.hidden_feature = hidden_feature
        self.hidden_readout = hidden_readout
        self.trainable_outer = trainable_outer
        self.pde = pde
        # Classical feature map: (x, t) -> 2 features
        self.feature_map = nn.Sequential(
            nn.Linear(2, self.hidden_feature),
            nn.Tanh(),
            nn.Linear(self.hidden_feature, self.feature_size), 
        )
        
        input_size = self.feature_size

        n_modes = input_size + 1
        if n_modes > 20:
            raise ValueError(
                "Input size too large for the simple layer construction. For large inputs (with larger size than 19), please use the CircuitBuilder. Here is a quick tutorial on how to use it: https://merlinquantum.ai/quickstart/first_quantum_layer.html#circuitbuilder-walkthrough"
            )
        if input_size < 1:
            raise ValueError(f"input_size must be at least 1, got {input_size}")

        input_state = n_modes * [0]
        for i in range(n_modes):
            if i % 2 == 0:
                input_state[i] = 1


        n_photons = sum(input_state)
        input_state = pcvl.BasicState(input_state)


        builder = ML.CircuitBuilder(n_modes=n_modes)
        # Trainable entangling layer before encoding
        builder.add_entangling_layer(trainable=True, name="LI_simple", trainable_outer=self.trainable_outer)

        # Angle encoding
        builder.add_angle_encoding(
            modes=list(range(input_size)),
            name="input",
            subset_combinations=False,
        )

        # Trainable entangling layer after encoding
        builder.add_entangling_layer(trainable=True, name="RI_simple", trainable_outer=self.trainable_outer)

        self.quantum = ML.QuantumLayer(
                input_size=self.feature_size,
                builder=builder,
                input_state=StateVector.from_basic_state(input_state),
                measurement_strategy=ML.MeasurementStrategy.amplitudes(computation_space='fock')
            )

        self.readout = nn.Sequential(
            nn.Linear(int(scipy.special.comb(n_modes+n_photons-1, n_photons)), self.hidden_readout),
            nn.Tanh(),
            nn.Linear(self.hidden_readout, 2),
        )
    
    def forward(self, xt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            xt: tensor of shape (batch, 2) with columns [x, t]
        
        Returns:
            u: solution field, shape (batch, 1)
            ux_hat: x-derivative approximation, shape (batch, 1)
        """
        x = xt[:, 0:1]
        t = xt[:, 1:2]
        z = self.feature_map(xt)
        q_amplitudes = self.quantum(z)
        # 3. Derive probabilities for classical readout: P(x) = |c|^2 -> (batch, 20) real
        q_probs = torch.abs(q_amplitudes) ** 2

        # 4. Classical Readout Pipeline
        out = self.readout(q_probs)
        
        q_u = out[:, 0:1]
        ux_hat = out[:, 1:2]
        
#        if self.config.use_hard_bc:
#            # Enforce homogeneous Dirichlet boundary conditions: u(0,t)=u(1,t)=0
#            u = x * (1.0 - x) * q_u
#        else:
#            u = q_u

        f_x = torch.zeros_like(x)
        if hasattr(self.pde, "initial_condition"):
            f_x = self.pde.initial_condition(x)

        envelope = 1 - torch.cos(0.5 * math.pi * t)
        u = (self.pde.x_min + x) * (self.pde.x_max - x) * q_u #(envelope * q_u) + f_x

        #u = q_u
        #u = x * (1.0 - x) * q_u
        
        return u, ux_hat


class BuilderMerlinHeatQPINN(nn.Module):
    """
    QPINN model using a custom MerLin CircuitBuilder.
    Incorporates data re-uploading and higher photon counts.
    """

    def __init__(self, feature_size : int = 4, quantum_output_size : int = 4, hidden_feature : int = 16, hidden_readout : int = 16):
        super().__init__()
        self.feature_size = feature_size
        self.quantum_output_size = quantum_output_size
        self.hidden_feature = hidden_feature
        self.hidden_readout = hidden_readout

        # Classical feature map: (x, t) -> 2 features
        self.feature_map = nn.Sequential(
            nn.Linear(2, self.hidden_feature),
            nn.Tanh(),
            nn.Linear(self.hidden_feature, self.feature_size),
        )

        try:
            import merlin as ML

            builder = ML.CircuitBuilder(n_modes=4)
            for _ in range(self.feature_size // 2):
                builder.add_entangling_layer(trainable=True, model="mzi")
                builder.add_angle_encoding(modes=[0, 1])  # Allocates input slots 0, 1
            builder.add_entangling_layer(trainable=True, model="mzi")

            pcvl_circuit = builder.to_pcvl_circuit()
            num_modes = pcvl_circuit.m
            critical_path = max(pcvl_circuit.depths())

            print(f'Builder circuit has {num_modes} modes and critical path length {critical_path}.')

            self.quantum = ML.QuantumLayer(
                # The circuit expects 4 inputs, so multiply feature_size by the number of uploads (2)
                input_size=self.feature_size,
                builder=builder,
                input_state=StateVector.from_basic_state([1, 1, 1, 0]),
                measurement_strategy=ML.MeasurementStrategy.amplitudes(computation_space='fock')
            )
        except ImportError as e:
            raise ImportError(
                "Could not import MerLin. Install with: pip install merlinquantum"
            ) from e

        self.readout = nn.Sequential(
            nn.Linear(self.quantum_output_size, self.hidden_readout),
            nn.Tanh(),
            nn.Linear(self.hidden_readout, 2),
        )
        # Precompute the index mapping for the partial trace (Subsystem A: modes 0,1)
        self._setup_partial_trace(photons=3, modes=4, subsys_A_size=2)

    def _setup_partial_trace(self, photons: int, modes: int, subsys_A_size: int):
        """Prepares the index map for tracing out Subsystem B."""
        states = generate_fock_basis(photons, modes)

        # Split each 4-mode state into A (modes 0,1) and B (modes 2,3)
        A_states = [s[:subsys_A_size] for s in states]
        B_states = [s[subsys_A_size:] for s in states]

        # Extract unique states for both subsystems while preserving order
        self.unique_A = list(dict.fromkeys(A_states))
        self.unique_B = list(dict.fromkeys(B_states))
        self.A_dim = len(self.unique_A)

        # Create a mapping: state_map[A_idx, B_idx] = flat_state_idx
        # If an (A, B) combination exceeds the photon limit, it remains -1
        state_map = torch.full((self.A_dim, len(self.unique_B)), -1, dtype=torch.long)

        for idx, (a, b) in enumerate(zip(A_states, B_states)):
            a_idx = self.unique_A.index(a)
            b_idx = self.unique_B.index(b)
            state_map[a_idx, b_idx] = idx

        # Register as a non-trainable buffer so it moves properly with model.to(device)
        self.register_buffer('state_map', state_map)

    def calculate_bipartite_entropy(self, amplitudes: torch.Tensor) -> torch.Tensor:
        """
        Calculates the Von Neumann entanglement entropy of Subsystem A.
        amplitudes shape: (batch_size, 20) complex
        """
        batch_size = amplitudes.size(0)

        # 1. Initialize the reduced density matrix rho_A
        rho_A = torch.zeros(
            (batch_size, self.A_dim, self.A_dim),
            dtype=amplitudes.dtype,
            device=amplitudes.device
        )

        # 2. Perform the Partial Trace over Subsystem B
        # rho_A[i,j] = sum_k (c_{i,k} * c_{j,k}^*)
        for i in range(self.A_dim):
            for j in range(self.A_dim):
                # Find indices in subsystem B that are valid for BOTH A_i and A_j
                valid_b = (self.state_map[i] != -1) & (self.state_map[j] != -1)
                idx_ik = self.state_map[i][valid_b]
                idx_jk = self.state_map[j][valid_b]

                if len(idx_ik) > 0:
                    rho_A[:, i, j] = torch.sum(
                        amplitudes[:, idx_ik] * torch.conj(amplitudes[:, idx_jk]),
                        dim=1
                    )

        # 3. Calculate Eigenvalues
        # Using eigvalsh because rho_A is a Hermitian matrix
        eigvals = torch.linalg.eigvalsh(rho_A)

        # 4. Calculate Von Neumann Entropy: S = -sum(lambda * ln(lambda))
        # Clamp to avoid log(0) and small negative eigenvalues from float imprecision
        eigvals = torch.clamp(eigvals, min=1e-12, max=1.0)
        entropy = -torch.sum(eigvals * torch.log(eigvals), dim=-1)

        return entropy

    def forward(self, xt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = xt[:, 0:1]
        z = self.feature_map(xt)

        # Duplicate the features for data re-uploading
        z_reuploaded = torch.cat([z, z], dim=1)

        # 1. Quantum Pass -> Returns complex amplitudes (batch, 20)
        q_amplitudes = self.quantum(z_reuploaded)

        # 2. Derive Entanglement Entropy
        entropy = self.calculate_bipartite_entropy(q_amplitudes)

        # 3. Derive probabilities for classical readout: P(x) = |c|^2 -> (batch, 20) real
        q_probs = torch.abs(q_amplitudes) ** 2

        # 4. Classical Readout Pipeline
        out = self.readout(q_probs)
        q_u = out[:, 0:1]
        ux_hat = out[:, 1:2]

#        if self.config.use_hard_bc:
#            u = x * (1.0 - x) * q_u
#        else:
        u = q_u

        return u, ux_hat, entropy.mean()

class MerlinTunableEntropy(nn.Module):
    """
    Similar to MerlinHeatQpinn but has architecture that allows same # weights as classical model
    """
    
    def __init__(self, feature_size : int = 4, hidden_feature : int = 16, hidden_readout : int = 4, trainable_outer : bool = True, entropy_modes_A : Optional[List[int]] = None, n_encoding_layers : int = 1, pde = None):
        """
        Args:
            entropy_modes_A: mode indices forming subsystem A for the entanglement
                entropy measurement; the remaining modes form subsystem B. Defaults
                to the encoded modes (0..feature_size-1), i.e. subsystem B is the
                single extra mode that carries no direct classical input.
            n_encoding_layers: number of angle-encoding passes over the same
                `feature_size` modes (data re-uploading), each fed a fresh chunk
                of features and interleaved with a trainable entangling layer.
                This increases expressivity in (x, t) without growing n_modes /
                the Fock-basis Hilbert space, unlike raising `feature_size`.
                Default 1 reproduces the original single-encoding circuit.
        """
        super().__init__()
        self.feature_size = feature_size
        self.hidden_feature = hidden_feature
        self.hidden_readout = hidden_readout
        self.trainable_outer = trainable_outer
        self.n_encoding_layers = n_encoding_layers

        self.pde = pde

        input_size = self.feature_size
        total_input_size = input_size * n_encoding_layers

        # Classical feature map: (x, t) -> total_input_size features, split into
        # one fresh feature_size-wide chunk per encoding pass.
        self.feature_map = nn.Sequential(
            nn.Linear(2, self.hidden_feature),
            nn.Tanh(),
            nn.Linear(self.hidden_feature, total_input_size),
        )

        n_modes = input_size + 1
        if n_modes > 20:
            raise ValueError(
                "Input size too large for the simple layer construction. For large inputs (with larger size than 19), please use the CircuitBuilder. Here is a quick tutorial on how to use it: https://merlinquantum.ai/quickstart/first_quantum_layer.html#circuitbuilder-walkthrough"
            )
        if input_size < 1:
            raise ValueError(f"input_size must be at least 1, got {input_size}")
        if n_encoding_layers < 1:
            raise ValueError(f"n_encoding_layers must be at least 1, got {n_encoding_layers}")

        input_state = n_modes * [0]
        for i in range(n_modes):
            if i % 2 == 0:
                input_state[i] = 1


        n_photons = sum(input_state)
        input_state = pcvl.BasicState(input_state)


        builder = ML.CircuitBuilder(n_modes=n_modes)
        # Trainable entangling layer before the first encoding pass
        builder.add_entangling_layer(trainable=True, name="LI_simple", trainable_outer=self.trainable_outer)

        for i in range(n_encoding_layers):
            # Angle encoding: re-uploads onto the same `input_size` modes each
            # pass, each consuming a fresh chunk of the feature map's output.
            # Names must not collapse to the same stem once merlin strips
            # trailing digits/underscores (CircuitBuilder._deduce_prefix), or
            # separate passes get merged into one undersized parameter family
            # (e.g. "input" + digit-suffixed "input_2X" both reduce to "input").
            # A trailing letter avoids that collision.
            layer_name = "input" if n_encoding_layers == 1 else f"input{chr(ord('a') + i)}"
            builder.add_angle_encoding(
                modes=list(range(input_size)),
                name=layer_name,
                subset_combinations=False,
            )
            # Trainable entangling layer after each encoding pass
            entangling_name = "RI_simple" if i == n_encoding_layers - 1 else f"MI_simple_{i + 1}"
            builder.add_entangling_layer(trainable=True, name=entangling_name, trainable_outer=self.trainable_outer)

        self.quantum = ML.QuantumLayer(
                input_size=total_input_size,
                builder=builder,
                input_state=StateVector.from_basic_state(input_state),
                measurement_strategy=ML.MeasurementStrategy.amplitudes(computation_space='fock')
            )

        self.readout = nn.Sequential(
            nn.Linear(int(scipy.special.comb(n_modes+n_photons-1, n_photons)), self.hidden_readout),
            nn.Tanh(),
            nn.Linear(self.hidden_readout, 2),
        )

    def forward(self, xt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            xt: tensor of shape (batch, 2) with columns [x, t]

        Returns:
            u: solution field, shape (batch, 1)
            ux_hat: x-derivative approximation, shape (batch, 1)
        """
        x = xt[:, 0:1]
        t = xt[:, 1:2]
        z = self.feature_map(xt)
        q_amplitudes = self.quantum(z)
        # 3. Derive probabilities for classical readout: P(x) = |c|^2 -> (batch, 20) real
        q_probs = torch.abs(q_amplitudes) ** 2

        # 4. Classical Readout Pipeline
        out = self.readout(q_probs)

        u = out[:, 0:1]
        ux_hat = out[:, 1:2]

#        if self.config.use_hard_bc:
#            # Enforce homogeneous Dirichlet boundary conditions: u(0,t)=u(1,t)=0
#            u = x * (1.0 - x) * q_u
#        else:
#            u = q_u

        f_x = torch.zeros_like(x)

        if hasattr(self.pde, "initial_condition"):
            f_x = self.pde.initial_condition(x)

        envelope = 1 - torch.cos(0.5 * math.pi * t)

        u = (self.pde.x_min + x) * (self.pde.x_max - x) * u #(envelope * u) + f_x


        return u, ux_hat


def _fock_basis_full(n_modes, n_photons):
    """Occupations over all m modes summing to exactly N: the true Fock basis."""
    from itertools import product as _product
    return sorted(v for v in _product(range(n_photons + 1), repeat=n_modes)
                  if sum(v) == n_photons)


class SymmetricUnitaryHeatQPINN(nn.Module):
    """Classical reimplementation of the photonic layer as tensor algebra: two
    m x m mode unitaries acting on an N-photon state via their symmetric power,
    with the phase encoding between them. Matches the quantum arm by
    construction -- a unit test / ablation platform, not an independent control.

    States are homogeneous degree-N polynomials, |n> <-> prod_j x_j^{n_j} /
    sqrt(prod n_j!). A mode unitary acts by substitution x -> U x, so the fixed
    input |1,0,1,0,...> becomes the outer product of the corresponding columns
    of W_L; the encoding multiplies mode-axis entry a by e^{i z_a} (z = 0 on the
    reference mode), reproducing e^{i omega.z}; W_R contracts each mode-axis.
    Folding the m^N tensor onto the D Fock states and scaling by sqrt(prod n_j!)
    gives amplitudes c_n; p = |c|^2, with sum_n p_n = 1 automatic.
    """

    def __init__(self, feature_size=4, quantum_output_size=20, hidden_feature=16,
                 hidden_readout=4, n_photons_per_pass=3, n_encoding_layers=1,
                 pde=None):
        super().__init__()
        self.feature_size = feature_size
        self.quantum_output_size = quantum_output_size   # schema compat only
        self.hidden_feature = hidden_feature
        self.hidden_readout = hidden_readout
        self.n_photons_per_pass = n_photons_per_pass
        self.n_encoding_layers = n_encoding_layers
        self.pde = pde

        m, N = feature_size + 1, n_photons_per_pass
        self.n_modes, self.N = m, N

        self.feature_map = nn.Sequential(
            nn.Linear(2, hidden_feature),
            nn.Tanh(),
            nn.Linear(hidden_feature, feature_size * n_encoding_layers),
        )

        occupied = [i for i in range(m) if i % 2 == 0]
        if len(occupied) != N:
            raise ValueError(f"input state has {len(occupied)} photons, expected N={N}")
        self.register_buffer("occupied", torch.tensor(occupied, dtype=torch.long))

        from itertools import product as _product
        basis = _fock_basis_full(m, N)
        index_of = {v: i for i, v in enumerate(basis)}
        flat_to_fock = []
        for a in _product(range(m), repeat=N):
            occ = [0] * m
            for i in a:
                occ[i] += 1
            flat_to_fock.append(index_of[tuple(occ)])
        mult = []
        for v in basis:
            f = 1.0
            for nj in v:
                f *= math.factorial(nj)
            mult.append(math.sqrt(f))
        self.register_buffer("flat_to_fock", torch.tensor(flat_to_fock, dtype=torch.long))
        self.register_buffer("norm_factor", torch.tensor(mult, dtype=torch.float32))
        self.D = len(basis)

        self.n_anti = m * (m - 1) // 2
        self.n_sym = m * (m + 1) // 2
        self.register_buffer("idx_anti", torch.tril_indices(m, m, offset=-1))
        self.register_buffer("idx_sym", torch.tril_indices(m, m, offset=0))
        self.gen_L = nn.Parameter(torch.randn(n_encoding_layers, self.n_anti + self.n_sym) * 0.5)
        self.gen_R = nn.Parameter(torch.randn(n_encoding_layers, self.n_anti + self.n_sym) * 0.5)

        self.readout = nn.Sequential(
            nn.Linear(n_encoding_layers * self.D, hidden_readout),
            nn.Tanh(),
            nn.Linear(hidden_readout, 2),
        )

    def _unitary(self, flat):
        """Cayley transform of a skew-Hermitian generator -> (m, m) unitary."""
        m = self.n_modes
        a_vals, s_vals = flat[:self.n_anti], flat[self.n_anti:]
        A = torch.zeros(m, m, device=flat.device, dtype=flat.dtype)
        A[self.idx_anti[0], self.idx_anti[1]] = a_vals
        A = A - A.T
        S = torch.zeros(m, m, device=flat.device, dtype=flat.dtype)
        S[self.idx_sym[0], self.idx_sym[1]] = s_vals
        S = S + S.T - torch.diag(torch.diagonal(S))
        K = torch.complex(A, S)
        eye = torch.eye(m, device=flat.device, dtype=K.dtype)
        return torch.linalg.solve(eye + K, eye - K)

    def _chunk_probs(self, z_chunk, layer):
        m, N = self.n_modes, self.N
        batch = z_chunk.shape[0]
        WL, WR = self._unitary(self.gen_L[layer]), self._unitary(self.gen_R[layer])

        T = None
        for mode in self.occupied.tolist():
            v = WL[:, mode]
            T = v if T is None else T.unsqueeze(-1) * v

        zfull = torch.cat(
            [z_chunk, torch.zeros(batch, m - self.feature_size,
                                  device=z_chunk.device, dtype=z_chunk.dtype)], dim=1)
        P = torch.exp(torch.complex(torch.zeros_like(zfull), zfull))
        Tb = T.unsqueeze(0).expand(batch, *([m] * N)).clone()
        for ax in range(N):
            shape = [batch] + [1] * N
            shape[ax + 1] = m
            Tb = Tb * P.view(shape)

        for ax in range(N):
            Tb = torch.tensordot(WR, Tb, dims=([1], [ax + 1])).movedim(0, ax + 1)

        flat = Tb.reshape(batch, -1)
        coeff = torch.zeros(batch, self.D, device=flat.device, dtype=flat.dtype)
        coeff = coeff.index_add(1, self.flat_to_fock, flat)
        c = coeff * self.norm_factor.to(coeff.dtype)
        return c.real ** 2 + c.imag ** 2

    def forward(self, xt):
        x = xt[:, 0:1]
        z = self.feature_map(xt)
        z = z.view(z.shape[0], self.n_encoding_layers, self.feature_size)
        p = torch.cat([self._chunk_probs(z[:, l], l)
                       for l in range(self.n_encoding_layers)], dim=1)
        out = self.readout(p)
        u = out[:, 0:1]
        ux_hat = out[:, 1:2]
        u = (self.pde.x_min + x) * (self.pde.x_max - x) * u
        return u, ux_hat

    @torch.no_grad()
    def self_check(self, n_probe=64, verbose=True):
        """sum_n p_n must be 1 for every input. Large deviation => broken."""
        z = torch.randn(n_probe, self.feature_size, device=self.gen_L.device)
        worst = 0.0
        for l in range(self.n_encoding_layers):
            total = self._chunk_probs(z, l).sum(dim=1)
            worst = max(worst, float((total - 1.0).abs().max()))
        if verbose:
            print(f"[symmetric] max |sum_n p_n - 1| = {worst:.2e} "
                  f"{'OK' if worst < 1e-4 else '** BROKEN **'}")
        return worst