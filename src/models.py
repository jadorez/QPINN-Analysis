import math
import torch
import torch.nn as nn
from typing import Tuple, List, Optional
from merlin.core.state_vector import StateVector
from src.utils import build_bipartition_state_map, bipartite_entanglement_entropy

import perceval as pcvl
import scipy

import merlin as ML


class MerlinHeatQPINN(nn.Module):
    """
    QPINN model using MerLin quantum layer for pde resolution.
    """

    def __init__(self, feature_size : int = 4, quantum_output_size : int = 20, hidden_feature : int = 16, hidden_readout : int = 4):
        super().__init__()
        self.feature_size = feature_size
        self.quantum_output_size = quantum_output_size
        self.hidden_feature = hidden_feature
        self.hidden_readout = hidden_readout

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
        z = self.feature_map(xt)
        q = self.quantum(z)
        out = self.readout(q)

        u = out[:, 0:1]
        ux_hat = out[:, 1:2]

        return u, ux_hat


class ClassicalHeatQPINN(nn.Module):
    """
    Classical neural network QPINN (baseline without quantum layer).
    Useful for comparing quantum vs classical performance.
    """

    def __init__(self, feature_size : int = 4, quantum_output_size : int = 20, hidden_feature : int = 16, hidden_readout : int = 4):
        super().__init__()
        self.feature_size = feature_size
        self.quantum_output_size = quantum_output_size
        self.hidden_feature = hidden_feature
        self.hidden_readout = hidden_readout

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
        out = self.network(xt)

        u = out[:, 0:1]
        ux_hat = out[:, 1:2]

        return u, ux_hat

class MerlinTunableSimple(nn.Module):
    """
    Similar to MerlinHeatQpinn but has architecture that allows same # weights as classical model
    """
    
    def __init__(self, feature_size : int = 4, hidden_feature : int = 16, hidden_readout : int = 4, trainable_outer : bool = True, entropy_modes_A : Optional[List[int]] = None, n_encoding_layers : int = 1):
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

        # Precompute the index mapping for the entanglement entropy partial trace.
        if entropy_modes_A is None:
            entropy_modes_A = list(range(input_size))  # encoded modes vs. the extra mode
        self.entropy_modes_A = entropy_modes_A
        state_map, self.entropy_A_dim = build_bipartition_state_map(
            photons=n_photons, modes=n_modes, modes_A=entropy_modes_A
        )
        self.register_buffer('entropy_state_map', state_map)

    def to(self, *args, **kwargs):
        # nn.Module.to() moves parameters/buffers via _apply and does not call
        # submodules' own .to(); merlin's QuantumLayer tracks its target device
        # in a plain attribute (self.device) that only that override updates,
        # so without this it silently keeps computing/returning on its original
        # device (e.g. CPU) even after the parent model is moved to MPS/CUDA.
        super().to(*args, **kwargs)
        self.quantum.to(*args, **kwargs)
        return self

    def amplitudes(self, xt: torch.Tensor) -> torch.Tensor:
        """
        Fock-basis amplitudes before the |.|^2 readout, shape (batch, hilbert_dim)
        complex. Exposed separately from forward() for computing quantities that
        need the raw state as a differentiable function of xt, e.g. the Quantum
        Fisher Information (see utils.quantum_fisher_information / eval.py's
        `compute_qfi`).
        """
        return self.quantum(self.feature_map(xt))

    def forward(self, xt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            xt: tensor of shape (batch, 2) with columns [x, t]

        Returns:
            u: solution field, shape (batch, 1)
            ux_hat: x-derivative approximation, shape (batch, 1)
            entropy: per-sample Von Neumann entanglement entropy, shape (batch,).
                Reduce with .mean() for a scalar summary, or reshape against the
                (x, t) grid it was evaluated on to plot entropy as a function of
                (x, t) — see eval.py's `entropy_grid`.
        """
        x = xt[:, 0:1]
        q_amplitudes = self.amplitudes(xt)
        # 3. Derive probabilities for classical readout: P(x) = |c|^2 -> (batch, 20) real
        q_probs = torch.abs(q_amplitudes) ** 2

        entropy = bipartite_entanglement_entropy(q_amplitudes, self.entropy_state_map, self.entropy_A_dim)

        # 4. Classical Readout Pipeline
        out = self.readout(q_probs)

        q_u = out[:, 0:1]
        ux_hat = out[:, 1:2]

#        if self.config.use_hard_bc:
#            # Enforce homogeneous Dirichlet boundary conditions: u(0,t)=u(1,t)=0
#            u = x * (1.0 - x) * q_u
#        else:
        u = q_u

        return u, ux_hat, entropy


def _hilbert_dim(feature_size: int) -> int:
    """Fock-basis dimension MerlinTunableSimple's circuit produces for a given feature_size."""
    n_modes = feature_size + 1
    input_state = n_modes * [0]
    for i in range(n_modes):
        if i % 2 == 0:
            input_state[i] = 1
    n_photons = sum(input_state)
    return int(scipy.special.comb(n_modes + n_photons - 1, n_photons))


class ClassicalTunableSimple(nn.Module):
    """
    Classical counterpart to MerlinTunableSimple: identical feature_map input/
    output shape, with the quantum layer + readout replaced by a plain
    classical MLP of matching input/output width. Used for parameter-matched
    quantum-vs-classical comparisons — pick the four hidden widths so that
    total trainable params equal a target MerlinTunableSimple's (see
    `match_classical_param_count`).
    """

    def __init__(self, feature_size: int = 4, hidden_feature: int = 16, hidden_transform: int = 16,
                 hidden_readout_1: int = 4, hidden_readout_2: int = 4, n_encoding_layers: int = 1):
        super().__init__()
        self.feature_size = feature_size
        self.hidden_feature = hidden_feature
        self.hidden_transform = hidden_transform
        self.hidden_readout_1 = hidden_readout_1
        self.hidden_readout_2 = hidden_readout_2
        self.n_encoding_layers = n_encoding_layers

        total_input_size = feature_size * n_encoding_layers
        hilbert_dim = _hilbert_dim(feature_size)

        # Same shape as MerlinTunableSimple.feature_map
        self.feature_map = nn.Sequential(
            nn.Linear(2, self.hidden_feature),
            nn.Tanh(),
            nn.Linear(self.hidden_feature, total_input_size),
        )

        # Stand-in for the quantum layer: maps onto the same Hilbert-space
        # dimension MerlinTunableSimple's readout would consume. Has its own
        # hidden width (unlike the quantum layer) purely so the total param
        # count is tunable enough to hit an exact match against the quantum
        # model's — see match_classical_param_count.
        self.transform = nn.Sequential(
            nn.Linear(total_input_size, self.hidden_transform),
            nn.Tanh(),
            nn.Linear(self.hidden_transform, hilbert_dim),
            nn.Tanh(),
        )

        # Two hidden widths (vs. MerlinTunableSimple.readout's one) so the
        # search in match_classical_param_count has enough independent knobs
        # to hit an exact total-param match for any target.
        self.readout = nn.Sequential(
            nn.Linear(hilbert_dim, self.hidden_readout_1),
            nn.Tanh(),
            nn.Linear(self.hidden_readout_1, self.hidden_readout_2),
            nn.Tanh(),
            nn.Linear(self.hidden_readout_2, 2),
        )

    def forward(self, xt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.feature_map(xt)
        q = self.transform(z)
        out = self.readout(q)

        u = out[:, 0:1]
        ux_hat = out[:, 1:2]

        return u, ux_hat


def match_classical_param_count(
    quantum_model: nn.Module,
    feature_size: int = 4,
    n_encoding_layers: int = 1,
    search_max: int = 64,
    reference_widths: Tuple[int, int, int, int] = (16, 16, 4, 4),
) -> "ClassicalTunableSimple":
    """
    Finds a ClassicalTunableSimple(feature_size, hidden_feature, hidden_transform,
    hidden_readout_1, hidden_readout_2, n_encoding_layers) whose total trainable
    parameter count exactly matches `quantum_model`'s.

    Many (hidden_feature, hidden_transform, hidden_readout_1, hidden_readout_2)
    combinations can hit the exact target, including degenerate ones (e.g.
    hidden_feature=1, a near-unusable 2->1 bottleneck, with the rest of the
    budget dumped into one wide layer). Returning the first match found biases
    toward exactly those degenerate shapes, which would make for an unfairly
    weak classical baseline. So instead every exact match within the search
    box is scored by relative distance from `reference_widths` (log-ratio, so
    e.g. half vs. double reference_widths score the same either direction),
    and the most balanced one is returned.

    Raises ValueError if no exact match is found within
    hidden_feature, hidden_transform, hidden_readout_{1,2} in [1, search_max].
    """
    target = sum(p.numel() for p in quantum_model.parameters())
    total_input_size = feature_size * n_encoding_layers
    hilbert_dim = _hilbert_dim(feature_size)
    ref_f, ref_t, ref_r1, ref_r2 = reference_widths

    best = None
    best_score = float("inf")

    for hidden_feature in range(1, search_max + 1):
        feature_map = (2 * hidden_feature + hidden_feature) + (hidden_feature * total_input_size + total_input_size)
        for hidden_transform in range(1, search_max + 1):
            transform = (total_input_size * hidden_transform + hidden_transform) + (hidden_transform * hilbert_dim + hilbert_dim)
            for hidden_readout_1 in range(1, search_max + 1):
                # readout = h_r1*(hilbert_dim+1) + h_r2*(h_r1+3) + 2 is linear in
                # h_r2 given everything else fixed, so solve for it directly
                # instead of a 4th nested loop.
                remaining = target - feature_map - transform - hidden_readout_1 * (hilbert_dim + 1) - 2
                coefficient = hidden_readout_1 + 3
                if remaining <= 0 or remaining % coefficient != 0:
                    continue
                hidden_readout_2 = remaining // coefficient
                if hidden_readout_2 < 1:
                    continue

                score = (
                    math.log(hidden_feature / ref_f) ** 2
                    + math.log(hidden_transform / ref_t) ** 2
                    + math.log(hidden_readout_1 / ref_r1) ** 2
                    + math.log(hidden_readout_2 / ref_r2) ** 2
                )
                if score < best_score:
                    best_score = score
                    best = (hidden_feature, hidden_transform, hidden_readout_1, hidden_readout_2)

    if best is None:
        raise ValueError(
            f"No (hidden_feature, hidden_transform, hidden_readout_1, hidden_readout_2) "
            f"in [1, {search_max}]^4 matches target param count {target} exactly. "
            f"Try a larger search_max."
        )

    hidden_feature, hidden_transform, hidden_readout_1, hidden_readout_2 = best
    return ClassicalTunableSimple(
        feature_size=feature_size,
        hidden_feature=hidden_feature,
        hidden_transform=hidden_transform,
        hidden_readout_1=hidden_readout_1,
        hidden_readout_2=hidden_readout_2,
        n_encoding_layers=n_encoding_layers,
    )


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
        state_map, self.A_dim = build_bipartition_state_map(photons=3, modes=4, modes_A=[0, 1])
        # Register as a non-trainable buffer so it moves properly with model.to(device)
        self.register_buffer('state_map', state_map)

    def calculate_bipartite_entropy(self, amplitudes: torch.Tensor) -> torch.Tensor:
        """
        Calculates the Von Neumann entanglement entropy of Subsystem A (modes 0,1).
        amplitudes shape: (batch_size, 20) complex
        """
        return bipartite_entanglement_entropy(amplitudes, self.state_map, self.A_dim)

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