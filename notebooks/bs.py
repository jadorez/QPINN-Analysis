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

        u = x * (1 - x) * (envelope * u) + f_x

        return u, ux_hat