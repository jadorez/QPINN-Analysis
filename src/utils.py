from typing import Callable, Tuple, List
import torch
import scipy
import numpy as np

def generate_fock_basis(photons: int, modes: int) -> List[Tuple[int, ...]]:
    # Helper function to generate the Fock basis in descending lexicographic order
    # This exactly matches Perceval and Merlin's internal state ordering.
    states = []
    def _gen(p_left, m_left, current_state):
        if m_left == 1:
            states.append(tuple(current_state + [p_left]))
            return
        for p in range(p_left, -1, -1):
            _gen(p_left - p, m_left - 1, current_state + [p])
    _gen(photons, modes, [])
    return states

def build_bipartition_state_map(photons: int, modes: int, modes_A: List[int]) -> Tuple[torch.Tensor, int]:
    """
    Builds the index map needed to partial-trace a Fock-basis state vector over an
    arbitrary bipartition of modes.

    Args:
        photons: total photon number of the Fock basis.
        modes:   total number of modes.
        modes_A: mode indices assigned to subsystem A; all remaining modes form
                 subsystem B (order among B is preserved from `range(modes)`).

    Returns:
        state_map: (A_dim, B_dim) long tensor mapping (a_idx, b_idx) -> flat Fock
                   basis index, or -1 where the (A, B) combination doesn't occur.
        A_dim: dimension of subsystem A's reduced Hilbert space.
    """
    modes_B = [m for m in range(modes) if m not in modes_A]
    states = generate_fock_basis(photons, modes)

    A_states = [tuple(s[m] for m in modes_A) for s in states]
    B_states = [tuple(s[m] for m in modes_B) for s in states]

    unique_A = list(dict.fromkeys(A_states))
    unique_B = list(dict.fromkeys(B_states))
    A_dim = len(unique_A)

    state_map = torch.full((A_dim, len(unique_B)), -1, dtype=torch.long)
    for idx, (a, b) in enumerate(zip(A_states, B_states)):
        state_map[unique_A.index(a), unique_B.index(b)] = idx

    return state_map, A_dim

def bipartite_entanglement_entropy(amplitudes: torch.Tensor, state_map: torch.Tensor, A_dim: int) -> torch.Tensor:
    """
    Von Neumann entanglement entropy of subsystem A, given complex Fock-basis
    amplitudes and the (A_dim, B_dim) state map from `build_bipartition_state_map`.

    Args:
        amplitudes: (batch, dim) complex tensor of Fock-basis amplitudes.
        state_map:  (A_dim, B_dim) index map, -1 for invalid combinations.
        A_dim:      dimension of subsystem A's reduced Hilbert space.

    Returns:
        entropy: (batch,) real tensor.
    """
    batch_size = amplitudes.size(0)
    # state_map may live on a different device than amplitudes (e.g. merlin's
    # quantum layer can return amplitudes on CPU even when the model is on MPS).
    state_map = state_map.to(amplitudes.device)

    # Partial trace over subsystem B: rho_A[i,j] = sum_k (c_{i,k} * c_{j,k}^*)
    rho_A = torch.zeros((batch_size, A_dim, A_dim), dtype=amplitudes.dtype, device=amplitudes.device)
    for i in range(A_dim):
        for j in range(A_dim):
            valid_b = (state_map[i] != -1) & (state_map[j] != -1)
            idx_ik = state_map[i][valid_b]
            idx_jk = state_map[j][valid_b]
            if len(idx_ik) > 0:
                rho_A[:, i, j] = torch.sum(
                    amplitudes[:, idx_ik] * torch.conj(amplitudes[:, idx_jk]),
                    dim=1
                )

    # rho_A is Hermitian -> eigvalsh. MPS doesn't support complex eigvalsh, so
    # compute on CPU and move back when running on Apple Silicon GPUs.
    if rho_A.device.type == "mps":
        eigvals = torch.linalg.eigvalsh(rho_A.cpu()).to(rho_A.device)
    else:
        eigvals = torch.linalg.eigvalsh(rho_A)
    eigvals = torch.clamp(eigvals, min=1e-12, max=1.0)
    entropy = -torch.sum(eigvals * torch.log(eigvals), dim=-1)

    return entropy

def gradient(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Compute gradient dy/dx using automatic differentiation.

    Args:
        y: output tensor
        x: input tensor with requires_grad=True

    Returns:
        grad: gradient tensor of same shape as x
    """
    return torch.autograd.grad(
        y,
        x,
        grad_outputs=torch.ones_like(y),
        create_graph=True,
        retain_graph=True,
    )[0]

def _amplitude_jacobian(amplitude_fn: Callable[[torch.Tensor], torch.Tensor], xt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Shared machinery for quantum_fisher_information / classical_fisher_information:
    the per-sample Jacobian of Fock-basis amplitudes w.r.t. the D real inputs in
    `xt` (D=2 for [x, t]; any D works, e.g. D=feature_size when probing the
    encoded features directly rather than (x, t) — see notebook section 6's
    phase-space constancy check), computed once so both Fisher information
    flavors (or `fisher_information` computing both) don't each pay for their
    own O(K) backward passes.

    Uses the grad-of-batch-sum trick (as in `gradient`, exact here because each
    sample's amplitudes depend only on its own xt row) once per real and
    imaginary part of each of the K Fock-basis amplitude components, so cost
    scales as O(K) backward passes through `amplitude_fn` — much more
    expensive than a single forward/backward. Intended for evaluation-time
    diagnostics on modest grids, not per-step training use.

    Args:
        amplitude_fn: callable mapping (batch, D) real xt -> (batch, K) complex
            Fock-basis amplitudes, e.g. a model's amplitude output before the
            |.|^2 readout (see MerlinTunableSimple.amplitudes).
        xt: (batch, D) tensor. A detached, grad-enabled copy is used
            internally, so the caller's tensor is left untouched.

    Returns:
        psi: (batch, K) complex amplitudes, detached.
        jac: (batch, K, D) complex, d psi_k / d xt_i, detached.
    """
    xt = xt.detach().clone().requires_grad_(True)
    psi = amplitude_fn(xt)
    batch, k_dim = psi.shape
    d_dim = xt.shape[1]

    real_part = psi.real
    imag_part = psi.imag
    ones = torch.ones(batch, device=psi.device, dtype=xt.dtype)

    real_jac = torch.empty(batch, k_dim, d_dim, device=psi.device, dtype=xt.dtype)
    imag_jac = torch.empty(batch, k_dim, d_dim, device=psi.device, dtype=xt.dtype)

    for k in range(k_dim):
        real_jac[:, k, :] = torch.autograd.grad(
            real_part[:, k], xt, grad_outputs=ones, retain_graph=True,
        )[0]
        imag_jac[:, k, :] = torch.autograd.grad(
            imag_part[:, k], xt, grad_outputs=ones, retain_graph=True,
        )[0]

    jac = torch.complex(real_jac, imag_jac)  # (batch, K, D): d psi_k / d xt_i
    return psi.detach(), jac  # values only from here on — the derivative info now lives in jac


def _qfi_from_jacobian(psi: torch.Tensor, jac: torch.Tensor) -> torch.Tensor:
    """QFIM from a precomputed (psi, jac) pair — see quantum_fisher_information."""
    inner = torch.einsum('bki,bkj->bij', jac.conj(), jac)  # <d_i psi | d_j psi>
    overlap = torch.einsum('bk,bki->bi', psi.conj(), jac)  # <psi | d_i psi>
    return 4.0 * (inner - torch.einsum('bi,bj->bij', overlap.conj(), overlap)).real


def _cfi_from_jacobian(psi: torch.Tensor, jac: torch.Tensor) -> torch.Tensor:
    """CFIM from a precomputed (psi, jac) pair — see classical_fisher_information."""
    p = psi.real ** 2 + psi.imag ** 2  # (batch, K) Born-rule probabilities
    dp = 2.0 * (psi.conj().unsqueeze(-1) * jac).real  # (batch, K, D): d p_k / d xt_i
    inv_p = 1.0 / p.clamp(min=1e-12)
    return torch.einsum('bk,bki,bkj->bij', inv_p, dp, dp)


def quantum_fisher_information(amplitude_fn: Callable[[torch.Tensor], torch.Tensor], xt: torch.Tensor) -> torch.Tensor:
    """
    Quantum Fisher Information Matrix (QFIM) of a pure state |psi> with
    respect to the D real inputs in `xt` (typically D=2, [x, t]; can also be
    the D=feature_size encoded features directly — see `_amplitude_jacobian`),
    evaluated per sample.

    For a normalized pure state, the QFIM is 4x the real part of the
    Fubini-Study metric:
        F_ij = 4 * Re[ <d_i psi | d_j psi> - <d_i psi | psi><psi | d_j psi> ]
    where d_i = d/d(xt[:, i]). It measures how distinguishable the encoded
    state becomes under an infinitesimal shift in that input under the *best
    possible* measurement: a larger value means the quantum state itself
    carries more resolving power for that input near that point — see
    `classical_fisher_information` for how much of that an actual (Fock-basis
    probability) measurement realizes.

    Args:
        amplitude_fn, xt: see `_amplitude_jacobian`.

    Returns:
        qfim: (batch, D, D) real tensor per sample.
    """
    psi, jac = _amplitude_jacobian(amplitude_fn, xt)
    return _qfi_from_jacobian(psi, jac)


def classical_fisher_information(amplitude_fn: Callable[[torch.Tensor], torch.Tensor], xt: torch.Tensor) -> torch.Tensor:
    """
    Classical Fisher Information Matrix (CFIM) of the Born-rule probability
    distribution p_k = |psi_k|^2 with respect to the D real inputs in `xt` —
    i.e. the information actually extractable by the Fock-basis photon-counting
    measurement this model's readout uses, as opposed to the best conceivable
    measurement (see `quantum_fisher_information`):
        F_ij = sum_k (1 / p_k) * (d p_k / d xt_i) * (d p_k / d xt_j)

    CFIM <= QFIM always (Loewner order); the gap is information the encoded
    state carries but the actual measurement basis can't access (e.g. a
    photon-number/computational-basis measurement is blind to relative phase).

    Args:
        amplitude_fn, xt: see `_amplitude_jacobian`.

    Returns:
        cfim: (batch, D, D) real tensor per sample.
    """
    psi, jac = _amplitude_jacobian(amplitude_fn, xt)
    return _cfi_from_jacobian(psi, jac)


def fisher_information(amplitude_fn: Callable[[torch.Tensor], torch.Tensor], xt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes both quantum_fisher_information and classical_fisher_information
    from a single shared Jacobian, instead of paying for the O(K) backward
    passes twice. Use this whenever both are wanted (e.g. to compare them).

    Args:
        amplitude_fn, xt: see `_amplitude_jacobian`.

    Returns:
        (qfim, cfim): each (batch, D, D) real tensor, see the respective functions.
    """
    psi, jac = _amplitude_jacobian(amplitude_fn, xt)
    return _qfi_from_jacobian(psi, jac), _cfi_from_jacobian(psi, jac)

def hilbert_space_dimension(feature_size: int):
    #Compute Hilbert space dimension
    
    n_modes = feature_size+1
    
    n_photons = np.ceil(n_modes/2)
    dimH = int(scipy.special.comb(n_modes+n_photons-1, n_photons))
    return dimH

def quantum_outputsize_for_param_match(feature_size: int, hidden_readout: int):
    #Compute Hilbert space dimension
    n_modes = feature_size+1

#    input_state = n_modes * [0]
#    for i in range(n_modes):                            #Non-opti, should be packed into a formula like n_photons = 1+(n_modes // 2)
#        if i % 2 == 0:
#            input_state[i] = 1
#    n_photons = sum(input_state)

    n_photons = np.ceil(n_modes/2)
    dimH = hilbert_space_dimension(feature_size)
    return (2 * feature_size * (feature_size+1) + dimH * hidden_readout) / (feature_size + hidden_readout + 1)
    #return (feature_size * (dimH+1) + dimH * (hidden_readout+1)) / (feature_size + hidden_readout + 1)





