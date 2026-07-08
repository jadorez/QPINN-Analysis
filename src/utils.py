from typing import Tuple, List
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





