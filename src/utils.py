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





