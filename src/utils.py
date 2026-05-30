from typing import Tuple, List
import torch


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




