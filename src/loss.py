import torch
import torch.nn as nn
import torch.nn.functional as F
mse = F.mse_loss
from src.pde import PDE
from src.utils import gradient



def total_loss(
    model,
    xt_f,
    xt_i,
    xt_b,
    pde,
    lambda_f= 5.0, #1.0,
    lambda_c= 2.0, #0.5,
    lambda_i= 2.0, #10.0
    lambda_b= 2.0, #2.0
):
    """
    Computes total weighted PINN/QPINN loss.

    Args:
        model: neural network model
        xt_f: sampled interior collocation points
        xt_i: sampled initial condition points
        xt_b: sampled boundary condition points
        pde: PDE object to pass to individual losses

    Returns:
        total loss tensor
        dictionary of scalar loss values
    """

    lf = pde_loss(model, xt_f, pde)
    lc = consistency_loss(model, xt_f)
    li = ic_loss(model, xt_i, pde)
    lb = bc_loss(model, xt_b, pde)

    total = (
        lambda_f * lf +
        lambda_c * lc +
        lambda_i * li +
        lambda_b * lb
    )

    return total, {
        "total": total.item(),
        "pde": lf.item(),
        "consistency": lc.item(),
        "initial": li.item(),
        "boundary": lb.item(),
    }




def pde_loss(model, xt_int, pde: PDE):
    """"
    Compute MSE Loss of residual of the PDE for the model output

    Args:
        model : pinn model
        xt_int : interior points with requires_grad= True
        pde : PDE object used for residual formula

    Returns:

        MSE Loss of residual of the PDE for the model output
    """


    u, u_x, *_ = model(xt_int) # model output

    # use the PDE method to compute the residual from the derivatives
    r = pde.residual(u, u_x, xt_int)

    return mse(r, torch.zeros_like(r))




def consistency_loss(model, xt_int):
    """
   Compute consistency residual: du/dx - ux_hat

   Ensures that the network's spatial derivative approximation matches
   the actual spatial derivative of the solution.

   Args:
       model: neural network model
       xt_interior: interior points with requires_grad=True

   Returns:
       residual: consistency residual tensor
   """

    u, u_x, *_ = model(xt_int)

    u_x_true = gradient(u, xt_int)[:, 0:1]

    return mse(u_x, u_x_true)

def ic_loss(model, xt_i, pde: PDE):
    """
    Compute initial condition loss.

    Uses the PDE's initial_condition() method to enforce u(x, 0) = f(x).

    Args:
        model: neural network model
        xt_initial: sampled initial condition points with t=0
        pde: PDE object
    Returns:
        loss: MSE loss at initial time
    """

    xt_i = xt_i.requires_grad_(True)
    u, _, *_ = model(xt_i)
    x = xt_i[:, 0:1]

    u_ic = pde.initial_condition(x)
    loss = mse(u, u_ic)
    if hasattr(pde, "initial_velocity"):
        u_t = gradient(u, xt_i)[:, 1:2]
        ut_ic = pde.initial_velocity(x)
        loss = loss + 1*mse(u_t, ut_ic)

    return loss


def bc_loss(model, xt_b, pde: PDE):
    """
    Compute boundary condition loss.

    Uses the PDE's boundary_condition_left() and boundary_condition_right() methods.

    Args:
        model: neural network model
        xt_boundary: boundary condition points (x=0 and x=1)
        pde: PDE object

    Returns:
        loss: MSE loss at boundaries
    """
    u, _, *_ = model(xt_b)

    x = xt_b[:, 0:1]
    t = xt_b[:, 1:2]

    loss = 0.0

    left_mask = torch.isclose(x, torch.full_like(x, pde.x_min))
    right_mask = torch.isclose(x, torch.full_like(x, pde.x_max))

    if left_mask.any():
        u_left = pde.boundary_condition_left(t[left_mask])
        loss += mse(u[left_mask], u_left)

    if right_mask.any():
        u_right = pde.boundary_condition_right(t[right_mask])
        loss += mse(u[right_mask], u_right)

    return loss

