"""
eval.py — standalone evaluator for QPINN / classical PINNs.

Usage example:
    from eval import evaluate

    metrics, U_pred, U_true, x_grid, t_grid = evaluate(
        model=model,
        pde=pde,
        nx=100,
        nt=100,
        device=device,
        dtype=dtype,
    )
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict


def evaluate(
    model: nn.Module,
    pde,
    nx: int = 60,
    nt: int = 60,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> Tuple[Dict, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Evaluate a trained PINN on a regular (x, t) grid.

    Args:
        model:   trained PyTorch model; forward(xt) -> (u, *extras)
        pde:     PDE object exposing:
                   .x_min, .x_max  — spatial domain
                   .t_min, .t_max  — temporal domain
                   .exact_solution(x, t) -> tensor
        nx:      number of spatial grid points
        nt:      number of temporal grid points
        device:  torch device (defaults to cuda if available else cpu)
        dtype:   torch dtype

    Returns:
        metrics:  dict with keys rel_l2, mae, max_abs_error, rmse
        U_pred:   predicted solution tensor of shape (nx, nt)  [on cpu]
        U_true:   exact solution tensor of shape (nx, nt)      [on cpu]
        x_grid:   spatial coordinates of shape (nx, 1)         [on cpu]
        t_grid:   temporal coordinates of shape (nt, 1)        [on cpu]
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Evaluating on {nx}×{nt} grid …")

    x = torch.linspace(pde.x_min, pde.x_max, nx, device=device, dtype=dtype).reshape(-1, 1)
    t = torch.linspace(pde.t_min, pde.t_max, nt, device=device, dtype=dtype).reshape(-1, 1)

    X, T = torch.meshgrid(x.squeeze(), t.squeeze(), indexing="ij")
    xt_grid = torch.stack([X.reshape(-1), T.reshape(-1)], dim=1)

    model.eval()
    with torch.no_grad():
        U_pred, *_ = model(xt_grid)
        U_pred = U_pred.reshape(nx, nt).cpu()

        U_true = pde.exact_solution(
            xt_grid[:, 0:1], xt_grid[:, 1:2]
        ).reshape(nx, nt).cpu()

    error = U_pred - U_true
    rel_l2 = float(torch.linalg.norm(error) / torch.linalg.norm(U_true))
    mae = float(torch.abs(error).mean())
    max_abs_error = float(torch.abs(error).max())
    rmse = float(torch.sqrt((error ** 2).mean()))

    metrics = {
        "rel_l2": rel_l2,
        "mae": mae,
        "max_abs_error": max_abs_error,
        "rmse": rmse,
    }

    print(f"  Relative L2 : {rel_l2:.4e}")
    print(f"  MAE         : {mae:.4e}")
    print(f"  Max abs err : {max_abs_error:.4e}")
    print(f"  RMSE        : {rmse:.4e}")

    return metrics, U_pred, U_true, x.cpu(), t.cpu()
