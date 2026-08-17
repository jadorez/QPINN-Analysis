"""
train.py — standalone training loop for QPINN / classical PINNs.


"""

import copy
import time
import numpy as np
import torch
import torch.nn as nn
from typing import Optional

from src.sampler import DataSampler
from src.pde import PDE
from src.loss import total_loss
from src.eval import evaluate


def train(
    model: nn.Module,
    pde: PDE,
    loss_fn = total_loss,
    # ---------- sampling ----------
    sampler = None,
    # ---------- optimisation ----------
    epochs: int = 300,
    lr: float = 1e-2,#0.5e-2, #1e-2,
    optimizer_name: str = "Adam",        # "adam" | "sgd"
    # ---------- loss weights ----------
    lambda_pde: float = 1.0,            #1.0
        lambda_consistency: float = 0.5,    #0.1
        lambda_initial: float = 5.0,    #10.0   #10.0
        lambda_boundary: float = 0.0,       #1.0
    # ---------- lr scheduling ----------
    lr_decay: Optional[str] = None,      # None | "exponential" | "step" | "cosine"
    lr_decay_factor: float = 0.1,
    lr_decay_steps: int = 100,
    # ---------- misc ----------
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
    log_every: int = 50,
    seed: int = 42,
    # ---------- best-checkpoint tracking (opt-in; off => identical to before) ----------
    track_best: bool = False,
    eval_every: int = 200,
    eval_nx: int = 60,
    eval_nt: int = 60,
    best_info: Optional[dict] = None,
) -> np.ndarray:
    """
    Train a PINN model.

    Args:
        model:            PyTorch model; forward(xt) -> (u, *extras)
        pde:              PDE object with .exact_solution(x, t) method
        loss_fn:          loss object with .total_loss(model, xt_f, xt_i, xt_b,
                              exact_fn, lambda_f, lambda_c, lambda_i, lambda_b)
                          returning (total_loss, loss_dict)
        sampler :         sampler instance used to sample points fed to loss function
        epochs:           number of training epochs
        lr:               learning rate
        optimizer_name:   "adam", "sgd", or "lbfgs"
        lambda_*:         loss term weights
        lr_decay:         scheduler type (None to disable)
        lr_decay_factor:  multiplicative decay factor for step/exponential
        lr_decay_steps:   steps between decay events
        device:           torch device (defaults to cuda if available else cpu)
        dtype:            torch dtype
        log_every:        print interval
        seed:             random seed
        track_best:       if True, every `eval_every` epochs evaluate on a
                          FIXED (not resampled) (eval_nx, eval_nt) grid via
                          src.eval.evaluate, and keep the state_dict with the
                          lowest fixed-grid rel_l2 seen. At the end of
                          training, `model` is loaded with those best
                          weights in place instead of the final epoch's.
                          This is deliberately NOT based on the noisy
                          per-epoch training loss (whose collocation points
                          are freshly resampled every epoch, so argmin over
                          it partly selects sampling luck, not model
                          quality) -- see the discussion this was added for.
                          Default False reproduces the exact prior behaviour
                          (final-epoch weights, same return value).
        eval_every:       epoch interval between fixed-grid checkpoint evals
                          (only used if track_best=True). Also always checks
                          epoch 1 and the final epoch.
        eval_nx, eval_nt: resolution of the fixed tracking grid (only used
                          if track_best=True); does not have to match the
                          grid used for final reporting.
        best_info:        optional dict; if provided and track_best=True, is
                          populated in place with "best_epoch",
                          "best_rel_l2", and "fixed_eval_history" (list of
                          (epoch, rel_l2) tuples) so callers can inspect the
                          tracking curve without changing this function's
                          return type.

    Returns:
        history: np.ndarray of shape (epochs, 6)
                 columns: [total, pde, consistency, initial, boundary]
                 :param sampler:
    """
    # ------------------------------------------------------------------ setup
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_default_dtype(dtype)

    model = model.to(device=device, dtype=dtype)
    for p in model.parameters():
        if p.is_floating_point():
            p.data = p.data.to(dtype)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters — total: {total_params:,}  trainable: {trainable_params:,}")

    # ------------------------------------------------------------------ sampler
    if sampler is None:
        sampler = DataSampler(pde=pde, device=device, dtype=dtype)

    # ------------------------------------------------------------------ optimizer
    opt_name = optimizer_name.lower()
    if opt_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif opt_name == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name!r}")

    # ------------------------------------------------------------------ scheduler
    scheduler = None
    if lr_decay is not None:
        decay = lr_decay.lower()
        if decay == "exponential":
            gamma = lr_decay_factor ** (1.0 / lr_decay_steps)
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
        elif decay == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=lr_decay_steps, gamma=lr_decay_factor,
            )
        elif decay == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs,
            )
        else:
            raise ValueError(f"Unknown lr_decay: {lr_decay!r}")

    # ------------------------------------------------------------------ loop
    print(f"\nTraining for {epochs} epochs on {device}  (dtype={dtype})\n{'='*60}")
    history = []
    start = time.time()

    best_rel_l2 = float("inf")
    best_epoch: Optional[int] = None
    best_state = None
    fixed_eval_history = []

    for epoch in range(1, epochs + 1):
        model.train()


        optimizer.zero_grad()
        xt_f = sampler.sample_interior()
        xt_i = sampler.sample_initial()
        xt_b = sampler.sample_boundary()

        loss, loss_dict = loss_fn(
            model, xt_f, xt_i, xt_b,
            pde,
            lambda_f=lambda_pde,
            lambda_c=lambda_consistency,
            lambda_i=lambda_initial,
            lambda_b=lambda_boundary,
        )
        loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        history.append([
            loss_dict["total"],
            loss_dict["pde"],
            loss_dict["consistency"],
            loss_dict["initial"],
            loss_dict["boundary"],
        ])

        if epoch == 1 or epoch % log_every == 0:
            elapsed = time.time() - start
            print(
                f"Epoch {epoch:5d}/{epochs} | "
                f"Loss: {loss_dict['total']:.4e} | "
                f"PDE: {loss_dict['pde']:.2e} | "
                f"Cons: {loss_dict['consistency']:.2e} | "
                f"IC: {loss_dict['initial']:.2e} | "
                f"BC: {loss_dict['boundary']:.2e} | "
                f"Time: {elapsed:.1f}s"
            )

        # ---- optional best-checkpoint tracking on a FIXED (non-resampled)
        # eval grid, deliberately separate from the noisy per-epoch training
        # loss above (see track_best docstring). ----
        if track_best and (epoch == 1 or epoch % eval_every == 0 or epoch == epochs):
            model.eval()
            with torch.no_grad():
                eval_metrics, *_ = evaluate(
                    model=model, pde=pde, nx=eval_nx, nt=eval_nt,
                    device=device, dtype=dtype,
                )
            model.train()
            fixed_eval_history.append((epoch, eval_metrics["rel_l2"]))
            if eval_metrics["rel_l2"] < best_rel_l2:
                best_rel_l2 = eval_metrics["rel_l2"]
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())

    print(f"\nDone — {time.time() - start:.1f}s total")

    if track_best and best_state is not None:
        final_rel_l2 = fixed_eval_history[-1][1] if fixed_eval_history else None
        print(
            f"[track_best] restoring epoch {best_epoch}/{epochs} weights "
            f"(fixed-grid rel_l2={best_rel_l2:.4e}) in place of the final "
            f"epoch's weights"
            + (f" (final-epoch fixed-grid rel_l2={final_rel_l2:.4e})"
               if final_rel_l2 is not None else "")
        )
        model.load_state_dict(best_state)
        if best_info is not None:
            best_info["best_epoch"] = best_epoch
            best_info["best_rel_l2"] = best_rel_l2
            best_info["final_epoch_rel_l2"] = final_rel_l2
            best_info["fixed_eval_history"] = fixed_eval_history

    return np.array(history)
