import torch
from src.pde import PDE


class DataSampler:
    """
    Like most PINNs, our methods samples point at random form the domain to compute the loss
    This class handles the selection of those points
    """

    def __init__(self, pde: PDE, device: torch.device, dtype: torch.dtype, n_interior: int = 128, n_initial: int = 128, n_boundary: int = 128
                 ):

        self.pde = pde
        self.device = device
        self.dtype = dtype

        self.n_interior = n_interior
        self.n_initial = n_initial
        self.n_boundary = n_boundary

    def sample_interior(self, n: int = None) -> torch.Tensor:
        """
        Sample interior points where PDE is enforced.

        Returns:
            xt: tensor of shape (n, 2) with columns [x, t], requires_grad=True
        """
        if n is None:
            n = self.n_interior

        x = torch.rand(n, 1, device=self.device, dtype=self.dtype)
        t = torch.rand(n, 1, device=self.device, dtype=self.dtype)

        # Scale to domain
        x = self.pde.x_min + x * (self.pde.x_max - self.pde.x_min)
        t = self.pde.t_min + t * (self.pde.t_max - self.pde.t_min)

        xt = torch.cat([x, t], dim=1)
        xt.requires_grad_(True)
        return xt

    def sample_initial(self, n: int = None) -> torch.Tensor:
        """
        Sample initial condition points (t=0).

        Returns:
            xt: tensor of shape (n, 2) with columns [x, t]
        """
        if n is None:
            n = self.n_initial

        x = torch.rand(n, 1, device=self.device, dtype=self.dtype)
        x = self.pde.x_min + x * (self.pde.x_max - self.pde.x_min)

        t = torch.full_like(x, self.pde.t_min)
        xt = torch.cat([x, t], dim=1)
        return xt

    def sample_boundary(self, n: int = None) -> torch.Tensor:
        """
        Sample boundary condition points (x=0 and x=1).

        Returns:
            xt: tensor of shape (n, 2) with columns [x, t]
        """
        if n is None:
            n = self.n_boundary

        t = torch.rand(n, 1, device=self.device, dtype=self.dtype)
        t = self.pde.t_min + t * (self.pde.t_max - self.pde.t_min)

        # Split between x=0 and x=1
        half = n // 2
        x0 = torch.full((half, 1), self.pde.x_min, device=self.device, dtype=self.dtype)
        x1 = torch.full((n - half, 1), self.pde.x_max, device=self.device, dtype=self.dtype)
        x = torch.cat([x0, x1], dim=0)

        # Concatenate [x, t] along dimension 1 (not 0!)
        t_boundary = torch.rand(n, 1, device=self.device, dtype=self.dtype)
        t_boundary = self.pde.t_min + t_boundary * (self.pde.t_max - self.pde.t_min)

        xt = torch.cat([x, t_boundary], dim=1)
        return xt