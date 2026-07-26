from dataclasses import dataclass, field
from src.utils import gradient
import torch
import math

@dataclass
class PDE:
# Base of the PDE class. This simply has a constant flat 0 solution

    x_min: float = 0.0  # bounds of spacetime interval
    x_max: float = 1.0
    t_min: float = 0.0
    t_max: float = 1.0

    def residual(self, u, u_x, points):
        """Takes model output and sample points and returns PDE residual."""
        return 0

    def exact_solution(self, x, t):
    # returns the exact solution of the PDE over the given spacetime interval
        shape = torch.broadcast_shapes(x.shape, t.shape)
        return torch.zeros(shape, dtype=x.dtype, device=x.device)

    def initial_condition(self, x):
    # returns IC : u(x,0) = f(x)
        return torch.zeros_like(x)

    def boundary_condition_left(self, t):
        """Return BC: u(0, t) = g1(t)"""
        return torch.zeros_like(t)

    def boundary_condition_right(self, t):
        """Return BC: u(1, t) = g2(t)"""
        return torch.zeros_like(t)


@dataclass
class HeatEquation(PDE):
# Implements the solution and IC/BC to the heat equation
# with constant BC and sine IC

# Defaults parameters : (can be called in init of class)
    freq: float = 0.5 # freq of IC - ! ONLY TAKES MULTIPLES OF 0.5 !
    alpha: float = 0.025 # heat diffusion parameter
    x_min: float = 0 #bounds of spacetime interval
    x_max: float = 1
    t_min: float = 0.0
    t_max: float = 1.0


    def residual(self, u, u_x, points):
        """Uses u and du/dx and sample points computes and returns residual du/dt - alpha * d²u/dx²"""

        u_xx = gradient(u_x, points)[:, 0:1]

        u_t = gradient(u, points)[:, 1:2]

        return u_t - self.alpha * u_xx


    def exact_solution(self, x, t):
        """Return exact solution u(x,t)."""
        return torch.exp(-self.alpha *4* self.freq * self.freq  * t) * torch.sin( #* math.pi ** 2
            2 * self.freq * x) #math.pi *
    def initial_condition(self, x):
        """Return IC: u(x, 0) = f(x)"""
        # return torch.sin(math.pi * x)
        return torch.sin(2 * self.freq * x) #* math.pi

    def boundary_condition_left(self, t):
        """Return BC: u(0, t) = g1(t)"""
        return torch.zeros_like(t)

    def boundary_condition_right(self, t):
        """Return BC: u(1, t) = g2(t)"""
        return torch.zeros_like(t)


@dataclass
class WaveEquation(PDE):

    c: float = 1.0  # Wave speed
    freq: float = 1.0

    # TODO : on pourrait faire une liste de waves pour avoir des interférences => spectre de frquences tres large

    def residual(self, u, u_x, points):
        """
        Compute Wave PDE residual.
        d²u/dt² - c² * d²u/dx² = 0
        """

        u_xx = gradient(u_x, points)[:, 0:1]

        u_t = gradient(u, points)[:, 1:2]
        u_tt = gradient(u_t, points)[:, 1:2]

        residual = u_tt - (self.c ** 2) * u_xx
        return residual
#    def residual(self, u, u_x, points):
#        """
#        Compute Wave PDE residual.
#        d²u/dt² - c² * d²u/dx² = 0
#        """
#        u_x_true = gradient(u, points)[:, 0:1]
#        u_xx = gradient(u_x_true, points)[:, 0:1]
#
#        u_t = gradient(u, points)[:, 1:2]
#        u_tt = gradient(u_t, points)[:, 1:2]
#
#        residual = u_tt - (self.c ** 2) * u_xx
#        return residual
    
    def exact_solution(self, x, t):

        return torch.sin(math.pi * self.freq * x) * torch.cos(math.pi * self.c * self.freq * t)

    def initial_condition(self, x):
        """IC: u(x, 0) = sin(pi*x)"""

        return torch.sin(math.pi * self.freq * x)

    def boundary_condition_left(self, t):
        """BC: u(0, t) = 0"""

        return torch.zeros_like(t)

    def boundary_condition_right(self, t):
        """BC: u(1, t) = 0"""

        return torch.zeros_like(t)
    
    def initial_velocity(self, x):
        """IC: u_t(x, 0) = 0"""
        return torch.zeros_like(x)