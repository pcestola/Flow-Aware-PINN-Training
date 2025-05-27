import math
import torch
import numpy as np
import torch.nn as nn
from typing import Tuple

def compute_gradient_norm(loss, model):
    # Pulisce i gradienti
    model.zero_grad()
    # Calcola i gradienti del singolo termine
    loss.backward(retain_graph=True)
    
    # Calcola la norma L2 di tutti i gradienti
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm


def gradient(f, variables):
    grad = f
    for var in variables:
        ones = torch.ones_like(grad)
        grad = torch.autograd.grad(grad, var, grad_outputs=ones, create_graph=True, retain_graph=True)[0]
    return grad

class ResidualCalculator:
    def compute_residual(self, u, coords):
        raise NotImplementedError("PDE residual not implemented")

class WaveEquation(ResidualCalculator):
    def __init__(self, c:float=1.0):
        self.c = c

    def compute_residual(self, u, t, x):
        u_tt = gradient(u, (t,t))
        u_xx = gradient(u, (x,x))
        return u_tt - self.c**2 * u_xx

class BurgerEquation(ResidualCalculator):
    def __init__(self, c:float=0.01/math.pi):
        self.c = c

    def compute_residual(self, u, t, x):
        u_t = gradient(u, (t,))
        u_x = gradient(u, (x,))
        u_xx = gradient(u, (x,x))
        return u_t + u * u_x - self.c * u_xx
    
    def initial_condition(self, x):
        if isinstance(x,torch.Tensor):
            return -torch.sin(math.pi*x)
        else:
            return -np.sin(np.pi*x)

class HeatEquation(ResidualCalculator):
    def __init__(self, alpha:float=1.0):
        self.alpha = alpha

    def compute_residual(self, u, x, t):
        u_t = gradient(u, (t,))
        u_xx = gradient(u, (x,x))
        return u_t - self.alpha * u_xx

class LaplaceEquation(ResidualCalculator):
    
    def compute_residual(self, u, x, y):
        u_xx = gradient(u, (x,x))
        u_yy = gradient(u, (y,y))
        return u_xx + u_yy

class EikonalEquation(ResidualCalculator):
    def __init__(self, c:float=1.0):
        super().__init__()
        self.c = c

    def compute_residual(self, u, x, y):
        ux = gradient(u, (x,))
        uy = gradient(u, (y,))
        return ux**2 + uy**2 - self.c

class PINN(nn.Module):
    def __init__(self, network:nn.Module, residual:ResidualCalculator):
        super(PINN, self).__init__()
        self.network = network
        self.residual = residual
        
    def forward(self, t:torch.Tensor, x:torch.Tensor):
        return self.network(torch.cat([t,x],dim=1))
    
    def bulk_loss(self, data:Tuple[torch.Tensor]):
        t, x = data
        u = self.forward(t, x)
        residual = self.residual.compute_residual(u, t, x)
        return torch.mean(residual**2)

    def initial_loss(self, data:Tuple[torch.Tensor]):
        t, x, u_true, v_true = data
        u = self.forward(t, x)
        ut = gradient(u,(t,))
        residual1 = u - u_true
        residual2 = ut - v_true
        return torch.mean(residual1**2) + torch.mean(residual2**2)

    def boundary_loss(self, data:Tuple[torch.Tensor]):
        t, x, u_true = data
        u = self.forward(t, x)
        residual = u - u_true
        return torch.mean(residual**2)