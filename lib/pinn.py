import math
import torch
import numpy as np
import torch.nn as nn
from typing import Tuple

# GRADIENT
def gradient(f, variables):
    grad = f
    for var in variables:
        ones = torch.ones_like(grad)
        grad = torch.autograd.grad(grad, var, grad_outputs=ones, create_graph=True, retain_graph=True)[0]
    return grad


# RESIDUALS
class ResidualCalculator:
    def compute_residual(self, u, coords):
        raise NotImplementedError("PDE residual not implemented")
    
    def initial_condition(self, data):
        raise NotImplementedError("PDE residual not implemented")
    
    def boundary_condition(self, data):
        raise NotImplementedError("PDE residual not implemented")

class WaveEquation(ResidualCalculator):
    def __init__(self, c:float=1.0):
        self.c = c

    def compute_residual(self, u, t, x):
        u_tt = gradient(u, (t,t))
        u_xx = gradient(u, (x,x))
        return u_tt - self.c**2 * u_xx

    def initial_condition(self, data):
        x = data[:,1:2]
        if isinstance(x,torch.Tensor):
            return torch.exp(-4*x**2), torch.zeros_like(x)
        else:
            return np.exp(-4*x**2), np.zeros_like(x)
    
    def boundary_condition(self, data):
        if isinstance(data,torch.Tensor):
            return torch.zeros((data.shape[0],1))
        else:
            return np.zeros((data.shape[0],1))

class HeatEquation(ResidualCalculator):
    def __init__(self, alpha:float=1.0):
        self.alpha = alpha

    def compute_residual(self, u, t, x):
        u_t = gradient(u, (t,))
        u_xx = gradient(u, (x,x))
        return u_t - self.alpha * u_xx
    
    def initial_condition(self, data):
        x = data[:,1]
        if isinstance(x,torch.Tensor):
            return torch.exp(-4*x**2)
        else:
            return np.exp(-4*x**2)
    
    def boundary_condition(self, data):
        if isinstance(data,torch.Tensor):
            return torch.zeros((data.shape[0],1))
        else:
            return np.zeros((data.shape[0],1))

class LaplaceEquation(ResidualCalculator):
    
    def compute_residual(self, u, x, y):
        u_xx = gradient(u, (x,x))
        u_yy = gradient(u, (y,y))
        return u_xx + u_yy
    
    def initial_condition(self, data):
        if isinstance(data,torch.Tensor):
            return torch.zeros((data.shape[0],1))
        else:
            return np.zeros((data.shape[0],1))

    def boundary_condition(self, data):
        if isinstance(data,torch.Tensor):
            return torch.zeros((data.shape[0],1))
        else:
            return np.zeros((data.shape[0],1))

class EikonalEquation(ResidualCalculator):
    def __init__(self, c:float=1.0):
        super().__init__()
        self.eps = 1
        self.c = c

    def compute_residual(self, u, x, y):
        ux = gradient(u, (x,))
        uy = gradient(u, (y,))
        if self.eps >= 0.0001:
            uxx = gradient(u, (x,x))
            uyy = gradient(u, (y,y))
            residual = -self.eps * (uxx + uyy) + (1.0 + 0.5*torch.sin(2*math.pi*x)*torch.sin(2*math.pi*y))*torch.sqrt(ux**2 + uy**2) - 1.0
            self.eps *= 0.99
        else:
            residual = (1.0 + 0.5*torch.sin(2*math.pi*x)*torch.sin(2*math.pi*y))*torch.sqrt(ux**2 + uy**2) - 1.0
        return residual
    
    def boundary_condition(self, data):
        if isinstance(data,torch.Tensor):
            return torch.zeros((data.shape[0],1))
        else:
            return np.zeros((data.shape[0],1))

# PINNACLE
class WaveEquation_1D(ResidualCalculator):
    def __init__(self, c:float=0.25):
        self.c = c

    def compute_residual(self, u, t, x):
        u_tt = gradient(u, (t,t))
        u_xx = gradient(u, (x,x))
        return u_tt - self.c * u_xx

    def initial_condition(self, data):
        x = data[:,1:2]
        if isinstance(x,torch.Tensor):
            return torch.sin(math.pi*x)+0.5*torch.sin(4*math.pi*x), torch.zeros_like(x)
        else:
            return np.sin(np.pi*x)+0.5*np.sin(4*np.pi*x), np.zeros_like(x)
    
    def boundary_condition(self, data):
        if isinstance(data,torch.Tensor):
            return torch.zeros((data.shape[0],1))
        else:
            return np.zeros((data.shape[0],1))

class Burgers_1D(ResidualCalculator):
    
    def __init__(self, c:float=0.01/math.pi):
        self.c = c

    def compute_residual(self, u, t, x):
        u_t = gradient(u, (t,))
        u_x = gradient(u, (x,))
        u_xx = gradient(u, (x,x))
        return u_t + u * u_x - self.c * u_xx
    
    def initial_condition(self, data):
        x = data[:,1:2]
        if isinstance(x,torch.Tensor):
            return -torch.sin(math.pi*x), None
        else:
            return -np.sin(np.pi*x), None
    
    def boundary_condition(self, data):
        if isinstance(data,torch.Tensor):
            return torch.zeros((data.shape[0],1))
        else:
            return np.zeros((data.shape[0],1))

class Poisson_2D_C(ResidualCalculator):
    
    def compute_residual(self, u, x, y):
        u_xx = gradient(u, (x,x))
        u_yy = gradient(u, (y,y))
        return u_xx + u_yy

    def boundary_condition(self, data):
        x = data[:,0]
        y = data[:,1]
        if isinstance(x,torch.Tensor):
            condition = torch.zeros((data.shape[0],1))
            mask = (torch.abs(x)>3.8) | (torch.abs(y)>3.8)
            condition[mask] = 1.0
            return condition
        else:
            condition = np.zeros((data.shape[0],1))
            mask = (np.abs(x)>3.8) | (np.abs(y)>3.8)
            condition[mask] = 1.0
            return condition

class Poisson_2D_CG(ResidualCalculator):
    def __init__(self):
        super().__init__()
        self.mu1 = 1.0
        self.mu2 = 4.0
        self.A = 10.0
        self.k = 8.0
    
    def compute_residual(self, u, x, y):
        u_xx = gradient(u, (x,x))
        u_yy = gradient(u, (y,y))
        f = self.A*(self.mu1**2 + self.mu2**2 + x**2 + y**2)*torch.sin(self.mu1*torch.pi*x)*torch.sin(self.mu2*torch.pi*y)
        return - u_xx - u_yy + self.k**2*u - f

    def boundary_condition(self, data):
        x = data[:,0]
        y = data[:,1]
        if isinstance(x,torch.Tensor):
            condition = torch.ones((data.shape[0],1))
            mask = (torch.abs(x)>0.98) | (torch.abs(y)>0.98)
            condition[mask] = 0.2
            return condition
        else:
            condition = np.ones((data.shape[0],1))
            mask = (np.abs(x)>0.98) | (np.abs(y)>0.98)
            condition[mask] = 0.2
            return condition

class NS_2D_C(ResidualCalculator):
    def __init__(self, nu: float = 0.01):
        super().__init__()
        self.nu = nu

    def compute_residual(self, u_pred, x, y):
        # u_pred: (N, 3) => [u, v, p]
        u = u_pred[:, 0:1]
        v = u_pred[:, 1:2]
        p = u_pred[:, 2:3]

        # Gradienti del campo vettoriale e della pressione
        u_x = gradient(u, (x,))
        u_y = gradient(u, (y,))
        v_x = gradient(v, (x,))
        v_y = gradient(v, (y,))

        p_x = gradient(p, (x,))
        p_y = gradient(p, (y,))

        u_xx = gradient(u, (x,x))
        u_yy = gradient(u, (y,y))
        v_xx = gradient(v, (x,x))
        v_yy = gradient(v, (y,y))

        # Equazioni di Navier-Stokes
        res_u = u * u_x + v * u_y + p_x - self.nu * (u_xx + u_yy)
        res_v = u * v_x + v * v_y + p_y - self.nu * (v_xx + v_yy)

        # Incompressibilità (divergenza zero)
        res_div = u_x + v_y

        return torch.cat([res_u, res_v, res_div], dim=1)

    def boundary_condition(self, data):
        x = data[:, 0]
        y = data[:, 1]
        if isinstance(x, torch.Tensor):
            condition = torch.full((data.shape[0], 3), float('nan'))  # NaN = "non imposto"
            # inflow
            mask = y == 1.0
            condition[mask, 0] = 4 * x[mask] * (1 - x[mask])  # u_x inlet
            condition[mask, 1] = 0.0                          # u_y inlet
            # walls
            mask = (y == 0.0) | (x == 0.0) | (x == 1.0)
            condition[mask, 0] = 0.0
            condition[mask, 1] = 0.0
            # pressure
            mask = (x == 0.0) & (y == 0.0)
            condition[mask, 2] = 0.0
            return condition
        else:
            condition = np.full((data.shape[0], 3), np.nan)
            # inflow
            mask = y == 1.0
            condition[mask, 0] = 4 * x[mask] * (1 - x[mask])  # u_x inlet
            condition[mask, 1] = 0.0                          # u_y inlet
            # walls
            mask = (y == 0.0) | (x == 0.0) | (x == 1.0)
            condition[mask, 0] = 0.0
            condition[mask, 1] = 0.0
            # pressure
            mask = (x == 0.0) & (y == 0.0)
            condition[mask, 2] = 0.0
            return condition

class NS_2D_CG(ResidualCalculator):
    def __init__(self, nu: float = 0.01):
        super().__init__()
        self.nu = nu

    def compute_residual(self, u_pred, x, y):
        # u_pred: (N, 3) => [u, v, p]
        u = u_pred[:, 0:1]
        v = u_pred[:, 1:2]
        p = u_pred[:, 2:3]

        # Gradienti del campo vettoriale e della pressione
        u_x = gradient(u, (x,))
        u_y = gradient(u, (y,))
        v_x = gradient(v, (x,))
        v_y = gradient(v, (y,))

        p_x = gradient(p, (x,))
        p_y = gradient(p, (y,))

        u_xx = gradient(u, (x,x))
        u_yy = gradient(u, (y,y))
        v_xx = gradient(v, (x,x))
        v_yy = gradient(v, (y,y))

        # Equazioni di Navier-Stokes
        res_u = u * u_x + v * u_y + p_x - self.nu * (u_xx + u_yy)
        res_v = u * v_x + v * v_y + p_y - self.nu * (v_xx + v_yy)

        # Incompressibilità (divergenza zero)
        res_div = u_x + v_y

        return torch.cat([res_u, res_v, res_div], dim=1)

    def boundary_condition(self, data):
        x = data[:, 0]
        y = data[:, 1]
        if isinstance(x, torch.Tensor):
            condition = torch.full((data.shape[0], 3), float('nan'))  # NaN = "non imposto"
            # walls
            mask = x != 4.0
            condition[mask, 0] = 0.0
            condition[mask, 1] = 0.0
            # inflow
            mask = x == 0.0
            condition[mask, 0] = 4 * y[mask] * (1 - y[mask])  # u_x inlet
            condition[mask, 1] = 0.0                          # u_y inlet
            # outflow
            mask = x == 4.0
            condition[mask, 2] = 0.0
            return condition
        else:
            condition = np.full((data.shape[0], 3), np.nan)
            # walls
            mask = x != 4.0
            condition[mask, 0] = 0.0
            condition[mask, 1] = 0.0
            # inflow
            mask = x == 0.0
            condition[mask, 0] = 4 * y[mask] * (1 - y[mask])  # u_x inlet
            condition[mask, 1] = 0.0                          # u_y inlet
            # outflow
            mask = x == 4.0
            condition[mask, 2] = 0.0
            return condition

# PINN
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
        return torch.mean(torch.sum(residual**2, dim=1))

    def initial_loss(self, data:Tuple[torch.Tensor]):
        if len(data) == 3:
            t, x, u_true = data
            u = self.forward(t, x)
            residual = u - u_true
            return torch.mean(torch.sum(residual**2, dim=1))
        else:
            t, x, u_true, v_true = data
            u = self.forward(t, x)
            ut = gradient(u,(t,))
            residual1 = u - u_true
            residual2 = ut - v_true
            return torch.mean(torch.sum(residual1**2, dim=1)) + torch.mean(torch.sum(residual2**2, dim=1))

    def boundary_loss(self, data:Tuple[torch.Tensor]):
        t, x, u_true = data
        u = self.forward(t, x)
        residual = 0.0
        for i in range(u_true.shape[1]):
            mask = ~torch.isnan(u_true[:,i])
            residual += torch.mean((u[mask,i] - u_true[mask,i])**2)
        return residual