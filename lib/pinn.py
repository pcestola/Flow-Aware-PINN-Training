import torch
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

class HyperbolicPINN(nn.Module):
    
    def __init__(self, network, residual, device):
        super().__init__()
        self.network = network
        self.residual = residual
        self.device = device
    
    def forward(self, t, x):
        return self.network(torch.cat((t,x),dim=1))
    
    def bulk_loss(self, bulk_data):
        t = bulk_data[0]
        x = bulk_data[1]
        u = self.forward(t,x)
        residual = self.residual.compute_residual(u,t,x)
        return torch.mean(residual**2)
    
    def boundary_loss(self, boundary_data):
        t = boundary_data[0]
        x = boundary_data[1]
        u0 = boundary_data[2]
        u = self.forward(t,x)
        residual = u-u0
        return torch.mean(residual**2)
    
    def initial_loss(self, intial_data):
        t = intial_data[0]
        x = intial_data[1]
        u0 = intial_data[2]
        v0 = intial_data[3]
        u = self.forward(t,x)
        ut = gradient(u,(t,))
        residual_1, residual_2 = u-u0, ut-v0
        return torch.mean(residual_1**2), torch.mean(residual_2**2)
    
    def total_loss(self, data: dict, weights: dict, multiple: bool = False):
        
        device = self.device
        # TODO: non mi piace cosi
        loss = torch.tensor(0.0, device=device)
        bulk_loss = torch.tensor(0.0, device=device)
        initial_loss1 = torch.tensor(0.0, device=device)
        initial_loss2 = torch.tensor(0.0, device=device)
        boundary_loss = torch.tensor(0.0, device=device)

        if data['bulk'] != None:
            w = weights.get('bulk', 1.0)
            bulk_loss = self.bulk_loss(data['bulk'])
            loss += w * bulk_loss

        if data['initial'] != None:
            w = weights.get('initial', 1.0)
            initial_loss1, initial_loss2 = self.initial_loss(data['initial'])
            loss += w * initial_loss1 + w * initial_loss2

        if data['boundary'] != None:
            w = weights.get('boundary', 1.0)
            boundary_loss = self.boundary_loss(data['boundary'])
            loss += w * boundary_loss

        if False:
            grad_bulk = compute_gradient_norm(bulk_loss, self.network)
            grad_boundary = compute_gradient_norm(boundary_loss, self.network)
            grad_initial = compute_gradient_norm(initial_loss1, self.network)
            grad_initial = compute_gradient_norm(initial_loss2, self.network)
            print(f"[GRADIENTI] bulk: {grad_bulk:.2e}, boundary: {grad_boundary:.2e}, initial: {grad_initial:.2e}")

        if multiple:
            return loss, bulk_loss.item(), initial_loss1.item(), initial_loss2.item(), boundary_loss.item()
        else:
            return loss
    
class EllipticPINN(nn.Module):
    
    def __init__(self, network, residual, device):
        super().__init__()
        self.network = network
        self.residual = residual
        self.device = device
    
    def forward(self, t, x):
        return self.network(torch.cat((t,x),dim=1))
    
    def bulk_loss(self, bulk_data):
        t = bulk_data[0]
        x = bulk_data[1]
        u = self.forward(t,x)
        residual = self.residual.compute_residual(u,t,x)
        return torch.mean(residual**2)
    
    def boundary_loss(self, boundary_data):
        t = boundary_data[0]
        x = boundary_data[1]
        u0 = boundary_data[2]
        u = self.forward(t,x)
        residual = u-u0
        return torch.mean(residual**2)
    
    def total_loss(self, data: dict, weights: dict, multiple: bool = False):
        
        device = self.device
        # TODO: non mi piace cosi
        loss = torch.tensor(0.0, device=device)
        bulk_loss = torch.tensor(0.0, device=device)
        boundary_loss = torch.tensor(0.0, device=device)

        if data['bulk'] != None:
            w = weights.get('bulk', 1.0)
            bulk_loss = self.bulk_loss(data['bulk'])
            loss += w * bulk_loss

        if data['boundary'] != None:
            w = weights.get('boundary', 1.0)
            boundary_loss = self.boundary_loss(data['boundary'])
            loss += w * boundary_loss

        if multiple:
            return loss, bulk_loss.item(), 0, 0, boundary_loss.item()
        else:
            return loss


'''
    OLD CODE
'''

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
        
class PINN_OLD(nn.Module):
    def __init__(self, net, residual):
        super(PINN_OLD, self).__init__()
        self.net = net
        self.residual = residual
        
    def forward(self, x, t):
        return self.net(torch.cat([x, t], dim=1))

    def bulk_loss(self, data):
        x, t = data
        u = self.forward(x, t)
        residual = self.residual.compute_residual(u, x, t)
        return torch.mean(residual**2)

    def initial_loss(self, data):
        x, t, u_true, v_true = data
        u = self.forward(x, t)
        ut = gradient(u,(t,))
        residual1 = u - u_true
        residual2 = ut - v_true
        return torch.mean(residual1**2) + torch.mean(residual2**2)

    def boundary_loss(self, data):
        x, t, u_true = data
        u = self.forward(x, t)
        residual = u - u_true
        return torch.mean(residual**2)
