import torch
import torch.nn as nn
import torch.autograd as autograd
import matplotlib.pyplot as plt

# 1) Rete come prima
class FCN(nn.Module):
    def __init__(self, in_dim=2, out_dim=1, hidden_dim=50, layers=4):
        super().__init__()
        layers_list = [nn.Linear(in_dim, hidden_dim), nn.Tanh()]
        for _ in range(layers-1):
            layers_list += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        layers_list += [nn.Linear(hidden_dim, out_dim, bias=False)]
        self.net = nn.Sequential(*layers_list)
    def forward(self, x):
        return self.net(x)

# 2) Generazione punti collocazione + BC interno/esterno
def generate_training_points(N_f=20000, N_b=400, r_in=0.3, r_out=1.0):
    theta_f = torch.rand(N_f,1) * 2*torch.pi
    r_f = torch.sqrt(torch.rand(N_f,1)*(r_out**2 - r_in**2) + r_in**2)
    x_f = r_f * torch.cos(theta_f)
    y_f = r_f * torch.sin(theta_f)
    X_f = torch.cat([x_f, y_f], dim=1).requires_grad_(True)

    theta_b = torch.rand(N_b,1) * 2*torch.pi
    x_bi = r_in  * torch.cos(theta_b[:N_b//2])
    y_bi = r_in  * torch.sin(theta_b[:N_b//2])
    x_be = r_out * torch.cos(theta_b[N_b//2:])
    y_be = r_out * torch.sin(theta_b[N_b//2:])
    X_b = torch.cat([
        torch.cat([x_bi, y_bi], dim=1),
        torch.cat([x_be, y_be], dim=1)
    ], dim=0).requires_grad_(True)
    U_b = torch.cat([
        torch.ones(N_b//2,1),
        torch.zeros(N_b//2,1)
    ], dim=0)

    return X_f, X_b, U_b

# 3) Residuo PDE di Poisson: u_xx + u_yy = 0
def pde_residual(model, X):
    u = model(X)
    grad_u = autograd.grad(u, X, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_x, u_y = grad_u[:,0:1], grad_u[:,1:2]

    u_xx = autograd.grad(u_x, X, grad_outputs=torch.ones_like(u_x), create_graph=True)[0][:,0:1]
    u_yy = autograd.grad(u_y, X, grad_outputs=torch.ones_like(u_y), create_graph=True)[0][:,1:2]

    return u_xx + u_yy

# 4) Loss BC e PDE
def loss_bc(model, X_b, U_b):
    return torch.mean((model(X_b) - U_b)**2)

def loss_pde(model, X_f):
    return torch.mean(pde_residual(model, X_f)**2)

# 5) Training con e senza proiezioni
model_norm = FCN()
model_proj = FCN()
model_proj.load_state_dict(model_norm.state_dict())

X_f, X_b, U_b = generate_training_points()
opt_norm = torch.optim.Adam(model_norm.parameters(), lr=1e-3)
opt_proj = torch.optim.Adam(model_proj.parameters(), lr=1e-3)

epochs = 500
normal_lb, normal_lf = [], []
proj_lb, proj_lf     = [], []

for epoch in range(epochs):
    # Normale
    opt_norm.zero_grad()
    lb = loss_bc(model_norm, X_b, U_b)
    lf = loss_pde(model_norm, X_f)
    (lb + lf).backward()
    opt_norm.step()

    normal_lb.append(lb.item())
    normal_lf.append(lf.item())

    # Proiezioni
    opt_proj.zero_grad()
    lb_p = loss_bc(model_proj, X_b, U_b)
    lf_p = loss_pde(model_proj, X_f)

    params = list(model_proj.parameters())
    g_b = torch.autograd.grad(lb_p, params,   retain_graph=True, create_graph=True, allow_unused=True)
    g_f = torch.autograd.grad(lf_p, params,   retain_graph=True, create_graph=True, allow_unused=True)

    proj_grads_b, proj_grads_f = [], []
    eps = 1e-8
    for p, gb, gf in zip(params, g_b, g_f):
        if gb is None: gb = torch.zeros_like(p)
        if gf is None: gf = torch.zeros_like(p)

        b = gb.view(-1)
        f = gf.view(-1)

        # grad BC ⟂ grad PDE
        comp_b_on_f = (torch.dot(b, f)/(f.norm()**2 + eps)) * f
        perp_b = (b - comp_b_on_f).view_as(p)
        proj_grads_b.append(perp_b)

        # grad PDE ⟂ grad BC
        comp_f_on_b = (torch.dot(f, b)/(b.norm()**2 + eps)) * b
        perp_f = (f - comp_f_on_b).view_as(p)
        proj_grads_f.append(perp_f)

    # 2) Primo passo: usa proj_grads_b
    for p, g in zip(params, proj_grads_b):
        p.grad = g
    opt_proj.step()
    opt_proj.zero_grad()

    # 3) Secondo passo: usa proj_grads_f
    for p, g in zip(params, proj_grads_f):
        p.grad = g
    opt_proj.step()
    opt_proj.zero_grad()

    proj_lb.append(lb_p.item())
    proj_lf.append(lf_p.item())

    if epoch % 50 == 0:
        print(f"Epoch {epoch:4d} | norm: BC={lb.item():.2e}, PDE={lf.item():.2e}"
              f" | proj: BC={lb_p.item():.2e}, PDE={lf_p.item():.2e}")

# Plots
plt.figure()
plt.plot(normal_lb, label='BC Loss Normale')
plt.plot(normal_lf, label='PDE Loss Normale')
plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.grid(); plt.savefig('normale.png')

plt.figure()
plt.plot(proj_lb,   label='BC Loss Proiezioni')
plt.plot(proj_lf,   label='PDE Loss Proiezioni')
plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.grid(); plt.savefig('proiezione.png')
