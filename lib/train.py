import os
import math
import torch
import logging
import imageio
import numpy as np
import torch.profiler
import matplotlib.pyplot as plt
import torch.autograd as autograd

from torch.func import functional_call, jacrev, vmap
from torch.nn.utils import parameters_to_vector

from math import inf
from typing import Tuple
from typing import Optional
from fvcore.nn import FlopCountAnalysis
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR

from lib.meshes import get_points_classes
from scipy.interpolate import LinearNDInterpolator

# NOTE: l'attuale scheduler è il migliore dal punto di vista della
# performance complessiva

class TrainerStep():
    def __init__(self, pinn, device=None, ckpt_dir=None, ckpt_interval=100):
        self.pinn = pinn
        self.device = device if device else torch.device('cpu')
        self.ckpt_dir = ckpt_dir
        self.ckpt_interval = ckpt_interval
        self.logger = logging.getLogger(__name__)

    def divide_epochs_linear(self, epochs: int, steps: int):
        base = [epochs // steps] * steps
        for i in range(epochs % steps):
            base[-(i + 1)] += 1  # assegna gli extra dalla fine
        scaled_parts = np.cumsum([0] + base).tolist()
        return scaled_parts

    def divide_epochs_exponential_growth(self, epochs:int, steps:int):
        base = np.exp(np.linspace(0, 1, steps))
        base = base / base.sum()
        scaled_parts = (base * epochs).round().astype(int)
        adjustment = epochs - scaled_parts.sum()
        scaled_parts[-1] += adjustment
        scaled_parts = list(scaled_parts.cumsum())
        scaled_parts.insert(0,0)
        return scaled_parts

    def train(self,
        bulk_data: Tuple[torch.Tensor],
        bdry_data: Tuple[torch.Tensor],
        init_data: Tuple[torch.Tensor],
        test_data: Tuple[torch.Tensor],
        indices: list,
        weights: dict,
        epochs: int,
        steps: int,
        divide: str,
        extra_epochs: int,
        lr_start: float,
        ckpt: bool):

        if ckpt:
            ckpt_path = os.path.join(self.ckpt_dir, f"weights_{steps}.pt")

        if divide == 'linear':
            decomposition_epochs = self.divide_epochs_linear(epochs, steps)
        elif divide == 'exponential':
            decomposition_epochs = self.divide_epochs_exponential_growth(epochs, steps)
        decomposition_epochs[-1] += extra_epochs

        optimizer = torch.optim.Adam(self.pinn.parameters(), lr=lr_start)
        scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=0.01, total_iters=decomposition_epochs[-1])

        self.logger.info(f"{'Epoch':>5} {'Step':>6} {'Error':>12} {'Total loss':>12} {'Bulk Loss':>12} {'Boundary Loss':>15} {'Initial Loss':>13} {'Learning rate':>14}")

        min_error = inf
        errors = [list() for _ in range(6)]
        flops = list()
        cumulative_flops = 0.0

        epoch_global = 0  # contatore epoche globale

        for step in range(steps):
            try:
                bulk_data_temp = tuple(x[:indices[step]] for x in bulk_data)
            except:
                break

            # range epoche per questo step
            epoch_start = decomposition_epochs[step]
            epoch_end = decomposition_epochs[step + 1]

            profiling_done = False
            for epoch in range(epoch_start, epoch_end):
                if not profiling_done:
                    with torch.profiler.profile(
                        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                        record_shapes=True,
                        with_flops=True,
                        with_stack=False
                    ) as prof:

                        optimizer.zero_grad()

                        bulk_loss = self.pinn.bulk_loss(bulk_data_temp)
                        total_loss = weights.get('bulk', 1.0) * bulk_loss

                        if bdry_data is not None:
                            boundary_loss = self.pinn.boundary_loss(bdry_data)
                            total_loss += weights.get('boundary', 1.0) * boundary_loss
                        else:
                            boundary_loss = torch.tensor(0.0, device=self.device)

                        if init_data is not None:
                            initial_loss = self.pinn.initial_loss(init_data)
                            total_loss += weights.get('initial', 1.0) * initial_loss
                        else:
                            initial_loss = torch.tensor(0.0, device=self.device)

                        total_loss.backward()
                        optimizer.step()
                        scheduler.step()

                    try:
                        step_flops = sum(e.flops for e in prof.key_averages() if e.flops is not None) / 1e9
                    except:
                        step_flops = 0.0

                    profiling_done = True
                    
                else:
                    # Normale epoca di training
                    optimizer.zero_grad()

                    bulk_loss = self.pinn.bulk_loss(bulk_data_temp)
                    total_loss = weights.get('bulk', 1.0) * bulk_loss

                    if bdry_data is not None:
                        boundary_loss = self.pinn.boundary_loss(bdry_data)
                        total_loss += weights.get('boundary', 1.0) * boundary_loss
                    else:
                        boundary_loss = torch.tensor(0.0, device=self.device)

                    if init_data is not None:
                        initial_loss = self.pinn.initial_loss(init_data)
                        total_loss += weights.get('initial', 1.0) * initial_loss
                    else:
                        initial_loss = torch.tensor(0.0, device=self.device)

                    total_loss.backward()
                    optimizer.step()
                    scheduler.step()
                
                cumulative_flops += step_flops

                # Logging e validazione
                if epoch % 100 == 0 or epoch == decomposition_epochs[-1] - 1:
                    self.pinn.eval()
                    with torch.no_grad():
                        u_pred = self.pinn(test_data[0], test_data[1])
                        u_real = test_data[2]

                        sol_l1 = torch.abs(u_real).mean().item()
                        sol_l2 = torch.sqrt((u_real ** 2).mean()).item()

                        mse = ((u_pred - u_real) ** 2).mean().item()
                        mae = torch.abs(u_pred - u_real).mean().item()
                        mxe = torch.max(torch.abs(u_pred - u_real)).item()
                        l1re = mae / sol_l1
                        l2re = math.sqrt(mse) / sol_l2
                        crmse = torch.abs((u_pred - u_real).mean()).item()

                    self.pinn.train()

                    if ckpt and mae < min_error:
                        min_error = mae
                        torch.save(self.pinn.state_dict(), ckpt_path)

                    errors[0].append(mae)
                    errors[1].append(mse)
                    errors[2].append(mxe)
                    errors[3].append(l1re)
                    errors[4].append(l2re)
                    errors[5].append(crmse)
                    flops.append(cumulative_flops)

                    self.logger.info(
                        f"{epoch:5d} {step:6d} {mae:12.6f} {total_loss.item():12.6f} {bulk_loss.item():12.6f} "
                        f"{boundary_loss.item():15.6f} {initial_loss.item():13.6f} "
                        f"{scheduler.get_last_lr()[0]:14.6f}"
                    )

                epoch_global += 1

        return flops, errors
    
    def train_with_gradient_gif(
        self,
        bulk_data: Tuple[torch.Tensor],
        bdry_data: Tuple[torch.Tensor],
        init_data: Tuple[torch.Tensor],
        indices: list,
        weights: dict,
        epochs: int,
        steps: int,
        divide: str,
        extra_epochs: int,
        lr_start: float,
        ckpt: bool,
        savepath: str,
        plotter: Optional[object] = None
    ):
        image_dir = savepath
        os.makedirs(image_dir, exist_ok=True)

        if ckpt:
            ckpt_path = os.path.join(self.ckpt_dir, f"weights_{steps}.pt")

        if divide == 'linear':
            decomposition_epochs = self.divide_epochs_linear(epochs, steps)
        elif divide == 'exponential':
            decomposition_epochs = self.divide_epochs_exponential_growth(epochs, steps)
        decomposition_epochs[-1] += extra_epochs

        flop_counter = FlopCountAnalysis(self.pinn.network, torch.rand((1, 2), device=self.device))
        flops_per_point = flop_counter.total()
        cumulative_flops = 0.0

        optimizer = torch.optim.Adam(self.pinn.parameters(), lr=lr_start)
        scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=steps)

        self.logger.info(f"{'Epoch':>5} {'Step':>6} {'Total loss':>12} {'Bulk Loss':>12} {'Boundary Loss':>15} {'Initial Loss':>13} {'Learning rate':>14}")

        min_loss = float('inf')
        losses = []
        flops = []

        for step in range(steps):
            bulk_data_temp = tuple(x[:indices[step]] for x in bulk_data)

            for epoch in range(decomposition_epochs[step], decomposition_epochs[step + 1]):
                optimizer.zero_grad()

                bulk_loss = self.pinn.bulk_loss(bulk_data_temp)
                total_loss = weights.get('bulk', 1.0) * bulk_loss

                if bdry_data is not None:
                    boundary_loss = self.pinn.boundary_loss(bdry_data)
                    total_loss += weights.get('boundary', 1.0) * boundary_loss
                else:
                    boundary_loss = torch.tensor(0.0)

                if init_data is not None:
                    initial_loss = self.pinn.initial_loss(init_data)
                    total_loss += weights.get('initial', 1.0) * initial_loss
                else:
                    initial_loss = torch.tensor(0.0)

                total_loss.backward()
                optimizer.step()

                if plotter:
                    plotter.accumulate_gradients()

                n_points = bulk_data_temp[0].shape[0]
                n_bdry = bdry_data[0].shape[0] if bdry_data is not None else 0
                n_init = init_data[0].shape[0] if init_data is not None else 0
                cumulative_flops += flops_per_point * (n_points + n_bdry + n_init)

                if (epoch % 100 == 0 or epoch == decomposition_epochs[-1] - 1):
                    self.pinn.eval()
                    bulk_loss = self.pinn.bulk_loss(bulk_data)
                    total_loss = weights.get('bulk', 1.0) * bulk_loss + \
                                weights.get('boundary', 1.0) * boundary_loss + \
                                weights.get('initial', 1.0) * initial_loss

                    if plotter:
                        plotter.average_gradients()

                        with torch.no_grad():
                            u = self.pinn(bulk_data[0], bulk_data[1]).detach().cpu().numpy()

                        x = bulk_data[0].detach().cpu().numpy()
                        y = bulk_data[1].detach().cpu().numpy()

                        # Crea figura combinata
                        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 8), gridspec_kw={'height_ratios': [2, 3]})

                        # Plot soluzione
                        sc = ax1.scatter(x, y, c=u, s=6, cmap='bwr')
                        ax1.axis('off')
                        fig.colorbar(sc, ax=ax1, orientation='horizontal', fraction=0.046, pad=0.04)

                        # Plot rete con gradienti
                        plotter.plot(ax2, show_gradients=True)

                        # Salva figura combinata
                        plt.tight_layout()
                        plt.savefig(os.path.join(image_dir, f"combined_{epoch:05d}.png"))
                        plt.close(fig)

                        plotter.reset_gradients()

                    self.pinn.train()

                    if ckpt and total_loss.item() < min_loss:
                        min_loss = total_loss.item()
                        torch.save(self.pinn.state_dict(), ckpt_path)

                    losses.append(total_loss.item())
                    flops.append(cumulative_flops)

                    self.logger.info(
                        f"{epoch:5d} {step:6d} {total_loss.item():12.6f} {bulk_loss.item():12.6f} "
                        f"{boundary_loss.item():15.6f} {initial_loss.item():13.6f} "
                        f"{scheduler.get_last_lr()[0]:14.6f}"
                    )

            scheduler.step()

        def make_combined_gif(prefix, output_name):
            files = sorted(
                f for f in os.listdir(image_dir)
                if f.startswith(prefix) and f.endswith(".png")
            )
            images = [imageio.v2.imread(os.path.join(image_dir, f)) for f in files]

            if images:
                imageio.mimsave(os.path.join(image_dir, output_name), images, fps=5)
                for f in files:
                    os.remove(os.path.join(image_dir, f))

        if plotter:
            make_combined_gif("combined_", "combined.gif")


        #if plotter:
            #make_gif("grad_", "gradients.gif")
            #make_gif("solution_", "solutions.gif")

        return flops, losses
    
    def train_with_error(self,
        bulk_data:Tuple[torch.Tensor],
        bdry_data:Tuple[torch.Tensor],
        init_data:Tuple[torch.Tensor],
        test_data:Tuple[torch.Tensor],
        indices:list,
        weights:dict,
        epochs:int,
        steps:int,
        lr_start:float,
        ckpt:bool,
        savepath:str):
        
        if ckpt:
            ckpt_path = os.path.join(self.ckpt_dir,f"weights_{steps}.pt")

        decomposition_epochs = self.divide_epochs_exponential_growth(epochs, steps)
        decomposition_epochs[-1] += 4000

        # Flops
        cumulative_flops = 0.0
        flop_counter = FlopCountAnalysis(self.pinn.network, torch.rand((1,2), device=self.device))
        flops_per_point = flop_counter.total()
        
        # Ottimizzatore
        optimizer = torch.optim.Adam(self.pinn.parameters(), lr=lr_start)
        # Scheduler
        scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=steps)
        #t_max = decomposition_epochs[-1]
        #scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=0.01, total_iters=t_max)
        #scheduler = CosineAnnealingLR(optimizer, T_max=t_max, eta_min=1e-4, last_epoch=-1)

        self.logger.info(f"{'Epoch':>5} {'Step':>6} {'Error':>12} {'Total loss':>12} {'Bulk Loss':>12} {'Boundary Loss':>15} {'Initial Loss':>13} {'Learning rate':>14}")

        min_error = inf
        errors = [list() for i in range(6)]
        flops = list()
        for step in range(steps):
            try:
                bulk_data_temp = tuple(x[:indices[step]] for x in bulk_data)
            except:
                break

            for epoch in range(decomposition_epochs[step],decomposition_epochs[step+1]):

                optimizer.zero_grad()

                bulk_loss = self.pinn.bulk_loss(bulk_data_temp)
                total_loss = weights.get('bulk', 1.0) * bulk_loss

                if bdry_data is not None:
                    boundary_loss = self.pinn.boundary_loss(bdry_data)
                    total_loss += weights.get('boundary', 1.0) * boundary_loss
                else:
                    boundary_loss = torch.tensor(0.0)

                if init_data is not None:
                    initial_loss = self.pinn.initial_loss(init_data)
                    total_loss += weights.get('initial', 1.0) * initial_loss
                else:
                    initial_loss = torch.tensor(0.0)

                total_loss.backward()
                optimizer.step()
                #scheduler.step()

                # flops
                n_points = bulk_data_temp[0].shape[0]
                n_bdry = bdry_data[0].shape[0] if bdry_data is not None else 0
                n_init = init_data[0].shape[0] if init_data is not None else 0
                cumulative_flops += flops_per_point * (n_points + n_bdry + n_init)

                if epoch % 100 == 0 or epoch == decomposition_epochs[-1]-1:
                    #scheduler.step()
                    self.pinn.eval()
                    u_pred = self.pinn(test_data[0], test_data[1])
                    u_real = test_data[2]

                    sol_l1 = torch.abs(u_real).mean().item()
                    sol_l2 = torch.sqrt((u_real**2).mean()).item()

                    mse = ((u_pred-u_real)**2).mean().item()
                    mae = torch.abs(u_pred-u_real).mean().item()
                    mxe = torch.max(torch.abs(u_pred-u_real)).item()
                    l1re = mae / sol_l1
                    l2re = math.sqrt(mse) / sol_l2
                    crmse = torch.abs((u_pred - u_real).mean()).item()
                    
                    '''
                    _, axs = plt.subplots(1, 1, figsize=(4, 3))
                    x = test_data[0].detach().cpu().numpy()
                    y = test_data[1].detach().cpu().numpy()
                    u = u_pred.detach().cpu().numpy()
                    axs.scatter(x, y, c=u, s=4, alpha=1.0, cmap='bwr')

                    x = bulk_data_temp[0].detach().cpu().numpy()
                    y = bulk_data_temp[1].detach().cpu().numpy()
                    axs.scatter(x, y, c='k', s=4, alpha=0.2)

                    axs.axis('off')
                    plt.savefig(os.path.join(savepath,f'solution_{steps}_{epoch}.png'))
                    plt.close()
                    '''

                    self.pinn.train()

                    if ckpt and mae < min_error:
                        min_error = mae
                        torch.save(self.pinn.state_dict(), ckpt_path)

                    errors[0].append(mae)
                    errors[1].append(mse)
                    errors[2].append(mxe)
                    errors[3].append(l1re)
                    errors[4].append(l2re)
                    errors[5].append(crmse)
                    flops.append(cumulative_flops)

                    self.logger.info(
                        f"{epoch:5d} {step:6d} {mae:12.6f} {total_loss.item():12.6f} {bulk_loss.item():12.6f} "
                        f"{boundary_loss.item():15.6f} {initial_loss.item():13.6f} "
                        f"{scheduler.get_last_lr()[0]:14.6f}"
                    )

            #scheduler.step()
                    
        return flops, errors
    
    def train_dot_prod(self,
        bulk_data: Tuple[torch.Tensor],
        bdry_data: Tuple[torch.Tensor],
        init_data: Tuple[torch.Tensor],
        test_data: Tuple[torch.Tensor],
        indices: list,
        weights: dict,
        epochs: int,
        steps: int,
        divide: str,
        extra_epochs: int,
        lr_start: float,
        ckpt: bool,
        savepath: str,
        mesh,
        time_axis,
        boundary_type):

        if ckpt:
            ckpt_path = os.path.join(self.ckpt_dir, f"weights_{steps}.pt")

        if divide == 'linear':
            decomposition_epochs = self.divide_epochs_linear(epochs, steps)
        elif divide == 'exponential':
            decomposition_epochs = self.divide_epochs_exponential_growth(epochs, steps)
        decomposition_epochs[-1] += extra_epochs

        #optimizer = torch.optim.Adam(self.pinn.parameters(), lr=lr_start)
        optimizer = torch.optim.SGD(self.pinn.parameters(), lr=lr_start)
        scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=0.01, total_iters=decomposition_epochs[-1])

        self.logger.info(f"{'Epoch':>5} {'Step':>6} {'Error':>12} {'Total loss':>12} {'Bulk Loss':>12} {'Boundary Loss':>15} {'Initial Loss':>13} {'Learning rate':>14}")

        # Prepare test data
        x_test, y_test, u_test = test_data
        import sys
        import matplotlib.pyplot as plt
        classes = get_points_classes(
            x_test,
            y_test,
            mesh,
            16, #steps
            time_axis,
            boundary_type
        )

        # Parameters
        min_error = inf
        errors = [list() for _ in range(6)]
        flops = list()

        for step in range(steps):
            try:
                bulk_data_temp = tuple(x[:indices[step]] for x in bulk_data)
            except:
                break

            for epoch in range(decomposition_epochs[step], decomposition_epochs[step + 1]):

                optimizer.zero_grad()
                if epoch % 1 == 0 or epoch == decomposition_epochs[-1] - 1:

                    bulk_loss = self.pinn.bulk_loss(bulk_data_temp)
                    total_loss = weights.get('bulk', 1.0) * bulk_loss
                    if bdry_data is not None:
                        boundary_loss = self.pinn.boundary_loss(bdry_data)
                        total_loss += weights.get('boundary', 1.0) * boundary_loss
                    else:
                        boundary_loss = torch.tensor(0.0)
                    if init_data is not None:
                        initial_loss = self.pinn.initial_loss(init_data)
                        total_loss += weights.get('initial', 1.0) * initial_loss
                    else:
                        initial_loss = torch.tensor(0.0)
                    grad = torch.autograd.grad(total_loss, self.pinn.parameters(), retain_graph=True, create_graph=False)
                    grad_loss = torch.cat([g.reshape(-1) for g in grad]).detach()
                    
                    params_dict = dict(self.pinn.named_parameters())

                    def pointwise_mse(params, x, y, u_target):
                        pred = functional_call(self.pinn, params, (x.unsqueeze(0), y.unsqueeze(0)))
                        diff = pred.squeeze(0) - u_target.squeeze(0)
                        return diff ** 2

                    grad_fn = jacrev(pointwise_mse, argnums=0)

                    grads_err = vmap(
                        lambda x, y, u: parameters_to_vector([
                            grad_fn(params_dict, x, y, u)[name] for name, _ in self.pinn.named_parameters()
                        ])
                    )(x_test, y_test, u_test)
                    
                    with torch.no_grad():
                        dot_products = - grads_err @ grad_loss

                if epoch % 1 == 0 or epoch == decomposition_epochs[-1] - 1:
                    
                    for i in range(1,17):
                        mask = classes==i
                        dot_products[mask] = dot_products[mask].sum()

                    points = np.column_stack((x_test.detach().cpu().numpy(), y_test.detach().cpu().numpy()))
                    values = dot_products.detach().cpu().numpy()
                    interp = LinearNDInterpolator(points, values)
                    val_v = interp(mesh.vertices[:,0],mesh.vertices[:,1])

                    # Visualizza
                    plt.figure(figsize=(8,6))
                    plt.tripcolor(mesh.vertices[:,0], mesh.vertices[:,1],
                          mesh.faces, val_v, shading='gouraud', cmap='seismic')
                    plt.xlim((0,4))
                    plt.ylim((-2,2))
                    plt.colorbar()
                    plt.savefig(savepath+f"/step_{step}_epoch_{epoch}.png")
                    plt.close()

                    if epoch == 20:
                        import sys
                        sys.exit()

                optimizer.zero_grad()

                bulk_loss = self.pinn.bulk_loss(bulk_data_temp)
                total_loss = weights.get('bulk', 1.0) * bulk_loss

                if bdry_data is not None:
                    boundary_loss = self.pinn.boundary_loss(bdry_data)
                    total_loss += weights.get('boundary', 1.0) * boundary_loss
                else:
                    boundary_loss = torch.tensor(0.0)

                if init_data is not None:
                    initial_loss = self.pinn.initial_loss(init_data)
                    total_loss += weights.get('initial', 1.0) * initial_loss
                else:
                    initial_loss = torch.tensor(0.0)

                total_loss.backward()
                optimizer.step()
                scheduler.step()

        return flops, errors



''' CAMPO VETTORIALE
# === 1. Costruzione della griglia regolare ===
                    x_lin = np.linspace(-4, 4, 100)
                    y_lin = np.linspace(-2, 2, 50)
                    x_grid, y_grid = np.meshgrid(x_lin, y_lin)  # shape (ny, nx)

                    # === 2. Interpolazione del campo scalare ===
                    # Supponiamo tu abbia già:
                    # - points: (N, 2) array di coordinate dei punti noti
                    # - values: (N,) array di valori scalari noti
                    interp = LinearNDInterpolator(points, values)

                    # Interpola sulla griglia
                    val_v_grid = interp(x_grid, y_grid)  # shape (ny, nx)

                    # === 3. Calcolo del gradiente numerico ===
                    dx = x_lin[1] - x_lin[0]  # passo in x
                    dy = y_lin[1] - y_lin[0]  # passo in y

                    # np.gradient restituisce (∂/∂y, ∂/∂x)
                    grad_y, grad_x = np.gradient(val_v_grid, dy, dx)
                    norm = np.sqrt(grad_x**2+grad_y**2)+1e-10
                    grad_y *= 0.1/norm
                    grad_x *= 0.1/norm

                    # === 4. Plot del campo scalare + campo vettoriale ===
                    plt.figure(figsize=(8, 6))
                    plt.contourf(x_grid, y_grid, val_v_grid, 100, cmap='seismic')
                    plt.quiver(x_grid, y_grid, grad_x, grad_y, color='black', scale=1, width=0.002,scale_units='xy',)
                    plt.xlim([-4, 4])
                    plt.ylim([-4, 4])
                    plt.colorbar(label='Interpolated scalar field')
                    plt.title('Gradient field of val_v')
                    plt.tight_layout()
                    plt.savefig(savepath + f"/step_{step}_epoch_{epoch}_gradient_grid.png", dpi=150)
                    plt.close()
'''