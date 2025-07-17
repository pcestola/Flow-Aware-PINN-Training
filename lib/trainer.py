import os
import sys
import math
import torch
import logging
import imageio
import numpy as np
import numpy.ma as ma
import torch.profiler
import matplotlib.pyplot as plt
import torch.autograd as autograd
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from math import inf
from typing import Tuple
from typing import Optional
from fvcore.nn import FlopCountAnalysis
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR

from lib.meshes import get_points_classes
from scipy.interpolate import LinearNDInterpolator

from functorch import vmap, jacrev, hessian
from torch.nn.utils import parameters_to_vector
from torch.func import functional_call  # se usi torch>=2.0


# TODO: controlla dal paper che la formula sia la stessa
def compute_ntk_boundary(model, x_bdry, x_init):
    """
    Calcola NTK usando jacrev + vmap + functional_call, compatibile con modelli che usano buffers
    """
    if len(x_bdry) == 2:
        x, y = x_bdry
    else:
        x, y, _ = x_bdry

    x = x.detach().clone().requires_grad_(True)
    y = y.detach().clone().requires_grad_(True)

    params_dict = dict(model.named_parameters())

    def function(params, x, y):
        pred = functional_call(model, params, (x.unsqueeze(1), y.unsqueeze(1)))
        return pred

    grad_fn = jacrev(function, argnums=0)

    grads = vmap(
        lambda x, y: parameters_to_vector([
            grad_fn(params_dict, x, y)[name] for name, _ in model.named_parameters()
        ])
    )(x, y)

    grad_fn = jacrev(function, argnums=0)
    
    return grads.T @ grads

# TODO: è fatto per funzionare solo con Possion_2d_c
def residual_functional(model_fn, params, t, x):
    # Funzione u(t, x) che restituisce scalare
    def u_scalar(ti, xi):
        ti, xi = ti.reshape((-1,1)), xi.reshape((-1,1))  # shape (1,1)
        return model_fn(params, (ti, xi)).squeeze()  # output: scalar

    # ∂²u/∂t²
    d2udt2 = vmap(lambda ti, xi: hessian(lambda t_: u_scalar(t_, xi), argnums=0)(ti))(t, x)
    # ∂²u/∂x²
    d2udx2 = vmap(lambda ti, xi: hessian(lambda x_: u_scalar(ti, x_), argnums=0)(xi))(t, x)

    residual = d2udt2.squeeze() + d2udx2.squeeze()

    return residual

def compute_ntk_residual(model, x_bulk):
    x, y = x_bulk
    params_dict = dict(model.named_parameters())
    
    # Wrappa il modello come funzione funzionale
    model_fn = lambda params, inputs: functional_call(model, params, inputs)

    def loss_fn(params, x, y):
        r = residual_functional(model_fn, params, x, y)
        return torch.mean(r**2)

    grad_fn = jacrev(loss_fn, argnums=0)

    grads = vmap(
        lambda x, y: parameters_to_vector([
            grad_fn(params_dict, x, y)[name] for name, _ in model.named_parameters()
        ])
    )(x, y)

    return grads.T @ grads

def compute_ntk_weights(model, x_bulk, x_bdry, x_init):
    
    trace_bulk = torch.trace(compute_ntk_boundary(model, x_bdry, x_init)).item()
    trace_bdry = torch.trace(compute_ntk_residual(model, x_bulk)).item()

    total_trace = trace_bulk + trace_bdry

    w_bulk = total_trace/trace_bulk
    w_bdry = total_trace/trace_bdry
    w_init = 0.0

    return w_bulk, w_bdry, w_init



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
        divide_mode: str,
        extra_epochs: int,
        lr_start: float,
        ckpt: bool):

        if ckpt:
            ckpt_path = os.path.join(self.ckpt_dir, f"weights_{steps}.pt")

        if divide_mode == 'linear':
            decomposition_epochs = self.divide_epochs_linear(epochs, steps)
        elif divide_mode == 'exponential':
            decomposition_epochs = self.divide_epochs_exponential_growth(epochs, steps)
        decomposition_epochs[-1] += extra_epochs

        optimizer = torch.optim.Adam(self.pinn.parameters(), lr=lr_start)
        scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=0.01, total_iters=decomposition_epochs[-1])

        self.logger.info(f"{'Epoch':>5} {'Step':>6} {'Error':>12} {'Total loss':>12} {'Bulk Loss':>12} {'Boundary Loss':>15} {'Initial Loss':>13} {'Learning rate':>14}")

        min_error = inf
        errors = [list() for _ in range(6)]
        flops = list()
        cumulative_flops = 0.0

        epoch_global = 0

        for step in range(steps):
            try:
                bulk_data_temp = tuple(x[:indices[step]] for x in bulk_data)
            except Exception as e:
                print(f"\n\n ERRORE: {e} \n\n")
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
    
    def train_ntk(self,
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

        epoch_global = 0

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
                        if bdry_data is not None:
                            boundary_loss = self.pinn.boundary_loss(bdry_data)
                        else:
                            boundary_loss = torch.tensor(0.0, device=self.device)
                        if init_data is not None:
                            initial_loss = self.pinn.initial_loss(init_data)
                        else:
                            initial_loss = torch.tensor(0.0, device=self.device)

                        w_bulk, w_bdry, w_init = compute_ntk_weights(
                            self.pinn,
                            bulk_data_temp,
                            bdry_data if bdry_data is not None else None,
                            init_data if init_data is not None else None
                        )

                        weights['bulk'] = w_bulk
                        weights['boundary'] = w_bdry
                        weights['initial'] = w_init
                        print(w_bulk, w_bdry, w_init)

                        total_loss = w_bulk * bulk_loss + w_bdry * boundary_loss + w_init * initial_loss

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
        divide:str,
        extra_epochs:int,
        lr_start:float,
        ckpt:bool,
        savepath:str,
        mesh):
        
        if ckpt:
            ckpt_path = os.path.join(self.ckpt_dir,f"weights_{steps}.pt")

        if divide == 'linear':
            decomposition_epochs = self.divide_epochs_linear(epochs, steps)
        elif divide == 'exponential':
            decomposition_epochs = self.divide_epochs_exponential_growth(epochs, steps)
        decomposition_epochs[-1] += extra_epochs

        # Flops
        cumulative_flops = 0.0
        flop_counter = FlopCountAnalysis(self.pinn.network, torch.rand((1,2), device=self.device))
        flops_per_point = flop_counter.total()
        
        # Ottimizzatore
        optimizer = torch.optim.Adam(self.pinn.parameters(), lr=lr_start)
        # Scheduler
        scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=0.01, total_iters=decomposition_epochs[-1])
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

            for epoch in range(decomposition_epochs[step], decomposition_epochs[step+1]):

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

                if epoch == decomposition_epochs[-1]-1:
                    self.pinn.eval()
                    u_pred = self.pinn(test_data[0], test_data[1])
                    u_real = test_data[2]
                    error = torch.mean((u_pred - u_real) ** 2).item()

                    target_pixels = 600
                    dpi_for_plot = 100
                    fig_size_inches = target_pixels / dpi_for_plot

                    fig = plt.figure(frameon=False)
                    fig.set_size_inches(fig_size_inches, fig_size_inches)
                    ax = plt.Axes(fig, [0., 0., 1., 1.])
                    ax.set_axis_off()
                    fig.add_axes(ax)

                    # Interpolazione dell'errore su mesh
                    x = test_data[0].detach().cpu().numpy()
                    y = test_data[1].detach().cpu().numpy()
                    #err = ((u_pred-u_real)**2).detach().cpu().numpy()
                    err = (u_pred).detach().cpu().numpy()

                    points = np.concatenate([x, y], axis=-1)
                    interp = LinearNDInterpolator(points, err)

                    xv = mesh.vertices[:, 0]
                    yv = mesh.vertices[:, 1]
                    err_on_vertices = interp(xv, yv)[:, 0]

                    # Maschera i NaN
                    masked_err = ma.masked_invalid(err_on_vertices)

                    # Calcola i limiti dei dati per impostare xlim e ylim
                    min_x, max_x = np.min(xv), np.max(xv)
                    min_y, max_y = np.min(yv), np.max(yv)

                    # Plot interpolazione
                    ax.tripcolor(xv, yv, masked_err, triangles=mesh.faces, shading='gouraud',
                                cmap='seismic')#, vmin=-1, vmax=1)
                    ax.triplot(xv,yv,'-k',triangles=mesh.faces,linewidth=0.5)

                    # Aggiungi punti di bulk, opzionale
                    #x_bulk = bulk_data_temp[0].detach().cpu().numpy()
                    #y_bulk = bulk_data_temp[1].detach().cpu().numpy()
                    #ax.scatter(x_bulk, y_bulk, c='w', s=8, alpha=0.4)

                    # Imposta esplicitamente i limiti degli assi in base ai dati plottati
                    ax.set_xlim(min_x, max_x)
                    ax.set_ylim(min_y, max_y)

                    # Assicurati che gli assi siano uguali per evitare distorsioni se i range sono diversi
                    # Questo può causare bordi neri se il rapporto dei limiti x e y non corrisponde al rapporto del riquadro.
                    # Se i tuoi dati sono in un quadrato, questo va bene.
                    ax.set_aspect('equal', adjustable='box')


                    # Salva immagine senza bordi extra e con la dimensione desiderata in pixel
                    plt.savefig(os.path.join(savepath, f'solution_{steps}_{epoch}.png'),
                                bbox_inches='tight', pad_inches=0, dpi=dpi_for_plot) # Usa il DPI calcolato o desiderato
                    plt.close()

                    self.pinn.train()

                    if ckpt and error < min_error:
                        min_error = error
                        torch.save(self.pinn.state_dict(), ckpt_path)

                    errors.append(error)
                    flops.append(cumulative_flops)

                    self.logger.info(
                        f"{epoch:5d} {step:6d} {error:12.6f} {total_loss.item():12.6f} {bulk_loss.item():12.6f} "
                        f"{boundary_loss.item():15.6f} {initial_loss.item():13.6f} "
                        f"{scheduler.get_last_lr()[0]:14.6f}"
                    )
                    
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
        scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=1.0, total_iters=decomposition_epochs[-1])

        self.logger.info(f"{'Epoch':>5} {'Step':>6} {'Error':>12} {'Total loss':>12} {'Bulk Loss':>12} {'Boundary Loss':>15} {'Initial Loss':>13} {'Learning rate':>14}")

        # Prepare test data
        x_test, y_test, u_test = test_data

        classes = get_points_classes(
            x_test,
            y_test,
            mesh,
            16, #steps
            time_axis,
            'all'#boundary_type
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

                if epoch % 1 == 0 or epoch == decomposition_epochs[-1] - 1:
                    
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
                        dot_products[mask] = dot_products[mask].mean()

                    target_pixels = 1000
                    dpi_for_plot = 100
                    fig_size_inches = target_pixels / dpi_for_plot

                    # parametri per la colorbar
                    cbar_height = 0.03   # 3% dell'altezza della figura
                    cbar_bottom = 0.01   # 5% dal fondo
                    main_bottom = cbar_bottom + cbar_height + 0.02

                    fig = plt.figure(frameon=False)
                    fig.set_size_inches(fig_size_inches, fig_size_inches)

                    # axes principale occupa dal main_bottom (es. 0.05+0.03+0.02=0.10) al top (1.0)
                    ax = fig.add_axes([
                        0.0,                # left = 0%
                        main_bottom,        # bottom = spazio lasciato sotto
                        1.0,                # width = 100%
                        1.0 - main_bottom   # height = resto
                    ])
                    ax.set_axis_off()

                    x = test_data[0].detach().cpu().numpy()
                    y = test_data[1].detach().cpu().numpy()
                    value = dot_products.detach().cpu().numpy()

                    points = np.concatenate([x, y], axis=-1)
                    interp = LinearNDInterpolator(points, value)

                    xv = mesh.vertices[:, 0]
                    yv = mesh.vertices[:, 1]
                    value = interp(xv, yv)

                    # Maschera i NaN
                    value = ma.masked_invalid(value)
                    print(epoch, value.min(), value.max())

                    # Calcola i limiti dei dati per impostare xlim e ylim
                    min_x, max_x = np.min(xv), np.max(xv)
                    min_y, max_y = np.min(yv), np.max(yv)

                    # Plot interpolazione
                    tri = ax.tripcolor(xv, yv, value, triangles=mesh.faces, shading='gouraud',
                                cmap='seismic', vmin=-3, vmax=3)
                    ax.triplot(xv,yv,'-k',triangles=mesh.faces,linewidth=0.5)

                    # Aggiungi punti di bulk, opzionale
                    #x_bulk = bdry_data[0].detach().cpu().numpy()
                    #y_bulk = bdry_data[1].detach().cpu().numpy()
                    #ax.scatter(x_bulk, y_bulk, c='w', s=8, alpha=0.4)

                    ax.set_xlim(min_x, max_x)
                    ax.set_ylim(min_y, max_y)
                    ax.set_aspect('equal', adjustable='box')
                    
                    cax = fig.add_axes([
                        0.1,           # left: 10%
                        cbar_bottom,   # bottom: 5%
                        0.8,           # width: 80%
                        cbar_height    # height: 3%
                    ])

                    # 4) Colorbar orizzontale
                    cb = fig.colorbar(tri, cax=cax, orientation='horizontal')
                    cb.ax.tick_params(labelsize=max(int(target_pixels / 80), 4))


                    plt.savefig(os.path.join(savepath, f'solution_{steps}_{epoch}.png'),
                                bbox_inches='tight', pad_inches=0, dpi=dpi_for_plot)
                    plt.close()

                    if epoch == 20:
                        import sys
                        sys.exit()

        return flops, errors
