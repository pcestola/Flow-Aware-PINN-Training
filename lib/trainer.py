import os
import math
import torch
import logging
import numpy as np
import numpy.ma as ma
import torch.profiler
import matplotlib.pyplot as plt

from math import inf
from typing import Tuple
from typing import Optional
from torch.func import functional_call
from fvcore.nn import FlopCountAnalysis
from lib.meshes import get_points_classes
from functorch import vmap, jacrev, hessian
from torch.optim.lr_scheduler import LinearLR
from torch.nn.utils import parameters_to_vector
from scipy.interpolate import LinearNDInterpolator


# Trainer
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

    def divide_epochs_proportional(self, epochs: int, indices: list, minimo: int = 1):
        
        # Calcola quanti punti ha ogni step
        counts = np.diff([0] + indices)
        total = sum(counts)

        # Calcola percentuali da conteggi
        percentuali = 100 * np.array(counts) / total

        # Assegna le epoche
        raw = epochs * percentuali / 100.0
        floored = np.floor(raw).astype(int)
        floored = np.maximum(floored, minimo)

        diff = epochs - floored.sum()
        if diff > 0:
            remainders = raw - floored
            indices_max = np.argsort(-remainders)
            for i in indices_max[:diff]:
                floored[i] += 1

        # Output: cumsum con 0 iniziale
        return np.cumsum([0] + floored.tolist()).tolist()

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
        elif divide_mode == 'proportional':
            decomposition_epochs = self.divide_epochs_proportional(epochs, indices, minimo=100)
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
    
    def train_with_error(self,
        bulk_data:Tuple[torch.Tensor],
        bdry_data:Tuple[torch.Tensor],
        init_data:Tuple[torch.Tensor],
        test_data:Tuple[torch.Tensor],
        indices:list,
        weights:dict,
        epochs:int,
        steps:int,
        divide_mode:str,
        extra_epochs:int,
        lr_start:float,
        ckpt:bool,
        savepath:str,
        mesh):
        
        if ckpt:
            ckpt_path = os.path.join(self.ckpt_dir,f"weights_{steps}.pt")

        if divide_mode== 'linear':
            decomposition_epochs = self.divide_epochs_linear(epochs, steps)
        elif divide_mode == 'exponential':
            decomposition_epochs = self.divide_epochs_exponential_growth(epochs, steps)
        decomposition_epochs[-1] += extra_epochs

        # Flops
        cumulative_flops = 0.0
        flop_counter = FlopCountAnalysis(self.pinn.network, torch.rand((1,2), device=self.device))
        flops_per_point = flop_counter.total()
        
        # Ottimizzatore
        optimizer = torch.optim.Adam(self.pinn.parameters(), lr=lr_start)
        # Scheduler
        scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=0.000001, total_iters=decomposition_epochs[-1])
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
                print(epoch,end='\r')

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

                if epoch % 100 == 0 or epoch == decomposition_epochs[-1]-1:
                    self.pinn.eval()
                    u_pred = self.pinn(test_data[0], test_data[1])
                    u_real = test_data[2]
                    error = torch.mean((u_pred - u_real) ** 2).item()

                    # === PARAMETRI PER L'EXPORT CORRETTO ===
                    dpi_for_plot = 300  # conforme alle linee guida (min 300 dpi per halftone)
                    width_mm = 84       # larghezza in mm (es. colonna singola)
                    width_inch = width_mm / 25.4
                    fig_size_inches = (width_inch, width_inch)  # quadrato, proporzioni corrette

                    # === CREA FIGURA SENZA BORDI, ETICHETTE O MARGINI ===
                    fig = plt.figure(frameon=False)
                    fig.set_size_inches(*fig_size_inches)
                    ax = plt.Axes(fig, [0., 0., 1., 1.])
                    ax.set_axis_off()
                    fig.add_axes(ax)

                    # === INTERPOLAZIONE DELL’ERRORE SULLA MESH ===
                    x = test_data[0].detach().cpu().numpy()
                    y = test_data[1].detach().cpu().numpy()
                    err = u_pred.detach().cpu().numpy()
                    #((u_pred - u_real) ** 2).detach().cpu().numpy()

                    points = np.concatenate([x, y], axis=-1)
                    interp = LinearNDInterpolator(points, err)

                    xv = mesh.vertices[:, 0]
                    yv = mesh.vertices[:, 1]
                    
                    err_on_vertices_raw = interp(xv, yv)
                    if err_on_vertices_raw is None:
                        err_on_vertices_raw = np.full_like(xv[:, None], np.nan)

                    # Verifica shape ed evita crash su None
                    if err_on_vertices_raw.ndim == 2 and err_on_vertices_raw.shape[1] == 1:
                        err_on_vertices = err_on_vertices_raw[:, 0]
                    else:
                        err_on_vertices = np.full_like(xv, np.nan)

                    # Riempi i NaN ai bordi con interpolazione nearest
                    if np.any(np.isnan(err_on_vertices)):
                        from scipy.interpolate import NearestNDInterpolator
                        interp_nearest = NearestNDInterpolator(points, err)
                        nan_mask = np.isnan(err_on_vertices)
                        err_on_vertices[nan_mask] = interp_nearest(xv[nan_mask], yv[nan_mask]).ravel()

                    masked_err = ma.masked_invalid(err_on_vertices)

                    min_x, max_x = np.min(xv), np.max(xv)
                    min_y, max_y = np.min(yv), np.max(yv)

                    # === PLOT ===
                    # === PLOT ===
                    tri = ax.tripcolor(xv, yv, masked_err, triangles=mesh.faces, shading='gouraud',
                                    cmap='inferno')#, vmin=0, vmax=1.0)

                    # === CREA UN NUOVO AXES PER LA COLORBAR ===
                    #from mpl_toolkits.axes_grid1 import make_axes_locatable
                    #divider = make_axes_locatable(ax)
                    #cax = divider.append_axes("right", size="5%", pad=0.05)

                    # === AGGIUNGI LA COLORBAR ===
                    #fig.colorbar(tri, cax=cax, orientation='vertical')

                    ax.set_xlim(min_x, max_x)
                    ax.set_ylim(min_y, max_y)
                    ax.set_aspect('equal', adjustable='box')

                    # === SALVA COME TIFF RGB A 300 DPI ===
                    output_path = os.path.join(savepath, f'solution_{steps}_{epoch}.png')
                    plt.savefig(output_path,
                                dpi=dpi_for_plot,
                                bbox_inches='tight',
                                pad_inches=0,
                                format='png')
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
        divide_mode: str,
        extra_epochs: int,
        lr_start: float,
        ckpt: bool,
        savepath: str,
        mesh,
        time_axis,
        boundary_type):

        if ckpt:
            ckpt_path = os.path.join(self.ckpt_dir, f"weights_{steps}.pt")

        if divide_mode == 'linear':
            decomposition_epochs = self.divide_epochs_linear(epochs, steps)
        elif divide_mode == 'exponential':
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

                    # === PREDIZIONE ===
                    self.pinn.eval()
                    u_pred = self.pinn(test_data[0], test_data[1])
                    u_real = test_data[2]
                    error = torch.mean((u_pred - u_real) ** 2).item()

                    # === PARAMETRI EXPORT ===
                    dpi_for_plot = 300
                    width_mm = 84
                    width_inch = width_mm / 25.4
                    fig_size_inches = (width_inch, width_inch)

                    # === CREA FIGURA PULITA ===
                    fig = plt.figure(frameon=False)
                    fig.set_size_inches(*fig_size_inches)
                    ax = plt.Axes(fig, [0., 0., 1., 1.])
                    ax.set_axis_off()
                    fig.add_axes(ax)

                    # === DATI ===
                    x = test_data[0].detach().cpu().numpy()
                    y = test_data[1].detach().cpu().numpy()
                    dot_vals = dot_products.detach().cpu().numpy()
                    points = np.concatenate([x, y], axis=-1)
                    interp = LinearNDInterpolator(points, dot_vals)

                    xv = mesh.vertices[:, 0]
                    yv = mesh.vertices[:, 1]

                    # === INTERPOLAZIONE + NEAREST AI BORDI ===
                    value_raw = interp(xv, yv)
                    if value_raw is None:
                        value_raw = np.full_like(xv[:, None], np.nan)

                    if value_raw.ndim == 2 and value_raw.shape[1] == 1:
                        value_on_vertices = value_raw[:, 0]
                    else:
                        value_on_vertices = np.full_like(xv, np.nan)

                    if np.any(np.isnan(value_on_vertices)):
                        from scipy.interpolate import NearestNDInterpolator
                        interp_nearest = NearestNDInterpolator(points, dot_vals)
                        nan_mask = np.isnan(value_on_vertices)
                        value_on_vertices[nan_mask] = interp_nearest(xv[nan_mask], yv[nan_mask]).ravel()

                    masked_value = ma.masked_invalid(value_on_vertices)

                    # === LIMITI GRAFICO ===
                    min_x, max_x = np.min(xv), np.max(xv)
                    min_y, max_y = np.min(yv), np.max(yv)

                    # === PLOT ===
                    tri = ax.tripcolor(xv, yv, masked_value, triangles=mesh.faces, shading='gouraud',
                                    cmap='seismic', vmin=-1.2, vmax=1.2)

                    # ✅ MESH (linee nere)
                    ax.triplot(xv, yv, triangles=mesh.faces, color='k', linewidth=0.27)

                    ax.set_xlim(min_x, max_x)
                    ax.set_ylim(min_y, max_y)
                    ax.set_aspect('equal', adjustable='box')

                    # === COLORBAR VERTICALE (5% larghezza + 5% pad) ===
                    from mpl_toolkits.axes_grid1 import make_axes_locatable
                    divider = make_axes_locatable(ax)
                    cax = divider.append_axes("right", size="5%", pad=0.05)
                    fig.colorbar(tri, cax=cax, orientation='vertical')

                    # === SALVA ===
                    output_path = os.path.join(savepath, f'solution_{steps}_{epoch}.png')
                    plt.savefig(output_path, dpi=dpi_for_plot, bbox_inches='tight', pad_inches=0, format='png')
                    plt.close()


                    if epoch == 20:
                        import sys
                        sys.exit()

        return flops, errors
