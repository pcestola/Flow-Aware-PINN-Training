import os
import math
import torch
import logging
import imageio
import numpy as np
import matplotlib.pyplot as plt

from math import inf
from typing import Tuple
from typing import Optional
from fvcore.nn import FlopCountAnalysis
from torch.optim.lr_scheduler import LinearLR

class TrainerStep():
    def __init__(self, pinn, device=None, ckpt_dir=None, ckpt_interval=100):
        self.pinn = pinn
        self.device = device if device else torch.device('cpu')
        self.ckpt_dir = ckpt_dir
        self.ckpt_interval = ckpt_interval
        self.logger = logging.getLogger(__name__)

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
        bulk_data:Tuple[torch.Tensor],
        bdry_data:Tuple[torch.Tensor],
        init_data:Tuple[torch.Tensor],
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

        # Flops
        cumulative_flops = 0.0
        flop_counter = FlopCountAnalysis(self.pinn.network, torch.rand((1,2), device=self.device))
        flops_per_point = flop_counter.total()
        
        # Ottimizzatore
        optimizer = torch.optim.Adam(self.pinn.parameters(), lr=lr_start)
        scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=steps)

        self.logger.info(f"{'Epoch':>5} {'Step':>6} {'Total loss':>12} {'Bulk Loss':>12} {'Boundary Loss':>15} {'Initial Loss':>13} {'Learning rate':>14}")

        min_loss = inf
        losses = list()
        flops = list()
        for step in range(steps):
            bulk_data_temp = tuple(x[:indices[step]] for x in bulk_data)

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

                # flops
                n_points = bulk_data_temp[0].shape[0]
                n_bdry = bdry_data[0].shape[0] if bdry_data is not None else 0
                n_init = init_data[0].shape[0] if init_data is not None else 0
                cumulative_flops += flops_per_point * (n_points + n_bdry + n_init)

                if epoch % 100 == 0 or epoch == decomposition_epochs[-1]-1:
                    self.pinn.eval()
                    bulk_loss = self.pinn.bulk_loss(bulk_data)
                    total_loss = weights.get('bulk', 1.0) * bulk_loss + weights.get('boundary', 1.0) * boundary_loss + weights.get('initial', 1.0) * initial_loss

                    x = bulk_data[0].detach().cpu().numpy()
                    y = bulk_data[1].detach().cpu().numpy()
                    with torch.no_grad():
                        u = self.pinn(bulk_data[0],bulk_data[1]).detach().cpu().numpy()
                    xmin, xmax = x.min(), x.max()
                    ymin, ymax = y.min(), y.max()

                    fig, axs = plt.subplots(1, 1, figsize=(4, 3))
                    axs.scatter(x, y, c=u, s=6, cmap='bwr')
                    axs.axis('off')
                    plt.savefig(savepath+f'_solution_{epoch}.png')
                    plt.close()

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

        return flops, losses

    def train_with_gradient_gif(
        self,
        bulk_data: Tuple[torch.Tensor],
        bdry_data: Tuple[torch.Tensor],
        init_data: Tuple[torch.Tensor],
        indices: list,
        weights: dict,
        epochs: int,
        steps: int,
        lr_start: float,
        ckpt: bool,
        savepath: str,
        plotter: Optional[object] = None
    ):
        image_dir = savepath
        os.makedirs(image_dir, exist_ok=True)

        if ckpt:
            ckpt_path = os.path.join(self.ckpt_dir, f"weights_{steps}.pt")

        decomposition_epochs = self.divide_epochs_exponential_growth(epochs, steps)

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
    
    # Controlla che sia corretto (confronta con train)
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
        #decomposition_epochs[-1] += 4000

        # Flops
        cumulative_flops = 0.0
        flop_counter = FlopCountAnalysis(self.pinn.network, torch.rand((1,2), device=self.device))
        flops_per_point = flop_counter.total()
        
        # Ottimizzatore
        optimizer = torch.optim.Adam(self.pinn.parameters(), lr=lr_start)
        scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=steps)

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

                # flops
                n_points = bulk_data_temp[0].shape[0]
                n_bdry = bdry_data[0].shape[0] if bdry_data is not None else 0
                n_init = init_data[0].shape[0] if init_data is not None else 0
                cumulative_flops += flops_per_point * (n_points + n_bdry + n_init)

                if epoch % 100 == 0 or epoch == decomposition_epochs[-1]-1:
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

            scheduler.step()
                    
        return flops, errors

class TrainerStepOLD():
    def __init__(self, pinn, device=None):
        self.pinn = pinn
        self.device = device if device else torch.device('cpu')

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
              bulk_data:Tuple[torch.Tensor],
              bdry_data:Tuple[torch.Tensor],
              init_data:Tuple[torch.Tensor],
              indices:list,
              weights:dict,
              epochs:int,
              steps:int,
              lr_start:float=1e-2):
        
        decomposition_epochs = self.divide_epochs_exponential_growth(epochs, steps)
        
        #optimizer = torch.optim.Adam(self.pinn.parameters(), lr=lr_start)
        #scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=steps)

        print(f"Epoch\tStep\tLRate\tTotal loss\tBulk Loss\tBoundary Loss\tInitial Loss")
        
        for step in range(steps):
            
            bulk_data_temp = tuple(x[:indices[step]] for x in bulk_data)

            optimizer = torch.optim.Adam(self.pinn.parameters(), lr=lr_start)

            for epoch in range(decomposition_epochs[step],decomposition_epochs[step+1]):

                optimizer.zero_grad()
                
                bulk_loss = self.pinn.bulk_loss(bulk_data_temp)
                total_loss = weights.get('bulk',1.0) * bulk_loss

                if bdry_data is not None:
                    boundary_loss = self.pinn.boundary_loss(bdry_data)
                    total_loss += weights.get('boundary',1.0) * boundary_loss
                else:
                    boundary_loss = torch.tensor(0.0)

                if init_data is not None:
                    initial_loss = self.pinn.initial_loss(init_data)
                    total_loss += weights.get('initial',1.0) * initial_loss
                else:
                    initial_loss = torch.tensor(0.0)

                total_loss.backward()
                optimizer.step()

                if epoch % 100 == 0:
                    print(f"{epoch}\t{step:4}\t{lr_start:.6f}\t{total_loss.item():.6f}\t{bulk_loss.item():.6f}\t{boundary_loss.item():.6f}\t{initial_loss.item():.6f}")

            #scheduler.step()
            lr_start -= 9e-3/steps