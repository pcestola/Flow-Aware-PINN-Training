# lib/train.py
import os
import torch
import logging
import numpy as np

from math import inf
from typing import Tuple
from torch.optim.lr_scheduler import ReduceLROnPlateau, LinearLR

class ProgressiveTrainer:
    def __init__(self, pinn, scheduler, device=None, ckpt_dir=None, ckpt_interval=100):
        """
        :param pinn:       Oggetto PINN da addestrare (nn.Module)
        :param device:     torch.device o stringa
        :param ckpt_dir:   Directory dove salvare i checkpoint (None per disabilitare)
        :param ckpt_interval: Salva checkpoint ogni ckpt_interval epoche
        """
        self.pinn = pinn
        self.scheduler = scheduler
        self.device = device or torch.device('cpu')
        self.ckpt_dir = ckpt_dir
        self.ckpt_interval = ckpt_interval
        self.logger = logging.getLogger(__name__)

        if self.ckpt_dir:
            os.makedirs(self.ckpt_dir, exist_ok=True)

    def train(self, bulk_data, boundary_data, initial_data, indices, weights, epochs=1000, lr=1e-3, ckpt=False, scheduler=False):
        """
        :param bulk_data:     tuple (x, t) con i dati bulk
        :param boundary_data: tuple (x, t, u0) con i dati di boundary
        :param indices:       lista di soglie progressive per il bulk
        :param weights:       tuple (w_bulk, w_boundary)
        :param epochs:        numero di epoche
        :param lr:            learning rate
        """
        optimizer = torch.optim.Adam(self.pinn.parameters(), lr=lr)
        #scheduler = StepLR(optimizer, step_size=1000, gamma=0.5)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, min_lr=1e-6)

        # Header a inizio training
        self.logger.info(f"{'Epoch':>6} | {'Total Loss':>12} | {'Bulk loss':>12} | {'Initial Loss':>12} | {'Initial Loss':>12} | {'Boundary Loss':>14}")
        self.logger.info("-" * 64)

        step = 0
        steps = len(indices)
        self.scheduler.insert(0,0)
        min_loss = inf
        for epoch in range(1,epochs+1):

            # TODO: troppo handcrafted
            #if epoch%100 == 0:
                #weights['boundary'] *= 0.9
                #print(weights['boundary'])

            if self.scheduler[step] == epoch-1:
                step += 1
                self.logger.info(f"Step {step}/{steps}, Using {indices[step-1]} points")
                n_pts = indices[step-1]
                data = {
                    'bulk': (bulk_data[0][:n_pts], bulk_data[1][:n_pts]),
                    'boundary': boundary_data,
                    'initial': initial_data
                }

            optimizer.zero_grad()

            # calcola le loss
            loss, bulk_loss, initial_loss1, initial_loss2, boundary_loss = self.pinn.total_loss(data, weights, multiple=True)

            # backward + step
            loss.backward()
            optimizer.step()
            if scheduler:
                scheduler.step(loss)

            # logging periodico
            if epoch % 100 == 0:
                self.logger.info(
                    f"{epoch:6} | "
                    f"{loss.item():12.6f} | "
                    f"{bulk_loss:12.6f} | "
                    f"{initial_loss1:12.6f} | "
                    f"{initial_loss2:12.6f} | "
                    f"{boundary_loss:14.6f}"
                )

            # checkpointing
            if ckpt and loss.item() < min_loss and (epoch % self.ckpt_interval == 0 or epoch == epochs):
                min_loss = loss.item()
                ckpt_path = os.path.join(self.ckpt_dir,"weights.pt")
                torch.save(self.pinn.state_dict(), ckpt_path)
                self.logger.info(f"Checkpoint salvato: {ckpt_path}")

class Trainer:
    def __init__(self, pinn, device=None):
        """
        Inizializza il Trainer per la PINN.
        
        :param pinn: Oggetto PINN da addestrare.
        :param loss_functions: Dizionario con le funzioni di loss per ciascun tipo di dato.
        :param weights: Dizionario con i pesi per ciascun tipo di loss.
        :param device: Device su cui eseguire i calcoli.
        """
        self.pinn = pinn
        self.device = device if device else torch.device('cpu')

    def train(self, bulk_data, boundary_data, init_data, weights, epochs=1000, lr=1e-3):
        """
        Allena la PINN.

        :param data: Dizionario con i dati (bulk, initial, boundary).
        :param epochs: Numero di epoche per l'ottimizzazione.
        :param lr: Tasso di apprendimento.
        """
        optimizer = torch.optim.Adam(self.pinn.parameters(), lr=lr)

        print(f"Epoch\tBulk Loss\tBoundary Loss\tInitial Loss\tTotal Loss")
        
        for epoch in range(epochs):
            optimizer.zero_grad()

            total_loss = 0.0

            # Calcola le loss per ciascun tipo di dato
            bulk_loss = self.pinn.bulk_loss(bulk_data)
            boundary_loss = self.pinn.boundary_loss(boundary_data)
            initial_loss = self.pinn.initial_loss(init_data)

            # Loss total
            # NOTE: la loss bilanciata è migliorativa (quando nello specifico?)
            #       è una loss che minimizza normalmente quanto tutti hanno la stessa importanza
            #       altrimenti cerca di tornare in una posizione nella quale hanno la stessa importanza
            #       si vede bene facendo il gradiente di (x^2+y^2)/(x+y)
            total_loss += weights['bulk']*bulk_loss + weights['boundary']*boundary_loss + weights['initial']*initial_loss

            #total_loss += weights['bulk']*bulk_loss**2 + weights['boundary']*boundary_loss**2 + weights['initial']*initial_loss**2
            #total_loss /= bulk_loss + boundary_loss + initial_loss + 1e-10
            
            # Backpropagation
            total_loss.backward()
            optimizer.step()

            if epoch % 100 == 0:
                print(f"{epoch}\t{bulk_loss.item():.6f}\t{boundary_loss.item():.6f}\t{initial_loss.item():.6f}\t{total_loss.item():.6f}")


'''
    OLD CODE
'''

# TODO: non ha senso salvare quando la loss è minimizzata
# salva solo all'ultimo stadio oppure usa una loss calcolata
# su tutto quanto il dataset
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
              ckpt:bool):
        
        if ckpt:
            ckpt_path = os.path.join(self.ckpt_dir,f"weights_{steps}.pt")

        decomposition_epochs = self.divide_epochs_exponential_growth(epochs, steps)
        
        optimizer = torch.optim.Adam(self.pinn.parameters(), lr=lr_start)
        scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=steps)

        self.logger.info(f"{'Epoch':>5} {'Step':>6} {'Total loss':>12} {'Bulk Loss':>12} {'Boundary Loss':>15} {'Initial Loss':>13} {'Learning rate':>14}")
        
        min_loss = inf
        losses = list()
        for step in range(steps):
            
            bulk_data_temp = tuple(x[:indices[step]] for x in bulk_data)

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
                    
                    self.pinn.eval()
                    bulk_loss = self.pinn.bulk_loss(bulk_data)
                    total_loss = weights.get('bulk',1.0) * bulk_loss + weights.get('boundary',1.0) * boundary_loss + weights.get('initial',1.0) * initial_loss
                    self.pinn.train()

                    if ckpt and total_loss.item() < min_loss:
                        min_loss = total_loss.item()
                        torch.save(self.pinn.state_dict(), ckpt_path)
                    
                    losses.append(total_loss.item())

                    self.logger.info(f"{epoch:5d} {step:6d} {total_loss.item():12.6f} {bulk_loss.item():12.6f} {boundary_loss.item():15.6f} {initial_loss.item():13.6f} {scheduler.get_last_lr()[0]:14.6f}")

            scheduler.step()

        return losses


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