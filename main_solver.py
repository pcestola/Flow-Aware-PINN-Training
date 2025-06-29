# scripts/train.py
# TODO: Le geometrie complesse suggeriscono di: imparare prima boundary e initial
#       e poi bulk (mantenendo stabili le altre due).
# TODO: Aggiungi loss di classificazione
# TODO: Sistema il codice, fa schifo
# TODO: Controlla che il conteggio dei flops sia corretto
# TODO: Per avere il grafico (flops vs error) corretto, devi fare checkpoint
#       in base ai flop consumati (ogni tot) non in base all'epoca.
#       Altrimenti usa interpolazione lineare.

import os
import yaml
import torch
import pickle
import logging
import argparse
import subprocess
import numpy as np
import matplotlib.pyplot as plt

from lib.schemes import WaveLeapFrog, HeatLeapFrog
from lib.models import SimpleNN, SIREN
from lib.train_old import TrainerStep
from lib.pinn import PINN, LaplaceEquation, BurgerEquation, WaveEquation, HeatEquation, EikonalEquation
from lib.meshes import mesh_preprocessing, visualize_scalar_field
from lib.dataset import plot_generated_data_3d

def parse_args():
    parser = argparse.ArgumentParser(
        description="Addestramento PINN da config YAML"
    )
    parser.add_argument(
        "--config", type=str, default="config/default.yaml",
        help="Path al file di configurazione YAML"
    )
    parser.add_argument(
        "--preview", action="store_true",
        help="Visualizza i grafici"
    )
    parser.add_argument(
        "--steps", type=int, default=0
    )
    parser.add_argument(
        "--repeat", type=int, default=1,
        help="Numero di ripetizioni per lo stesso esperimento (diversi seed)"
    )
    return parser.parse_args()


def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def setup_logging(log_cfg, path, steps):
    os.makedirs(path, exist_ok=True)
    log_file = os.path.join(path, f"train_{steps}.log")
    logging.basicConfig(
        filename=log_file,
        level=getattr(logging, log_cfg["level"]),
        #format="%(asctime)s [%(levelname)s] %(message)s"
        format="[%(levelname)s] %(message)s"
    )

def get_equation(name:str):
    eqs = {
        'wave': WaveEquation,
        'heat': HeatEquation,
        'laplace': LaplaceEquation,
        'eikonal': EikonalEquation,
        'burger': BurgerEquation
    }
    if name not in eqs:
        raise ValueError(f"Equazione sconosciuta: {name}")
    return eqs[name]()

def get_solver(name:str):
    solvers = {
        'wave': WaveLeapFrog,
        'heat': HeatLeapFrog
    }
    if name not in solvers:
        raise ValueError(f"Equazione sconosciuta: {name}")
    return solvers[name]

def get_free_gpu(exclude_gpu=2):
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,nounits,noheader"],
            encoding="utf-8"
        )
        memory_usage = [int(x) for x in output.strip().split("\n")]

        indexed_usage = [(i, mem) for i, mem in enumerate(memory_usage) if i != exclude_gpu]

        if not indexed_usage:
            print("Nessuna GPU disponibile tranne quella esclusa.")
            return None

        free_gpu = min(indexed_usage, key=lambda x: x[1])[0]
        return free_gpu

    except Exception as e:
        print(f"Errore nel controllo GPU: {e}")
        return None


def main():
    args = parse_args()
    cfg = load_config(args.config)

    if args.steps > 0:
        cfg["decomposition"]["steps"] = args.steps

    for run_id in range(args.repeat):
        seed = 42 + run_id
        torch.manual_seed(seed)
        np.random.seed(seed)


        # Cartella di salvataggio
        save_dir = os.path.join("results", cfg["name"], f"run_{run_id}")
        log_dir = os.path.join(save_dir, "logging")
        img_dir = os.path.join(save_dir, "images")
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(img_dir, exist_ok=True)

        if cfg["checkpoint"]["enabled"]:
            ckpt_dir = os.path.join(save_dir, "checkpoint")
            os.makedirs(ckpt_dir, exist_ok=True)
        else:
            ckpt_dir = None

        # Logging locale
        if cfg["logging"]["enabled"]:
            setup_logging(cfg["logging"], log_dir, cfg["decomposition"]["steps"])
            logging.info(f"Esecuzione {run_id+1}/{args.repeat} con seed={seed}")

        if torch.cuda.is_available():
            #gpu_id = get_free_gpu()
            gpu_id = 0
            device = torch.device(f"cuda:{gpu_id}")
            logging.info(f'Using GPU: {device}')
        else:
            device = torch.device("cpu")
            logging.info('Using CPU')

        # Caricamento dati mesh
        mesh, bulk_points, boundary_points, initial_points, indices, scheduler = mesh_preprocessing(
            path = cfg["mesh"]["file"],
            epochs = cfg["training"]["epochs"],
            steps = cfg["decomposition"]["steps"],
            time_axis = cfg["decomposition"]["time_axis"],
            boundary_type = cfg["decomposition"]["type"],
            bulk_n = cfg["mesh"]["bulk_points"],
            boundary_n = cfg["mesh"]["boundary_points"],
            init_n = cfg["mesh"]["initial_points"],
            preview=args.preview
        )

        '''
            Costruzione dati bulk e boundary
        '''
        # Caricamento equazione
        equation = get_equation(cfg["equation"])

        # Bulk
        bulk_data = (
            torch.from_numpy(bulk_points[:, 0:1]).to(device=device, dtype=torch.float32).requires_grad_(),
            torch.from_numpy(bulk_points[:, 1:2]).to(device=device, dtype=torch.float32).requires_grad_()
        )
        data_min = bulk_points.min(axis=0)
        data_max = bulk_points.max(axis=0)
        del bulk_points

        # Boundary
        boundary_value = np.zeros_like(boundary_points[:,0:1])
        boundary_data = (
            torch.from_numpy(boundary_points[:, 0:1]).to(device=device, dtype=torch.float32).requires_grad_(),
            torch.from_numpy(boundary_points[:, 1:2]).to(device=device, dtype=torch.float32).requires_grad_(),        
            torch.from_numpy(boundary_value).to(device=device, dtype=torch.float32)
        )
        del boundary_points

        # Initial
        if not initial_points is None:
            initial_value = equation.initial_condition(initial_points[:,1:2])
            initial_vel = np.zeros_like(initial_value)
            initial_data = (
                torch.from_numpy(initial_points[:, 0:1]).to(device=device, dtype=torch.float32).requires_grad_(),
                torch.from_numpy(initial_points[:, 1:2]).to(device=device, dtype=torch.float32).requires_grad_(),
                torch.from_numpy(initial_value).to(device=device, dtype=torch.float32),
                torch.from_numpy(initial_vel).to(device=device, dtype=torch.float)
            )
            del initial_points
        else:
            initial_data = None

        # test
        solver = get_solver(cfg["equation"])((4,100,8,200,0.01))
        test_t = np.linspace(-4,4,200)
        test_x = np.linspace(-2,2,100)
        test_t, test_x = np.meshgrid(test_t, test_x)
        test_t, test_x = np.expand_dims(test_t.flatten(),1), np.expand_dims(test_x.flatten(),1)
        test_u = solver.solve(equation.initial_condition).T.reshape((20000,1))
        test_t = torch.from_numpy(test_t).to(device=device, dtype=torch.float32)
        test_x = torch.from_numpy(test_x).to(device=device, dtype=torch.float32)
        test_u = torch.from_numpy(test_u).to(device=device, dtype=torch.float32)
        test_data = (test_t, test_x, test_u)

        if args.preview:
            plot_generated_data_3d(bulk_data, boundary_data, initial_data)

        '''
            MODELLO
        '''
        # Costruzione modello
        net_type = cfg["model"]["type"]
        if net_type == 'simple':
            network = SimpleNN(cfg["model"]["dims"], data_min, data_max)
        elif net_type == 'siren':
            network = SIREN(cfg["model"]["dims"])
        
        # Costruzione PINN
        pinn = PINN(network, equation)
        pinn.to(device)

        '''
            TRAINING
        '''
        trainer = TrainerStep(
            pinn,
            device=device,
            ckpt_dir=ckpt_dir,
            ckpt_interval = cfg["checkpoint"]["interval"]
        )

        flops, errors = trainer.train_with_error(
            bulk_data=bulk_data,
            bdry_data=boundary_data,
            init_data=initial_data,
            test_data=test_data,
            indices=indices,
            weights=cfg["training"]["weights"],
            epochs=int(cfg["training"]["epochs"]),
            steps=int(cfg["decomposition"]["steps"]),
            lr_start=float(cfg["training"]["lr"]),
            ckpt=cfg["checkpoint"]["enabled"],
            savepath=os.path.join(img_dir,f'image_{cfg["decomposition"]["steps"]}')
        )
        
        # Show results
        solution = pinn.network(torch.tensor(mesh.vertices, dtype=torch.float32, device=device))
        solution = solution.detach().cpu().flatten().numpy()

        #visualize_scalar_field(mesh, solution, save_path=os.path.join(img_dir,f'solution_{cfg["decomposition"]["steps"]}'))

        with open(os.path.join(log_dir,f'loss_{cfg["decomposition"]["steps"]}.pkl'), "wb") as f:
            pickle.dump((flops, errors), f)

'''
    Implementa un sistema che aggiorna automaticamente i pesi dei termini di loss durante il training, per esempio:

    SoftAdapt (Heydari et al.)

    GradNorm (Chen et al.)

    NTK-based reweighting (Wang et al. 2021)
'''

if __name__ == "__main__":
    main()
