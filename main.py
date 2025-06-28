import os
import yaml
import torch
import pickle
import logging
import argparse
import subprocess
import numpy as np

from lib.models import SimpleNN, SIREN
from lib.train import TrainerStep
from lib.pinn import (
    PINN, LaplaceEquation,
    Burgers_1D, WaveEquation, HeatEquation,
    EikonalEquation, Poisson_2D_C, Poisson_2D_CG,
    Kuramoto_Shivashinsky, Example
)
from lib.meshes import mesh_preprocessing, visualize_scalar_field
from lib.gif import generate_gif

from lib.plotter import Plotter

# UTILS
def parse_args():
    parser = argparse.ArgumentParser(description="Addestramento PINN da config YAML")
    parser.add_argument("--path", type=str, help="Path alla cartella")
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--device", type=int, default=0)
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
        format="[%(levelname)s] %(message)s"
    )

def get_equation(name:str):
    eqs = {
        'wave': WaveEquation,
        'heat': HeatEquation,
        'laplace': LaplaceEquation,
        'poisson_1': Poisson_2D_C,
        'poisson_2': Poisson_2D_CG,
        'eikonal': EikonalEquation,
        'burger': Burgers_1D,
        'kuramoto': Kuramoto_Shivashinsky,
        'example': Example
    }
    if name not in eqs:
        raise ValueError(f"Equazione sconosciuta: {name}")
    return eqs[name]()

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

# MAIN
def main():
    args = parse_args()
    cfg = load_config(os.path.join(args.path,'config.yaml'))

    if args.steps > 0:
        cfg["decomposition"]["steps"] = args.steps

    for run_id in range(args.repeat):
        seed = 42 + run_id
        torch.manual_seed(seed)
        np.random.seed(seed)

        name = cfg["name"]+'_'+cfg["decomposition"]["epochs_division"]+'_'+str(cfg["decomposition"]["epochs_extra"])
        save_dir = os.path.join("results", name, f"run_{run_id}")
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
            gpu_id = args.device
            device = torch.device(f"cuda:{gpu_id}")
            logging.info(f'Using GPU: {device}')
        else:
            device = torch.device("cpu")
            logging.info('Using CPU')

        # Caricamento dati mesh
        mesh, bulk_points, boundary_points, initial_points, indices, scheduler = mesh_preprocessing(
            path = os.path.join(args.path,'mesh.obj'),
            epochs = cfg["training"]["epochs"],
            steps = cfg["decomposition"]["steps"],
            time_axis = cfg["decomposition"]["time_axis"],
            boundary_type = cfg["decomposition"]["type"],
            bulk_n = cfg["mesh"]["bulk_points"],
            boundary_n = cfg["mesh"]["boundary_points"],
            init_n = cfg["mesh"]["initial_points"],
            preview=False
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
        boundary_value = equation.boundary_condition(boundary_points)
        boundary_data = (
            torch.from_numpy(boundary_points[:, 0:1]).to(device=device, dtype=torch.float32).requires_grad_(),
            torch.from_numpy(boundary_points[:, 1:2]).to(device=device, dtype=torch.float32).requires_grad_(),        
            torch.from_numpy(boundary_value).to(device=device, dtype=torch.float32)
        )
        del boundary_points

        # Initial
        if not initial_points is None:
            initial_value = equation.initial_condition(initial_points)
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

        '''
            TEST DATA
        '''
        data = np.loadtxt(os.path.join(args.path, 'solution.txt'), comments='%', dtype=float)
        test_data = tuple(
            torch.from_numpy(data[:, i:i+1]).to(device=device, dtype=torch.float32)
            for i in range(data.shape[1])
        )
        del data

        '''
        import matplotlib.pyplot as plt
        #plt.scatter(test_data[0].detach().cpu(),test_data[1].detach().cpu(),c=test_data[2].detach().cpu())
        t = test_data[0].detach().cpu().numpy()
        x = test_data[1].detach().cpu().numpy()
        u = test_data[2].detach().cpu().numpy()
        mask1 = x==0
        mask2 = x==2*np.pi
        t = np.concatenate((t[mask1],t[mask2]),axis=0).reshape((-1,1))
        x = np.concatenate((x[mask1],x[mask2]),axis=0).reshape((-1,1))
        u = np.concatenate((u[mask1],u[mask2]),axis=0).reshape((-1,1))
        boundary_data = (
            torch.from_numpy(t).to(device=device, dtype=torch.float32).requires_grad_(),
            torch.from_numpy(x).to(device=device, dtype=torch.float32).requires_grad_(),        
            torch.from_numpy(u).to(device=device, dtype=torch.float32)
        )
        '''

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

        # Plotter
        #plotter = Plotter(network)
        #plotter.prepare()

        '''
        TRAINING
        '''
        trainer = TrainerStep(
            pinn,
            device=device,
            ckpt_dir=ckpt_dir,
            ckpt_interval = cfg["checkpoint"]["interval"]
        )

        flops, errors = trainer.train(
            bulk_data=bulk_data,
            bdry_data=boundary_data,
            init_data=initial_data,
            test_data=test_data,
            indices=indices,
            weights=cfg["training"]["weights"],
            epochs=int(cfg["training"]["epochs"]),
            steps=int(cfg["decomposition"]["steps"]),
            divide=cfg["decomposition"]["epochs_division"],
            extra_epochs=cfg["decomposition"]["epochs_extra"],
            lr_start=float(cfg["training"]["lr"]),
            ckpt=cfg["checkpoint"]["enabled"],
            #savepath=img_dir,
        )
        
        # Results
        #solution = pinn.network(torch.tensor(mesh.vertices[:,:2], dtype=torch.float32, device=device))
        #solution = solution.detach().cpu().flatten().numpy()
        #visualize_scalar_field(mesh, solution, save_path=os.path.join(save_dir,f'solution_{cfg["decomposition"]["steps"]}'))

        #generate_gif(img_dir, save_dir, cfg["decomposition"]["steps"])

        with open(os.path.join(log_dir,f'error_{cfg["decomposition"]["steps"]}.pkl'), "wb") as f:
            pickle.dump((flops, errors), f)

if __name__ == "__main__":
    main()
