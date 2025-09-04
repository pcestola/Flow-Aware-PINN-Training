# main.py
import os
import yaml
import torch
import pickle
import logging
import argparse
import subprocess
import numpy as np

from lib.models import SimpleNN, SIREN
from lib.trainer import TrainerStep
from lib.pinn import (
    PINN, LaplaceEquation, Burgers_1D, WaveEquation_1D, HeatEquation,
    Poisson_2D_C, NS_2D_C, NS_2D_CG, EikonalEquation
)
from lib.meshes import mesh_preprocessing, visualize_scalar_field


# ---------------- UTILS ---------------- #
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
        format="%(asctime)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

def get_equation(name:str):
    eqs = {
        'wave': WaveEquation_1D,
        'heat': HeatEquation,
        'laplace': LaplaceEquation,
        'poisson_1': Poisson_2D_C,
        'burger': Burgers_1D,
        'ns_1': NS_2D_C,
        'ns_2': NS_2D_CG,
        'eikonal': EikonalEquation,
        'eikonal_2': EikonalEquation
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


# ---------------- PIPELINE ---------------- #
def setup_experiment(cfg, args, run_id, seed):
    name = cfg["name"]
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

    if cfg["logging"]["enabled"]:
        setup_logging(cfg["logging"], log_dir, cfg["decomposition"]["subdomains"])
        logging.info(f"Esecuzione {run_id+1}/{args.repeat} con seed={seed}")

    if torch.cuda.is_available():
        if args.device >= 0:
            gpu_id = args.device
        else:
            gpu_id = get_free_gpu()
        device = torch.device(f"cuda:{gpu_id}")
        logging.info(f'Using GPU: {device}')
    else:
        device = torch.device("cpu")
        logging.info('Using CPU')
    
    return device, ckpt_dir, log_dir, save_dir

def prepare_data(cfg, args, device):
    mesh, bulk_points, boundary_points, initial_points, indices = mesh_preprocessing(
            path = os.path.join(args.path,'mesh.obj'),
            epochs          = cfg["training"]["epochs"],
            steps           = cfg["decomposition"]["subdomains"],
            time_axis       = cfg["decomposition"]["time_axis"],
            directed_axis   = cfg["decomposition"]["directed_axis"],
            boundary_type   = cfg["decomposition"]["information_boundary"],
            bulk_n          = cfg["mesh"]["bulk_points"],
            boundary_n      = cfg["mesh"]["boundary_points"],
            init_n          = cfg["mesh"]["initial_points"]
        )

    equation = get_equation(cfg["equation"])

    # Bulk Data
    bulk_data = (
        torch.from_numpy(bulk_points[:, 0:1]).to(device=device, dtype=torch.float32).requires_grad_(),
        torch.from_numpy(bulk_points[:, 1:2]).to(device=device, dtype=torch.float32).requires_grad_()
    )
    data_min = bulk_points.min(axis=0)
    data_max = bulk_points.max(axis=0)

    # Boundary Data
    if not boundary_points is None:
        boundary_value = equation.boundary_condition(boundary_points)
        boundary_data = (
            torch.from_numpy(boundary_points[:, 0:1]).to(device=device, dtype=torch.float32).requires_grad_(),
            torch.from_numpy(boundary_points[:, 1:2]).to(device=device, dtype=torch.float32).requires_grad_(),        
            torch.from_numpy(boundary_value).to(device=device, dtype=torch.float32)
        )
    else:
        boundary_data = None

    # Initial Data
    if not initial_points is None:
        initial_value, initial_velocity = equation.initial_condition(initial_points)
        if not initial_velocity is None:
            initial_data = (
                torch.from_numpy(initial_points[:, 0:1]).to(device=device, dtype=torch.float32).requires_grad_(),
                torch.from_numpy(initial_points[:, 1:2]).to(device=device, dtype=torch.float32).requires_grad_(),
                torch.from_numpy(initial_value).to(device=device, dtype=torch.float32),
                torch.from_numpy(initial_velocity).to(device=device, dtype=torch.float)
            )
        else:
            initial_data = (
                torch.from_numpy(initial_points[:, 0:1]).to(device=device, dtype=torch.float32).requires_grad_(),
                torch.from_numpy(initial_points[:, 1:2]).to(device=device, dtype=torch.float32).requires_grad_(),
                torch.from_numpy(initial_value).to(device=device, dtype=torch.float32)
            )
    else:
        initial_data = None

    # Test Data
    test_points_value = np.loadtxt(os.path.join(args.path, 'solution.txt'), comments='%', dtype=float)
    test_data = (
        torch.from_numpy(test_points_value[:, 0:1]).to(device=device, dtype=torch.float32),
        torch.from_numpy(test_points_value[:, 1:2]).to(device=device, dtype=torch.float32),
        torch.from_numpy(test_points_value[:, 2:]).to(device=device, dtype=torch.float32)
    )

    return mesh, bulk_data, boundary_data, initial_data, test_data, data_min, data_max, indices, equation

def build_network(cfg, data_min=None, data_max=None):
    net_type = cfg["model"]["type"]
    if net_type == 'simple':
        network = SimpleNN(cfg["model"]["dims"], data_min, data_max)
    elif net_type == 'siren':
        network = SIREN(cfg["model"]["dims"])
    return network

def run_train(cfg, trainer, pinn, device, bulk_data, boundary_data, initial_data, test_data, mesh, indices, save_dir, plot=False):

    flops, errors = trainer.train(
        bulk_data=bulk_data,
        bdry_data=boundary_data,
        init_data=initial_data,
        test_data=test_data,
        indices=indices,
        weights=cfg["training"]["weights"],
        epochs=int(cfg["training"]["epochs"]),
        steps=int(cfg["decomposition"]["subdomains"]),
        divide_mode=cfg["decomposition"]["epochs_scheduling"],
        extra_epochs=cfg["decomposition"]["epochs_extra"],
        lr_start=float(cfg["training"]["lr"]),
        ckpt=cfg["checkpoint"]["enabled"]
    )
    
    # Results
    if plot:
        solution = pinn.network(torch.tensor(mesh.vertices[:,:2], dtype=torch.float32, device=device))
        solution = solution.detach().cpu().flatten().numpy()
        visualize_scalar_field(mesh, solution, save_path=os.path.join(save_dir,f'solution_{cfg["decomposition"]["subdomains"]}'))

    return flops, errors    


def main():
    args = parse_args()
    cfg = load_config(os.path.join(args.path,'config.yaml'))

    if args.steps > 0:
        cfg["decomposition"]["subdomains"] = args.steps

    for run_id in range(args.repeat):
        seed = 42 + run_id
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Folders Setup
        device, ckpt_dir, log_dir, save_dir = setup_experiment(cfg, args, run_id, seed)

        # Data Preparation
        mesh, bulk_data, boundary_data, initial_data, test_data, data_min, data_max, indices, equation = prepare_data(cfg, args, device)

        # Costruzione modello
        network = build_network(cfg, data_min, data_max)
        
        # Costruzione PINN
        pinn = PINN(network, equation)
        pinn.to(device)

        # Trainer
        trainer = TrainerStep(
            pinn,
            device=device,
            ckpt_dir=ckpt_dir,
            ckpt_interval = cfg["checkpoint"]["interval"]
        )

        # Training
        flops, errors = run_train(
            cfg, trainer, pinn, device,
            bulk_data, boundary_data, initial_data, test_data,
            mesh, indices,
            save_dir,
            plot=False)
        
        #solution = pinn.network(torch.tensor(mesh.vertices[:,:2], dtype=torch.float32, device=device))
        #solution = solution.detach().cpu().numpy().flatten()
        #visualize_scalar_field(mesh, solution, save_path=os.path.join(save_dir,f'solution_{cfg["decomposition"]["subdomains"]}'))

        with open(os.path.join(log_dir,f'error_{cfg["decomposition"]["subdomains"]}.pkl'), "wb") as f:
            pickle.dump((flops, errors), f)

if __name__ == "__main__":
    main()