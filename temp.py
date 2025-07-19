# main.py
import os
import yaml
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt

from lib.meshes import mesh_preprocessing
from lib.pinn import (
    PINN, LaplaceEquation, Burgers_1D, WaveEquation, HeatEquation,
    Poisson_2D_C, Poisson_2D_CG, NS_2D_C, NS_2D_CG
)

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

# ---------------- PIPELINE ---------------- #
def get_equation(name:str):
    eqs = {
        'wave': WaveEquation,
        'heat': HeatEquation,
        'laplace': LaplaceEquation,
        'poisson_1': Poisson_2D_C,
        'poisson_2': Poisson_2D_CG,
        'burger': Burgers_1D,
        'ns_1': NS_2D_C,
        'ns_2': NS_2D_CG
    }
    if name not in eqs:
        raise ValueError(f"Equazione sconosciuta: {name}")
    return eqs[name]()

def prepare_data(cfg, args, device):
    mesh, bulk_points, boundary_points, initial_points, indices, scheduler = mesh_preprocessing(
            path = os.path.join(args.path,'mesh.obj'),
            epochs          = cfg["training"]["epochs"],
            steps           = cfg["decomposition"]["steps"],
            time_axis       = cfg["decomposition"]["time_axis"],
            boundary_type   = cfg["decomposition"]["type"],
            bulk_n          = cfg["mesh"]["bulk_points"],
            boundary_n      = cfg["mesh"]["boundary_points"],
            init_n          = cfg["mesh"]["initial_points"],
            preview         = False
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


def main():
    args = parse_args()
    cfg = load_config(os.path.join(args.path,'config.yaml'))

    if args.steps > 0:
        cfg["decomposition"]["steps"] = args.steps

    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Folders Setup
    device = torch.device('cuda:0')

    # Data Preparation
    mesh, bulk_data, boundary_data, initial_data, test_data, data_min, data_max, indices, equation = prepare_data(cfg, args, device)

    plt.figure(figsize=(512/100,512/100))
    plt.triplot(mesh.vertices[:,0],mesh.vertices[:,1],triangles=mesh.faces,color='k',linewidth=0.5,alpha=0.8)
    classes = []
    prev = 0
    for k, idx in enumerate(indices):
        count = idx - prev
        classes.extend([k] * count)
        prev = idx
    plt.scatter(bulk_data[0].detach().cpu(), bulk_data[1].detach().cpu(), c=classes, s=10, cmap='tab20')
    plt.scatter(boundary_data[0].detach().cpu(), boundary_data[1].detach().cpu(), c='r', s=2)
    if initial_data != None:
        plt.scatter(initial_data[0].detach().cpu(), initial_data[1].detach().cpu(),c='g',s=2)
    plt.savefig('cancella.png')


if __name__ == "__main__":
    main()