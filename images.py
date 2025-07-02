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

    if torch.cuda.is_available():
        gpu_id = args.device
        device = torch.device(f"cuda:{gpu_id}")
        logging.info(f'Using GPU: {device}')
    else:
        device = torch.device("cpu")
        logging.info('Using CPU')

    # Caricamento dati mesh
    import trimesh
    import matplotlib.pyplot as plt
    from lib.meshes import get_topological_boundary, progressive_domain_decomposition, visualize_scalar_field, get_geodesic_distance, get_progressive_dataset, get_progressive_dataset_bhv
    
    mesh = trimesh.load(os.path.join(args.path,'mesh.obj'))
    boundary = get_topological_boundary(mesh)
    distance = get_geodesic_distance(mesh, boundary, True)
    subs, _ = progressive_domain_decomposition(mesh,boundary,100,4,'linear',True)
    
    all_vertices = []
    all_faces = []
    all_values = []
    offset = 0

    for i, sub in enumerate(subs):
        all_vertices.append(sub.vertices)
        all_faces.append(sub.faces + offset)
        all_values.append(np.full(sub.faces.shape[0], i))
        offset += sub.vertices.shape[0]

    all_vertices = np.vstack(all_vertices)
    all_faces = np.vstack(all_faces)
    all_values = np.concatenate(all_values)

    bulk_points, _, _, idxs, _ = get_progressive_dataset(mesh,steps=4,bulk_n=1000,boundary_n=100)

    fig, axs = plt.subplots(1,4,figsize=(24,6),constrained_layout=True)
    axs[0].tripcolor(
        mesh.vertices[:, 0], 
        mesh.vertices[:, 1], 
        mesh.faces,
        facecolors=np.ones(mesh.faces.shape[0]),
        cmap='Greys',
        edgecolors='k'
    )
    for edge in boundary:
        p1 = mesh.vertices[edge[0]]
        p2 = mesh.vertices[edge[1]]
        axs[0].plot([p1[0], p2[0]], [p1[1], p2[1]], color='red', linewidth=1.5)
    axs[0].axis('off')
    axs[0].axis('equal')

    axs[1].tripcolor(mesh.vertices[:,0], mesh.vertices[:,1], mesh.faces, distance, cmap='magma', edgecolors='k')
    axs[1].axis('off')
    axs[1].axis('equal')

    axs[2].tripcolor(all_vertices[:, 0], all_vertices[:, 1], all_faces, facecolors=all_values, cmap='tab10', edgecolors='k')
    axs[2].axis('off')
    axs[2].axis('equal')

    for edge in boundary:
        p1 = mesh.vertices[edge[0]]
        p2 = mesh.vertices[edge[1]]
        axs[3].plot([p1[0], p2[0]], [p1[1], p2[1]], color='red', linewidth=1.5)
    
    colors = np.zeros((bulk_points.shape[0]))
    for color, (start, end) in enumerate(zip(idxs[:-1],idxs[1:])):
        colors[start:end] = color+1
    axs[3].scatter(bulk_points[:,0],bulk_points[:,1],s=40,c=colors,cmap='tab10',edgecolor='k',linewidth=0.8)
    axs[3].axis('off')
    axs[3].axis('equal')
    
    plt.savefig('decomposition.png', dpi=300, bbox_inches='tight', pad_inches=0)


if __name__ == "__main__":
    main()
