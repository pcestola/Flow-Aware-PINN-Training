# scripts/train.py
# TODO: Aggiungi uno scheduler per il training (si puo scegliere
#       in maniera particolare nel caso della domain deocmposition?)
# TODO: Dal punto di vista teorico argomento l'andamento concorrente della loss
#       chi domina? quando? perche? (fare la media rimedia al class umbalance?)
#       Le geometrie complesse suggeriscono di: imparare prima boundary e initial
#       e poi bulk (mantenendo stabili le altre due).
# TODO: automatizza, in output metti tutto, anche la loss etc...
# TODO: ripensa come costruisci e carichi i dati di boundary e initial
#       attualmente nel file yaml non usi "initial_points"

# NOTE: Attualmente il rettangolo con dato iniziale esponenziale converge lentamente
#       e mostra solo un ramo di propagazione dell'onda (questa cosa era già)
#       successa, come avevamo risolto?

# TODO: La boundary_loss deve avere peso decrescente che tende ad 1.0
import os
import yaml
import torch
import logging
import argparse
import numpy as np
import matplotlib.pyplot as plt

from lib.models import SimpleNN, SIREN
from lib.train import ProgressiveTrainer, Trainer, TrainerStep
from lib.pinn import PINN, LaplaceEquation, WaveEquation, HeatEquation, EikonalEquation
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
    return parser.parse_args()


def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def setup_logging(log_cfg, path):
    os.makedirs(path, exist_ok=True)
    log_file = os.path.join(path, "train.log")
    logging.basicConfig(
        filename=log_file,
        level=getattr(logging, log_cfg["level"]),
        format="%(asctime)s [%(levelname)s] %(message)s"
    )

def get_equation(name:str):
    eqs = {
        "wave": WaveEquation,
        "heat": HeatEquation,
        "laplace": LaplaceEquation,
        "eikonal": EikonalEquation
    }
    if name not in eqs:
        raise ValueError(f"Equazione sconosciuta: {name}")
    return eqs[name]()

def main():
    args = parse_args()
    cfg = load_config(args.config)

    # Cartella di salvataggio
    save_dir = os.path.join("results", cfg["name"])
    ckpt_dir = os.path.join(save_dir, "checkpoint")
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    # Logging locale
    if cfg["logging"]["enabled"]:
        setup_logging(cfg["logging"], save_dir)
        logging.info("Inizio training")

    if torch.cuda.is_available():
        device = "cuda"
        torch.cuda.init()
        torch.cuda.current_device()
    else:
        device = "cpu"

    # Caricamento dati mesh
    mesh, bulk_points, boundary_points, initial_points, indices, scheduler = mesh_preprocessing(
        path = cfg["mesh"]["file"],
        epochs = cfg["training"]["epochs"],
        steps = cfg["mesh"]["steps"],
        time_axis = cfg["mesh"]["time_axis"],
        bulk_n = cfg["mesh"]["bulk_points"],
        boundary_n = cfg["mesh"]["boundary_points"],
        init_n = cfg["mesh"]["initial_points"],
        preview=args.preview
    )

    '''
        Costruzione dati bulk e boundary
    '''
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
        initial_value = np.exp(-4*initial_points[:,1:2]**2)
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

    # Caricamento equazione
    equation = get_equation(cfg["equation"])
    
    # Costruzione PINN
    pinn = PINN(network, equation)
    pinn.to(device)

    '''
        TRAINING
    '''
    trainer = TrainerStep(pinn, device=torch.device('cuda:0'))

    trainer.train(
        bulk_data=bulk_data,
        bdry_data=boundary_data,
        init_data=initial_data,
        indices=indices,
        weights=cfg["training"]["weights"],
        epochs=cfg["training"]["epochs"],
        steps=cfg["mesh"]["steps"],
        lr_start=1e-2
    )

    '''
    scheduler_enabled = cfg["training"].get("use_scheduler", False)

    trainer = ProgressiveTrainer(
        pinn,
        scheduler,
        device,
        ckpt_dir=ckpt_dir,
        ckpt_interval=int(cfg["checkpoint"]["interval"])
    )
    
    trainer.train(
        bulk_data=bulk_data,
        boundary_data=boundary_data,
        initial_data=initial_data,
        indices=indices,
        weights=cfg["training"]["weights"],
        epochs=int(cfg["training"]["epochs"]),
        lr=float(cfg["training"]["lr"]),
        ckpt=cfg["checkpoint"]["enabled"],
        scheduler=scheduler_enabled
    )
    '''
    
    # Show results
    solution = pinn.network(torch.tensor(mesh.vertices, dtype=torch.float32, device=device))
    solution = solution.detach().cpu().flatten().numpy()

    # TODO: aggiungi un plot 3D
    visualize_scalar_field(mesh, solution)
    

'''
    Implementa un sistema che aggiorna automaticamente i pesi dei termini di loss durante il training, per esempio:

    SoftAdapt (Heydari et al.)

    GradNorm (Chen et al.)

    NTK-based reweighting (Wang et al. 2021)
'''

if __name__ == "__main__":
    main()
