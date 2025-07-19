import torch
import meshio
import trimesh
import numpy as np
import pyvista as pv
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
import point_cloud_utils as pcu
from scipy.spatial import cKDTree
from matplotlib import cm

# https://fwilliams.info/point-cloud-utils/sections/mesh_sampling/#:~:text=at%20the%20points.-,Generating%20blue%2Dnoise%20random%20samples%20on%20a%20mesh,are%20separated%20by%20some%20radius.&text=Generating%20blue%20noise%20samples%20on,a%20target%20radius%20(right).

'''
    MESHIO
'''
def meshio_get_topological_boundary(mesh:meshio.Mesh, vertices=False):
    edge_count = defaultdict(int)
    for cell in mesh.cells[0].data:
        n = len(cell)
        for i in range(n):
            edge = (cell[i],cell[(i+1)%n])
            edge = tuple(sorted(edge))
            edge_count[edge] += 1

    boundary_edges = [edge for edge, count in edge_count.items() if count == 1]

    if vertices:
        return list(np.unique(np.array(boundary_edges).flatten()))
    else:
        return boundary_edges

def meshio_classify_boundary(mesh:meshio.Mesh, time_axis:int):
    minimum = np.min(mesh.points[:, time_axis])
    classes = []

    edges = get_topological_boundary(mesh)

    for edge in edges:
        p1 = mesh.points[edge[0]]
        p2 = mesh.points[edge[1]]
        t1, t2 = p1[time_axis], p2[time_axis]

        if np.isclose(t1, minimum) and np.isclose(t2, minimum):
            classes.append(0)
        elif np.isclose(t1, t2):
            classes.append(2)
        else:
            classes.append(1)

    return classes

'''
    TRIMESH
'''
def load_mesh(path):
    return trimesh.load(path, process=False)

def get_topological_boundary(mesh:trimesh.Trimesh) -> np.ndarray:
    edges = mesh.edges_sorted
    view = np.ascontiguousarray(edges).view(
        np.dtype((np.void, edges.dtype.itemsize * edges.shape[1]))
    )
    _, idx, counts = np.unique(view, return_index=True, return_counts=True)
    boundary = edges[idx[counts == 1]]
    return boundary

# NOTE: prima di navier stokes
def classify_boundary_old(mesh:trimesh.Trimesh, time_axis:int):
    
    minimum = np.min(mesh.vertices[:, time_axis])
    classes = []

    edges = get_topological_boundary(mesh)

    for edge in edges:
        p1 = mesh.vertices[edge[0]]
        p2 = mesh.vertices[edge[1]]
        t1, t2 = p1[time_axis], p2[time_axis]

        if np.isclose(t1, minimum) and np.isclose(t2, minimum):
            classes.append(0)
        elif np.isclose(t1, t2):
            classes.append(2)
        else:
            classes.append(1)

    classes = np.array(classes)

    return classes

def classify_boundary(mesh:trimesh.Trimesh, time_axis:int):
    
    minimum = np.min(mesh.vertices[:, time_axis])
    maximum = np.max(mesh.vertices[:, time_axis])
    classes = []

    edges = get_topological_boundary(mesh)

    for edge in edges:
        p1 = mesh.vertices[edge[0]]
        p2 = mesh.vertices[edge[1]]
        t1, t2 = p1[time_axis], p2[time_axis]

        if np.isclose(t1, minimum) and np.isclose(t2, minimum):
            classes.append(0)
        elif np.isclose(t1, maximum) and np.isclose(t2, maximum):
            classes.append(2)
        else:
            classes.append(1)

    classes = np.array(classes)

    return classes

def get_geodesic_distance(mesh:trimesh.Trimesh, boundary:np.ndarray, normalize=True) -> np.array:
    
    G = nx.Graph()
    for (v1, v2), w in zip(mesh.edges_unique, mesh.edges_unique_length):
        G.add_edge(int(v1), int(v2), weight=float(w))
    
    sources = list(np.unique(boundary.flatten()))

    dist_map = nx.multi_source_dijkstra_path_length(G, sources)
    distance = np.full(mesh.vertices.shape[0], np.inf)
    for i, val in dist_map.items():
        distance[i] = val
    if normalize:
        m = np.isfinite(distance)
        mn, mx = distance[m].min(), distance[m].max()
        distance[m] = (distance[m] - mn) / (mx - mn)
    
    return distance

def progressive_domain_decomposition(
    mesh,
    boundary,
    epochs: int = 100,
    steps: int = 10,
    scheduler: str = 'linear',
    submeshes: bool = False):
    
    d = get_geodesic_distance(mesh, boundary)
    
    if scheduler == 'linear':
        sched = list(np.linspace(0, epochs, steps + 1, dtype=int)[1:])
    else:
        raise ValueError('unsupported scheduler')

    if submeshes:
        #fd = d[mesh.faces].mean(axis=1)
        fd = d[mesh.faces].min(axis=1)
        subs = []
        for i in range(steps):
            m = (fd >= i/steps) & (fd < (i+1)/steps)
            if m.any():
                subs.append(mesh.submesh([m], append=True))
        return subs, sched
    
    bins = np.linspace(0, 1, steps + 1)
    cls = np.clip(np.digitize(d, bins), 0, steps)

    return cls, sched

def sample_points_on_mesh(mesh:trimesh.Trimesh, n:int) -> np.ndarray:
    
    tri = mesh.vertices[mesh.faces][:, :, :2]
    v0, v1, v2 = tri[:, 0], tri[:, 1], tri[:, 2]
    area = 0.5 * np.abs(
        (v1[:,0]-v0[:,0])*(v2[:,1]-v0[:,1]) -
        (v1[:,1]-v0[:,1])*(v2[:,0]-v0[:,0])
    )

    prob = area / area.sum()
    idx = np.random.choice(len(tri), size=n, p=prob)
    t0, t1, t2 = v0[idx], v1[idx], v2[idx]
    u = np.random.rand(n,1)
    v = np.random.rand(n,1)
    o = u+v>1
    u[o] = 1-u[o]
    v[o] = 1-v[o]
    w = 1-u-v
    pts = t0*u + t1*v + t2*w

    return pts

def sample_points_on_mesh_poisson_disk(mesh:trimesh.Trimesh, num_samples:int, radius:float=None):
    
    #OLD: v = np.concatenate((mesh.vertices, np.zeros((mesh.vertices.shape[0],1))), axis=1)
    v = mesh.vertices
    f = mesh.faces

    if radius == None:
        fid, bc = pcu.sample_mesh_poisson_disk(v, f, num_samples=num_samples)
        rand_positions = pcu.interpolate_barycentric_coords(f, fid, bc, v)
    else:
        fid, bc = pcu.sample_mesh_poisson_disk(v, f, num_samples=-1, radius=radius)
        rand_positions = pcu.interpolate_barycentric_coords(f, fid, bc, v)
    
    return rand_positions[:,:2]

def sample_points_on_boundary(mesh: trimesh.Trimesh, boundary: np.ndarray, n: int) -> np.ndarray:
    # Estrai i vertici 2D
    v2d = mesh.vertices[:, :2]

    # Estrai i segmenti (archi)
    s = v2d[boundary[:, 0]]
    e = v2d[boundary[:, 1]]

    # Calcola lunghezza di ciascun segmento
    seg_lengths = np.linalg.norm(e - s, axis=1)
    total_length = np.sum(seg_lengths)

    # Punti equispaziati lungo tutto il contorno
    target_distances = np.linspace(0, total_length, n, endpoint=False)

    # Posizioni cumulative dei segmenti
    cumulative = np.insert(np.cumsum(seg_lengths), 0, 0.0)

    # Costruzione dei punti equispaziati
    points = []
    seg_idx = 0
    for d in target_distances:
        # Avanza finché trovi il segmento giusto
        while seg_idx + 1 < len(cumulative) and cumulative[seg_idx + 1] < d:
            seg_idx += 1

        # Interpolazione lineare
        local_d = d - cumulative[seg_idx]
        t = local_d / seg_lengths[seg_idx] if seg_lengths[seg_idx] > 0 else 0.0
        pt = (1 - t) * s[seg_idx] + t * e[seg_idx]
        points.append(pt)

    # Aggiungi estremi di ogni arco
    arc_endpoints = np.vstack([s, e])  # shape (2M, 2)

    # Combina e rimuovi duplicati (se desiderato)
    all_points = np.vstack([points, arc_endpoints])
    all_points = np.unique(all_points, axis=0)

    return all_points


def get_progressive_dataset_old(
    mesh,
    epochs: int = 100,
    steps: int = 10,
    time_axis: int = -1,
    boundary_type: str = "all",
    bulk_n: int = 1000,
    boundary_n: int = 100,
    init_n: int = 0,
    preview=False):

    boundary = get_topological_boundary(mesh)

    if time_axis >= 0:
        boundary_classes = classify_boundary(mesh, time_axis=time_axis)
        initial_boundary = boundary[boundary_classes == 0]
        boundary = boundary[boundary_classes == 1]
    # NOTE: da scrivere meglio e generalizzare
    elif boundary_type == 'initial':
        boundary_classes = classify_boundary(mesh, time_axis=time_axis)
        initial_boundary = boundary[boundary_classes == 0]
    else:
        initial_boundary = None

    # 1. Generazione dei punti bulk sull'intero dominio
    bulk_points = sample_points_on_mesh_poisson_disk(mesh, bulk_n)

    # 2. Mappa di distanza geodetica su tutti i vertici
    if boundary_type == "initial":
        distance_map = get_geodesic_distance(mesh, initial_boundary)
    elif boundary_type == "boundary":
        distance_map = get_geodesic_distance(mesh, boundary)
    elif boundary_type == "all":
        if initial_boundary is not None:
            distance_map = get_geodesic_distance(mesh, np.vstack((initial_boundary,boundary)))
        else:
            distance_map = get_geodesic_distance(mesh, boundary)

    # 3. Assegnazione della distanza ai punti: proiezione sul vertice più vicino
    tree = cKDTree(mesh.vertices[:, :2])
    _, nn_idx = tree.query(bulk_points, k=1)
    point_distances = distance_map[nn_idx]

    # Classificazione in steps intervallati
    bins = np.linspace(0, 1, steps + 1)
    cls = np.clip(np.digitize(point_distances, bins), 0, steps)

    # Ordina i punti in base alla classe
    sorted_idx = np.argsort(cls)
    bulk_points = bulk_points[sorted_idx]
    cls = cls[sorted_idx]

    # Scheduler lineare
    sched = list(np.linspace(0, epochs, steps + 1, dtype=int)[1:])

    # Indici per separare le classi
    idxs = np.cumsum(np.unique(cls, return_counts=True)[1])

    # Campionamento dei punti di bordo
    boundary_points = sample_points_on_boundary(mesh, boundary, boundary_n)

    # Campionamento iniziale se richiesto
    if init_n > 0 and initial_boundary is not None:
        initial_points = sample_points_on_boundary(mesh, initial_boundary, init_n)
    else:
        initial_points = None

    return bulk_points, boundary_points, initial_points, idxs, sched


def get_progressive_dataset(
    mesh,
    epochs: int = 100,
    steps: int = 10,
    time_axis: int = -1,
    boundary_type: str = "all",
    bulk_n: int = 1000,
    boundary_n: int = 100,
    init_n: int = 0,
    preview=False):

    boundary = get_topological_boundary(mesh)

    if time_axis >= 0:
        boundary_classes = classify_boundary(mesh, time_axis=time_axis)
        initial_boundary = boundary[boundary_classes == 0]
        boundary = boundary[boundary_classes == 1]
        if boundary_type == 'all':
            information_boundary = np.vstack((initial_boundary,boundary))
        elif boundary_type == 'boundary':
            information_boundary = boundary
        elif boundary_type == 'initial':
            information_boundary = initial_boundary
    elif boundary_type == 'initial':
        boundary_classes = classify_boundary(mesh, time_axis=0)
        information_boundary = boundary[boundary_classes == 0]
        initial_boundary = None
    else:
        information_boundary = boundary
        initial_boundary = None

    # 1. Generazione dei punti bulk sull'intero dominio
    bulk_points = sample_points_on_mesh_poisson_disk(mesh, bulk_n)

    # 2. Mappa di distanza geodetica su tutti i vertici
    distance_map = get_geodesic_distance(mesh, information_boundary)

    # Allo stato attuale la divisione è uniforme nella distanza
    # Pertanto non lo è necessariamente nel numero di elementi per classe
    distance_map = distance_map[mesh.faces].min(axis=1)
    bins = np.linspace(0, 1, steps + 1)
    faces_class = np.clip(np.digitize(distance_map, bins), 0, steps)

    points_3d = np.concatenate((bulk_points, np.zeros((bulk_points.shape[0],1))),axis=1)
    _, _, faces_id = mesh.nearest.on_surface(points_3d)
    points_class = faces_class[faces_id]

    # Ordina i punti in base alla classe
    sorted_idx = np.argsort(points_class)
    bulk_points = bulk_points[sorted_idx]
    points_class = points_class[sorted_idx]

    # Scheduler lineare
    sched = list(np.linspace(0, epochs, steps + 1, dtype=int)[1:])

    # Indici per separare le classi
    idxs = np.cumsum(np.unique(points_class, return_counts=True)[1]).tolist()

    # Gestione casi particolari
    if len(idxs) < steps:
        idxs += [idxs[-1]] * (steps - len(idxs))
    elif len(idxs) > steps:
        idxs = idxs[:steps]

    # Campionamento dei punti di bordo
    boundary_points = sample_points_on_boundary(mesh, boundary, boundary_n)

    # Campionamento iniziale se richiesto
    if init_n > 0 and initial_boundary is not None:
        initial_points = sample_points_on_boundary(mesh, initial_boundary, init_n)
    else:
        initial_points = None

    return bulk_points, boundary_points, initial_points, idxs, sched


def get_progressive_dataset_bhv(
    mesh,
    epochs: int = 100,
    steps: int = 10,
    time_axis: int = -1,
    boundary_type: str = "all",
    bulk_n: int = 1000,
    boundary_n: int = 100,
    init_n: int = 0,
    preview=False):

    assert boundary_type in {"initial", "boundary", "all"}

    boundary = get_topological_boundary(mesh)

    if time_axis >= 0:
        boundary_classes = classify_boundary(mesh, time_axis=time_axis)
        initial_boundary = boundary[boundary_classes == 0]
        boundary = boundary[boundary_classes == 1]
    else:
        initial_boundary = None

    # 1. Sample bulk points
    bulk_points = sample_points_on_mesh_poisson_disk(mesh, bulk_n)

    # 2. Compute geodesic distance
    if boundary_type == "all":
        ref_boundary = boundary if initial_boundary is None else np.vstack((initial_boundary, boundary))
    elif boundary_type == "boundary":
        ref_boundary = boundary
    elif boundary_type == "initial":
        ref_boundary = initial_boundary
    distance_map = get_geodesic_distance(mesh, ref_boundary)

    # 3. Classify faces by min vertex distance
    bins = np.linspace(0, 1, steps + 1)
    faces_distance = distance_map[mesh.faces].min(axis=1)
    faces_class = np.clip(np.digitize(faces_distance, bins), 0, steps) - 1

    # 4. Classify points using BVH
    _, _, face_ids = mesh.nearest.on_surface(np.hstack((bulk_points, np.zeros((bulk_points.shape[0], 1)))))
    cls = faces_class[face_ids]

    # 5. Sort points by class
    sorted_idx = np.argsort(cls)
    bulk_points = bulk_points[sorted_idx]
    cls = cls[sorted_idx]
    counts = np.bincount(cls, minlength=steps+1)
    idxs = np.cumsum(counts[counts > 0])

    # Scheduler lineare
    sched = list(np.linspace(0, epochs, steps + 1, dtype=int)[1:])

    # Campionamento dei punti di bordo
    boundary_points = sample_points_on_boundary(mesh, boundary, boundary_n)

    # Campionamento iniziale se richiesto
    if init_n > 0 and initial_boundary is not None:
        initial_points = sample_points_on_boundary(mesh, initial_boundary, init_n)
    else:
        initial_points = None

    return bulk_points, boundary_points, initial_points, idxs, sched


'''
    GRAPHIC
'''
def visualize_boundary(mesh:trimesh.Trimesh, boundary_edges:list, classes:list=None):

    #OLD: points_3d = np.column_stack((mesh.vertices, np.zeros(len(mesh.vertices))))
    points_3d = mesh.vertices

    surface = pv.PolyData(points_3d)
    
    lines = []
    colors = []
    for i, edge in enumerate(boundary_edges):
        lines.append([2, edge[0], edge[1]])
        if classes is not None:
            colors.append(classes[i])
        else:
            colors.append(0)

    edge_lines = np.hstack(lines).astype(np.int32)
    edges = pv.PolyData()
    edges.points = points_3d
    edges.lines = edge_lines

    plotter = pv.Plotter()
    plotter.add_mesh(surface, color="white", opacity=0.1, point_size=5)
    plotter.add_mesh(edges, scalars=colors, line_width=3, cmap=["blue", "red", "black"], show_scalar_bar=False)
    plotter.show()
    import random
    plotter.screenshot(f"cancella_{random.randint(0,100)}.png")

def visualize_dataset(
        bulk_points: np.ndarray,
        boundary_points: np.ndarray,
        initial_points: np.ndarray,
        idxs: list
    ):
    fig, ax = plt.subplots(figsize=(8, 6))

    # Boundary
    ax.scatter(boundary_points[:, 0], boundary_points[:, 1], s=20, c='k', label='Boundary')

    # Initial
    if initial_points is not None:
        ax.scatter(initial_points[:, 0], initial_points[:, 1], s=30, c='red', label='Initial')

    # Bulk: crea una mappa di classi per gruppi progressivi
    colors = np.zeros((bulk_points.shape[0]), dtype=int)
    for i, idx in enumerate(idxs):
        colors[idx:] += 1

    # TODO: sistema questo pezzo di codice
    cmap = cm.get_cmap('viridis', colors.shape[0])

    scatter = ax.scatter(
        bulk_points[:, 0], bulk_points[:, 1],
        c=colors,
        cmap=cmap,
        s=10,
        label='Bulk',
        alpha=0.7
    )

    ax.set_title("Generated Dataset")

    # Sposta la leggenda fuori dal plot
    ax.legend(loc='center left', bbox_to_anchor=(1.05, 0.5))
    plt.tight_layout()
    plt.savefig('./dataset.png', dpi=300, bbox_inches='tight')

def visualize_scalar_field(mesh, s, face=False, cmap='bwr', save_path=None):
    v = mesh.vertices[:, :2]
    f = mesh.faces
    if face:
        plt.tripcolor(v[:,0], v[:,1], f, facecolors=s, cmap=cmap, edgecolors='k')
    else:
        plt.tripcolor(v[:,0], v[:,1], f, s, cmap=cmap, edgecolors='k')
    plt.colorbar()
    plt.axis('equal')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

'''
    EXTRA
'''
def get_points_classes(
    x, y,
    mesh,
    steps,
    time_axis,
    boundary_type
    ):

    boundary = get_topological_boundary(mesh)

    if time_axis >= 0:
        boundary_classes = classify_boundary(mesh, time_axis=time_axis)
        initial_boundary = boundary[boundary_classes == 0]
        boundary = boundary[boundary_classes == 1]
    else:
        initial_boundary = None

    # 2. Mappa di distanza geodetica su tutti i vertici
    if boundary_type == "initial":
        distance_map = get_geodesic_distance(mesh, initial_boundary)
    elif boundary_type == "boundary":
        distance_map = get_geodesic_distance(mesh, boundary)
    elif boundary_type == "all":
        if initial_boundary is not None:
            distance_map = get_geodesic_distance(mesh, np.vstack((initial_boundary,boundary)))
        else:
            distance_map = get_geodesic_distance(mesh, boundary)

    # 3. Concatena
    points = torch.cat((x,y),dim=1).detach().cpu()

    # 3. Assegnazione della distanza ai punti: proiezione sul vertice più vicino
    tree = cKDTree(mesh.vertices[:, :2])
    _, nn_idx = tree.query(points, k=1)
    point_distances = distance_map[nn_idx]

    # Classificazione in steps intervallati
    bins = np.linspace(0, 1, steps + 1)
    cls = np.clip(np.digitize(point_distances, bins), 0, steps)

    return cls


'''
    MAIN
'''
def mesh_preprocessing(
        path:str,
        epochs:int,
        steps:int,
        time_axis:int,
        boundary_type:str,
        bulk_n:int,
        boundary_n:int,
        init_n:int,
        preview=False):

    mesh = trimesh.load(path, process=False)

    bulk_points, boundary_points, initial_points, idxs, scheduler = get_progressive_dataset(
        mesh,
        epochs,
        steps,
        time_axis,
        boundary_type,
        bulk_n,
        boundary_n,
        init_n,
        preview
    )

    if preview:
        visualize_dataset(bulk_points, boundary_points, initial_points, idxs)
    
    return mesh, bulk_points, boundary_points, initial_points, idxs, scheduler
    
if __name__ == '__main__':
    mesh_preprocessing('meshes/holes.obj',preview=True)