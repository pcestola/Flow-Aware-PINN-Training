import torch
import trimesh
import numpy as np
import pyvista as pv
import networkx as nx
import matplotlib.pyplot as plt
import point_cloud_utils as pcu

from matplotlib import cm
from scipy.spatial import cKDTree


# Geometry
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

def boundary_point_normals_2d(mesh, topo_boundary, boundary_points,
                              eps=1e-12, strict=True, vertex_tol=1e-9):
    V = np.asarray(mesh.vertices)[:, :2]
    F = np.asarray(mesh.faces, dtype=int)
    P = np.asarray(boundary_points, dtype=float)

    if isinstance(topo_boundary, np.ndarray) and topo_boundary.ndim == 2 and topo_boundary.shape[1] == 2:
        E = topo_boundary.astype(int)
    else:
        E = np.concatenate([
            np.stack([loop, np.roll(loop, -1)], axis=1) for loop in topo_boundary
        ], axis=0).astype(int)
    if E.size == 0:
        raise ValueError("topo_boundary vuoto.")

    edge2faces = {}
    for f_idx, (a, b, c) in enumerate(F):
        for u, v in ((a, b), (b, c), (c, a)):
            key = (u, v) if u < v else (v, u)
            edge2faces.setdefault(key, []).append(f_idx)

    A = V[E[:, 0]]; B = V[E[:, 1]]
    EV = B - A
    L2 = np.einsum('ij,ij->i', EV, EV)
    good = L2 > eps
    if not np.all(good):
        EV[~good] = 0.0
        L2[~good] = 1.0

    T = EV / np.sqrt(L2)[:, None]
    n_right = np.stack([T[:, 1], -T[:, 0]], axis=1)

    N_edge = np.empty_like(n_right)
    for m, (i, j) in enumerate(E):
        key = (i, j) if i < j else (j, i)
        faces = edge2faces.get(key, [])
        if strict and len(faces) != 1:
            raise ValueError(f"Edge {i}-{j}: {len(faces)} facce adiacenti (atteso 1).")
        if len(faces) == 1:
            f = faces[0]; tri = F[f]
            k = tri[~np.isin(tri, [i, j])][0]
            e = EV[m]
            cross_z = e[0]*(V[k,1]-V[i,1]) - e[1]*(V[k,0]-V[i,0])
            N_edge[m] = n_right[m] if cross_z > 0 else -n_right[m]
        else:
            N_edge[m] = n_right[m]

    PA = P[:, None, :] - A[None, :, :]
    t = np.einsum('nmi,mi->nm', PA, EV) / L2[None, :]
    t_clamped = np.clip(t, 0.0, 1.0)
    Pproj = A[None, :, :] + t_clamped[..., None] * EV[None, :, :]

    # >>> FIX QUI <<<
    delta = P[:, None, :] - Pproj
    d2 = np.einsum('nmi,nmi->nm', delta, delta)
    # <<< FIX QUI >>>

    edge_idx = np.argmin(d2, axis=1)
    normals = N_edge[edge_idx].copy()

    vert2edges = {}
    for m, (i, j) in enumerate(E):
        vert2edges.setdefault(int(i), []).append(m)
        vert2edges.setdefault(int(j), []).append(m)

    t_best = t[np.arange(P.shape[0]), edge_idx]
    near_i = np.isclose(t_best, 0.0, atol=vertex_tol)
    near_j = np.isclose(t_best, 1.0, atol=vertex_tol)
    if np.any(near_i) or np.any(near_j):
        for idx_point in np.where(near_i | near_j)[0]:
            m = edge_idx[idx_point]
            i, j = E[m]
            v = i if near_i[idx_point] else j
            inc = vert2edges.get(int(v), [])
            if inc:
                n_avg = N_edge[inc].mean(axis=0)
                nl = np.linalg.norm(n_avg)
                if nl > eps:
                    normals[idx_point] = n_avg / nl

    return normals

def get_progressive_dataset(
    mesh,
    epochs: int = 100,
    steps: int = 10,
    time_axis: int = None,
    directed_axis: int = None,
    boundary_type: str = "all",
    bulk_n: int = 1000,
    boundary_n: int = 100,
    initial_n: int = 0):

    boundary = get_topological_boundary(mesh)

    # Get Spatial, Initial, Information Boundaries
    if time_axis is not None:
        boundary_classes = classify_boundary(mesh, time_axis=time_axis)
        initial_boundary = boundary[boundary_classes == 0]
        spatial_boundary = boundary[boundary_classes == 1]
        if boundary_type == 'all':
            information_boundary = np.vstack((initial_boundary,spatial_boundary))
        elif boundary_type == 'spatial':
            information_boundary = spatial_boundary
        elif boundary_type == 'initial':
            information_boundary = initial_boundary
    elif directed_axis is not None:
        boundary_classes = classify_boundary(mesh, time_axis=directed_axis)
        initial_boundary = None
        spatial_boundary = boundary
        if boundary_type == 'all' or boundary_type == 'spatial':
            information_boundary = boundary
        elif boundary_type == 'directed':
            information_boundary = boundary[boundary_classes == 0]
    else:
        initial_boundary = None
        spatial_boundary = boundary
        information_boundary = boundary

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
    boundary_points = sample_points_on_boundary(mesh, spatial_boundary, boundary_n)

    # Campionamento iniziale se richiesto
    if initial_boundary is not None:
        initial_points = sample_points_on_boundary(mesh, initial_boundary, initial_n)
    else:
        initial_points = None

    return bulk_points, boundary_points, initial_points, idxs


# Drawing
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


# Extra  
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


# Main
def mesh_preprocessing(
        path:str,
        epochs:int,
        steps:int,
        time_axis:int,
        directed_axis:int,
        boundary_type:str,
        bulk_n:int,
        boundary_n:int,
        init_n:int):

    mesh = trimesh.load(path, process=False)

    bulk_points, boundary_points, initial_points, idxs = get_progressive_dataset(
        mesh,
        epochs,
        steps,
        time_axis,
        directed_axis,
        boundary_type,
        bulk_n,
        boundary_n,
        init_n
    )
    
    return mesh, bulk_points, boundary_points, initial_points, idxs
    
if __name__ == '__main__':
    mesh_preprocessing('meshes/holes.obj')