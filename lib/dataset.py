import torch
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

class DataGenerator:
    def __init__(self, algorithm="grid"):
        """
        Generatore di dati flessibile per PINNs.
        
        :param algorithm: Algoritmo di generazione ("grid", "sobol", "random").
        """
        self.algorithm = algorithm
    
    def generate_bulk(self, bounds, num_points, transpose=False):
        """
        Genera dati bulk (intervallo interno del dominio).
        
        :param bounds: Lista di tuple [(xmin, xmax), (tmin, tmax)].
        :param num_points: Numero totale di punti da generare.
        :return: Tensor di dati generati.
        """
        if self.algorithm == "sobol":
            x, t = self._generate_sobol(bounds, num_points, 2)
        elif self.algorithm == "random":
            x, t = self._generate_random(bounds, num_points, 2)
        elif self.algorithm == "grid":
            x, t = self._generate_grid(bounds, num_points, transpose)
        else:
            raise ValueError(f"Algoritmo sconosciuto: {self.algorithm}")
        
        return x.unsqueeze(1).requires_grad_(), t.unsqueeze(1).requires_grad_()
    
    def generate_boundary(self, bounds, num_points, boundary_function):
        """
        Genera dati boundary (solo bordi del dominio).
        
        :param bounds: Lista di tuple [(xmin, xmax), (tmin, tmax)].
        :param num_points: Numero di punti da generare.
        :param boundary_function: Funzione che specifica il valore di u ai bordi.
        :return: Tensor di punti e corrispondenti valori della funzione.
        """
        x_bounds, t_bounds = bounds
        x_min, x_max = x_bounds
        t_min, t_max = t_bounds
        dt = (t_max-t_min)/num_points
        
        # Generazione di punti sui bordi
        x = torch.cat([
            torch.full((num_points,), x_min, dtype=torch.float),
            torch.full((num_points,), x_max, dtype=torch.float)
        ],dim=0)
        
        t = torch.cat([
            torch.linspace(t_min+dt, t_max, num_points),
            torch.linspace(t_min+dt, t_max, num_points)
        ],dim=0)

        u = boundary_function(x, t)

        return x.unsqueeze(1).requires_grad_(), t.unsqueeze(1).requires_grad_(), u.unsqueeze(1)
    
    def generate_initial(self, bounds, num_points, initial_function):
        """
        Genera dati iniziali (t=0, dominio spaziale completo).
        
        :param bounds: Lista di tuple [(xmin, xmax), (tmin, tmax)].
        :param num_points: Numero di punti da generare.
        :param initial_function: Funzione che specifica il valore iniziale u(x, 0).
        :return: Tensor di punti e corrispondenti valori della funzione.
        """
        x_bounds, t_bounds = bounds
        x_min, x_max = x_bounds
        t_min, _ = t_bounds

        # Generazione punti iniziali
        x = torch.linspace(x_min, x_max, num_points)
        
        t = torch.full((num_points,), t_min, dtype=torch.float)

        u = initial_function(x)

        return x.unsqueeze(1).requires_grad_(), t.unsqueeze(1).requires_grad_(), u.unsqueeze(1)
    
    def _generate_sobol(self, bounds, num_points, dim):
        """
        Genera punti usando la sequenza di Sobol.
        """
        sobol_engine = torch.quasirandom.SobolEngine(dimension=dim)
        sobol_points = sobol_engine.draw(num_points)
        scaled_points = self._scale_points(sobol_points, bounds)
        return scaled_points[:,0], scaled_points[:,1]
    
    def _generate_random(self, bounds, num_points, dim):
        """
        Genera punti casuali uniformi.
        """
        random_points = torch.rand(num_points, dim)
        scaled_points = self._scale_points(random_points, bounds)
        return scaled_points[:,0], scaled_points[:,1]
    
    def _generate_grid(self, bounds, num_points, transpose=False):
        """
        Genera punti su una griglia regolare.
        """
        step_size_0 = (bounds[0][1] - bounds[0][0]) / (num_points[0]+1)
        step_size_1 = (bounds[1][1] - bounds[1][0]) / num_points[1]
        intervals = [
            (bounds[0][0]+step_size_0, bounds[0][1]-step_size_0),
            (bounds[1][0]+step_size_1, bounds[1][1])
        ]
        grids = [torch.linspace(i[0], i[1], n) for i, n in zip(intervals,num_points)]
        grid = torch.meshgrid(*grids, indexing='ij')
        if transpose:
            data = torch.cat([g.T.flatten().unsqueeze(1) for g in grid], dim=1)
        else:
            data = torch.cat([g.flatten().unsqueeze(1) for g in grid], dim=1)
        return data[:,0], data[:,1]
    
    def _scale_points(self, points, bounds):
        """
        Scala i punti generati nell'intervallo specificato dai bounds.
        """
        for i, (lower, upper) in enumerate(bounds):
            points[:, i] = points[:, i] * (upper - lower) + lower
        return points

def plot_generated_data(bulk_data, boundary_points, initial_points, bounds, figsize=(12,6)):
    """
    Crea un plot per visualizzare i dati bulk, boundary e initial.
    
    :param bulk_data: Tensore dei dati bulk.
    :param boundary_points: Tensore dei punti boundary.
    :param initial_points: Tensore dei punti initial.
    :param bounds: Limiti del dominio [(xmin, xmax), (tmin, tmax)].
    """
    xmin, xmax = bounds[0]
    tmin, tmax = bounds[1]

    fig, axs = plt.subplots(2,2,figsize=figsize)

    # Dati bulk
    if bulk_data is not None:
        axs[0,0].scatter(bulk_data[1].detach().numpy(), bulk_data[0].detach().numpy(), s=10, c='gray', label="Bulk Points", alpha=0.4, edgecolors='k')

    # Dati boundary
    if boundary_points is not None:
        axs[0,0].scatter(boundary_points[1].detach().numpy(), boundary_points[0].detach().numpy(), s=20, c='red', label="Boundary Points", alpha=0.8, edgecolors='k')

    # Dati initial
    if initial_points is not None:
        axs[0,0].scatter(initial_points[1].detach().numpy(), initial_points[0].detach().numpy(), s=20, c='green', label="Initial Points", alpha=0.8, edgecolors='k')

    # Miglioramenti estetici
    axs[0,0].set_ylabel("Space (x)", fontsize=10)

    # Imposta i limiti e i tick sugli assi
    axs[0,0].set_xlim(tmin*1.05, tmax*1.05)
    axs[0,0].set_ylim(xmin*1.05, xmax*1.05)
    axs[0,0].set_xticks(torch.linspace(tmin, tmax, 5).tolist())
    axs[0,0].set_yticks(torch.linspace(xmin, xmax, 5).tolist())

    # Griglia e legenda
    axs[0,0].grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
    axs[0,0].legend(fontsize=9, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3)

    axs[0,1].plot(initial_points[0].detach().numpy(), initial_points[2].detach().numpy(), '-k')

    length = boundary_points[2].shape[0]//2
    axs[1,0].plot(boundary_points[1][:length].detach().numpy(), boundary_points[2][:length].detach().numpy(), '-k')
    axs[1,1].plot(boundary_points[1][length:].detach().numpy(), boundary_points[2][length:].detach().numpy(), '-k')
    
    axs[1,0].set_xlabel("Time (t)", fontsize=10)
    axs[1,0].set_ylabel("Space (x)", fontsize=10)
    axs[1,1].set_xlabel("Time (t)", fontsize=10)
    plt.show()

class CircularDataGenerator:
    def __init__(self, algorithm="polar"):
        """
        Generatore di dati per un dominio circolare.

        :param algorithm: Algoritmo di generazione ("grid", "sobol", "random").
        """
        self.algorithm = algorithm

    def generate_bulk(self, center, radius, num_points):
        """
        Genera dati bulk all'interno del dominio circolare.

        :param center: Centro del cerchio (x0, y0).
        :param radius: Raggio del cerchio.
        :param num_points: Numero di punti da generare.
        :return: Due tensori con le coordinate x e y dei punti bulk generati.
        """
        if self.algorithm == "sobol":
            x, y = self._generate_sobol(center, radius, num_points)
        elif self.algorithm == "random":
            x, y = self._generate_random(center, radius, num_points)
        elif self.algorithm == "polar":
            x, y = self._generate_polar(center, radius, num_points)
        else:
            raise ValueError(f"Algoritmo sconosciuto: {self.algorithm}")

        return x.unsqueeze(1).requires_grad_(), y.unsqueeze(1).requires_grad_()

    def generate_boundary(self, center, radius, num_points, boundary_function):
        """
        Genera dati boundary sulla circonferenza del cerchio.

        :param center: Centro del cerchio (x0, y0).
        :param radius: Raggio del cerchio.
        :param num_points: Numero di punti sulla circonferenza.
        :param boundary_function: Funzione che specifica il valore di u al bordo.
        :return: Tre tensori con le coordinate x, y e i valori della funzione.
        """
        theta = torch.linspace(0, 2 * math.pi, num_points)
        x = center[0] + radius * torch.cos(theta)
        y = center[1] + radius * torch.sin(theta)

        u = boundary_function(x, y)

        return x.unsqueeze(1).requires_grad_(), y.unsqueeze(1).requires_grad_(), u.unsqueeze(1)

    def _generate_sobol(self, center, radius, num_points):
        """
        Genera punti bulk usando la sequenza di Sobol.
        """
        sobol_engine = torch.quasirandom.SobolEngine(dimension=2)
        sobol_points = sobol_engine.draw(num_points)
        
        # Scala i punti nel rettangolo circoscritto al cerchio
        scaled_points = sobol_points * (2 * radius) - radius
        
        # Trasla i punti al centro del cerchio
        scaled_points += torch.tensor(center, dtype=torch.float32)

        # Filtra i punti fuori dal cerchio
        mask = ((scaled_points[:, 0] - center[0]) ** 2 + (scaled_points[:, 1] - center[1]) ** 2) <= radius ** 2
        valid_points = scaled_points[mask]
        
        return valid_points[:, 0], valid_points[:, 1]

    def _generate_random(self, center, radius, num_points):
        """
        Genera punti bulk casuali uniformemente distribuiti.
        """
        points_x, points_y = [], []
        while len(points_x) < num_points:
            candidate = torch.rand(2) * (2 * radius) - radius
            candidate += torch.tensor(center, dtype=torch.float32)

            # Aggiungi il punto se è dentro il cerchio
            if ((candidate[0] - center[0]) ** 2 + (candidate[1] - center[1]) ** 2) <= radius ** 2:
                points_x.append(candidate[0])
                points_y.append(candidate[1])

        return torch.tensor(points_x), torch.tensor(points_y)
    
    def _generate_polar(self, center, radius, num_points):
        """
        Genera punti bulk in coordinate polari, riempiendo il cerchio in modo uniforme.

        :param center: Centro del cerchio (x0, y0).
        :param radius: Raggio del cerchio.
        :param num_points: Numero di punti da generare.
        :return: Due tensori con le coordinate x e y dei punti bulk generati.
        """
        # Calcola il numero di punti per livello radiale
        num_r = int(torch.sqrt(torch.tensor(num_points * 2 / math.pi)))
        r = torch.linspace(0, radius, num_r)

        # Distribuisci i punti angolari in funzione del raggio per densità uniforme
        x_list, y_list = [], []
        for radius_value in r:
            if radius_value == 0:
                theta = torch.tensor([0.0])  # Un solo punto al centro
            else:
                num_theta = max(1, int(2 * math.pi * radius_value / (radius / num_r)))  # Densità angolare adattiva
                theta = torch.linspace(0, 2 * math.pi - (2 * math.pi / num_theta), num_theta)

            # Coordinate cartesiane
            x_list.append(center[0] + radius_value * torch.cos(theta))
            y_list.append(center[1] + radius_value * torch.sin(theta))

        # Concatenazione di tutti i livelli radiali
        x = torch.cat(x_list)
        y = torch.cat(y_list)

        return x, y

    def plot_generated_data(self, bulk_data, boundary_points, center, radius, figsize=(8, 8)):
        """
        Visualizza i dati bulk e boundary per il dominio circolare.

        :param bulk_data: Tensore dei dati bulk.
        :param boundary_points: Tensore dei punti boundary.
        :param center: Centro del cerchio (x0, y0).
        :param radius: Raggio del cerchio.
        """
        fig, ax = plt.subplots(figsize=figsize)

        # Dati bulk
        if bulk_data is not None:
            ax.scatter(bulk_data[0].detach().numpy(), bulk_data[1].detach().numpy(), s=5, c='gray', label="Bulk Points", alpha=0.6, edgecolor='k')

        # Dati boundary
        if boundary_points is not None:
            ax.scatter(boundary_points[0].detach().numpy(), boundary_points[1].detach().numpy(), s=15, c='red', label="Boundary Points", alpha=0.8, edgecolor='k')

        # Circonferenza del dominio
        theta = torch.linspace(0, 2 * math.pi, 100)
        circle_x = center[0] + radius * torch.cos(theta)
        circle_y = center[1] + radius * torch.sin(theta)
        #ax.plot(circle_x.numpy(), circle_y.numpy(), linestyle="--", color="black", alpha=0.5, label="Domain Boundary")

        ax.set_aspect('equal', adjustable='datalim')
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3)
        ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
        plt.show()


def plot_generated_data_3d(bulk_data, boundary_data, initial_data, figsize=(10, 6)):

    # Estrai i dati (CPU + detach)
    bt, bx, bv = boundary_data
    if initial_data != None:
        if len(initial_data) == 4:
            it, ix, iv, _ = initial_data
        else:
            it, ix, iv = initial_data

    bkx_np = bulk_data[0].detach().cpu().numpy()
    bkt_np = bulk_data[1].detach().cpu().numpy()

    bx_np = boundary_data[0].detach().cpu().numpy()
    bt_np = boundary_data[1].detach().cpu().numpy()
    bv_np = boundary_data[2].detach().cpu().numpy()

    if initial_data != None:
        ix_np = initial_data[0].detach().cpu().numpy()
        it_np = initial_data[1].detach().cpu().numpy()
        iv_np = initial_data[2].detach().cpu().numpy()

    # Plot 3D
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(bkt_np, bkx_np, np.zeros_like(bkt_np), color='green', label='Bulk', alpha=0.6)
    ax.scatter(bt_np, bx_np, bv_np, color='red', label='Boundary', alpha=0.6)
    if initial_data != None:
        ax.scatter(it_np, ix_np, iv_np, color='blue', label='Initial', alpha=0.6)

    ax.set_xlabel('t')
    ax.set_ylabel('x')
    ax.set_zlabel('u')
    ax.set_title('Dati al Bordo e Iniziali')
    ax.legend()

    plt.savefig('./3d.png', dpi=300, bbox_inches='tight')