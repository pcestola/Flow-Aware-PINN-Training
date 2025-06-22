import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn

class Plotter:
    def __init__(self, model):
        self.model = model
        self.gradient_accumulator = None
        self.gradient_counter = 0

    def prepare(self):
        self.layers = [m for m in self.model.modules() if isinstance(m, nn.Linear)]
        self.dims = [layer.in_features for layer in self.layers]
        self.dims.append(self.layers[-1].out_features)

        self.node_positions = []
        self.edges = []
        self.node_indices = []
        node_id = 0
        x_gap, y_gap = 20.0, 1.0

        for layer_idx, n in enumerate(self.dims):
            y0 = - (n - 1) * y_gap / 2
            ids = []
            for j in range(n):
                self.node_positions.append((layer_idx * x_gap, y0 + j * y_gap))
                ids.append(node_id)
                node_id += 1
            self.node_indices.append(ids)

        for l, layer in enumerate(self.layers):
            src, tgt = self.node_indices[l], self.node_indices[l+1]
            for i, s in enumerate(src):
                for j, t in enumerate(tgt):
                    self.edges.append((s, t))

    def accumulate_gradients(self):
        if not self.gradient_accumulator:
            self.gradient_accumulator = []
            for module in self.model.modules():
                if isinstance(module, nn.Linear) and module.weight.grad is not None:
                    self.gradient_accumulator.append(module.weight.grad.detach().clone())
                else:
                    self.gradient_accumulator.append(None)
        else:
            for i, module in enumerate(self.model.modules()):
                if isinstance(module, nn.Linear) and module.weight.grad is not None:
                    if self.gradient_accumulator[i] is not None:
                        self.gradient_accumulator[i] += module.weight.grad.detach()
        self.gradient_counter += 1

    def average_gradients(self):
        for i, module in enumerate(self.model.modules()):
            if isinstance(module, nn.Linear) and self.gradient_accumulator[i] is not None:
                module.weight.grad = self.gradient_accumulator[i] / self.gradient_counter

    def reset_gradients(self):
        self.gradient_accumulator = []
        self.gradient_counter = 0

    def plot(self, ax, show_gradients=False):
        # Preleva valori: pesi o gradienti
        values = []
        for layer in self.layers:
            tensor = layer.weight.grad if show_gradients else layer.weight
            arr = tensor.detach().cpu().numpy()
            values.extend(np.abs(arr).flatten())

        values = np.array(values)
        if values.ptp() < 1e-8:  # evita divisione per zero
            norm = np.zeros_like(values)
        else:
            norm = (values - values.min()) / (values.ptp() + 1e-8)

        # Scegli la colormap
        cmap_raw = plt.cm.inferno if show_gradients else plt.cm.viridis
        cmap = cmap_raw(norm)

        # Disegna archi
        for idx, (src, tgt) in enumerate(self.edges):
            x1, y1 = self.node_positions[src]
            x2, y2 = self.node_positions[tgt]
            ax.plot([x1, x2], [y1, y2], color=cmap[idx], linewidth=1)

        # Disegna nodi
        xs, ys = zip(*self.node_positions)
        ax.scatter(xs, ys, color='gray', s=20, zorder=3)

        ax.set_aspect('equal')
        ax.axis('off')
