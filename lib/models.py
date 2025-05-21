import math
import torch
import torch.nn as nn

class SimpleNN(nn.Module):
    def __init__(self, dimensions, xmin, xmax):
        super(SimpleNN, self).__init__()
        self.register_buffer("xmin", torch.tensor(xmin,dtype=torch.float))
        self.register_buffer("xmax", torch.tensor(xmax,dtype=torch.float))
        
        layers = []
        for i in range(len(dimensions) - 1):
            layers.append(nn.Linear(dimensions[i], dimensions[i+1]))
            if i < len(dimensions) - 2:
                layers.append(nn.Tanh())
        self.fc = nn.Sequential(*layers)

        for layer in self.fc:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

    def normalize_input(self, x):
        return 2 * (x - self.xmin) / (self.xmax - self.xmin) - 1

    def forward(self, x):
        x = self.normalize_input(x)
        return self.fc(x)
    

class Sine(nn.Module):
    def __init__(self, omega_0=1.0):
        super().__init__()
        self.omega_0 = omega_0

    def forward(self, x):
        return torch.sin(self.omega_0 * x)

class SIRENLayer(nn.Module):
    def __init__(self, in_features, out_features, omega_0=1.0, is_first=False):
        super().__init__()
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features)
        self.omega_0 = omega_0

        self.init_weights(is_first)

    def init_weights(self, is_first):
        with torch.no_grad():
            if is_first:
                bound = 1 / self.in_features
            else:
                bound = math.sqrt(6 / self.in_features) / self.omega_0
            self.linear.weight.uniform_(-bound, bound)
            self.linear.bias.uniform_(-bound, bound)

    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))

class SIREN(nn.Module):
    def __init__(self, dimensions, omega_0=30.0):
        super(SIREN, self).__init__()
        layers = []

        for i in range(len(dimensions) - 1):
            is_first = (i == 0)
            layer_omega = omega_0 if is_first else 1.0
            layers.append(SIRENLayer(dimensions[i], dimensions[i+1], omega_0=layer_omega, is_first=is_first))

        self.net = nn.Sequential(*layers[:-1],  # All but last with sine
                                 layers[-1])   # Last layer without sine if you want raw output

    def forward(self, x):
        return self.net(x)