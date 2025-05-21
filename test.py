import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

# === Modello ===
class SimpleNN(nn.Module):
    def __init__(self, dimensions):
        super().__init__()
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
    def forward(self, x):
        return self.fc(x)

# === Dati ===
x = torch.linspace(-np.pi, np.pi, 1000).unsqueeze(1)
y = torch.sin(x)

# === Modello, loss, ottimizzatore ===
net = SimpleNN([1, 32, 32, 32, 1])
loss_fn = nn.MSELoss()
opt = torch.optim.Adam(net.parameters(), lr=1e-3)

print(net)

# === Training ===
for epoch in range(2000):
    opt.zero_grad()
    pred = net(x)
    loss = loss_fn(pred, y)
    loss.backward()
    opt.step()
    if epoch % 200 == 0:
        print(f"Epoch {epoch} | Loss: {loss.item():.6f}")

# === Plot ===
with torch.no_grad():
    y_pred = net(x).squeeze().numpy()

plt.plot(x.numpy(), y.numpy(), label='sin(x)')
plt.plot(x.numpy(), y_pred, label='MLP output')
plt.legend()
plt.grid(True)
plt.title("Fit di sin(x) con SimpleNN")
plt.show()