import trimesh
import triangle
import numpy as np
import matplotlib.pyplot as plt

# Burgers Equation 1D
vertices = np.array([
    [0.0, -1.0],
    [1.0, -1.0],
    [1.0,  1.0],
    [0.0,  1.0]
])
segments = np.array([
    [0, 1],
    [1, 2],
    [2, 3],
    [3, 0]
])
Geometry = dict(vertices=vertices, segments=segments)
# 'p': PSLG, 'q30': min angle 30°, 'a0.01': max area
t = triangle.triangulate(Geometry, 'pq30a0.01')
mesh = trimesh.Trimesh(vertices=t['vertices'], faces=t['triangles'])

plt.figure(figsize=(6, 6))
for tri in t['triangles']:
    pts = t['vertices'][tri]
    pts = np.vstack([pts, pts[0]])  # chiude il triangolo
    plt.plot(pts[:, 0], pts[:, 1], 'k-')

plt.gca().set_aspect('equal')
plt.axis('off')
plt.tight_layout()
plt.savefig("mesh.png", dpi=300)
plt.close()

mesh.export('burger_1d.obj')