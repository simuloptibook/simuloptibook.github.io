import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# Parameters: independent standard normals
mu = np.array([0.0, 0.0])
Sigma = np.eye(2)  # [[1, 0], [0, 1]]

# Grid over which to evaluate the PDF
x = np.linspace(-4, 4, 200)
y = np.linspace(-4, 4, 200)
X, Y = np.meshgrid(x, y)

# Evaluate the PDF (closed-form for independent standard normals)
Z = (1 / (2 * np.pi)) * np.exp(-(X**2 + Y**2) / 2)

# Create figure with vertical stacking (top: 3D surface, bottom: 2D contour)
fig = plt.figure(figsize=(8, 10))

# Top subplot: 3D surface
ax_top = fig.add_subplot(2, 1, 1, projection='3d')
surf = ax_top.plot_surface(X, Y, Z, cmap=cm.viridis, linewidth=0, antialiased=True)
ax_top.set_title('Bivariate Standard Normal PDF (Surface)')
ax_top.set_xlabel('x')
ax_top.set_ylabel('y')
ax_top.set_zlabel('f(x, y)')
fig.colorbar(surf, shrink=0.7, aspect=12, ax=ax_top, pad=0.1)

# Bottom subplot: 2D contour
ax_bottom = fig.add_subplot(2, 1, 2)
contours = ax_bottom.contour(X, Y, Z, levels=10, cmap=cm.viridis)
ax_bottom.clabel(contours, inline=True, fontsize=8)
ax_bottom.set_title('Contours (Isodensity Curves)')
ax_bottom.set_xlabel('x')
ax_bottom.set_ylabel('y')
ax_bottom.set_aspect('equal', adjustable='box')

plt.tight_layout()
plt.savefig('bivariate_normal.png', dpi=300, bbox_inches='tight')
plt.show()