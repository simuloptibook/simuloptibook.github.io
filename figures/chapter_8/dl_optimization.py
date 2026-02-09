import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Set random seed for reproducibility
np.random.seed(42)

# Define the grid for the landscape
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)

# Create a rugged landscape using a combination of Gaussian functions
Z = np.zeros_like(X)
for _ in range(20):  # Add 20 random Gaussian "bumps"
    x_center = np.random.uniform(-4, 4)
    y_center = np.random.uniform(-4, 4)
    amplitude = np.random.uniform(0.5, 2.0)
    sigma = np.random.uniform(0.5, 1.5)
    Z += amplitude * np.exp(-((X - x_center)**2 + (Y - y_center)**2) / (2 * sigma**2))

# Add noise to make the landscape more rugged
Z += 0.1 * np.random.randn(*X.shape)

# Add a global minimum
global_min_x, global_min_y = 2.0, 2.0
Z -= 3.0 * np.exp(-((X - global_min_x)**2 + (Y - global_min_y)**2) / 0.5)

# Plot the 3D landscape
fig = plt.figure(figsize=(12, 6))

# 3D Surface Plot
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
ax1.set_title('3D Rugged Optimization Landscape')
ax1.set_xlabel('Weight 1')
ax1.set_ylabel('Weight 2')
ax1.set_zlabel('Loss')

# Contour Plot
ax2 = fig.add_subplot(122)
contour = ax2.contour(X, Y, Z, levels=20, cmap='viridis')
ax2.set_title('Contour Plot of Rugged Landscape')
ax2.set_xlabel('Weight 1')
ax2.set_ylabel('Weight 2')
plt.colorbar(contour, ax=ax2, label='Loss')

plt.tight_layout()

# Save the plot as an image file
plt.savefig('rugged_landscape.png', dpi=300, bbox_inches='tight')
print("Plot saved as 'rugged_landscape.png'")
