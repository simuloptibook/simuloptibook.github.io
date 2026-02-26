import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

# Generate time series data
n = 100
t = np.arange(n)

# Stationary process: AR(1) with |phi| < 1
phi_stationary = 0.7
epsilon = np.random.normal(0, 1, n)
stationary_process = np.zeros(n)
for i in range(1, n):
    stationary_process[i] = phi_stationary * stationary_process[i-1] + epsilon[i]

# Non-stationary process: Random walk (AR(1) with phi = 1)
non_stationary_process = np.zeros(n)
for i in range(1, n):
    non_stationary_process[i] = non_stationary_process[i-1] + epsilon[i]

# Create the figure and subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot stationary process
ax1.plot(t, stationary_process, linewidth=1)
ax1.axhline(y=0, color='r', linestyle='--', alpha=0.7)
ax1.set_title('Stationary Stochastic Process', fontsize=14, fontweight='bold')
ax1.set_xlabel('Time', fontsize=12)
ax1.set_ylabel('Value', fontsize=12)
ax1.grid(True, alpha=0.3)

# Add summary statistics for stationary process
mean_stationary = np.mean(stationary_process)
std_stationary = np.std(stationary_process)
ax1.text(0.02, 0.98, f'Mean: {mean_stationary:.3f}\nStd: {std_stationary:.3f}', 
         transform=ax1.transAxes, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Plot non-stationary process
ax2.plot(t, non_stationary_process, linewidth=1)
ax2.set_title('Non-Stationary Stochastic Process (Random Walk)', fontsize=14, fontweight='bold')
ax2.set_xlabel('Time', fontsize=12)
ax2.set_ylabel('Value', fontsize=12)
ax2.grid(True, alpha=0.3)

# Add summary statistics for non-stationary process
mean_non_stationary = np.mean(non_stationary_process)
std_non_stationary = np.std(non_stationary_process)
ax2.text(0.02, 0.98, f'Mean: {mean_non_stationary:.3f}\nStd: {std_non_stationary:.3f}', 
         transform=ax2.transAxes, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

plt.tight_layout()

# Save as PNG file with high resolution
output_file = 'stationary_vs_nonstationary.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')

print(f"Graph saved as {output_file}")

plt.show()
