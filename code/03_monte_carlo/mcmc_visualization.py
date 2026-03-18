import numpy as np
import matplotlib.pyplot as plt

# Set a random seed for reproducible visualizations
np.random.seed(42)

# Define the rugged probability landscape (Target Distribution)
def probability_landscape(x, y):
    # A broad Gaussian envelope to keep the distribution bounded
    envelope = np.exp(-(x**2 + y**2) / 10)
    # High-frequency sine and cosine waves to create rugged peaks and valleys
    ruggedness = 1.5 + np.sin(2.5 * x) * np.cos(2.5 * y)
    return envelope * ruggedness

# Create a meshgrid for plotting the 3D surface
x_range = np.linspace(-4, 4, 150)
y_range = np.linspace(-4, 4, 150)
X, Y = np.meshgrid(x_range, y_range)
Z = probability_landscape(X, Y)

# Create a single wide figure
fig = plt.figure(figsize=(20, 8))

# ==========================================
# SUBPLOT 1: Many points on the landscape
# ==========================================
ax1 = fig.add_subplot(121, projection='3d')
ax1.set_title("1. Many Samples", fontsize=16)

# Plot the slightly transparent surface
ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.6, edgecolor='none')

# Generate many points using simple rejection sampling
num_points = 400
rx = np.random.uniform(-4, 4, num_points * 10)
ry = np.random.uniform(-4, 4, num_points * 10)
rz = probability_landscape(rx, ry)
# Accept points based on their probability height
accept_mask = np.random.uniform(0, np.max(Z), num_points * 10) < rz

px = rx[accept_mask][:num_points]
py = ry[accept_mask][:num_points]
pz = probability_landscape(px, py) + 0.05 # Small offset

# Scatter the points
ax1.scatter(px, py, pz, color='red', s=20, alpha=0.8, edgecolors='black', linewidth=0.5)

ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Probability Density')


# ==========================================
# SUBPLOT 2: Single point trajectory with arrows
# ==========================================
ax2 = fig.add_subplot(122, projection='3d')
ax2.set_title("2. Single MCMC Trajectory", fontsize=16)

# Plot the same surface (slightly more transparent to help visibility)
ax2.plot_surface(X, Y, Z, cmap='viridis', alpha=0.5, edgecolor='none')

# Simulate a single MCMC Trajectory using the Metropolis-Hastings algorithm
steps = 30

# MODIFICATION: Start in the negative Y area (the "front" relative to the camera).
# The MCMC walker will naturally climb up the front-facing slope.
current_x, current_y = -1.5, -3.5 
history_x, history_y =[current_x], [current_y]

for _ in range(steps):
    # Propose a new state via a random Gaussian jump
    prop_x = current_x + np.random.normal(0, 0.7)
    prop_y = current_y + np.random.normal(0, 0.7)
    
    # Calculate acceptance probability
    p_current = probability_landscape(current_x, current_y)
    p_prop = probability_landscape(prop_x, prop_y)
    
    acceptance_ratio = p_prop / p_current
    
    # Accept or reject the proposal
    if np.random.rand() < acceptance_ratio:
        current_x, current_y = prop_x, prop_y
        
    history_x.append(current_x)
    history_y.append(current_y)

hx = np.array(history_x)
hy = np.array(history_y)

# MODIFICATION: Increased Z offset (0.15 instead of 0.05) to ensure 
# arrows hover clearly above the bumps and avoid 3D clipping issues.
hz = probability_landscape(hx, hy) + 0.15 

# Draw the 3D arrows connecting the trajectory steps
for i in range(len(hx) - 1):
    dx = hx[i+1] - hx[i]
    dy = hy[i+1] - hy[i]
    dz = hz[i+1] - hz[i]
    
    # Only draw an arrow if the proposed state was accepted
    if dx != 0 or dy != 0:
        ax2.quiver(hx[i], hy[i], hz[i], 
                   dx, dy, dz, 
                   color='red', arrow_length_ratio=0.15, linewidths=2.5)

# Plot the single final current point (the "walker")
ax2.scatter(hx[-1], hy[-1], hz[-1], color='red', s=80, edgecolors='black', zorder=5)

# Explicitly set the camera angle so the front slope is perfectly visible
ax2.view_init(elev=30, azim=-50)

ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Probability Density')

# Show the combined plot
plt.tight_layout()
plt.savefig("mcmc_visualization.png", dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.show()