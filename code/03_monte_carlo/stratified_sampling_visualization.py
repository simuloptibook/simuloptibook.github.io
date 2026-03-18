import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib as mpl

# Reproducibility
rng = np.random.default_rng(42)

# Monotonic function f(x)
# Choose a monotonic increasing function. You can switch to x**2 or np.log1p(x) if you prefer.
def f(x):
    return np.exp(0.4 * x)  # Smoothly increasing over the domain

# Domain and strata settings
x_min, x_max = 0.0, 10.0
n_strata = 6

# Samples per stratum (can be uniform or customized per stratum)
samples_per_stratum = [4, 4, 4, 4, 4, 4]  # total 24 samples

# Generate the function curve for plotting
x_curve = np.linspace(x_min, x_max, 800)
y_curve = f(x_curve)

# Define stratum boundaries (equal-width strata)
bounds = np.linspace(x_min, x_max, n_strata + 1)

# Prepare colormap for strata
cmap = mpl.cm.get_cmap('Set3', n_strata)

# Set up the plot
fig, ax = plt.subplots(figsize=(10, 6))

# Plot the function curve
ax.plot(x_curve, y_curve, color='black', linewidth=2, label='f(x)')

# Shade each stratum and sample points
sample_x = []
sample_y = []
for i in range(n_strata):
    left, right = bounds[i], bounds[i + 1]
    # Shade stratum region (vertical band)
    rect = Rectangle((left, ax.get_ylim()[0] if ax.get_ylim()[0] != -np.inf else 0),
                     right - left,
                     1,  # temporary height; will normalize after setting y-limits
                     facecolor=cmap(i), alpha=0.12, edgecolor='none')
    # We'll add rectangles after we know y-limits; store for later adjustment
    ax.add_patch(rect)

# Plot the curve first to set y-limits accurately
ax.plot(x_curve, y_curve, color='black', linewidth=2)

# Now adjust rectangles to span the y-range properly
ymin, ymax = ax.get_ylim()
for i, patch in enumerate(ax.patches):
    left = bounds[i]
    right = bounds[i + 1]
    patch.set_xy((left, ymin))
    patch.set_height(ymax - ymin)
    patch.set_width(right - left)

# Draw vertical lines for stratum boundaries
for b in bounds:
    ax.axvline(b, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)

# Sample uniformly within each stratum and plot points on f(x)
for i in range(n_strata):
    left, right = bounds[i], bounds[i + 1]
    k = samples_per_stratum[i]
    xs = rng.uniform(left, right, size=k)
    ys = f(xs)
    sample_x.extend(xs)
    sample_y.extend(ys)
    ax.scatter(xs, ys, s=60, color=cmap(i), edgecolor='k', linewidth=0.6,
               label=f'Stratum {i+1} samples' if i == 0 else None)  # single legend entry

# Optional: annotate strata centers
for i in range(n_strata):
    mid = 0.5 * (bounds[i] + bounds[i + 1])
    ax.text(mid, ymin + 0.03*(ymax - ymin), f'S{i+1}', ha='center', va='bottom',
            fontsize=10, color='dimgray')

# Labels and legend
ax.set_title('Visualization of Stratified Sampling on a Monotonic Function')
ax.set_xlabel('x (stratification variable)')
ax.set_ylabel('f(x)')
ax.grid(True, alpha=0.2)
ax.set_xlim(x_min, x_max)
ax.legend(loc='upper left')

plt.tight_layout()
plt.savefig("stratified_sampling_visualization.png", dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.show()