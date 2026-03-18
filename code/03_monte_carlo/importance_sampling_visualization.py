import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D projection)
from scipy.stats import multivariate_normal, chi2

# ----------------------------
# Configuration
# ----------------------------
np.random.seed(42)

# Define a bivariate normal distribution
mu = np.array([0.0, 0.0])
Sigma = np.array([[1.5, 0.8],
                  [0.8, 1.0]])

rv = multivariate_normal(mean=mu, cov=Sigma)

# Grid for evaluating the PDF
eigvals, eigvecs = np.linalg.eigh(Sigma)
max_std = np.sqrt(eigvals.max())
span = 4.5 * max_std  # cover ~ +/- 4.5 std along the largest axis

num = 200
x = np.linspace(mu[0] - span, mu[0] + span, num)
y = np.linspace(mu[1] - span, mu[1] + span, num)
X, Y = np.meshgrid(x, y)
pos = np.dstack((X, Y))
Z = rv.pdf(pos)

# Log-PDF for tail visibility (use log10 to compress dynamic range)
eps = 1e-300  # avoid log(0)
Zlog = np.log10(Z + eps)

# ----------------------------
# Tail sampling
# ----------------------------
n_samples = 100_000
samples = rv.rvs(size=n_samples)

# Mahalanobis distance squared: d^2 ~ chi2(df=2)
Sigma_inv = np.linalg.inv(Sigma)
diff = samples - mu
d2 = np.einsum('ij,jk,ik->i', diff, Sigma_inv, diff)

# Choose a tail threshold via chi-square quantile (e.g., 95% or 99%)
tail_quantile = 0.95
threshold = chi2.ppf(tail_quantile, df=2)
tail_mask = d2 > threshold
tail_samples = samples[tail_mask]

# ----------------------------
# Plotting
# ----------------------------
fig = plt.figure(figsize=(14, 6))

# Compute a visually helpful offset so points sit above the surfaces
offset_pdf = 0.07 * Z.max()  # 7% of peak height
offset_log = 0.07 * (Zlog.max() - Zlog.min())  # 7% of log-range

# 1) Linear PDF surface with tail points overlaid (lifted above the surface)
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
surf1 = ax1.plot_surface(
    X, Y, Z, cmap=cm.viridis, linewidth=0, antialiased=True,
    alpha=0.6, zorder=0
)

tail_pdf_vals = rv.pdf(tail_samples)
ax1.scatter(
    tail_samples[:, 0], tail_samples[:, 1], tail_pdf_vals + offset_pdf,
    c='crimson', s=18, alpha=0.9, edgecolor='black', linewidths=0.5,
    label=f'Tail samples (>{int(tail_quantile*100)}% contour)', depthshade=False, zorder=10
)

ax1.set_title('Bivariate Normal PDF (linear scale)')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('pdf')
fig.colorbar(surf1, ax=ax1, shrink=0.6, pad=0.1)
ax1.legend(loc='upper right', frameon=True)

# Analytic tail contour on the base (z=0) plane:
# For multivariate normal, f(x) = f(mu) * exp(-0.5 * d^2).
# So the pdf level for d^2 = threshold is:
f_mu = rv.pdf(mu)
level_on_ellipse = f_mu * np.exp(-0.5 * threshold)
cset = ax1.contour(X, Y, Z, levels=[level_on_ellipse], zdir='z', offset=0.0,
                   colors='crimson', linestyles='--', linewidths=2)

# 2) Log-PDF surface for tail visibility (lifted points)
ax2 = fig.add_subplot(1, 2, 2, projection='3d')
surf2 = ax2.plot_surface(
    X, Y, Zlog, cmap=cm.inferno, linewidth=0, antialiased=True,
    alpha=0.7, zorder=0
)
ax2.scatter(
    tail_samples[:, 0], tail_samples[:, 1], np.log10(tail_pdf_vals + eps) + offset_log,
    c='cyan', s=18, alpha=0.9, edgecolor='black', linewidths=0.5,
    label='Tail samples', depthshade=False, zorder=10
)

ax2.set_title('Bivariate Normal log10-PDF (tails emphasized)')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_zlabel('log10(pdf)')
fig.colorbar(surf2, ax=ax2, shrink=0.6, pad=0.1)
ax2.legend(loc='upper right', frameon=True)

# Improve layout and view
for ax in (ax1, ax2):
    ax.view_init(elev=30, azim=-50)
    ax.set_box_aspect((1, 1, 0.5))

plt.tight_layout()
plt.savefig("importance_sampling_visualization.png", dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.show()