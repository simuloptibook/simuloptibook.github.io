import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

def monte_carlo_pi(n_points: int, seed: int | None = None, show: bool = True):
    """
    Monte Carlo estimation of pi with visualization.

    Parameters
    ----------
    n_points : int
        Number of random points to sample in [-1, 1] x [-1, 1].
    seed : int | None
        Random seed for reproducibility.
    show : bool
        If True, displays the plot.

    Returns
    -------
    pi_est : float
        Estimated value of pi.
    fig, ax : matplotlib Figure and Axes
        The visualization figure and axes.
    """
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1.0, 1.0, n_points)
    y = rng.uniform(-1.0, 1.0, n_points)

    # Points inside the unit circle: x^2 + y^2 <= 1
    inside = (x**2 + y**2) <= 1.0
    pi_est = 4.0 * inside.mean()  # pi ≈ 4 * (# inside / N)

    # Plot
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(x[inside], y[inside], c='blue', s=10, alpha=0.7, label='Inside circle')
    ax.scatter(x[~inside], y[~inside], c='green', s=10, alpha=0.7, label='Outside circle')

    # Draw the unit circle boundary
    circle = Circle((0, 0), 1.0, fill=False, edgecolor='black', linewidth=2)
    ax.add_patch(circle)

    # Formatting
    ax.set_title(f"Monte Carlo π estimate: {pi_est:.6f}  (N={n_points})")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')

    if show:
        plt.savefig("monte_carlo_pi.png", dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.show()

    return pi_est, fig, ax

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monte Carlo visualization for estimating pi.")
    parser.add_argument("-n", "--n_points", type=int, default=5000,
                        help="Number of random points to generate (default: 5000).")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility (default: None).")
    args = parser.parse_args()

    monte_carlo_pi(n_points=args.n_points, seed=args.seed, show=True)