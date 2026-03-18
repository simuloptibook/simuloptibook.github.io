import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def proposal_distribution(x):
    """Proposal distribution: uniform distribution"""
    return np.ones_like(x) * 0.5

def target_distribution(x):
    """Target distribution: standard normal PDF"""
    return norm.pdf(x)

def rejection_sampling_normal(n_samples, seed=None):
    """
    Generate random samples from standard normal distribution using rejection sampling.
    
    Parameters:
    -----------
    n_samples : int
        Number of samples to generate
    seed : int, optional
        Random seed for reproducibility
        
    Returns:
    --------
    samples : ndarray
        Array of samples from standard normal distribution
    """
    if seed is not None:
        np.random.seed(seed)
    
    samples = []
    iterations = 0
    
    # For standard normal, we'll use uniform proposal in range [-4, 4]
    # M is chosen such that M * proposal(x) >= target(x) for all x
    # Since max of normal PDF is ~0.4, and proposal is 0.5 in [-4,4], M=1 should work
    M = 1.0  # Safety margin to ensure coverage
    
    while len(samples) < n_samples:
        # Generate candidate from proposal distribution (uniform in [-4, 4])
        x_candidate = np.random.uniform(-4, 4)
        
        # Generate uniform random number for acceptance test
        u = np.random.uniform(0, M * proposal_distribution(x_candidate))
        
        # Acceptance test
        if u < target_distribution(x_candidate):
            samples.append(x_candidate)
        
        iterations += 1
    
    return np.array(samples), iterations

# Run the sampling
print("Running rejection sampling for standard normal distribution...")
print("=" * 60)

# Example with small sample for demonstration
samples, iterations = rejection_sampling_normal(10000, seed=42)
acceptance_rate = len(samples) / iterations * 100

print(f"Generated {len(samples)} samples from standard normal distribution")
print(f"Total iterations: {iterations}")
print(f"Acceptance rate: {acceptance_rate:.2f}%")
print(f"Sample mean: {np.mean(samples):.4f}")
print(f"Sample std: {np.std(samples):.4f}")

# Compare with exact normal distribution statistics
print(f"\nTheoretical mean: 0.0000")
print(f"Theoretical std: 1.0000")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
M = 1.0
# Histogram of samples with normal curve
x = np.linspace(-4, 4, 10000)
axes[0, 0].hist(samples, bins=50, density=True, alpha=0.7, color='steelblue', edgecolor='black')
axes[0, 0].plot(x, norm.pdf(x), 'r-', lw=2, label='Normal(0,1) PDF')
axes[0, 0].set_xlabel('Value')
axes[0, 0].set_ylabel('Density')
axes[0, 0].set_title('Rejection Sampling: Histogram vs Theoretical Distribution')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Q-Q plot
from scipy import stats
stats.probplot(samples, dist="norm", plot=axes[0, 1])
axes[0, 1].set_title('Q-Q Plot')
axes[0, 1].grid(True, alpha=0.3)

# Convergence demonstration - running mean
running_mean = []
for i in range(1, len(samples) + 1):
    running_mean.append(np.mean(samples[:i]))
axes[1, 0].plot(running_mean, 'b-', alpha=0.7)
axes[1, 0].axhline(y=0, color='r', linestyle='--', label='Theoretical mean')
axes[1, 0].set_xlabel('Number of samples')
axes[1, 0].set_ylabel('Running mean')
axes[1, 0].set_title('Convergence of Running Mean')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Rejection sampling visualization in 2D
x_range = np.linspace(-4, 4, 200)
target_vals = target_distribution(x_range)
proposal_vals = proposal_distribution(x_range)
axes[1, 1].plot(x_range, target_vals, 'r-', lw=2, label='Target (Normal PDF)')
axes[1, 1].plot(x_range, M * proposal_vals, 'b--', lw=2, label=f'M * Proposal (Uniform), M={M}')
axes[1, 1].fill_between(x_range, 0, target_vals, alpha=0.3, color='red')
axes[1, 1].set_xlabel('x')
axes[1, 1].set_ylabel('Density')
axes[1, 1].set_title('Rejection Sampling Visualization')
axes[1, 1].legend()
axes[1, 1].set_ylim(0, 0.6)
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('rejection_sampling_normal.png', dpi=300, bbox_inches='tight')
plt.show()

# Demonstrate with different sample sizes to show efficiency
print("\n" + "=" * 60)
print("Efficiency comparison with different sample sizes:")
print("=" * 60)

for n in [100, 500, 1000]:
    samples_test, iters_test = rejection_sampling_normal(n, seed=42)
    rate = n / iters_test * 100
    print(f"Sample size: {n:4d} | Iterations: {iters_test:4d} | Acceptance rate: {rate:5.2f}%")

# Show example of generated samples
print("\n" + "=" * 60)
print("First 10 generated samples:")
print("=" * 60)
print(samples[:10])