import random
import math

def estimate_pi_monte_carlo(num_iterations):
    points_inside_circle = 0
    
    inside_x, inside_y = [], []
    outside_x, outside_y = [], []

    for _ in range(num_iterations):
        x = random.uniform(0, 1)
        y = random.uniform(0, 1)

        distance_squared = x**2 + y**2

        if distance_squared <= 1:
            points_inside_circle += 1
            inside_x.append(x)
            inside_y.append(y)
        else:
            outside_x.append(x)
            outside_y.append(y)

    pi_estimate = 4 * (points_inside_circle / num_iterations)
    return pi_estimate

N = 100000  # 100,000 darts
estimated_pi = estimate_pi_monte_carlo(N)
estimated_pi

import numpy as np
import scipy.stats as stats

def monte_carlo_with_ci(num_iterations):
    x = np.random.uniform(0, 1, num_iterations)
    y = np.random.uniform(0, 1, num_iterations)
    
    inside_circle = (x**2 + y**2) <= 1
    values = inside_circle * 4.0
    
    mean_estimate = np.mean(values)
    std_dev = np.std(values, ddof=1) # ddof=1 for sample standard deviation
    standard_error = std_dev / np.sqrt(num_iterations)
    
    # 4. Calculate 95% Confidence Interval
    # z-score for 95% is 1.96, or strictly: stats.norm.ppf(0.025)
    z_score = stats.norm.ppf(0.025) 
    margin_of_error = z_score * standard_error
    
    lower_bound = mean_estimate + margin_of_error
    upper_bound = mean_estimate - margin_of_error
    
    return mean_estimate, lower_bound, upper_bound

# Run
mu, lower, upper = monte_carlo_with_ci(10000)

print(f"Estimate: {mu:.4f}")
print(f"95% CI:   [{lower:.4f}, {upper:.4f}]")

import matplotlib.pyplot as plt

def simulation_control_variates(n_iterations):
    u = np.random.uniform(0, 1, n_iterations)
    y_samples = np.exp(u**2)
    x_samples = 1 + u**2
    expected_x = 4 / 3
    
    # Calculate Optimal 'c'
    cov_matrix = np.cov(x_samples, y_samples)
    covariance_xy = cov_matrix[0, 1]
    variance_x = cov_matrix[0, 0]
    
    c_optimal = covariance_xy / variance_x
    
    # Formula: Y_cv = Y - c * (X - E[X])
    cv_samples = y_samples - c_optimal * (x_samples - expected_x)

    naive_estimate = np.mean(y_samples)
    naive_variance = np.var(y_samples)
    
    # Control Variates Estimate
    cv_estimate = np.mean(cv_samples)
    cv_variance = np.var(cv_samples)
    
    return {
        "naive_est": naive_estimate,
        "cv_est": cv_estimate,
        "naive_var": naive_variance,
        "cv_var": cv_variance,
        "correlation": np.corrcoef(x_samples, y_samples)[0,1],
        "c_optimal": c_optimal
    }

N = 10000
results = simulation_control_variates(N)

# The "True" value (calculated via high-precision numerical integration for comparison)
# integral of e^(x^2) from 0 to 1 is approx 1.4626517459...
true_value = 1.4626517459

print(f"--- Results with N={N:,} ---")
print(f"True Value:               {true_value:.6f}")
print(f"Naive MC Estimate:        {results['naive_est']:.6f} (Error: {abs(results['naive_est'] - true_value):.6f})")
print(f"Control Variate Estimate: {results['cv_est']:.6f} (Error: {abs(results['cv_est'] - true_value):.6f})")

print(f"\n--- Variance Analysis ---")
print(f"Correlation (X, Y):       {results['correlation']:.4f}")
print(f"Naive Variance:           {results['naive_var']:.6f}")
print(f"CV Variance:              {results['cv_var']:.6f}")

reduction_factor = results['naive_var'] / results['cv_var']
print(f"Variance Reduction:       {reduction_factor:.1f}x lower variance")

def importance_sampling_demo(n_samples, threshold=5):
    samples_naive = np.random.normal(0, 1, n_samples)
    
    hits_naive = samples_naive > threshold
    prob_naive = np.mean(hits_naive)
    
    var_naive = (prob_naive * (1 - prob_naive)) / n_samples
    
    shift_mean = threshold
    samples_is = np.random.normal(loc=shift_mean, scale=1, size=n_samples)
    
    hits_is = samples_is > threshold # This will be True for roughly 50% of samples
    f_x = hits_is.astype(float)
    
    p_x = stats.norm.pdf(samples_is, loc=0, scale=1)
    q_x = stats.norm.pdf(samples_is, loc=shift_mean, scale=1)
    
    weights = p_x / q_x
    
    weighted_values = f_x * weights
    prob_is = np.mean(weighted_values)
    
    var_is = np.var(weighted_values) / n_samples
    
    return prob_naive, var_naive, prob_is, var_is

N = 10000 # Only 10,000 samples (Small for a rare event!)
target = 5

naive_p, naive_var, is_p, is_var = importance_sampling_demo(N, target)
true_p = 1 - stats.norm.cdf(target) # Analytical solution

print(f"Target: P(X > {target})")
print(f"True Probability: {true_p:.10f}")
print(f"Naive MC Estimate: {naive_p:.10f}")
print(f"Naive Variance:    {naive_var:.10f} (Likely zero if no hits occurred)")
print(f"Imp. Samp Estimate:{is_p:.10f}")
print(f"Imp. Samp Variance:{is_var:.20f}")

# Check the ratio of variance reduction (avoid div by zero)
if naive_var == 0:
    print("Naive method failed completely (0 hits). Importance Sampling is infinitely better here.")
else:
    print(f"Variance Reduction Factor: {naive_var / is_var:.1f}x")

def simulation_stratified(n_samples):
    u_naive = np.random.uniform(0, 1, n_samples)
    y_naive = np.exp(u_naive)
    
    est_naive = np.mean(y_naive)
    var_naive = np.var(y_naive)
    
    strata_starts = np.arange(n_samples) / n_samples
    offsets = np.random.uniform(0, 1/n_samples, n_samples)
    x_stratified = strata_starts + offsets
    
    y_stratified = np.exp(x_stratified)
    est_stratified = np.mean(y_stratified)
    
    return est_naive, est_stratified

experiments = 1000
N = 100 # Samples per experiment

naive_results = []
stratified_results = []

for _ in range(experiments):
    n_res, s_res = simulation_stratified(N)
    naive_results.append(n_res)
    stratified_results.append(s_res)

true_value = np.e - 1

# Calculate statistics of the *Estimators*
var_of_naive_method = np.var(naive_results)
var_of_stratified_method = np.var(stratified_results)

print(f"True Value: {true_value:.6f}")
print("-" * 30)
print(f"Naive MC Variance (across {experiments} runs):      {var_of_naive_method:.8f}")
print(f"Stratified MC Variance (across {experiments} runs): {var_of_stratified_method:.8f}")
print("-" * 30)
print(f"Variance Reduction Factor: {var_of_naive_method / var_of_stratified_method:.1f}x")

# --- Visualization (First 50 points) ---

# Plot Standard
plt.subplot(1, 2, 1)
u_naive = np.random.uniform(0, 1, 20)
plt.scatter(u_naive, np.zeros_like(u_naive), color='red', alpha=0.6, s=50)
plt.title("Standard MC (Clumping & Gaps)")
plt.yticks([])
plt.xlim(0, 1)
plt.grid(axis='x', alpha=0.3)

# Plot Stratified
plt.subplot(1, 2, 2)
strata_starts = np.arange(20) / 20
offsets = np.random.uniform(0, 1/20, 20)
x_stratified = strata_starts + offsets
plt.scatter(x_stratified, np.zeros_like(x_stratified), color='green', alpha=0.6, s=50)
# Draw grid lines to show strata
for i in range(21):
    plt.axvline(i/20, color='gray', alpha=0.2)
plt.title("Stratified Sampling (Even Spread)")
plt.yticks([])
plt.xlim(0, 1)

plt.tight_layout()
plt.show()

def latin_hypercube_sampling(n_samples, n_dim):
    samples = np.zeros((n_samples, n_dim))
    
    for d in range(n_dim):
        permutation = np.random.permutation(n_samples)
        
        jitter = np.random.uniform(0, 1, n_samples)
        
        samples[:, d] = (permutation + jitter) / n_samples
        
    return samples

N = 10 # 10 samples
D = 2  # 2 dimensions

# Generate Data
random_samples = np.random.uniform(0, 1, (N, D))
lhs_samples = latin_hypercube_sampling(N, D)

# Plot 1: Random Sampling
plt.subplot(1, 2, 1)
plt.scatter(random_samples[:, 0], random_samples[:, 1], color='red', s=100)
plt.title("Random Sampling")
plt.xlim(0, 1); plt.ylim(0, 1)
# Draw grid to show bins
for i in range(1, N):
    plt.axvline(i/N, color='gray', alpha=0.2)
    plt.axhline(i/N, color='gray', alpha=0.2)
plt.xlabel("X")
plt.ylabel("Y")

# Plot 2: Latin Hypercube Sampling
plt.subplot(1, 2, 2)
plt.scatter(lhs_samples[:, 0], lhs_samples[:, 1], color='green', s=100)
plt.title("Latin Hypercube Sampling")
plt.xlim(0, 1); plt.ylim(0, 1)
# Draw grid to show bins
for i in range(1, N):
    plt.axvline(i/N, color='gray', alpha=0.2)
    plt.axhline(i/N, color='gray', alpha=0.2)
plt.xlabel("X")
plt.ylabel("Y")

plt.tight_layout()
plt.show()

from scipy.stats import qmc

def target_function(x):
    return np.sum(x**2, axis=1)

dim = 5
sample_sizes = np.arange(10, 1000, 10) 

errors_mc = []
errors_lhs = []
true_mean = dim * (1/3)

for n in sample_sizes:
    mc_samples = np.random.uniform(low=-1, high=1, size=(n, dim))
    mc_results = target_function(mc_samples)
    mc_estimate = np.mean(mc_results)
    errors_mc.append(abs(mc_estimate - true_mean))

    sampler = qmc.LatinHypercube(d=dim)
    lhs_unit = sampler.random(n=n)

    lhs_samples = qmc.scale(lhs_unit, l_bounds=[-1]*dim, u_bounds=[1]*dim)
    
    lhs_results = target_function(lhs_samples)
    lhs_estimate = np.mean(lhs_results)
    errors_lhs.append(abs(lhs_estimate - true_mean))

plt.semilogy(sample_sizes, errors_mc, color='red', alpha=0.4, label='Standard Monte Carlo Noise')
plt.semilogy(sample_sizes, errors_lhs, color='green', alpha=0.4, label='LHS Noise')

# Add trendlines (moving average) to make it clearer
def moving_average(a, n=10) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

plt.semilogy(sample_sizes[9:], moving_average(errors_mc), color='darkred', linewidth=2, label='Standard MC Trend')
plt.semilogy(sample_sizes[9:], moving_average(errors_lhs), color='darkgreen', linewidth=2, label='LHS Trend')

plt.title(f"Convergence Comparison: LHS vs Standard MC ({dim} Dimensions)")
plt.xlabel("Number of Samples (Iterations)")
plt.ylabel("Absolute Error (Log Scale)")
plt.legend()
plt.grid(True, which="both", ls="-", alpha=0.2)
plt.show()

# Final comparison for the largest N
print(f"Final Error (N={sample_sizes[-1]}):")
print(f"Standard MC Error: {errors_mc[-1]:.6f}")
print(f"LHS Error:         {errors_lhs[-1]:.6f}")
print(f"Improvement Factor: {errors_mc[-1]/errors_lhs[-1]:.1f}x more accurate")