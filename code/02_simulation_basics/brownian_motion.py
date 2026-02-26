import numpy as np
import matplotlib.pyplot as plt

def simulate_brownian_motion(total_time, num_steps):
    if num_steps <= 0 or total_time <= 0:
        raise ValueError("num_steps and total_time must be positive.")

    dt = total_time / num_steps
    time_points = np.linspace(0, total_time, num_steps + 1)

    standard_deviation = np.sqrt(dt)
    
    Z_increments = np.random.normal(loc=0.0, scale=1.0, size=num_steps)
    dW_increments = standard_deviation * Z_increments

    brownian_path = np.concatenate(([0.0], np.cumsum(dW_increments)))

    return time_points, brownian_path

TOTAL_TIME = 1.0     # Simulate up to time T=1
NUM_STEPS = 1000     # Use 1000 steps for a smooth path

np.random.seed(42) # Seed for reproducibility
time, path = simulate_brownian_motion(TOTAL_TIME, NUM_STEPS)

plt.plot(time, path, label='Wiener Process Path', color='navy')

# The theoretical standard deviation at time t is sqrt(t)
std_dev = np.sqrt(time)
plt.plot(time, 2 * std_dev, 'r--', label=r'Theoretical Bounds ($\pm 2\sqrt{t}$)', alpha=0.6)
plt.plot(time, -2 * std_dev, 'r--', alpha=0.6)

plt.title('Simulated 1D Brownian Motion (Wiener Process)')
plt.xlabel('Time (t)')
plt.ylabel('State ($W(t)$)')
plt.axhline(0, color='black', linewidth=0.5)
plt.xlim(0, TOTAL_TIME)
plt.grid(True, linestyle='--')
plt.legend()
output_file = 'brownian_motion.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')

plt.show()

NUM_PATHS = 5
for i in range(NUM_PATHS):
    np.random.seed(i) # Use a different seed for each path
    _, multi_path = simulate_brownian_motion(TOTAL_TIME, NUM_STEPS)
    plt.plot(time, multi_path, alpha=0.7)

plt.plot(time, 2 * np.sqrt(time), 'k--', label='Theoretical Bounds', linewidth=1.5)
plt.plot(time, -2 * np.sqrt(time), 'k--', linewidth=1.5)
plt.title(f'Multiple Simulated Brownian Motion Paths')
plt.xlabel('Time (t)')
plt.ylabel('State ($W(t)$)')
plt.axhline(0, color='black', linewidth=0.5)
plt.xlim(0, TOTAL_TIME)
plt.grid(True, linestyle='--')
output_file = 'brownian_motion_paths.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.show()