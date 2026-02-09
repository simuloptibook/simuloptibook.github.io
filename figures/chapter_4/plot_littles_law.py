import numpy as np
import matplotlib.pyplot as plt

def generate_mm1_data(lam, mu, num_customers):
    """
    Simulates an M/M/1 queue to generate arrival and departure timestamps.
    """
    # 1. Generate Arrival Times (Poisson Process)
    inter_arrival_times = np.random.exponential(scale=1/lam, size=num_customers)
    arrival_times = np.cumsum(inter_arrival_times)
    
    # 2. Generate Departure Times
    service_times = np.random.exponential(scale=1/mu, size=num_customers)
    departure_times = []
    
    current_time = 0
    for i in range(num_customers):
        # Service starts when customer arrives OR server frees up
        start_service = max(arrival_times[i], current_time)
        end_service = start_service + service_times[i]
        departure_times.append(end_service)
        current_time = end_service
        
    return arrival_times, np.array(departure_times)

# --- Configuration ---
np.random.seed(42) 
LAMBDA = 1.0       
MU = 1.5           
CUSTOMERS = 10     

# --- Generate Data ---
arrivals, departures = generate_mm1_data(LAMBDA, MU, CUSTOMERS)

# Prepare staircase plotting data
plot_arrivals_t = np.insert(arrivals, 0, 0.0)
plot_arrivals_y = np.arange(len(plot_arrivals_t))
plot_departures_t = np.insert(departures, 0, 0.0)
plot_departures_y = np.arange(len(plot_departures_t))

# --- Plotting ---
fig, ax = plt.subplots(figsize=(10, 6))

# Plot A(t) and D(t)
ax.step(plot_arrivals_t, plot_arrivals_y, where='post', label='$A(t)$ Arrivals', color='royalblue', lw=2)
ax.step(plot_departures_t, plot_departures_y, where='post', label='$D(t)$ Departures', color='crimson', lw=2, linestyle='--')

# Fill the area
all_times = np.sort(np.concatenate([plot_arrivals_t, plot_departures_t]))
indices_A = np.searchsorted(plot_arrivals_t, all_times, side='right') - 1
indices_D = np.searchsorted(plot_departures_t, all_times, side='right') - 1
ax.fill_between(all_times, plot_arrivals_y[indices_A], plot_departures_y[indices_D], 
                step='post', color='gray', alpha=0.15, label='Total Time (Area)')

# Annotate W (Horizontal)
idx = 5
ax.hlines(y=idx, xmin=arrivals[idx-1], xmax=departures[idx-1], colors='green', lw=2.5)
ax.text((arrivals[idx-1]+departures[idx-1])/2.04, idx+0.2, f'$W_{idx}$', color='green', ha='center', weight='bold')

# Annotate L (Vertical)
sample_t = (arrivals[5] + departures[4]) / 2
val_A = plot_arrivals_y[np.searchsorted(plot_arrivals_t, sample_t, side='right') - 1]
val_D = plot_departures_y[np.searchsorted(plot_departures_t, sample_t, side='right') - 1]
ax.vlines(x=sample_t, ymin=val_D, ymax=val_A, colors='purple', lw=2.5)
ax.text(sample_t + 0.2, (val_A + val_D)/2, f'$L(t)$', color='purple', weight='bold')

# Final touches
# ax.set_title('Geometric Visualization of Little\'s Law', fontsize=14)
ax.set_xlabel('Time ($t$)')
ax.set_ylabel('Cumulative Entities')
ax.legend(loc='upper left')
ax.grid(True, linestyle=':', alpha=0.5)

# --- EXPORT TO PNG ---
plt.tight_layout()
plt.savefig('littles_law_illustration.png', dpi=300, bbox_inches='tight')
print("Plot successfully saved as 'littles_law_illustration.png'")

# Optional: Still show it in the window if running locally
plt.show()