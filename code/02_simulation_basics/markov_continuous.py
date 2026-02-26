import numpy as np
import matplotlib.pyplot as plt

def simulate_ctmc(Q_matrix, start_state, max_time):
    current_time = 0.0
    current_state = start_state
    
    times = [0.0]
    states = [current_state]
    
    num_states = Q_matrix.shape[0]

    while current_time < max_time:
        exit_rate = -Q_matrix[current_state, current_state]
        
        if exit_rate <= 0:
            break
            
        time_step = np.random.exponential(scale=1/exit_rate)
        
        if current_time + time_step > max_time:
            break
            
        rates_row = Q_matrix[current_state, :].copy()
        
        rates_row[current_state] = 0
        
        jump_probabilities = rates_row / exit_rate
        
        next_state = np.random.choice(np.arange(num_states), p=jump_probabilities)
        
        current_time += time_step
        current_state = next_state
        
        times.append(current_time)
        states.append(current_state)

    times.append(max_time)
    states.append(states[-1])

    return times, states

# Define the Generator Matrix (Q)
# Rows must sum to 0.
# Units: events per hour (rates)
Q = np.array([
# To:   Op(0)   Deg(1)  Fail(2)
    [-0.6,   0.5,    0.1],  # From Operational (Exit rate = 0.6)
    [ 0.0,  -2.0,    2.0],  # From Degraded    (Exit rate = 2.0 -> Very fast exit)
    [ 1.0,   0.0,   -1.0]   # From Failed      (Exit rate = 1.0 -> Repair time)
])

STATE_NAMES = {0: 'Operational', 1: 'Degraded', 2: 'Failed'}
START_STATE = 0
DURATION = 24.0  # Simulate 24 hours

# --- Run Simulation ---
np.random.seed(101)
sim_times, sim_states = simulate_ctmc(Q, START_STATE, DURATION)

plt.step(sim_times, sim_states, where='post', color='purple', linewidth=2)

plt.yticks([0, 1, 2], [STATE_NAMES[0], STATE_NAMES[1], STATE_NAMES[2]])
plt.xlabel('Time (Hours)')
plt.ylabel('Machine State')
plt.title('CTMC Trajectory: Variable Holding Times')
plt.grid(True, axis='y', linestyle='--')
plt.xlim(0, DURATION)
plt.ylim(-0.5, 2.5)

for i in range(len(sim_times)-1):
    if sim_states[i] == 0 and (sim_times[i+1] - sim_times[i] > 2.0):
        mid_point = (sim_times[i] + sim_times[i+1]) / 2
        plt.text(mid_point, 0.1, "Long Holding Time\n(Random)", ha='center', fontsize=9, color='green')
        break

plt.show()
