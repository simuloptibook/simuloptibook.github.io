import numpy as np
import matplotlib.pyplot as plt

def simulate_dtmc(start_state, transition_matrix, num_steps):
    # 1. Initialize the chain
    num_states = transition_matrix.shape[0]
    current_state = start_state
    state_history = [current_state]

    # 2. Iterate for the specified number of steps
    for _ in range(num_steps):
        # Get the probability distribution for the next state, based on the current state (row of the matrix)
        probabilities = transition_matrix[current_state, :]
        
        # Select the next state using the probabilities
        # np.random.choice selects a state from the possible states (0 to num_states-1) 
        # based on the corresponding probabilities.
        next_state = np.random.choice(
            a=np.arange(num_states), 
            p=probabilities
        )
        
        # Update and record
        current_state = next_state
        state_history.append(current_state)
        
    return state_history


# Define the Transition Probability Matrix (P)
P = np.array([
    [0.80, 0.15, 0.05],  # From Sunny (0)
    [0.20, 0.60, 0.20],  # From Cloudy (1)
    [0.10, 0.40, 0.50]   # From Rainy (2)
])

# Define the State Space for interpretation
STATE_MAP = {0: 'Sunny', 1: 'Cloudy', 2: 'Rainy'}

START_STATE_INDEX = 0  # Start on a Sunny day
NUM_SIMULATION_STEPS = 50 # Simulate 50 days

# 3. Run the Simulation
np.random.seed(42) # Seed for reproducibility
simulated_states = simulate_dtmc(START_STATE_INDEX, P, NUM_SIMULATION_STEPS)
simulated_weather = [STATE_MAP[s] for s in simulated_states]
simulated_weather[:10]

time_steps = np.arange(NUM_SIMULATION_STEPS + 1)

plt.plot(time_steps, simulated_states, marker='o', linestyle='-', drawstyle='steps-post', markersize=4)
plt.yticks([0, 1, 2], [STATE_MAP[0], STATE_MAP[1], STATE_MAP[2]])
plt.title('Discrete-Time Markov Chain State Trajectory (Weather)')
plt.xlabel('Time Step (Day)')
plt.ylabel('State')
plt.grid(True, axis='y', linestyle='--')

# Save as PNG file with high resolution
output_file = 'markov_discrete.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')

plt.show()
