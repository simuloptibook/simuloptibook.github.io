import numpy as np
from scipy.stats import chisquare
from collections import defaultdict


class LCG:
    """
    X(n+1) = (a * X(n) + c) mod m
    """
    def __init__(self, seed, a, c, m):
        self._state = seed
        self.a = a
        self.c = c
        self.m = m
        self.seed = seed

    def next_int(self):
        """Generates the next pseudo-random integer 
        in the sequence."""
        self._state = (self.a * self._state + self.c) % self.m
        return self._state

    def generate(self, size):
        """Generates a sequence of integers and 
        normalizes them to [0, 1)."""
        sequence_int = []
        sequence_float = []
        # Reset state to seed for sequence generation
        self._state = self.seed 
        
        for _ in range(size):
            next_val = self.next_int()
            sequence_int.append(next_val)
            # Normalize to a float in [0, 1) by dividing by the modulus
            sequence_float.append(next_val / self.m)
            
        return np.array(sequence_int), np.array(sequence_float)
    
SEED = 42
A = 65  # Multiplier
C = 1   # Increment
M = 2**10  # Modulus (1024) - A small M is used for demonstration purposes
SEQUENCE_SIZE = 100000

# 1. Initialize and Generate Sequence
prng = LCG(SEED, A, C, M)
int_sequence, float_sequence = prng.generate(SEQUENCE_SIZE)
int_sequence[:10]

def calculate_period(lcg_generator):
    """
    Calculates the period (cycle length) of the LCG.
    The period is the number of values generated before the sequence repeats.
    """
    initial_state = lcg_generator.seed
    current_state = initial_state
    
    # Check for the next state immediately after the seed to start the loop
    current_state = (lcg_generator.a * current_state + lcg_generator.c) % lcg_generator.m
    period = 1
    
    # Loop until the state returns to the initial seed
    while current_state != initial_state:
        current_state = (lcg_generator.a * current_state + lcg_generator.c) % lcg_generator.m
        period += 1
        
        # Safety break for potentially infinite loops in case of a non-standard LCG
        if period > lcg_generator.m:
           return f"Period is greater than modulus m ({lcg_generator.m}). Check parameters."
            
    return period

period = calculate_period(prng)
print(period)

def chi_squared_uniformity_test(data_float, num_bins=10):
    """
    Statistical Test: Chi-Squared Goodness-of-Fit Test for Uniformity.
    """
    N = len(data_float)
    
    # 1. Bin the data to get observed frequencies
    # The bins are equal-sized intervals in [0, 1).
    observed_frequencies, _ = np.histogram(data_float, bins=num_bins, range=(0, 1))
    
    # 2. Calculate expected frequencies for a perfectly uniform distribution
    expected_frequency = N / num_bins
    expected_frequencies = np.full(num_bins, expected_frequency)
    
    # 3. Perform the Chi-Squared test
    # The 'chisquare' function compares observed and expected frequencies.
    # A small p-value (e.g., < 0.05) leads to rejection of H0, meaning non-uniformity.
    chi2_stat, p_value = chisquare(f_obs=observed_frequencies, f_exp=expected_frequencies)
    
    return chi2_stat, p_value, num_bins

chi2_stat, p_value_uniformity, num_bins = chi_squared_uniformity_test(float_sequence)
print(f'Chi2 statistic: {chi2_stat}, p-value: {p_value_uniformity}, number of bins: {num_bins}')


def serial_correlation_check(data_float):
    """
    Characterization: Autocorrelation (Serial Correlation) Check.
    """
    # X_n: all values except the last one
    X_n = data_float[:-1]
    # X_{n+1}: all values except the first one
    X_n_plus_1 = data_float[1:]
    
    # Calculate the Pearson correlation coefficient (r)
    # The result is an array, we take the correlation between the two sequences (index 0, 1)
    correlation_matrix = np.corrcoef(X_n, X_n_plus_1)
    lag_1_correlation = correlation_matrix[0, 1]
    
    return lag_1_correlation

lag_1_correlation = serial_correlation_check(float_sequence)
print(f'The lag 1 correlation coefficient is {lag_1_correlation}')

import random

random.seed(SEED)

# Generate a sequence of random floats in the range [0.0, 1.0)
float_sequence_mt = np.array([random.uniform(0, 1) for _ in range(SEQUENCE_SIZE)])

# Serial Correlation Check
lag_1_correlation_mt = serial_correlation_check(float_sequence_mt)
print(f'The lag 1 correlation coefficient is {lag_1_correlation_mt}')

def simulate_bernoulli_process(num_trials, success_probability):
    """
    Simulates a Bernoulli process for a given number of trials and success probability.
    """
    if not 0 <= success_probability <= 1:
        raise ValueError("Success probability must be between 0 and 1.")

    bernoulli_outcomes = [
        1 if random.random() < success_probability else 0
        for _ in range(num_trials)
    ]
    
    return bernoulli_outcomes

import matplotlib.pyplot as plt
import numpy as np
P_SUCCESS = 0.3  # Probability of success (p)
NUM_TRIALS = 50  # Total number of trials
random.seed(42)  # Set seed for reproducibility
outcomes = simulate_bernoulli_process(NUM_TRIALS, P_SUCCESS)
counting_process = np.cumsum(outcomes)

trials = np.arange(1, NUM_TRIALS + 1)

# Subplot 1: The Bernoulli Process (Outcomes)
plt.subplot(2, 1, 1)
plt.step(trials, outcomes, where='post', color='blue', linewidth=2)
plt.yticks([0, 1], ['Failure (0)', 'Success (1)'])
plt.title('Bernoulli Process (Sequence of Trials)')
plt.xlabel('Trial Number (i)')
plt.ylabel('Outcome ($X_i$)')
plt.grid(axis='y', linestyle='--')
plt.ylim(-0.1, 1.1)

# Subplot 2: The Bernoulli Counting Process (Cumulative Sum)
plt.subplot(2, 1, 2)
plt.step(trials, counting_process, where='post', color='red', linewidth=2)
plt.title('Bernoulli Counting Process (Cumulative Successes)')
plt.xlabel('Trial Number (t)')
plt.ylabel('Count ($N(t)$)')
plt.axhline(NUM_TRIALS * P_SUCCESS, color='green', linestyle=':', label='Expected Count')
plt.legend()
plt.grid(True, linestyle='--')

plt.tight_layout()
plt.show()

def simulate_poisson_process(rate_lambda, max_time):
    if rate_lambda <= 0 or max_time <= 0:
        raise ValueError("Rate and max_time must be positive.")

    # Set up the event simulation
    current_time = 0.0
    event_times = [0.0]  # Start at time 0 with the first "event"

    while current_time < max_time:
        # 1. Generate the next inter-arrival time
        time_until_next_event = random.expovariate(rate_lambda)
        
        # 2. Update the cumulative time
        current_time += time_until_next_event
        
        # 3. Record the event time if it's within the simulation duration
        if current_time < max_time:
            event_times.append(current_time)

    # Calculate the event count at each recorded time
    num_events = len(event_times) - 1 # N(0)=0 is the first entry
    num_events_at_t = list(range(num_events + 1))
    
    return event_times, num_events_at_t

RATE_LAMBDA = 2.0  # Average rate of 2 events per unit of time
MAX_TIME = 10.0    # Simulate over 10 time units

random.seed(42) # Set seed for reproducibility
event_times, cumulative_counts = simulate_poisson_process(RATE_LAMBDA, MAX_TIME)
event_times[:5]

# Plot the step function of the counting process N(t)
# We use drawstyle='steps-post' to create the classic step-function look
plt.plot(event_times, cumulative_counts, drawstyle='steps-post', color='darkorange', linewidth=2, marker='o', markersize=4)

# Plot the expected count line
plt.axhline(RATE_LAMBDA * MAX_TIME, color='gray', linestyle='--', alpha=0.6, label='Expected Final Count')

plt.title(f'Poisson Counting Process $N(t)$ with $lambda = {RATE_LAMBDA}$')
plt.xlabel('Time (t)')
plt.ylabel('Number of Events ($N(t)$)')
plt.xlim(0, MAX_TIME)
plt.ylim(bottom=0)
plt.grid(True, linestyle='--')
plt.legend()
plt.show()

import numpy as np
import matplotlib.pyplot as plt
import random

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
plt.show()

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
