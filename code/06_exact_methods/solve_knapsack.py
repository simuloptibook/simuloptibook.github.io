import pulp
from pulp import PULP_CBC_CMD

# 1. Initialize the Model (Maximization)
model = pulp.LpProblem("The_Knapsack_Problem", pulp.LpMaximize)

# 2. Define Decision Variables
# cat='Binary' tells the solver these are 0 or 1 integers.
# This triggers the Branch and Bound algorithm under the hood.
items = ['A', 'B', 'C', 'D']
weights = {'A': 4, 'B': 3, 'C': 2, 'D': 1}
values  = {'A': 15, 'B': 10, 'C': 9, 'D': 5}

x = pulp.LpVariable.dicts("Select", items, cat='Binary')

# 3. Define Objective Function
model += pulp.lpSum([values[i] * x[i] for i in items]), "Total Value"

# 4. Define Constraint (Capacity <= 10)
model += pulp.lpSum([weights[i] * x[i] for i in items]) <= 10, "Capacity"

# 5. Solve
model.solve(PULP_CBC_CMD(msg=False))

# 6. Output Results
print(f"Status: {pulp.LpStatus[model.status]}")
print(f"Max Value: ${pulp.value(model.objective)}")
print("Selected Items:")
for i in items:
    if pulp.value(x[i]) == 1:
        print(f" - Item {i} (Value: {values[i]}, Weight: {weights[i]})")
