import numpy as np
from scipy.optimize import linprog

# Coefficients for the objective function (to be maximized)
c = [-3, -2]  # Negated for maximization
# Coefficients for the inequality constraints
A = [[1, 1],
     [2, 1]]
# Right-hand side of the inequality constraints
b = [4, 6]
# Solve the linear program using the Simplex method
res = linprog(c, A_ub=A, b_ub=b, method='highs')
# Print the results
print("Optimal value (max revenue):", -res.fun)  # Negate to get the original maximized value
print("Optimal solution (Product A, Product B):", res.x)
