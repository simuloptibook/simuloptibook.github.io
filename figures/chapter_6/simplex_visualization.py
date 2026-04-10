import numpy as np
from scipy.spatial import HalfspaceIntersection, ConvexHull
import plotly.graph_objects as go

# ---------------------------------------------------------
# 1. DEFINE THE LINEAR PROGRAM
# ---------------------------------------------------------
c = np.array([5, 4, 3])

halfspaces = np.array([
    [1, 2, 1, -10],
    [3, 1, 2, -15],
    [1, 1, 3, -12],
    [-1, 0, 0, 0],
    [0, -1, 0, 0],
    [0, 0, -1, 0]
])

interior_point = np.array([1.0, 1.0, 1.0])

# ---------------------------------------------------------
# 2. CALCULATE VERTICES & CONVEX HULL (The Feasible Region)
# ---------------------------------------------------------
hs = HalfspaceIntersection(halfspaces, interior_point)
vertices = hs.intersections
hull = ConvexHull(vertices)

z_values = np.dot(vertices, c)
optimal_idx = np.argmax(z_values)
optimal_vertex = vertices[optimal_idx]
max_z = z_values[optimal_idx]

# ---------------------------------------------------------
# 3. VISUALIZATION WITH PLOTLY
# ---------------------------------------------------------
fig = go.Figure()

# A. Plot the Feasible Region (Polyhedron Faces)
i = hull.simplices[:, 0]
j = hull.simplices[:, 1]
k = hull.simplices[:, 2]

fig.add_trace(go.Mesh3d(
    x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
    i=i, j=j, k=k,
    color='cyan', opacity=0.3,
    name='Feasible Region'
))

# B. Plot the Vertices (Basic Feasible Solutions)
fig.add_trace(go.Scatter3d(
    x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
    mode='markers',
    marker=dict(size=5, color='black'),
    name='Vertices (BFS)'
))

# C. Highlight the Optimal Vertex
fig.add_trace(go.Scatter3d(
    x=[optimal_vertex[0]], y=[optimal_vertex[1]], z=[optimal_vertex[2]],
    mode='markers',
    marker=dict(size=10, color='red', symbol='diamond'),
    name=f'Optimal Solution<br>Z = {round(max_z, 2)}'
))

# D. Plot the Objective Function Plane (ARTIFACT-FREE FIX)
# The plane is 5x + 4y + 3z = max_z. 
# We find where this plane intersects the X, Y, and Z axes.
x_intercept = max_z / c[0]
y_intercept = max_z / c[1]
z_intercept = max_z / c[2]

# These 3 points form a perfect triangle representing the plane in the positive octant
plane_x = [x_intercept, 0, 0]
plane_y = [0, y_intercept, 0]
plane_z = [0, 0, z_intercept]

fig.add_trace(go.Mesh3d(
    x=plane_x, y=plane_y, z=plane_z,
    i=[0], j=[1], k=[2], # Connect the 3 points into a single triangle
    color='red', opacity=0.4,
    name='Objective Plane'
))

# ---------------------------------------------------------
# 4. FORMATTING AND DISPLAY
# ---------------------------------------------------------
fig.update_layout(
    title='3D Linear Programming & Simplex Method Visualization',
    scene=dict(
        xaxis_title='X Axis',
        yaxis_title='Y Axis',
        zaxis_title='Z Axis',
        xaxis=dict(range=[0, 8]),
        yaxis=dict(range=[0, 8]),
        zaxis=dict(range=[0, 10])
    ),
    margin=dict(l=0, r=0, b=0, t=40)
)

fig.show()