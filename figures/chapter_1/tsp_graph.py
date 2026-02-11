import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

# Create a graph with 6 cities (nodes)
G = nx.Graph()

# Define city positions (nodes)
cities = {
    0: (1, 1),   # City A
    1: (4, 1),   # City B
    2: (6, 3),   # City C
    3: (4, 5),   # City D
    4: (2, 5),   # City E
    5: (0, 3)    # City F
}

# Add nodes with positions
G.add_nodes_from(cities.keys())
pos = cities

# Add edges with distances (weighted graph)
edges_with_distances = [
    (0, 1, 3.0),  # A-B
    (0, 5, 3.16), # A-F
    (1, 2, 2.24), # B-C
    (2, 3, 2.24), # C-D
    (3, 4, 2.83), # D-E
    (4, 5, 2.83), # E-F
    (0, 4, 4.12), # A-E
    (1, 3, 4.12), # B-D
    (2, 5, 4.24), # C-F
    (0, 2, 4.47), # A-C
    (1, 4, 4.12), # B-E
    (3, 5, 4.47)  # D-F
]

G.add_weighted_edges_from(edges_with_distances)

# Define a valid tour (example: 0 -> 1 -> 2 -> 3 -> 4 -> 5 -> 0)
tour = [0, 1, 2, 3, 4, 5, 0]
tour_edges = list(zip(tour[:-1], tour[1:]))

# Create figure
plt.figure(figsize=(10, 8))

# Draw all edges
edge_colors = ['lightgray' if (u, v) not in tour_edges and (v, u) not in tour_edges else 'red' 
               for u, v in G.edges()]

# Draw the graph
nx.draw(G, pos, with_labels=True, node_size=500, node_color='lightblue', 
        edge_color=edge_colors, width=[2 if (u, v) in tour_edges or (v, u) in tour_edges else 1 
        for u, v in G.edges()], font_size=12, font_weight='bold')

# Add edge labels with distances
edge_labels = nx.get_edge_attributes(G, 'weight')
for (u, v), distance in edge_labels.items():
    # Format distance to 2 decimal places
    formatted_distance = f"{distance:.2f}"
    nx.draw_networkx_edge_labels(G, pos, edge_labels={(u, v): formatted_distance}, 
                                 font_size=10, label_pos=0.5)

# Highlight tour edges by drawing them again on top
# for i in range(len(tour_edges)):
#     u, v = tour_edges[i]
#     nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], edge_color='red', 
#                           width=3, arrows=True) # arrowstyle='-|>', arrowsize=15)

# Add title
plt.title('Traveling Salesman Problem - Valid Tour Highlighted in Red', fontsize=14)

# Save as PNG
plt.savefig('tsp_graph.png', dpi=300, bbox_inches='tight')

print("Graph saved as tsp_graph.png")

# Show the plot (optional)
plt.show()
