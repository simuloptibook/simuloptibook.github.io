import tsplib95
import numpy as np


def read_tsp_distance_matrix(file_path: str) -> np.ndarray:
    """
    Reads a TSPLIB file and returns a distance matrix (NumPy array).
    Automatically handles EDGE_WEIGHT_TYPE (GEO, EUC_2D, ATT, etc.).
    
    Args:
        file_path: Path to .tsp file
        
    Returns:
        np.ndarray of shape (n, n) with integer distances
    """
    # Load the problem
    problem = tsplib95.load(file_path)
    
    # Generate distance matrix
    # `get_distance_matrix()` returns a dict or matrix depending on weight type
    # But to be safe, we use the direct distance function
    n = problem.dimension
    matrix = np.zeros((n, n), dtype=int)
    
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            if i != j:
                matrix[i-1, j-1] = problem.get_weight(i, j)
            # diagonal remains 0
    
    return matrix


if __name__ == "__main__":
    dist_mat = read_tsp_distance_matrix("ulysses16.tsp")
    print(dist_mat)
    