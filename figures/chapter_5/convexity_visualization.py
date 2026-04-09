"""
3D Visualization of a Convex Function with Convex Constraint Set
=================================================================

This script visualizes:
1. A convex function (quadratic bowl) in 3D
2. A convex constraint set (circular or polygonal region)
3. The intersection showing feasible region
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


def convex_function(x, y):
    """
    A simple convex function: f(x,y) = x^2 + y^2
    This is a quadratic bowl - strictly convex
    """
    return x**2 + y**2


def constraint_circle(x, y, radius=1.5):
    """
    Circular constraint: x^2 + y^2 <= radius^2
    Returns True if point is inside the constraint
    """
    return x**2 + y**2 <= radius**2


def constraint_rectangle(x, y, x_range=(-1, 1), y_range=(-1.2, 1.2)):
    """
    Rectangular constraint: x_range[0] <= x <= x_range[1], same for y
    Returns True if point is inside the rectangle
    """
    return (x_range[0] <= x) & (x <= x_range[1]) & (y_range[0] <= y) & (y <= y_range[1])


def create_3d_convex_visualization():
    """
    Create a 3D visualization with both the convex function and constraint set
    """
    # Create grid
    x = np.linspace(-2.5, 2.5, 100)
    y = np.linspace(-2.5, 2.5, 100)
    X, Y = np.meshgrid(x, y)
    
    # Compute function values
    Z = convex_function(X, Y)
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(16, 6))
    
    # ========== Plot 1: Convex function (full) ==========
    ax1 = fig.add_subplot(131, projection='3d')
    surf1 = ax1.plot_surface(X, Y, Z, cmap=cm.viridis, alpha=0.8, 
                              linewidth=0, antialiased=True)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('f(x,y)')
    ax1.set_title('Convex Function: f(x,y) = x² + y²')
    fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=10)
    
    # ========== Plot 2: Convex function with circular constraint ==========
    ax2 = fig.add_subplot(132, projection='3d')
    
    # Plot the surface
    surf2 = ax2.plot_surface(X, Y, Z, cmap=cm.viridis, alpha=0.4, 
                              linewidth=0, antialiased=True)
    
    # Create constraint region mask
    mask = constraint_circle(X, Y, radius=1.5)
    Z_constraint = np.ma.masked_where(~mask, Z)
    
    # Plot constrained region with different colormap
    surf_constraint = ax2.plot_surface(X, Y, Z_constraint, 
                                      cmap=cm.plasma, alpha=0.9,
                                      linewidth=0, antialiased=True)
    
    # Add contour lines for the constraint boundary
    ax2.contour(X, Y, X**2 + Y**2, levels=[1.5**2], colors='red', 
                linewidths=2, linestyles='--', offset=0)
    
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_zlabel('f(x,y)')
    ax2.set_title('With Circular Constraint (radius=1.5)')
    ax2.set_zlim(0, np.max(Z))
    
    # ========== Plot 3: Convex function with rectangular constraint ==========
    ax3 = fig.add_subplot(133, projection='3d')
    
    # Plot the surface
    surf3 = ax3.plot_surface(X, Y, Z, cmap=cm.viridis, alpha=0.4, 
                              linewidth=0, antialiased=True)
    
    # Create rectangular constraint mask
    mask_rect = constraint_rectangle(X, Y)
    Z_rect = np.ma.masked_where(~mask_rect, Z)
    
    # Plot constrained region
    surf_rect = ax3.plot_surface(X, Y, Z_rect, cmap=cm.coolwarm, alpha=0.9,
                                  linewidth=0, antialiased=True)
    
    # Add contour lines for constraint boundaries
    x_range = (-1, 1)
    y_range = (-1.2, 1.2)
    ax3.contour(X, Y, X**2 + Y**2, levels=[4, 5.76], colors='black', 
                linewidths=2, linestyles='--', offset=0)
    
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    ax3.set_zlabel('f(x,y)')
    ax3.set_title('With Rectangular Constraint')
    ax3.set_zlim(0, np.max(Z))
    
    plt.tight_layout()
    plt.savefig('/home/rubenruiz/Repos/simuloptibook.github.io/figures/chapter_5/convexity_visualization.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Visualization saved to: figures/chapter_5/convexity_visualization.png")


def create_3d_constraint_only_visualization():
    """
    Alternative view: Show only the constraint set in 3D space
    """
    x = np.linspace(-2.5, 2.5, 100)
    y = np.linspace(-2.5, 2.5, 100)
    X, Y = np.meshgrid(x, y)
    Z = convex_function(X, Y)
    
    fig = plt.figure(figsize=(12, 5))
    
    # Left: Circular constraint
    ax1 = fig.add_subplot(121, projection='3d')
    surf = ax1.plot_surface(X, Y, Z, cmap=cm.viridis, alpha=0.3, linewidth=0)
    
    # Plot constraint boundary as a curve
    theta = np.linspace(0, 2*np.pi, 100)
    r = 1.5
    x_circ = r * np.cos(theta)
    y_circ = r * np.sin(theta)
    z_circ = convex_function(x_circ, y_circ)
    ax1.plot(x_circ, y_circ, z_circ, 'r-', linewidth=3, label='Constraint boundary')
    
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('f(x,y)')
    ax1.set_title('Circular Constraint Set')
    ax1.legend()
    
    # Right: Rectangular constraint
    ax2 = fig.add_subplot(122, projection='3d')
    surf2 = ax2.plot_surface(X, Y, Z, cmap=cm.viridis, alpha=0.3, linewidth=0)
    
    # Plot rectangular constraint boundaries
    x_range = (-1, 1)
    y_range = (-1.2, 1.2)
    
    # Four edges of the rectangle
    x_edge1 = np.linspace(x_range[0], x_range[1], 50)
    y_edge1 = np.full(50, y_range[0])
    z_edge1 = convex_function(x_edge1, y_edge1)
    ax2.plot(x_edge1, y_edge1, z_edge1, 'b-', linewidth=3)
    
    y_edge2 = np.linspace(y_range[0], y_range[1], 50)
    x_edge2 = np.full(50, x_range[1])
    z_edge2 = convex_function(x_edge2, y_edge2)
    ax2.plot(x_edge2, y_edge2, z_edge2, 'b-', linewidth=3)
    
    x_edge3 = np.linspace(x_range[1], x_range[0], 50)
    y_edge3 = np.full(50, y_range[1])
    z_edge3 = convex_function(x_edge3, y_edge3)
    ax2.plot(x_edge3, y_edge3, z_edge3, 'b-', linewidth=3)
    
    y_edge4 = np.linspace(y_range[1], y_range[0], 50)
    x_edge4 = np.full(50, x_range[0])
    z_edge4 = convex_function(x_edge4, y_edge4)
    ax2.plot(x_edge4, y_edge4, z_edge4, 'b-', linewidth=3)
    
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_zlabel('f(x,y)')
    ax2.set_title('Rectangular Constraint Set')
    
    plt.tight_layout()
    plt.savefig('/home/rubenruiz/Repos/simuloptibook.github.io/figures/chapter_5/convex_constraint_boundaries.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Constraint boundaries saved to: figures/chapter_5/convex_constraint_boundaries.png")


if __name__ == "__main__":
    print("Creating convex function visualization...")
    create_3d_convex_visualization()
    print("\nCreating constraint boundary visualization...")
    create_3d_constraint_only_visualization()
    print("\nDone!")
