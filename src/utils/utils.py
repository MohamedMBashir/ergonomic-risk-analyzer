import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def visualize_3d_keypoints(points, point_labels=None, figsize=(10, 10)):
    """
    Visualize 3D keypoints and their connections.
    
    Args:
        points: List of numpy arrays containing 3D coordinates [(x1,y1,z1), (x2,y2,z2), ...]
        point_labels: List of labels for the points (optional)
        figsize: Tuple for figure size (width, height)
    """
    # Create 3D plot
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # Extract coordinates
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    zs = [p[2] for p in points]
    
    # Plot points
    ax.scatter(xs, ys, zs, c='blue', marker='o', s=100)
    
    # Plot connections between consecutive points
    for i in range(len(points)-1):
        x_line = [points[i][0], points[i+1][0]]
        y_line = [points[i][1], points[i+1][1]]
        z_line = [points[i][2], points[i+1][2]]
        ax.plot(x_line, y_line, z_line, 'r-', linewidth=2)
    
    # Add labels if provided
    if point_labels:
        for i, label in enumerate(point_labels):
            ax.text(xs[i], ys[i], zs[i], label)
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Keypoints Visualization')
    
    # Set equal aspect ratio
    ax.set_box_aspect([1,1,1])
    
    plt.show()