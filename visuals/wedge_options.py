import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def create_wedge(dx, dy, y_center, phi_x, phi_y):
    # Define corners of the wedge on the XZ plane at origin
    corners = np.array([
        [-dx/2, 0, -dy/2],
        [dx/2, 0, -dy/2],
        [dx/2, 0, dy/2],
        [-dx/2, 0, dy/2]
    ])

    # Rotation around the original X-axis (now Z-axis in the new system)
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(np.radians(phi_x)), -np.sin(np.radians(phi_x))],
        [0, np.sin(np.radians(phi_x)), np.cos(np.radians(phi_x))]
    ])

    # Rotation around the original Z-axis (now X-axis in the new system)
    Rz = np.array([
        [np.cos(np.radians(phi_y)), -np.sin(np.radians(phi_y)), 0],
        [np.sin(np.radians(phi_y)), np.cos(np.radians(phi_y)), 0],
        [0, 0, 1]
    ])

    # Apply rotations
    corners = np.dot(corners, Rx)
    corners = np.dot(corners, Rz)

    # Shift along the Y axis
    corners[:, 1] += y_center

    return corners