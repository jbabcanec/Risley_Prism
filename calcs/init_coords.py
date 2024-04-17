import numpy as np
from inputs import *
from utils.funs import *

def initialize_coordinates(rx, ry, thetax, thetay, phix, phiy, int_dist):
    # Initialize original coordinates
    orig_coordx = np.array([rx, 0, 0])  # Initial x coordinates
    orig_coordy = np.array([0, ry, 0])  # Initial y coordinates
    orig_coordz = np.array([0, 0, 0])   # Initial z coordinates

    # X calculations
    x1, x2 = rx, rx + np.tan(np.radians(thetax))
    x3, x4 = 0, 1 if phix[0] == 0 else 1 / cotd(phix[0])
    z1, z2 = 0, 1
    z3, z4 = int_dist[0], int_dist[0] + (1 if phix[0] != 0 else 0)

    Px = ((x1 * z2 - z1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * z4 - z3 * x4)) / ((x1 - x2) * (z3 - z4) - (z1 - z2) * (x3 - x4))
    Pz = ((x1 * z2 - z1 * x2) * (z3 - z4) - (z1 - z2) * (x3 * z4 - z3 * x4)) / ((x1 - x2) * (z3 - z4) - (z1 - z2) * (x3 - x4))
    new_coordx = np.array([Px, 0, Pz])  # Calculated x position

    # Y calculations
    y1, y2 = ry, ry + np.tan(np.radians(thetay))
    y3, y4 = 0, 1 if phiy[0] == 0 else 1 / cotd(phiy[0])
    z1, z2 = 0, 1
    z3, z4 = int_dist[0], int_dist[0] + (1 if phiy[0] != 0 else 0)

    Py = ((y1 * z2 - z1 * y2) * (y3 - y4) - (y1 - y2) * (y3 * z4 - z3 * y4)) / ((y1 - y2) * (z3 - z4) - (z1 - z2) * (y3 - y4))
    new_coordy = np.array([0, Py, Pz])  # Calculated y position

    # The z coordinates follow the calculated positions from the x coordinates
    new_coordz = np.array([0, 0, Pz])  # Calculated z position based on x calculations

    return (orig_coordx, new_coordx), (orig_coordy, new_coordy), (orig_coordz, new_coordz), Px, Py, Pz
