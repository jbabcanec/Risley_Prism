import numpy as np
from inputs import *
from utils.funs import *

def initialize_coordinates(rx, ry, thetax, thetay, phix, phiy, int_dist):
    # Initialize original coordinates
    orig_coordx = [rx, 0, 0]    # Initial x coordinates
    orig_coordy = [0, ry, 0]    # Initial y coordinates
    orig_coordz = [0, 0, 0]     # Initial z coordinates

    # X calculations
    x1, x2 = rx, rx + tand(thetax)
    x3, x4 = 0, 1 if phix[0] == 0 else cotd(phix[0])
    z1, z2 = 0, 1
    z3, z4 = int_dist[0], int_dist[0] + (1 if phix[0] != 0 else 0)

    Px = ((x1 * z2 - z1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * z4 - z3 * x4)) / ((x1 - x2) * (z3 - z4) - (z1 - z2) * (x3 - x4))
    Pz_x = ((x1 * z2 - z1 * x2) * (z3 - z4) - (z1 - z2) * (x3 * z4 - z3 * x4)) / ((x1 - x2) * (z3 - z4) - (z1 - z2) * (x3 - x4))
    new_coordx = [Px, 0, Pz_x]  # Calculated x position

    # Y calculations
    y1, y2 = ry, ry + tand(thetay)
    y3, y4 = 0, 1 if phiy[0] == 0 else cotd(phiy[0])
    z1, z2 = 0, 1
    z3, z4 = int_dist[0], int_dist[0] + (1 if phiy[0] != 0 else 0)

    Py = ((y1 * z2 - z1 * y2) * (y3 - y4) - (y1 - y2) * (y3 * z4 - z3 * y4)) / ((y1 - y2) * (z3 - z4) - (z1 - z2) * (y3 - y4))
    Pz_y = ((y1 * z2 - z1 * y2) * (z3 - z4) - (z1 - z2) * (y3 * z4 - z3 * y4)) / ((y1 - y2) * (z3 - z4) - (z1 - z2) * (y3 - y4))

    new_coordy = [0, Py, Pz_y]  # Calculated y position

    print('initial y posns')
    print(f'y1: {y1}, y2: {y2}, y3: {y3}, y4: {y4}')
    print(f'z1: {z1}, z2: {z2}, z3: {z3}, z4: {z4}')

    return (orig_coordx, new_coordx), (orig_coordy, new_coordy), (orig_coordz), Px, Py, Pz_x, Pz_y
