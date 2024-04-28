import numpy as np
from utils.funs import cosd, sind, tand
from inputs import *

def calc_z_coord(idx, phix, phiy, gamma, cum_dist, coordx, coordy):
    # Initialize z coordinates array with the same length as phi arrays
    z_coords = []

    # Calculate z coordinates based on provided formulas
    for i in range(len(phix)):  # Starting from 1 because z_coords[0] is already set
        x_component = coordx[idx][i + 1][0]  # Extract the x-component directly
        y_component = coordy[idx][i + 1][1]  # Extract the y-component directly

        # Calculate the Z coordinate for each wedge
        z = cum_dist[i] + ((x_component * cosd(gamma[i]) - y_component * sind(gamma[i])) * tand(phix[i]))
        z_coords.append([0, 0, z])

        # Before the calculation line, add these print statements:
        print("\n-------------------------")
        print(f"Iteration {i+1} on Coord z:")
        print("-------------------------")
        print(f"    Cumulative Distance (cum_dist[i - 1]): {cum_dist[i - 1]}")
        print(f"    x_component (coordx[{idx}][{i}][0]): {coordx[idx][i][0]}")  # Assuming coordx[i] is a tuple like (Px, 0, Pz)
        print(f"    y_component (coordy[{idx}][{i}][1]): {coordy[idx][i][1]}")  # Assuming coordy[i] is a tuple like (0, Py, Pz)
        print(f"    Gamma for wedge i-1 (gamma[i - 1]): {gamma[i - 1]} degrees")
        print(f"    Phi_x for wedge i-1 (phix[i - 1]): {phix[i - 1]} degrees")
        print(f"    Next Positions for z: {z_coords}")


    return z_coords
