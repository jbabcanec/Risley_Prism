import numpy as np
import os
import pickle

from utils.funs import *
from utils.saving import save_data
from calcs.calc_x import calc_x
from calcs.calc_y import calc_y
from calcs.calc_z import calc_z
from calcs.init_coords import initialize_coordinates
from visuals.plot import plot
from inputs import *


def main():
    # Initialize phi angles for each wedge, gamma, and cumulative distances
    phix, phiy, thetax, thetay, gamma, cum_dist = initialize()

    # Get initial coordinates and Px, Py, Pz values
    ((orig_coordx, new_coordx), (orig_coordy, new_coordy), (orig_coordz), px0, py0, pz0) = initialize_coordinates(RX, RY, thetax, thetay, phix, phiy, int_dist)

    # Initialize history storage
    history_phix = [phix]
    history_phiy = [phiy]
    history_thetax = [thetax]
    history_thetay = [thetay]
    all_x_coords = [orig_coordx, new_coordx]  # Collect both initial sets of coordinates
    all_y_coords = [orig_coordy, new_coordy]
    all_z_coords = [orig_coordz]
    Px = [px0]  # Start with the initial P's
    Py = [py0]
    Pz = [pz0]

    # Main simulation loop
    for idx, current_time in enumerate(time):
        print(f"\nTime step {idx+1}/{len(time)} at time {current_time:.2f} sec")

        if(1):
            print(f"Px: {Px}, Py: {Py}, Pz: {Pz}")
            exit()

        # Update angles and vectors for each wedge as it spins and creates a new wedge angle
        update_angles_and_vectors(current_time, phix, phiy, gamma)

        # Perform X, Y, and Z calculations within the loop to use updated phi values
        x_coords, thetax = calc_x(phix, gamma, cum_dist, thetax)
        y_coords, thetay = calc_y(phiy, gamma, cum_dist, thetay)
        z_coords = calc_z(phix, phiy, gamma, cum_dist)

        # Collect coordinate data at each time step and tore the updated angles and phis into their history
        all_x_coords.append(x_coords)
        all_y_coords.append(y_coords)
        all_z_coords.append(z_coords)
        history_thetax.append(thetax.copy())
        history_thetay.append(thetay.copy())
        history_phix.append(phix.copy())
        history_phiy.append(phiy.copy())

    # Save all the data
    save_data(history_phix, history_phiy, history_thetax, history_thetay, all_x_coords, all_y_coords, all_z_coords, Px, Py, Pz)

    # Plot the results
    plot(all_x_coords, all_y_coords, all_z_coords, history_phix, history_phiy, history_thetax, history_thetay)


def initialize():
    phix = np.array(STARTPHIX, dtype=float)
    phiy = np.array(STARTPHIY, dtype=float)
    thetax = float(STARTTHETAX)
    thetay = float(STARTTHETAY)
    gamma = np.zeros(WEDGENUM)
    cum_dist = np.cumsum([0] + int_dist)
    print("Initial conditions:", dict(phix=phix, phiy=phiy, thetax=thetax, thetay=thetay))
    return phix, phiy, thetax, thetay, gamma, cum_dist

def update_angles_and_vectors(current_time, phix, phiy, gamma):
    # Reinitialize phix and phiy to START values at each time step
    phix[:] = STARTPHIX
    phiy[:] = STARTPHIY
    for i in range(WEDGENUM):
        update_individual_wedge(i, current_time, phix, phiy, gamma)

def update_individual_wedge(i, current_time, phix, phiy, gamma):
    gamma[i] = (360 * N[i] * current_time) % 360
    n1, nx, ny = compute_vectors(gamma[i], phix[i])
    cos_angle_nx, cos_angle_ny = compute_angles(n1, nx, ny)
    print_wedge_status(i, gamma, n1, cos_angle_nx, cos_angle_ny)
    phix[i], phiy[i] = update_phi(cos_angle_nx, cos_angle_ny)

def compute_vectors(gamma, phi):
    n1 = np.array([cosd(gamma) * tand(phi), sind(gamma) * tand(phi), -1])
    nx = np.array([1, 0, 0])
    ny = np.array([0, 1, 0])
    return n1, nx, ny

def compute_angles(n1, nx, ny):
    cos_angle_nx = np.dot(n1, nx) / (np.linalg.norm(n1) * np.linalg.norm(nx))
    cos_angle_ny = np.dot(n1, ny) / (np.linalg.norm(n1) * np.linalg.norm(ny))
    return cos_angle_nx, cos_angle_ny

def print_wedge_status(i, gamma, n1, cos_angle_nx, cos_angle_ny):
    print(f"Wedge {i+1}:")
    print(f"  Gamma: {gamma[i]:.2f} degrees")
    print(f"  n1 vector: {n1}")
    print(f"  Cosine angles - nx: {cos_angle_nx:.4f}, ny: {cos_angle_ny:.4f}")

def update_phi(cos_angle_nx, cos_angle_ny):
    phix = 90 - acosd(cos_angle_nx)
    phiy = 90 - acosd(cos_angle_ny)
    return phix, phiy


if __name__ == "__main__":
    main()
