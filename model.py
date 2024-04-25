import numpy as np
import os
import pickle

from utils.funs import *
from utils.saving import save_data
from calcs.init_coords import initialize_coordinates
from calcs.calc_proj_coord import calc_proj_coord
from calcs.calc_z_coord import calc_z_coord
from visuals.plot import plot
from inputs import *


def main():
    # Initialize phi angles for each wedge, gamma, and cumulative distances
    phix, phiy, thetax, thetay, gamma, cum_dist = initialize()
    time = np.linspace(0, TIMELIM, INC)

    # Get initial coordinates and Px, Py, Pz values
    ((orig_coordx, new_coordx), (orig_coordy, new_coordy), (orig_coordz), PX0, PY0, PZ_X0, PZ_Y0) = initialize_coordinates(RX, RY, thetax, thetay, phix, phiy, int_dist)

    # Initialize history storage
    history_phix = {'0': phix.copy()}
    history_phiy = {'0': phiy.copy()}
    history_thetax = {'0': thetax}
    history_thetay = {'0': thetay}
    all_x_coords = {'0': [orig_coordx, new_coordx]}
    all_y_coords = {'0': [orig_coordy, new_coordy]}
    all_z_coords = {'0': [orig_coordz]}

    print(all_x_coords)
    print(all_y_coords)

    # Main simulation loop
    for idx, current_time in enumerate(time):
        print(f"\nTime step {idx+1}/{len(time)} at time {current_time:.2f} sec")

        # Update angles and vectors for each wedge as it spins and creates a new wedge angle
        update_angles_and_vectors(current_time, phix, phiy, gamma)

        x_coords, new_thetax = calc_proj_coord(str(idx), phix, cum_dist, thetax, PX0, PZ_X0, all_x_coords, 'x')
        y_coords, new_thetay = calc_proj_coord(str(idx), phiy, cum_dist, thetay, PY0, PZ_Y0, all_y_coords, 'y')
        z_coords = calc_z_coord(str(idx), phix, phiy, gamma, cum_dist, x_coords, y_coords)

        if(1):
            print(f'x_coords: {x_coords}')
            print(f'y_coords: {y_coords}')
            print(f'z_coords: {z_coords}')
            exit()

        # Collect coordinate data at each time step and store the updated angles and phis into their history
        all_x_coords[str(idx)] = [x_coords] + all_x_coords['0']
        all_y_coords[str(idx)] = [y_coords] + all_y_coords['0']
        all_z_coords[str(idx)] = [z_coords] + all_z_coords['0']
        history_thetax[str(idx)] = thetax
        history_thetay[str(idx)] = thetay
        history_phix[str(idx)] = phix.copy()
        history_phiy[str(idx)] = phiy.copy()

    # Save all the data
    save_data(history_phix, history_phiy, history_thetax, history_thetay, all_x_coords, all_y_coords, all_z_coords)

    # Plot the results
    plot(all_x_coords, all_y_coords, all_z_coords, history_phix, history_phiy, history_thetax, history_thetay)

def initialize():
    phix = np.array(STARTPHIX + [0.0], dtype=float)  # Adding 0.0 for workpiece
    phiy = np.array(STARTPHIY + [0.0], dtype=float)
    thetax = float(STARTTHETAX)
    thetay = float(STARTTHETAY)
    gamma = np.zeros(WEDGENUM + 1) # Adding final 0 gamma for workpiece
    cum_dist = np.cumsum(int_dist) # K is the cumulative distance in MATLAB
    print("Initial conditions:", dict(phix=phix, phiy=phiy, thetax=thetax, thetay=thetay))
    return phix, phiy, thetax, thetay, gamma, cum_dist

def update_angles_and_vectors(current_time, phix, phiy, gamma):
    # Reinitialize phix and phiy to START values at each time step
    phix[:] = STARTPHIX + [0.0]
    phiy[:] = STARTPHIY + [0.0]
    for i in range(WEDGENUM):
        update_individual_wedge(i, current_time, phix, phiy, gamma)

def update_individual_wedge(i, current_time, phix, phiy, gamma):
    gamma[i] = (360 * N[i] * current_time) % 360
    n1, nx, ny = compute_vectors(gamma[i], phix[i], phiy[i])
    cos_angle_nx, cos_angle_ny = compute_angles(n1, nx, ny)
    print_wedge_status(i, gamma, n1, cos_angle_nx, cos_angle_ny)
    phix[i], phiy[i] = update_phi(cos_angle_nx, cos_angle_ny)

def compute_vectors(gamma, phix, phiy):
    n1 = np.array([cosd(gamma + phiy) * tand(phix), sind(gamma + phiy) * tand(phix), -1])
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
