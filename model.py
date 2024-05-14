import numpy as np
import os
import pickle

from utils.funs import *
from utils.saving import save_data
from utils.format import *
from calcs.init_coords import initialize_coordinates
from calcs.calc_proj_coord import calc_proj_coord
from calcs.calc_z_coord import calc_z_coord
from visuals.plot import plot
from visuals.axes_options import set_axes_equal
from inputs import *

def main():
    # Initialize phi angles for each wedge, gamma, and cumulative distances
    phix, phiy, thetax, thetay, gamma, cum_dist = initialize()
    time = np.arange(0, TIMELIM + 1, TIMELIM // INC)
    print(f'time: {time}')

    # Initialize history storage
    history_phix = {}
    history_phiy = {}
    history_thetax = {}
    history_thetay = {}
    all_x_coords = {}
    all_y_coords = {}
    all_z_coords = {}
    Laser_coords = {}

    # Main calculation loop
    for idx, current_time in enumerate(time):
        print(f"\nTime step {idx+1}/{len(time)} at time {current_time:.2f} sec")

        # Update angles and vectors for each wedge as it spins and creates a new wedge angle
        update_angles_and_vectors(current_time, phix, phiy, gamma)
        ((orig_coordx, new_coordx), (orig_coordy, new_coordy), (orig_coordz), PX0, PY0, PZ_X0, PZ_Y0) = initialize_coordinates(RX, RY, thetax, thetay, phix, phiy, int_dist)

        x_coords, new_thetax = calc_proj_coord(str(idx), orig_coordx, new_coordx, phix, cum_dist, thetax, PX0, PZ_X0, 'x')
        y_coords, new_thetay = calc_proj_coord(str(idx), orig_coordy, new_coordy, phiy, cum_dist, thetay, PY0, PZ_Y0, 'y')
        z_coords = calc_z_coord(str(idx), orig_coordz, phix, phiy, gamma, cum_dist, x_coords, y_coords)

        # Collect coordinate data at each time step and store the updated angles and phis into their history
        all_x_coords[str(idx)] = x_coords[str(idx)]
        all_y_coords[str(idx)] = y_coords[str(idx)]
        all_z_coords[str(idx)] = z_coords[str(idx)]
        Laser_coords[idx] = [[x[0], y[1], z[2]] for x, y, z in zip(all_x_coords[str(idx)], all_y_coords[str(idx)], all_z_coords[str(idx)])]

        history_thetax[str(idx)] = new_thetax.tolist()
        history_thetay[str(idx)] = new_thetay.tolist()
        history_phix[str(idx)] = phix.copy()
        history_phiy[str(idx)] = phiy.copy()

    # Round n' print
    print(f'\n------------')
    print(f'    Laser coords:\n{format_dict(round_dict_values(Laser_coords, 2))}')
    print(f'    All Thetax:\n{format_dict(round_dict_values(history_thetax, 2))}')
    print(f'    All Thetay:\n{format_dict(round_dict_values(history_thetay, 2))}')
    print(f'    All Phix:\n{format_dict(round_dict_values(history_phix, 2))}')
    print(f'    All Phiy:\n{format_dict(round_dict_values(history_phiy, 2))}')
    print(f'\n------------')

    # Save all the data
    save_data(history_phix, history_phiy, history_thetax, history_thetay, Laser_coords)

    # Plot the results
    #plot(Laser_coords, history_phix, history_phiy, history_thetax, history_thetay, int_dist)

def initialize():
    phix = STARTPHIX + [0.0]  # Adding 0.0 for workpiece
    phiy = STARTPHIY + [0.0]
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
    n1 = [cosd(gamma + phiy) * tand(phix), sind(gamma + phiy) * tand(phix), -1]
    nx = [1, 0, 0]
    ny = [0, 1, 0]
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
