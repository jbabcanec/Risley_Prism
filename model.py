import numpy as np
import os
import pickle
from inputs import *
from utils.funs import *


def main():
    # Initialize phi angles for each wedge
    phix, phiy, gamma, cum_dist = initialize()
    history_phix = []  # List to store history of phix
    history_phiy = []  # List to store history of phiy

    # Main simulation loop
    for idx, current_time in enumerate(time):
        print(f"\nTime step {idx+1}/{len(time)} at time {current_time:.2f} sec")
        
        # Update angles and vectors for each wedge
        update_angles_and_vectors(current_time, phix, phiy, gamma)

        # Store historical values
        history_phix.append(phix.copy())
        history_phiy.append(phiy.copy())

        # Calculate cumulative distances
        update_cumulative_distances(phix, phiy, gamma, cum_dist)

    # Perform X, Y, and Z calculations
    x_coords = calc_x(phix, gamma)
    y_coords = calc_y(phiy, gamma)
    z_coords = calc_z(phix, phiy, gamma)

    # Save all the data
    save_data(history_phix, history_phiy, x_coords, y_coords, z_coords)

    # Plot the results
    plot(x_coords, y_coords, z_coords, history_phix, history_phiy)

def initialize():
    phix = np.array(STARTPHIX, dtype=float)
    phiy = np.array(STARTPHIY, dtype=float)
    gamma = np.zeros(WEDGENUM)
    cum_dist = np.zeros(WEDGENUM + 1)
    print("Initial conditions:", dict(phix=phix, phiy=phiy))
    return phix, phiy, gamma, cum_dist

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

def update_cumulative_distances(phix, phiy, gamma, cum_dist):
    sumk = 0
    for i in range(WEDGENUM):
        sumk += int_dist[i]
        cum_dist[i] = sumk
    cum_dist[WEDGENUM] = sumk + int_dist[-1]

def calc_x(phix, gamma):
    # Placeholder function for X calculations
    pass

def calc_y(phiy, gamma):
    # Placeholder function for Y calculations
    pass

def calc_z(phix, phiy, gamma):
    # Placeholder function for Z calculations
    pass

def plot(x_coords, y_coords, z_coords, history_phix, history_phiy):
    # Placeholder function for plotting
    pass

def save_data(history_phix, history_phiy, x_coords, y_coords, z_coords):
    data_directory = "data"
    os.makedirs(data_directory, exist_ok=True)  # Ensure the directory exists
    filepath = os.path.join(data_directory, "simulation_data.pkl")

    # Collect all data into a dictionary
    data = {
        "history_phix": history_phix,
        "history_phiy": history_phiy,
        "x_coords": x_coords,
        "y_coords": y_coords,
        "z_coords": z_coords
    }

    # Write the data to a file using pickle
    with open(filepath, 'wb') as file:
        pickle.dump(data, file)

    print(f"Data saved to {filepath}")


if __name__ == "__main__":
    main()
