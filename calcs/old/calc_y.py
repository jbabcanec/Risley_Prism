import numpy as np
from utils.funs import *
from inputs import ref_ind

def calc_y(phiy, gamma, cum_dist, thetay, py, pz):
    y_coords = {}
    # Start with the initial theta value for the first wedge, then calculate subsequent values.
    new_thetay = np.zeros(len(phiy))  # This ensures we have space for each wedge, including workpiece
    new_thetay[0] = thetay  # Initial thetay used for the first calculation

    # Initialize py and pz for subsequent wedges based on initial values
    py_next = py
    pz_next = pz

    # Compute new angles and positions for each wedge
    for i in range(len(phiy) - 1):  # Exclude the last element which is the workpiece
        # Normalizing the vectors
        N_hat = np.array([tand(phiy[i]), 0, -1])
        N_hat = N_hat / np.linalg.norm(N_hat)
        s_i = np.array([tand(new_thetay[i]), 0, 1])  # Use updated theta for each wedge
        s_i = s_i / np.linalg.norm(s_i)
        z_hat = np.array([0, 0, 1])

        # Governing equation of refraction
        s_f = (ref_ind[i] / ref_ind[i + 1]) * np.cross(N_hat, np.cross(-N_hat, s_i)) - \
              N_hat * np.sqrt(1 - ((ref_ind[i] / ref_ind[i + 1]) ** 2) * np.dot(np.cross(N_hat, s_i), np.cross(N_hat, s_i)))
        new_thetay[i + 1] = np.sign(s_f[0]) * acosd(np.dot(z_hat, s_f) / (np.linalg.norm(s_f) * np.linalg.norm(z_hat)))

        # Calculate new coordinates
        y1 = py_next
        y2 = y1 + tand(new_thetay[i + 1])
        y3 = 0
        z1 = pz_next
        z2 = pz_next + 1
        z3 = cum_dist[i + 1]

        y4 = 1 if phiy[i + 1] == 0 else cotd(phiy[i + 1])
        z4 = cum_dist[i + 1] + (1 if phiy[i + 1] != 0 else 0)

        py_next = ((y1 * z2 - z1 * y2) * (y3 - y4) - (y1 - y2) * (y3 * z4 - z3 * y4)) / \
                    ((y1 - y2) * (z3 - z4) - (z1 - z2) * (y3 - y4))
        pz_next = ((y1 * z2 - z1 * y2) * (z3 - z4) - (z1 - z2) * (y3 * z4 - z3 * y4)) / \
                    ((y1 - y2) * (z3 - z4) - (z1 - z2) * (y3 - y4))

        y_coords[i + 1] = (0, py_next, pz_next)

        # print("-------------------------")
        # print(f"\nIteration {i+1}:")
        # print("-------------------------")
        # print(f"    Normalized normal vector N_hat: {N_hat}")
        # print(f"    Initial direction vector s_i: {s_i}")
        # print(f"    Final direction vector s_f before normalization: {s_f}")
        # print(f"    Updated angle new_thetay[i+1]: {new_thetay[i + 1]}")
        # print(f"    cum_dist: {cum_dist}, cum_dist[i+1]: {cum_dist[i + 1]}")
        # print(f"    y1: {y1}, y2: {y2}, y3: {y3}, y4: {y4}")
        # print(f"    z1: {z1}, z2: {z2}, z3: {z3}, z4: {z4}")
        # print(f"    Next positions py_next, pz_next: {py_next}, {pz_next}")

    return y_coords, new_thetay