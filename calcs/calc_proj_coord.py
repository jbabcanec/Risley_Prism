import numpy as np
from utils.funs import *
from inputs import ref_ind

def calc_proj_coord(idx, phi, cum_dist, theta, p, pz, all_coords, axis):
    coords = {}
    new_theta = np.zeros(len(phi))  # This ensures we have space for each wedge, including workpiece
    new_theta[0] = theta  # Initial theta used for the first calculation

    # Initialize p and pz for subsequent wedges based on initial values
    p_next = p
    pz_next = pz

    # Compute new angles and positions for each wedge
    for i in range(len(phi) - 1):  # Exclude the last element which is the workpiece
        # Normalizing the vectors
        N_hat = np.array([tand(phi[i]), 0, -1])
        N_hat = N_hat / np.linalg.norm(N_hat)
        s_i = np.array([tand(new_theta[i]), 0, 1])  # Use updated theta for each wedge
        s_i = s_i / np.linalg.norm(s_i)
        z_hat = np.array([0, 0, 1])

        # Governing equation of refraction
        s_f = (ref_ind[i] / ref_ind[i + 1]) * np.cross(N_hat, np.cross(-N_hat, s_i)) - \
              N_hat * np.sqrt(1 - ((ref_ind[i] / ref_ind[i + 1]) ** 2) * np.dot(np.cross(N_hat, s_i), np.cross(N_hat, s_i)))
        new_theta[i + 1] = np.sign(s_f[0]) * acosd(np.dot(z_hat, s_f) / (np.linalg.norm(s_f) * np.linalg.norm(z_hat)))

        # Calculate new coordinates
        p1 = p_next
        p2 = p1 + tand(new_theta[i + 1])
        p3 = 0
        z1 = pz_next
        z2 = pz_next + 1
        z3 = cum_dist[i + 1]

        p4 = 1 if phi[i + 1] == 0 else cotd(phi[i + 1])
        z4 = cum_dist[i + 1] + (1 if phi[i + 1] != 0 else 0)

        p_next = ((p1 * z2 - z1 * p2) * (p3 - p4) - (p1 - p2) * (p3 * z4 - z3 * p4)) / \
                    ((p1 - p2) * (z3 - z4) - (z1 - z2) * (p3 - p4))
        pz_next = ((p1 * z2 - z1 * p2) * (z3 - z4) - (z1 - z2) * (p3 * z4 - z3 * p4)) / \
                    ((p1 - p2) * (z3 - z4) - (z1 - z2) * (p3 - p4))

        new_coord = (0, p_next, pz_next) if axis == 'y' else (p_next, 0, pz_next)
        all_coords[idx].append(new_coord)

        print("\n-------------------------")
        print(f"Iteration {i+1} on Coord {axis}:")
        print("-------------------------")
        print(f"    Normalized normal vector N_hat: {N_hat}")
        print(f"    Initial direction vector s_i: {s_i}")
        print(f"    Final direction vector s_f before normalization: {s_f}")
        print(f"    Updated angle new_theta{axis}{i + 1}: {new_theta[i + 1]}")
        print(f"    cum_dist: {cum_dist}, cum_dist[{i + 1}]: {cum_dist[i + 1]}")
        print(f"    {axis}1: {p1}, {axis}2: {p2}, {axis}3: {p3}, {axis}4: {p4}")
        print(f"    z1: {z1}, z2: {z2}, z3: {z3}, z4: {z4}")
        print(f"    Next positions p{axis}_next, pz_next: {p_next}, {pz_next}")
        print(f'    p{axis} = {p_next}')
        print(f'    pz = {p_next}')

    return all_coords, new_theta