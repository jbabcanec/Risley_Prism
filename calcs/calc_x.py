import numpy as np
from utils.funs import *
from inputs import ref_ind

def calc_x(phix, gamma, cum_dist, thetax, px, pz):
    x_coords = {}
    # Start with the initial theta value for the first wedge, then calculate subsequent values.
    new_thetax = np.zeros(len(phix))  # Ensures we have space for each wedge, including workpiece
    new_thetax[0] = thetax  # Initial thetax used for the first calculation

    # Initialize px and pz for subsequent wedges based on initial values
    px_next = px
    pz_next = pz

    # Compute new angles and positions for each wedge
    for i in range(len(phix) - 1):  # Exclude the last element which is the workpiece
        # Normalizing the vectors
        N_hat = np.array([tand(phix[i]), 0, -1])
        N_hat = N_hat / np.linalg.norm(N_hat)
        s_i = np.array([tand(new_thetax[i]), 0, 1])  # Use updated theta for each wedge
        s_i = s_i / np.linalg.norm(s_i)
        z_hat = np.array([0, 0, 1])

        # Governing equation of refraction
        s_f = (ref_ind[i] / ref_ind[i + 1]) * np.cross(N_hat, np.cross(-N_hat, s_i)) - \
              N_hat * np.sqrt(1 - ((ref_ind[i] / ref_ind[i + 1]) ** 2) * np.dot(np.cross(N_hat, s_i), np.cross(N_hat, s_i)))
        new_thetax[i + 1] = np.sign(s_f[0]) * acosd(np.dot(z_hat, s_f) / (np.linalg.norm(s_f) * np.linalg.norm(z_hat)))

        # Calculate new coordinates
        x1 = px_next
        x2 = x1 + tand(new_thetax[i + 1])
        x3 = 0
        z1 = pz_next
        z2 = pz_next + 1
        z3 = cum_dist[i + 1]

        x4 = 1 if phix[i + 1] == 0 else cotd(phix[i + 1])
        z4 = cum_dist[i + 1] + (1 if phix[i + 1] != 0 else 0)

        px_next = ((x1 * z2 - z1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * z4 - z3 * x4)) / \
                    ((x1 - x2) * (z3 - z4) - (z1 - z2) * (x3 - x4))
        pz_next = ((x1 * z2 - z1 * x2) * (z3 - z4) - (z1 - z2) * (x3 * z4 - z3 * x4)) / \
                    ((x1 - x2) * (z3 - z4) - (z1 - z2) * (x3 - x4))

        x_coords[i + 1] = (px_next, 0, pz_next)

    return x_coords, new_thetax