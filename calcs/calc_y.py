import numpy as np
from utils.funs import *
from inputs import *

def calc_y(phiy, gamma, cum_dist, thetay):
    # Simple placeholder vector for y_coords
    y_coords = np.array([0, 1, 1])

    # Return the current thetay without changes
    new_thetay = thetay

    return y_coords, new_thetay