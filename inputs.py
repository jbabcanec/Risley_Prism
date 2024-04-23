import numpy as np

# Constants
WEDGENUM = 3  # Number of interfaces
TIMELIM = 10  # Amount of time in seconds
INC = 100     # Number of time-steps

# Hardcoded inputs
time = np.linspace(0, TIMELIM, INC)  # Time array, don't touch
N = [1, 1, 1]                  		 # Rotations per second for each wedge
STARTTHETAX = 30             		 # Initial laser angle with respect to x-axis for each interface
STARTTHETAY = 45              		 # Initial laser angle with respect to y-axis for each interface
STARTPHIX = [15, 20, 25]             # Initial phi for each interface
STARTPHIY = [0, 0, 0]             	 # assuming wedges are never starting at an oblique angle
RX = 0                   			 # Initial laser height with respect to x-axis for each interface
RY = 0                   			 # Initial laser height with respect to y-axis for each interface

D = 5                                # Wedge diameter, determines if laser goes out of analytical space
int_dist = [2, 3, 4]				 # Optical axis distance between interfaces, called k in MATLAB code, K is the cumulative distance in MATLAB
int_dist.append(5)        			 # Distance from last wedge to workpiece
ref_ind = [1.1, 1.2, 1.3, 1.4]  	 # Refractive index before each interface
