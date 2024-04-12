# settings.py
# Storing initial settings for the 3D spinning model

# Number of interfaces
wedgenum = 3

# Time settings
timelim = 10.0  # total time in seconds
inc = 100       # number of time steps

# Rotations per second for each wedge
N = [1.2, 0.8, 0.5]

# Initial angles and heights
thetax_1 = 30
thetay_1 = 45
rx_1 = 2
ry_1 = 2

# Wedge diameter
d = 5

# Starting angles (phi) for interfaces
startphix = [10, 20, 30]

# Optical axis distances between interfaces
k = [1, 1.5, 2, 2.5]

# Refractive indices before interfaces
n = [1.0, 1.33, 1.5, 1.0]
