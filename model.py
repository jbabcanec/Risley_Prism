import numpy as np
import matplotlib.pyplot as plt
from inputs import *  # Assuming all necessary settings are stored here

# Time array
time = np.linspace(0, timelim, inc)

# Preallocate arrays
phix = np.array(startphix)
phiy = np.array(startphix)  # Assuming initial phiy similar to phix
gamma = np.zeros(wedgenum)
K = np.zeros(wedgenum + 1)
coordx = np.zeros((wedgenum + 2, 3))
coordy = np.zeros((wedgenum + 2, 3))
coordz = np.zeros((wedgenum + 2, 3))
LaserCoord = np.zeros((wedgenum + 2, 3))

# Initial coordinates
coordx[1] = [rx_1, 0, 0]
coordy[1] = [0, ry_1, 0]
coordz[1] = [0, 0, 0]  # Assuming Z coordinate starts at 0

# Sum of k values to compute distances
sumk = 0
for i in range(wedgenum + 1):
    sumk += k[i]
    K[i] = sumk

# Simulation loop
for iter in range(len(time)):
    for i in range(wedgenum):
        gamma[i] = (360 * N[i] * time[iter]) % 360
        n1 = np.array([np.cos(np.radians(gamma[i])) * np.tan(np.radians(phix[i])),
                       np.sin(np.radians(gamma[i])) * np.tan(np.radians(phix[i])),
                       -1])
        nx = np.array([1, 0, 0])
        ny = np.array([0, 1, 0])

        phix[i] = 90 - np.degrees(np.arccos(np.dot(n1, nx) / (np.linalg.norm(nx) * np.linalg.norm(n1))))
        phiy[i] = 90 - np.degrees(np.arccos(np.dot(n1, ny) / (np.linalg.norm(ny) * np.linalg.norm(n1))))

    # Initial laser path calculation
    x1, x2, x3 = coordx[1][0], coordx[1][0] + np.tan(np.radians(thetax_1)), 0
    y1, y2, y3 = coordy[1][1], coordy[1][1] + np.tan(np.radians(thetay_1)), 0
    z1, z2, z3 = 0, 1, K[0]

    # Iterate over each wedge and calculate new coordinates
    for i in range(1, wedgenum + 1):
        x4 = np.cot(np.radians(phix[i-1])) if phix[i-1] != 0 else 1
        y4 = np.cot(np.radians(phiy[i-1])) if phiy[i-1] != 0 else 1
        z4 = K[i-1] + 1

        Px = ((x1 * z2 - z1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * z4 - z3 * x4)) / ((x1 - x2) * (z3 - z4) - (z1 - z2) * (x3 - x4))
        Py = ((y1 * z2 - z1 * y2) * (y3 - y4) - (y1 - y2) * (y3 * z4 - z3 * y4)) / ((y1 - y2) * (z3 - z4) - (z1 - z2) * (y3 - y4))
        Pz = ((x1 * z2 - z1 * x2) * (z3 - z4) - (z1 - z2) * (x3 * z4 - z3 * x4)) / ((x1 - x2) * (z3 - z4) - (z1 - z2) * (x3 - x4))

        coordx[i+1] = [Px, 0, Pz]
        coordy[i+1] = [0, Py, Pz]
        coordz[i+1] = [0, 0, K[i] + ((Px * np.cos(np.radians(gamma[i-1])) - Py * np.sin(np.radians(gamma[i-1]))) * np.tan(np.radians(phix[i-1])))]  # Simplified

        # Update starting points for next calculation
        x1, x2, x3 = Px, Px + np.tan(np.radians(phix[i])), Px
        y1, y2, y3 = Py, Py + np.tan(np.radians(phiy[i])), Py
        z1, z2, z3 = Pz, Pz + 1, K[i]

    # Store final coordinates in LaserCoord
    for i in range(1, wedgenum + 2):
        LaserCoord[i] = [coordx[i][0], coordy[i][1], coordz[i][2]]

# Plotting
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for i in range(1, wedgenum + 2):
    ax.scatter(LaserCoord[i][0], LaserCoord[i][1], LaserCoord[i][2], c='r', marker='o')
ax.set_xlabel('X Coordinate')
ax.set_ylabel('Y Coordinate')
ax.set_zlabel('Z Coordinate')
ax.grid(True)
plt.show()
