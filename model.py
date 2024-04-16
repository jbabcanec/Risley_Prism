import numpy as np
from inputs import *
from funs import *

def main():
    # Initialize phi angles for each wedge
    phix = np.array(STARTPHIX, dtype=float)
    phiy = np.array(STARTPHIY, dtype=float)
    gamma = np.zeros(WEDGENUM)
    cum_dist = np.zeros(WEDGENUM + 1)  # Array to store cumulative distances

    print("Initial conditions:", dict(phix=phix, phiy=phiy))

    # Iterate over each time step
    for idx, current_time in enumerate(time):
        print(f"\nTime step {idx+1}/{len(time)} at time {current_time:.2f} sec")
        
        # Reinitialize phix and phiy to START values at each time step
        phix[:] = STARTPHIX
        phiy[:] = STARTPHIY

        # Update gamma for each wedge and compute vectors
        for i in range(WEDGENUM):
            gamma[i] = (360 * N[i] * current_time) % 360
            n1 = np.array([cosd(gamma[i]) * tand(phix[i]),
                           sind(gamma[i]) * tand(phix[i]),
                           -1])
            nx = np.array([1, 0, 0])
            ny = np.array([0, 1, 0])

            # Calculate angles
            cos_angle_nx = np.dot(n1, nx) / (np.linalg.norm(n1) * np.linalg.norm(nx))
            cos_angle_ny = np.dot(n1, ny) / (np.linalg.norm(n1) * np.linalg.norm(ny))

            # Debug outputs for each wedge
            print(f"Wedge {i+1}:")
            print(f"  Gamma: {gamma[i]:.2f} degrees")
            print(f"  n1 vector: {n1}")
            print(f"  Cosine angles - nx: {cos_angle_nx:.4f}, ny: {cos_angle_ny:.4f}")

            # Update phix and phiy
            phix[i] = 90 - acosd(cos_angle_nx)
            phiy[i] = 90 - acosd(cos_angle_ny)

            print(f"  Updated phix: {phix[i]:.4f} degrees")
            print(f"  Updated phiy: {phiy[i]:.4f} degrees")

        # Calculate cumulative distances
        sumk = 0
        for i in range(WEDGENUM):
            sumk += int_dist[i]
            cum_dist[i] = sumk
        cum_dist[WEDGENUM] = sumk + int_dist[-1]

    # Print final outcomes
    print("\nFinal results:")
    print("Final phix:", phix)
    print("Final phiy:", phiy)
    print("Cumulative Distances:", cum_dist)

if __name__ == "__main__":
    main()
