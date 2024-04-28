import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from visuals.axes_options import set_axes_equal
from visuals.wedge_options import create_wedge


def plot(Laser_coords, history_phix, history_phiy, history_thetax, history_thetay, int_dist):
    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Assuming 'Laser_coords' is a dictionary containing lists of coordinates indexed by integers
    for key, coords in Laser_coords.items():
        # Extract x, y, z coordinates
        x_coords, y_coords, z_coords = zip(*coords)
        
        # Plot the points more bold
        ax.scatter(y_coords, z_coords, x_coords, s=25, marker='o', label=f'Loop {key}')  # Increased marker size

        # Connect points with a line
        ax.plot(y_coords, z_coords, x_coords, linestyle='-', color='blue')  # Add this line to connect the points

        # Add dotted lines from each point down to the Z-plane
        for (x, y, z) in zip(x_coords, y_coords, z_coords):
            ax.plot([0, y], [z, z], [0, x], 'grey', linestyle='dotted', linewidth=1)  # Adjust color and style if needed

    # Calculate the sum of all components of int_dist
    total_z = sum(int_dist)
    
    # Add a line from the origin (0, 0, 0) to (0, 0, total_z)
    ax.plot([0, 0], [0, total_z], [0, 0], 'lightcoral', linestyle='-', linewidth=1, label='Principal Axis')

    # Add wedges at each position
    DX, DY = 10, 10  # Dimensions of each wedge
    y_positions = np.cumsum(int_dist)  # Compute positions for wedges

    for y, phix, phiy in zip(y_positions, history_phix['0'], history_phiy['0']):
        wedge = create_wedge(DX, DY, y, phi_x=phix, phi_y=phiy)
        poly = Poly3DCollection([wedge], color='cyan', alpha=0.5, edgecolor='black')
        ax.add_collection3d(poly)

    # Set labels according to the left hand rule
    ax.set_xlabel('Y axis')
    ax.set_ylabel('Z axis')
    ax.set_zlabel('X axis')
    ax.view_init(elev=30, azim=135+180)  # Isometric view
    
    # Make the axes to scale
    set_axes_equal(ax)
    
    # Adding legend
    ax.legend()

    # Show the plot
    plt.show()

# Example usage
Laser_coords = {
    0: [
        [0, 0, 0],
        [1.1104258082893717, 0.5249319811555441, 6.297537698585813],
        [1.9899604982941679, 1.0058257604516667, 12.724286388744616],
        [2.563406806822984, 1.449516946397829, 18.686862783760663],
        [2.932258375740169, 1.861360813404719, 24.0]
    ]
}

history_phix = {'0': [15.0, 20.0, 15.0, 0.0]}
history_phiy = {'0': [0.0, 0.0, 0.0, 0.0]}
history_thetax = {'0': [10.0, 7.792820795621765, 5.493482858375487, 3.971247874417968]}
history_thetay = {'0': [5.0, 4.582401846600678, 4.229240896997841, 3.92666009251296]}
int_dist = [6, 6, 6, 6]

#plot(Laser_coords, history_phix, history_phiy, history_thetax, history_thetay, int_dist)
