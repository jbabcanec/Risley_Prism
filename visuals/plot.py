import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from visuals.axes_options import set_axes_equal

def plot(Laser_coords, history_phix, history_phiy, history_thetax, history_thetay, int_dist):
    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Assuming 'Laser_coords' is a dictionary containing lists of coordinates indexed by integers
    for key, coords in Laser_coords.items():
        # Extract x, y, z coordinates
        x_coords, y_coords, z_coords = zip(*coords)
        
        # Plot the points with proper orientation according to LHR
        ax.scatter(y_coords, z_coords, x_coords, marker='o', linestyle='dotted', label=f'Loop {key}')
    
    # Set labels according to the left hand rule
    ax.set_xlabel('Y axis (Middle Finger - Right Arm)')
    ax.set_ylabel('Z axis (Index Finger - Facing)')
    ax.set_zlabel('X axis (Thumb - Up)')
    
    # Rotate the view to look down Z-axis with X-axis vertical and Y-axis to the right
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

plot(Laser_coords, history_phix, history_phiy, history_thetax, history_thetay, int_dist)
