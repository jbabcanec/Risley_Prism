import numpy as np
import pickle
import os

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from visuals.axes_options import set_axes_equal
from visuals.wedge_options import create_wedge

def plot(Laser_coords=None, history_phix=None, history_phiy=None, history_thetax=None, history_thetay=None, int_dist=None):
    def read_data():
        output_directory = "output"
        filepath = os.path.join(output_directory, "simulation_data.pkl")
        
        with open(filepath, 'rb') as file:
            data = pickle.load(file)

        return data

    def print_data(data):
        print("Laser_coords for the first key '0':")
        print(data['Laser_coords'][0])
        
        print("\nHistory phix:")
        for key, value in data['history_phix'].items():
            print(f"Key {key}: {value}")
        
        print("\nHistory phiy:")
        for key, value in data['history_phiy'].items():
            print(f"Key {key}: {value}")
        
        print("\nHistory thetax:")
        for key, value in data['history_thetax'].items():
            print(f"Key {key}: {value}")
        
        print("\nHistory thetay:")
        for key, value in data['history_thetay'].items():
            print(f"Key {key}: {value}")

    # Use provided data if available, otherwise read from file
    if Laser_coords is None or history_phix is None:
        data = read_data()
        print_data(data)
        Laser_coords = data['Laser_coords']
        history_phix = data['history_phix']
        history_phiy = data['history_phiy']
        history_thetax = data['history_thetax']
        history_thetay = data['history_thetay']
        int_dist = [6, 6, 6, 6] if int_dist is None else int_dist
    else:
        # Use provided data directly
        if int_dist is None:
            int_dist = [6, 6, 6, 6]

    # Create an enhanced 3D plot with better styling
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Set dark background for better visibility
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    
    # Enhanced visualization with color-coded temporal progression
    import matplotlib.cm as cm
    import matplotlib.colors as colors
    
    # Create a colormap for temporal progression
    cmap = cm.get_cmap('viridis')
    norm = colors.Normalize(vmin=0, vmax=len(Laser_coords)-1)
    
    # Track all beam paths for better visualization
    all_beam_paths = []
    
    for key, coords in Laser_coords.items():
        # Extract x, y, z coordinates
        x_coords, y_coords, z_coords = zip(*coords)
        all_beam_paths.append((x_coords, y_coords, z_coords))
        
        # Use color to show temporal progression
        color = cmap(norm(key))
        
        # Plot the laser path with enhanced styling
        ax.scatter(y_coords, z_coords, x_coords, s=50, marker='o', 
                  color=color, alpha=0.8, edgecolors='white', linewidth=0.5,
                  label=f'T={key*0.1:.1f}s' if key % 10 == 0 else "")
        
        # Connect points with enhanced lines
        ax.plot(y_coords, z_coords, x_coords, linestyle='-', color=color, 
               linewidth=2, alpha=0.7)
        
        # Add beam divergence lines for first few time steps
        if key < 5:
            for i, (x, y, z) in enumerate(zip(x_coords, y_coords, z_coords)):
                if i > 0:  # Skip the initial point
                    ax.plot([0, y], [z, z], [0, x], color='cyan', 
                           linestyle=':', linewidth=0.8, alpha=0.5)

    # Calculate the sum of all components of int_dist
    total_z = sum(int_dist)
    
    # Add enhanced principal axis with gradient
    ax.plot([0, 0], [0, total_z], [0, 0], 'red', linestyle='-', 
           linewidth=3, label='Principal Axis', alpha=0.8)
    
    # Add coordinate system indicators
    ax.plot([0, 2], [0, 0], [0, 0], 'red', linewidth=2, alpha=0.7)  # X-axis
    ax.plot([0, 0], [0, 2], [0, 0], 'green', linewidth=2, alpha=0.7)  # Y-axis
    ax.plot([0, 0], [0, 0], [0, 2], 'blue', linewidth=2, alpha=0.7)  # Z-axis
    
    # Add text labels for axes
    ax.text(2.5, 0, 0, 'X', color='red', fontsize=12, weight='bold')
    ax.text(0, 2.5, 0, 'Y', color='green', fontsize=12, weight='bold')
    ax.text(0, 0, 2.5, 'Z', color='blue', fontsize=12, weight='bold')

    # Add enhanced wedge visualization
    DX, DY = 10, 10  # Dimensions of each wedge
    y_positions = np.cumsum(int_dist[:-1])  # Compute positions for wedges (excluding workpiece)
    
    # Use different colors for each wedge
    wedge_colors = ['lightblue', 'lightgreen', 'lightyellow', 'lightpink']

    for i, (y, phix, phiy) in enumerate(zip(y_positions, history_phix['0'][:-1], history_phiy['0'][:-1])):
        wedge = create_wedge(DX, DY, y, phi_x=phix, phi_y=phiy)
        color = wedge_colors[i % len(wedge_colors)]
        poly = Poly3DCollection([wedge], color=color, alpha=0.6, 
                               edgecolor='darkblue', linewidth=1.5)
        ax.add_collection3d(poly)
        
        # Add wedge labels
        ax.text(0, y, DY/2 + 1, f'Wedge {i+1}', fontsize=10, 
               ha='center', weight='bold')

    # Enhanced axis labels and styling
    ax.set_xlabel('Y axis', fontsize=12, weight='bold', color='white')
    ax.set_ylabel('Z axis', fontsize=12, weight='bold', color='white')
    ax.set_zlabel('X axis', fontsize=12, weight='bold', color='white')
    
    # Style the tick labels
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.tick_params(axis='z', colors='white')
    
    # Set optimal viewing angle for Risley prism visualization
    ax.view_init(elev=20, azim=45)
    
    # Make the axes to scale
    set_axes_equal(ax)
    
    # Add enhanced title and legend
    plt.title('Risley Prism Beam Steering Simulation', 
             fontsize=16, weight='bold', color='white', pad=20)
    
    # Customize legend
    legend = ax.legend(loc='upper left', bbox_to_anchor=(0.02, 0.98), 
                      fontsize=10, frameon=True, fancybox=True, shadow=True)
    legend.get_frame().set_facecolor('black')
    legend.get_frame().set_alpha(0.8)
    for text in legend.get_texts():
        text.set_color('white')
    
    # Add grid for better depth perception
    ax.grid(True, alpha=0.3)
    
    # Adjust layout and show
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot()
