import numpy as np
import os
import pickle

def save_data(history_phix, history_phiy, history_thetax, history_thetay, all_x_coords, all_y_coords, all_z_coords, Px, Py, Pz):
    data_directory = "data"
    os.makedirs(data_directory, exist_ok=True)  # Ensure the directory exists
    filepath = os.path.join(data_directory, "simulation_data.pkl")

    # Collect all data into a dictionary
    data = {
        "history_phix": history_phix,
        "history_phiy": history_phiy,
        "history_thetax": history_thetax,
        "history_thetay": history_thetay,
        "all_x_coords": all_x_coords,
        "all_y_coords": all_y_coords,
        "all_z_coords": all_z_coords,
        "Px": Px,
        "Py": Py,
        "Pz": Pz
    }

    # Write the data to a file using pickle
    with open(filepath, 'wb') as file:
        pickle.dump(data, file)

    print(f"Data saved to {filepath}")
