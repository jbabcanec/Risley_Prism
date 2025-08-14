import numpy as np
import os
import pickle

from utils.funs import *
from utils.saving import save_data
from utils.format import *
from calcs.init_coords import initialize_coordinates
from calcs.calc_proj_coord import calc_proj_coord
from calcs.calc_z_coord import calc_z_coord
from visuals.plot import plot
from visuals.axes_options import set_axes_equal
from inputs import *

def main():
    try:
        # Validate inputs before starting simulation
        validate_inputs()
        
        # Initialize phi angles for each wedge, gamma, and cumulative distances
        phix, phiy, thetax, thetay, gamma, cum_dist = initialize()
        time = np.arange(0, TIMELIM, TIMELIM / INC)
        print(f'time: {time}')
    except ValueError as e:
        print(f"Input validation error: {e}")
        return
    except Exception as e:
        print(f"Initialization error: {e}")
        return

    # Initialize history storage
    history_phix = {}
    history_phiy = {}
    history_thetax = {}
    history_thetay = {}
    all_x_coords = {}
    all_y_coords = {}
    all_z_coords = {}
    Laser_coords = {}

    # Main calculation loop
    for idx, current_time in enumerate(time):
        print(f"\nTime step {idx+1}/{len(time)} at time {current_time:.2f} sec")

        # Update angles and vectors for each wedge as it spins and creates a new wedge angle
        update_angles_and_vectors(current_time, phix, phiy, gamma)
        ((orig_coordx, new_coordx), (orig_coordy, new_coordy), (orig_coordz), PX0, PY0, PZ_X0, PZ_Y0) = initialize_coordinates(RX, RY, thetax, thetay, phix, phiy, int_dist)

        x_coords, new_thetax = calc_proj_coord(str(idx), orig_coordx, new_coordx, phix, cum_dist, thetax, PX0, PZ_X0, 'x')
        y_coords, new_thetay = calc_proj_coord(str(idx), orig_coordy, new_coordy, phiy, cum_dist, thetay, PY0, PZ_Y0, 'y')
        z_coords = calc_z_coord(str(idx), orig_coordz, phix, phiy, gamma, cum_dist, x_coords, y_coords)

        # Collect coordinate data at each time step and store the updated angles and phis into their history
        all_x_coords[str(idx)] = x_coords[str(idx)]
        all_y_coords[str(idx)] = y_coords[str(idx)]
        all_z_coords[str(idx)] = z_coords[str(idx)]
        # Extract coordinates properly - each coord is a 3-element list [x, y, z]
        Laser_coords[idx] = [[x[0] if isinstance(x, list) else x, 
                             y[1] if isinstance(y, list) else y, 
                             z[2] if isinstance(z, list) else z] 
                            for x, y, z in zip(all_x_coords[str(idx)], all_y_coords[str(idx)], all_z_coords[str(idx)])]

        history_thetax[str(idx)] = new_thetax.tolist()
        history_thetay[str(idx)] = new_thetay.tolist()
        history_phix[str(idx)] = phix.copy()
        history_phiy[str(idx)] = phiy.copy()

    # Round n' print
    print(f'\n------------')
    print(f'    Laser coords:\n{format_dict(round_dict_values(Laser_coords, 2))}')
    print(f'    All Thetax:\n{format_dict(round_dict_values(history_thetax, 2))}')
    print(f'    All Thetay:\n{format_dict(round_dict_values(history_thetay, 2))}')
    print(f'    All Phix:\n{format_dict(round_dict_values(history_phix, 2))}')
    print(f'    All Phiy:\n{format_dict(round_dict_values(history_phiy, 2))}')
    print(f'\n------------')

    # Extract workpiece projections for quick visualization
    workpiece_projections = []
    for idx in range(len(time)):
        if str(idx) in Laser_coords and Laser_coords[idx]:
            final_pos = Laser_coords[idx][-1]  # Last position (at workpiece)
            workpiece_projections.append([final_pos[0], final_pos[1], idx*0.1])  # x, y, time
    
    # Display workpiece projection summary
    print(f'\n{"="*60}')
    print("WORKPIECE LASER PROJECTIONS")
    print(f'{"="*60}')
    print("Time (s) | X-Position | Y-Position | Distance from Center")
    print(f'{"-"*60}')
    
    for i, (x, y, t) in enumerate(workpiece_projections[::10]):  # Show every 10th point
        dist = np.sqrt(x**2 + y**2)
        print(f"{t:8.1f} | {x:10.2f} | {y:10.2f} | {dist:18.2f}")
    
    print(f'{"-"*60}')
    if workpiece_projections:
        x_vals = [p[0] for p in workpiece_projections]
        y_vals = [p[1] for p in workpiece_projections]
        x_range = max(x_vals) - min(x_vals)
        y_range = max(y_vals) - min(y_vals)
        scan_area = x_range * y_range
        print(f"Scan range: X = {x_range:.2f}, Y = {y_range:.2f}")
        print(f"Total scan area: {scan_area:.2f} square units")
        print(f"Center position: ({np.mean(x_vals):.2f}, {np.mean(y_vals):.2f})")
    print(f'{"="*60}')

    # Save all the data
    save_data(history_phix, history_phiy, history_thetax, history_thetay, Laser_coords)

    # Plot the results
    if plotit == 'on':
        plot(Laser_coords, history_phix, history_phiy, history_thetax, history_thetay, int_dist)

def initialize():
    phix = STARTPHIX + [0.0]  # Adding 0.0 for workpiece
    phiy = STARTPHIY + [0.0]
    thetax = float(STARTTHETAX)
    thetay = float(STARTTHETAY)
    gamma = np.zeros(WEDGENUM + 1) # Adding final 0 gamma for workpiece
    cum_dist = np.cumsum(int_dist) # K is the cumulative distance in MATLAB
    print("Initial conditions:", dict(phix=phix, phiy=phiy, thetax=thetax, thetay=thetay))
    return phix, phiy, thetax, thetay, gamma, cum_dist

def update_angles_and_vectors(current_time, phix, phiy, gamma):
    # Reinitialize phix and phiy to START values at each time step
    phix[:] = STARTPHIX + [0.0]
    phiy[:] = STARTPHIY + [0.0]
    for i in range(WEDGENUM):
        update_individual_wedge(i, current_time, phix, phiy, gamma)

def update_individual_wedge(i, current_time, phix, phiy, gamma):
    gamma[i] = (360 * N[i] * current_time) % 360
    n1, nx, ny = compute_vectors(gamma[i], phix[i], phiy[i])
    cos_angle_nx, cos_angle_ny = compute_angles(n1, nx, ny)
    print_wedge_status(i, gamma, n1, cos_angle_nx, cos_angle_ny)
    phix[i], phiy[i] = update_phi(cos_angle_nx, cos_angle_ny)

def compute_vectors(gamma, phix, phiy):
    # Precompute trigonometric values for efficiency
    gamma_rad = np.radians(gamma + phiy)
    phix_tan = tand(phix)
    cos_gamma = np.cos(gamma_rad)
    sin_gamma = np.sin(gamma_rad)
    
    n1 = [cos_gamma * phix_tan, sin_gamma * phix_tan, -1]
    nx = [1, 0, 0]
    ny = [0, 1, 0]
    return n1, nx, ny

def compute_angles(n1, nx, ny):
    # Cache the norm of n1 since it's used twice
    n1_norm = np.linalg.norm(n1)
    # nx and ny are unit vectors, so their norms are 1
    cos_angle_nx = np.dot(n1, nx) / n1_norm
    cos_angle_ny = np.dot(n1, ny) / n1_norm
    return cos_angle_nx, cos_angle_ny

def print_wedge_status(i, gamma, n1, cos_angle_nx, cos_angle_ny):
    print(f"Wedge {i+1}:")
    print(f"  Gamma: {gamma[i]:.2f} degrees")
    print(f"  n1 vector: {n1}")
    print(f"  Cosine angles - nx: {cos_angle_nx:.4f}, ny: {cos_angle_ny:.4f}")

def update_phi(cos_angle_nx, cos_angle_ny):
    # Clamp values to [-1, 1] to avoid numerical errors in arccos
    cos_angle_nx = np.clip(cos_angle_nx, -1, 1)
    cos_angle_ny = np.clip(cos_angle_ny, -1, 1)
    
    phix = 90 - acosd(cos_angle_nx)
    phiy = 90 - acosd(cos_angle_ny)
    return phix, phiy

def validate_inputs():
    """Validate input parameters to ensure simulation stability."""
    if WEDGENUM <= 0:
        raise ValueError("WEDGENUM must be positive")
    
    if TIMELIM <= 0 or INC <= 0:
        raise ValueError("TIMELIM and INC must be positive")
    
    if len(STARTPHIX) != WEDGENUM or len(STARTPHIY) != WEDGENUM:
        raise ValueError(f"STARTPHIX and STARTPHIY must have {WEDGENUM} elements")
    
    if len(N) != WEDGENUM:
        raise ValueError(f"N (rotation speeds) must have {WEDGENUM} elements")
    
    if len(int_dist) != WEDGENUM + 1:
        raise ValueError(f"int_dist must have {WEDGENUM + 1} elements")
    
    if len(ref_ind) != WEDGENUM + 1:
        raise ValueError(f"ref_ind must have {WEDGENUM + 1} elements")
    
    # Check for valid phi ranges
    for i, phi in enumerate(STARTPHIX):
        if not (-90 < phi < 90):
            raise ValueError(f"STARTPHIX[{i}] = {phi} must be in range (-90, 90) degrees")
    
    for i, phi in enumerate(STARTPHIY):
        if not (-90 <= phi <= 90):
            raise ValueError(f"STARTPHIY[{i}] = {phi} must be in range [-90, 90] degrees")
    
    # Check for valid refractive indices
    for i, n in enumerate(ref_ind):
        if n < 1.0:
            raise ValueError(f"ref_ind[{i}] = {n} must be >= 1.0")
    
    print("Input validation passed!")


if __name__ == "__main__":
    main()
