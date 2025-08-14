#!/usr/bin/env python3
"""
Quick validation test - Compare key mathematical operations between Python implementation
and the expected MATLAB behavior using a single time step.
"""

import numpy as np
import sys
import inputs

# Import Python simulation components
from model import initialize, update_angles_and_vectors, validate_inputs
from calcs.calc_proj_coord import calc_proj_coord
from calcs.init_coords import initialize_coordinates
from calcs.calc_z_coord import calc_z_coord
from utils.funs import tand, cosd, sind, acosd, cotd

def setup_test_case():
    """Setup a simple test case matching MATLAB structure."""
    # Simple 3-wedge test case
    inputs.WEDGENUM = 3
    inputs.TIMELIM = 1.0
    inputs.INC = 10
    inputs.N = [1.0, 0.5, 1.5]
    inputs.STARTPHIX = [5.0, 8.0, 3.0]
    inputs.STARTPHIY = [0.0, 0.0, 0.0]
    inputs.STARTTHETAX = 10.0
    inputs.STARTTHETAY = 5.0
    inputs.RX = 0.0
    inputs.RY = 0.0
    inputs.int_dist = [6.0, 6.0, 6.0, 6.0]
    inputs.ref_ind = [1.0, 1.2, 1.3, 1.4]
    inputs.plotit = 'off'

def matlab_single_step(time_val):
    """
    Replicate MATLAB calculation for a single time step.
    Based on the MATLAB model.m structure.
    """
    print(f"\n🔬 Testing single time step at t = {time_val}")
    
    # MATLAB-style variable setup
    wedgenum = inputs.WEDGENUM
    startphix = inputs.STARTPHIX.copy()
    startphiy = inputs.STARTPHIY.copy()
    N = inputs.N
    k = inputs.int_dist
    n = inputs.ref_ind
    rx_1 = inputs.RX
    ry_1 = inputs.RY
    thetax_1 = inputs.STARTTHETAX
    thetay_1 = inputs.STARTTHETAY
    
    # Initialize phi values (MATLAB line 45)
    phix = startphix.copy()
    phiy = startphiy.copy()
    
    # Calculate gamma (MATLAB lines 49-51)
    gamma = []
    for i in range(wedgenum):
        gamma_i = (360 * N[i] * time_val) % 360
        gamma.append(gamma_i)
    
    print(f"Gamma values: {gamma}")
    
    # Modified phi calculations (MATLAB lines 54-64)
    for i in range(wedgenum):
        # MATLAB: n1 = [cosd(gamma_{i})*tand(phix_{i}); sind(gamma_{i})*tand(phix_{i}); -1];
        n1 = np.array([
            cosd(gamma[i]) * tand(phix[i]),
            sind(gamma[i]) * tand(phix[i]),
            -1
        ])
        
        ny = np.array([0, 1, 0])
        nx = np.array([1, 0, 0])
        
        # MATLAB: phix_{i} = 90 - acosd((dot(n1,nx))/(norm(nx)*norm(n1)));
        phix[i] = 90 - acosd(np.dot(n1, nx) / (np.linalg.norm(nx) * np.linalg.norm(n1)))
        phiy[i] = 90 - acosd(np.dot(n1, ny) / (np.linalg.norm(ny) * np.linalg.norm(n1)))
    
    print(f"Updated phix: {phix}")
    print(f"Updated phiy: {phiy}")
    
    # Calculate cumulative distances K (MATLAB lines 67-72)
    K = []
    sumk = 0
    for i in range(wedgenum):
        sumk += k[i]
        K.append(sumk)
    K.append(K[-1] + k[-1])  # K{wedgenum+1}
    
    print(f"Cumulative distances K: {K}")
    
    # X calculations (MATLAB lines 79-97)
    # Initial laser path x
    x1 = rx_1
    x2 = rx_1 + tand(thetax_1)
    x3 = 0
    z1 = 0
    z2 = 1
    z3 = k[0]
    
    if phix[0] == 0:
        x4 = 1  # arbitrary
        z4 = k[0]
    else:
        x4 = cotd(phix[0])
        z4 = k[0] + 1
    
    # Intersection calculation (MATLAB lines 95-96)
    Px_1 = ((x1*z2 - z1*x2)*(x3 - x4) - (x1 - x2)*(x3*z4 - z3*x4)) / ((x1 - x2)*(z3 - z4) - (z1 - z2)*(x3 - x4))
    Pz_1 = ((x1*z2 - z1*x2)*(z3 - z4) - (z1 - z2)*(x3*z4 - z3*x4)) / ((x1 - x2)*(z3 - z4) - (z1 - z2)*(x3 - x4))
    
    print(f"First intersection: Px_1 = {Px_1}, Pz_1 = {Pz_1}")
    
    # Refraction calculations through wedges (MATLAB lines 105-137)
    Px = [None, Px_1]  # Px_0 undefined, Px_1 calculated
    Pz = [None, Pz_1]
    thetax = [thetax_1]  # thetax_1
    
    phix.append(0)  # phix_{wedgenum+1} = 0
    
    for i in range(wedgenum):
        # MATLAB lines 106-108: Vectors
        N_vec = np.array([tand(phix[i]), 0, -1])
        s_i = np.array([tand(thetax[i]), 0, 1])
        zmeasure = np.array([0, 0, 1])
        
        # Normalize
        N_vec = N_vec / np.linalg.norm(N_vec)
        s_i = s_i / np.linalg.norm(s_i)
        
        # Snell's law (MATLAB line 114)
        cross_N_si = np.cross(N_vec, s_i)
        cross_N_neg_N_si = np.cross(N_vec, np.cross(-N_vec, s_i))
        
        term1 = (n[i] / n[i+1]) * cross_N_neg_N_si
        term2_inner = 1 - ((n[i] / n[i+1])**2) * np.dot(cross_N_si, cross_N_si)
        term2 = N_vec * np.sqrt(term2_inner)
        
        s_f = term1 - term2
        
        # New angle (MATLAB line 117)
        sign_factor = abs(s_f[0]) / s_f[0] if s_f[0] != 0 else 1
        thetax_new = sign_factor * acosd(np.dot(zmeasure, s_f) / (np.linalg.norm(s_f) * np.linalg.norm(zmeasure)))
        thetax.append(thetax_new)
        
        # New coordinates (MATLAB lines 120-135)
        x1 = Px[i+1]
        x2 = Px[i+1] + tand(thetax_new)
        x3 = 0
        z1 = Pz[i+1]
        z2 = Pz[i+1] + 1
        z3 = K[i+1]
        
        if phix[i+1] == 0:
            x4 = 1  # arbitrary
            z4 = K[i+1]
        else:
            x4 = cotd(phix[i+1])
            z4 = K[i+1] + 1
        
        Px_new = ((x1*z2 - z1*x2)*(x3 - x4) - (x1 - x2)*(x3*z4 - z3*x4)) / ((x1 - x2)*(z3 - z4) - (z1 - z2)*(x3 - x4))
        Pz_new = ((x1*z2 - z1*x2)*(z3 - z4) - (z1 - z2)*(x3*z4 - z3*x4)) / ((x1 - x2)*(z3 - z4) - (z1 - z2)*(x3 - x4))
        
        Px.append(Px_new)
        Pz.append(Pz_new)
        
        print(f"Wedge {i+1}: thetax = {thetax_new:.6f}, Px = {Px_new:.6f}, Pz = {Pz_new:.6f}")
    
    # Y calculations (MATLAB lines 148-201)
    # Initial laser path y
    y1 = ry_1
    y2 = ry_1 + tand(thetay_1)
    y3 = 0
    z1 = 0
    z2 = 1
    z3 = k[0]
    
    if phiy[0] == 0:
        y4 = 1  # arbitrary
        z4 = k[0]
    else:
        y4 = cotd(phiy[0])
        z4 = k[0] + 1
    
    # First Y intersection (MATLAB lines 163-164)
    Py_1 = ((y1*z2 - z1*y2)*(y3 - y4) - (y1 - y2)*(y3*z4 - z3*y4)) / ((y1 - y2)*(z3 - z4) - (z1 - z2)*(y3 - y4))
    Pz_1_y = ((y1*z2 - z1*y2)*(z3 - z4) - (z1 - z2)*(y3*z4 - z3*y4)) / ((y1 - y2)*(z3 - z4) - (z1 - z2)*(y3 - y4))
    
    print(f"First Y intersection: Py_1 = {Py_1}, Pz_1_y = {Pz_1_y}")
    
    # Y refraction calculations through wedges (MATLAB lines 170-200)
    Py = [None, Py_1]  # Py_0 undefined, Py_1 calculated
    thetay = [thetay_1]  # thetay_1
    
    phiy.append(0)  # phiy_{wedgenum+1} = 0
    
    for i in range(wedgenum):
        # MATLAB lines 171-173: Y-direction vectors
        N_vec = np.array([tand(phiy[i]), 0, -1])
        s_i = np.array([tand(thetay[i]), 0, 1])
        zmeasure = np.array([0, 0, 1])
        
        # Normalize
        N_vec = N_vec / np.linalg.norm(N_vec)
        s_i = s_i / np.linalg.norm(s_i)
        
        # Snell's law for Y direction (MATLAB line 178)
        cross_N_si = np.cross(N_vec, s_i)
        cross_N_neg_N_si = np.cross(N_vec, np.cross(-N_vec, s_i))
        
        term1 = (n[i] / n[i+1]) * cross_N_neg_N_si
        term2_inner = 1 - ((n[i] / n[i+1])**2) * np.dot(cross_N_si, cross_N_si)
        term2 = N_vec * np.sqrt(term2_inner)
        
        s_f = term1 - term2
        
        # New Y angle (MATLAB line 181)
        sign_factor = abs(s_f[0]) / s_f[0] if s_f[0] != 0 else 1
        thetay_new = sign_factor * acosd(np.dot(zmeasure, s_f) / (np.linalg.norm(s_f) * np.linalg.norm(zmeasure)))
        thetay.append(thetay_new)
        
        # New Y coordinates (MATLAB lines 184-199)
        y1 = Py[i+1]
        y2 = Py[i+1] + tand(thetay_new)
        y3 = 0
        z1 = Pz[i+1]  # Use Z from X calculations
        z2 = Pz[i+1] + 1
        z3 = K[i+1]
        
        if phiy[i+1] == 0:
            y4 = 1  # arbitrary
            z4 = K[i+1]
        else:
            y4 = cotd(phiy[i+1])
            z4 = K[i+1] + 1
        
        Py_new = ((y1*z2 - z1*y2)*(y3 - y4) - (y1 - y2)*(y3*z4 - z3*y4)) / ((y1 - y2)*(z3 - z4) - (z1 - z2)*(y3 - y4))
        Py.append(Py_new)
        
        print(f"Y Wedge {i+1}: thetay = {thetay_new:.6f}, Py = {Py_new:.6f}")
    
    # Return final workpiece position
    final_x = Px[-1]  # Px_{wedgenum+1}
    final_y = Py[-1]  # Py_{wedgenum+1}
    final_z = K[-1]   # At workpiece
    
    return final_x, final_y, final_z

def python_single_step(time_val):
    """Run Python simulation for single time step."""
    print(f"\n🐍 Python simulation at t = {time_val}")
    
    # Initialize
    phix, phiy, thetax, thetay, gamma, cum_dist = initialize()
    
    # Update angles
    update_angles_and_vectors(time_val, phix, phiy, gamma)
    
    # Initialize coordinates
    ((orig_coordx, new_coordx), (orig_coordy, new_coordy), (orig_coordz), PX0, PY0, PZ_X0, PZ_Y0) = initialize_coordinates(
        inputs.RX, inputs.RY, thetax, thetay, phix, phiy, inputs.int_dist
    )
    
    # Calculate projections
    x_coords, new_thetax = calc_proj_coord('0', orig_coordx, new_coordx, phix, cum_dist, thetax, PX0, PZ_X0, 'x')
    y_coords, new_thetay = calc_proj_coord('0', orig_coordy, new_coordy, phiy, cum_dist, thetay, PY0, PZ_Y0, 'y')
    z_coords = calc_z_coord('0', orig_coordz, phix, phiy, gamma, cum_dist, x_coords, y_coords)
    
    # Extract final workpiece position
    final_coords = [x_coords['0'][-1], y_coords['0'][-1], z_coords['0'][-1]]
    final_x = final_coords[0][0] if isinstance(final_coords[0], list) else final_coords[0]
    final_y = final_coords[1][1] if isinstance(final_coords[1], list) else final_coords[1]
    final_z = final_coords[2][2] if isinstance(final_coords[2], list) else final_coords[2]
    
    print(f"Python final position: x={final_x:.6f}, y={final_y:.6f}, z={final_z:.6f}")
    
    return final_x, final_y, final_z

def compare_single_step():
    """Compare MATLAB and Python for a single time step."""
    setup_test_case()
    validate_inputs()
    
    test_time = 0.5  # Test at t = 0.5 seconds
    
    print("="*80)
    print("🔬 SINGLE TIME STEP VALIDATION")
    print("="*80)
    
    # Run MATLAB-style calculation
    matlab_x, matlab_y, matlab_z = matlab_single_step(test_time)
    
    # Run Python calculation
    python_x, python_y, python_z = python_single_step(test_time)
    
    # Compare results
    print("\n" + "="*80)
    print("📊 COMPARISON RESULTS")
    print("="*80)
    
    print(f"MATLAB result: ({matlab_x:.6f}, {matlab_y:.6f}, {matlab_z:.6f})")
    print(f"Python result: ({python_x:.6f}, {python_y:.6f}, {python_z:.6f})")
    
    diff_x = abs(matlab_x - python_x)
    diff_y = abs(matlab_y - python_y)
    diff_z = abs(matlab_z - python_z)
    max_diff = max(diff_x, diff_y, diff_z)
    
    print(f"Differences:   ({diff_x:.2e}, {diff_y:.2e}, {diff_z:.2e})")
    print(f"Maximum difference: {max_diff:.2e}")
    
    tolerance = 2e-2  # 0.02 tolerance for engineering precision
    if max_diff < tolerance:
        print("✅ SINGLE STEP VALIDATION PASSED!")
        print("Python implementation matches MATLAB reference within engineering tolerance.")
        return True
    else:
        print("❌ SINGLE STEP VALIDATION FAILED!")
        print("Differences exceed engineering tolerance threshold.")
        return False

if __name__ == "__main__":
    success = compare_single_step()
    sys.exit(0 if success else 1)