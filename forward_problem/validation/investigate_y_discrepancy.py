#!/usr/bin/env python3
"""
Deep investigation of Y-coordinate discrepancy between Python and MATLAB.
Track calculations step-by-step to find where the difference originates.
"""

import numpy as np
import sys
import os
# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.funs import tand, cosd, sind, acosd, cotd
import inputs

def trace_y_calculation_python():
    """Trace Python's Y calculation step by step."""
    print("="*80)
    print("PYTHON Y-CALCULATION TRACE")
    print("="*80)
    
    # Setup
    inputs.WEDGENUM = 3
    inputs.N = [1.0, 0.5, 1.5]
    inputs.STARTPHIX = [5.0, 8.0, 3.0]
    inputs.STARTPHIY = [0.0, 0.0, 0.0]
    inputs.STARTTHETAY = 5.0
    inputs.RY = 0.0
    inputs.int_dist = [6.0, 6.0, 6.0, 6.0]
    inputs.ref_ind = [1.0, 1.2, 1.3, 1.4]
    
    time_val = 0.5
    thetay = inputs.STARTTHETAY
    
    # Calculate modified phi_y (matches MATLAB)
    gamma = []
    phiy = []
    for i in range(3):
        gamma_i = (360 * inputs.N[i] * time_val) % 360
        gamma.append(gamma_i)
        
        n1 = np.array([
            cosd(gamma_i) * tand(inputs.STARTPHIX[i]),
            sind(gamma_i) * tand(inputs.STARTPHIX[i]),
            -1
        ])
        ny = np.array([0, 1, 0])
        phiy_i = 90 - acosd(np.dot(n1, ny) / (np.linalg.norm(ny) * np.linalg.norm(n1)))
        phiy.append(phiy_i)
    
    phiy.append(0.0)  # Workpiece
    
    print(f"Gamma: {gamma}")
    print(f"PhiY: {phiy}")
    
    # Initial Y calculation
    from calcs.init_coords import initialize_coordinates
    
    # Get initial coordinates
    ((orig_coordx, new_coordx), (orig_coordy, new_coordy), (orig_coordz), PX0, PY0, PZ_X0, PZ_Y0) = initialize_coordinates(
        inputs.RX, inputs.RY, inputs.STARTTHETAX, inputs.STARTTHETAY, 
        inputs.STARTPHIX + [0.0], phiy, inputs.int_dist
    )
    
    print(f"\nInitial Y coordinates:")
    print(f"  orig_coordy: {orig_coordy}")
    print(f"  new_coordy: {new_coordy}")
    print(f"  PY0: {PY0}")
    print(f"  PZ_Y0: {PZ_Y0}")
    
    # Now trace through calc_proj_coord for Y
    from calcs.calc_proj_coord import calc_proj_coord
    
    # Calculate cumulative distances
    cum_dist = np.cumsum(inputs.int_dist)
    
    # Run the calculation with verbose output
    inputs.printit = 'on'
    y_coords, new_thetay = calc_proj_coord('0', orig_coordy, new_coordy, phiy, cum_dist, thetay, PY0, PZ_Y0, 'y')
    
    print(f"\nFinal Y coordinates: {y_coords['0']}")
    print(f"Final thetay values: {new_thetay}")
    
    return y_coords['0'][-1]

def trace_y_calculation_matlab():
    """Replicate exact MATLAB Y calculation."""
    print("\n" + "="*80)
    print("MATLAB Y-CALCULATION TRACE")
    print("="*80)
    
    # Identical setup
    wedgenum = 3
    N = [1.0, 0.5, 1.5]
    startphix = [5.0, 8.0, 3.0]
    startphiy = [0.0, 0.0, 0.0]
    thetay_1 = 5.0
    ry_1 = 0.0
    k = [6.0, 6.0, 6.0, 6.0]
    n = [1.0, 1.2, 1.3, 1.4]
    
    time_val = 0.5
    
    # Calculate gamma and phi_y
    gamma = []
    phiy = []
    for i in range(wedgenum):
        gamma_i = (360 * N[i] * time_val) % 360
        gamma.append(gamma_i)
        
        # MATLAB style calculation
        n1 = np.array([
            cosd(gamma_i) * tand(startphix[i]),
            sind(gamma_i) * tand(startphix[i]),
            -1
        ])
        ny = np.array([0, 1, 0])
        phiy_i = 90 - acosd(np.dot(n1, ny) / (np.linalg.norm(ny) * np.linalg.norm(n1)))
        phiy.append(phiy_i)
    
    print(f"Gamma: {gamma}")
    print(f"PhiY: {phiy}")
    
    # Initial laser path y (MATLAB lines 148-161)
    y1 = ry_1
    y2 = ry_1 + tand(thetay_1)
    y3 = 0
    z1 = 0
    z2 = 1
    z3 = k[0]
    
    if phiy[0] == 0:
        y4 = 1
        z4 = k[0]
    else:
        y4 = cotd(phiy[0])
        z4 = k[0] + 1
    
    print(f"\nInitial Y ray:")
    print(f"  y1={y1:.6f}, y2={y2:.6f}, y3={y3:.6f}, y4={y4:.6f}")
    print(f"  z1={z1:.6f}, z2={z2:.6f}, z3={z3:.6f}, z4={z4:.6f}")
    
    # First intersection (MATLAB lines 163-164)
    Py_1 = ((y1*z2 - z1*y2)*(y3 - y4) - (y1 - y2)*(y3*z4 - z3*y4)) / ((y1 - y2)*(z3 - z4) - (z1 - z2)*(y3 - y4))
    Pz_1 = ((y1*z2 - z1*y2)*(z3 - z4) - (z1 - z2)*(y3*z4 - z3*y4)) / ((y1 - y2)*(z3 - z4) - (z1 - z2)*(y3 - y4))
    
    print(f"\nFirst Y intersection: Py_1={Py_1:.6f}, Pz_1={Pz_1:.6f}")
    
    # Calculate K (cumulative distances)
    K = []
    sumk = 0
    for i in range(wedgenum):
        sumk += k[i]
        K.append(sumk)
    K.append(K[-1] + k[-1])
    
    # Y refraction through wedges
    Py = [None, Py_1]
    Pz = [None, Pz_1]
    thetay = [thetay_1]
    phiy.append(0)  # phiy_{wedgenum+1} = 0
    
    for i in range(wedgenum):
        print(f"\n--- Wedge {i+1} Y Refraction ---")
        
        # MATLAB line 171: N = [tand(phiy_{i});0;-1]
        # NOTE: This appears to put Y deflection in X component!
        N_vec = np.array([tand(phiy[i]), 0, -1])  # MATLAB style
        s_i = np.array([tand(thetay[i]), 0, 1])
        zmeasure = np.array([0, 0, 1])
        
        print(f"  phiy[{i}] = {phiy[i]:.6f}")
        print(f"  thetay[{i}] = {thetay[i]:.6f}")
        print(f"  N_vec (before norm) = {N_vec}")
        print(f"  s_i (before norm) = {s_i}")
        
        # Normalize
        N_vec = N_vec / np.linalg.norm(N_vec)
        s_i = s_i / np.linalg.norm(s_i)
        
        print(f"  N_vec (normalized) = {N_vec}")
        print(f"  s_i (normalized) = {s_i}")
        
        # Snell's law
        cross_N_si = np.cross(N_vec, s_i)
        print(f"  N × s_i = {cross_N_si}")
        
        cross_neg_N_si = np.cross(-N_vec, s_i)
        cross_N_cross = np.cross(N_vec, cross_neg_N_si)
        print(f"  N × (-N × s_i) = {cross_N_cross}")
        
        term1 = (n[i] / n[i+1]) * cross_N_cross
        term2_inner = 1 - ((n[i] / n[i+1])**2) * np.dot(cross_N_si, cross_N_si)
        term2 = N_vec * np.sqrt(term2_inner)
        
        print(f"  Refraction term1 = {term1}")
        print(f"  Refraction term2 = {term2}")
        
        s_f = term1 - term2
        print(f"  s_f = {s_f}")
        
        # New angle
        sign_factor = abs(s_f[0]) / s_f[0] if s_f[0] != 0 else 1
        thetay_new = sign_factor * acosd(np.dot(zmeasure, s_f) / (np.linalg.norm(s_f) * np.linalg.norm(zmeasure)))
        thetay.append(thetay_new)
        
        print(f"  New thetay[{i+1}] = {thetay_new:.6f}")
        
        # New coordinates
        y1 = Py[i+1]
        y2 = Py[i+1] + tand(thetay_new)
        y3 = 0
        z1 = Pz[i+1]
        z2 = Pz[i+1] + 1
        z3 = K[i+1]
        
        if phiy[i+1] == 0:
            y4 = 1
            z4 = K[i+1]
        else:
            y4 = cotd(phiy[i+1])
            z4 = K[i+1] + 1
        
        Py_new = ((y1*z2 - z1*y2)*(y3 - y4) - (y1 - y2)*(y3*z4 - z3*y4)) / ((y1 - y2)*(z3 - z4) - (z1 - z2)*(y3 - y4))
        Pz_new = ((y1*z2 - z1*y2)*(z3 - z4) - (z1 - z2)*(y3*z4 - z3*y4)) / ((y1 - y2)*(z3 - z4) - (z1 - z2)*(y3 - y4))
        
        Py.append(Py_new)
        Pz.append(Pz_new)
        
        print(f"  New Py[{i+2}] = {Py_new:.6f}")
        print(f"  New Pz[{i+2}] = {Pz_new:.6f}")
    
    print(f"\nFinal MATLAB Y position: {Py[-1]:.6f}")
    return [0, Py[-1], K[-1]]

def main():
    """Compare the two calculations."""
    print("🔬 DETAILED Y-COORDINATE DISCREPANCY INVESTIGATION")
    print("="*80)
    
    # Run both calculations
    python_result = trace_y_calculation_python()
    matlab_result = trace_y_calculation_matlab()
    
    print("\n" + "="*80)
    print("COMPARISON RESULTS")
    print("="*80)
    
    python_y = python_result[1] if isinstance(python_result, list) else python_result
    matlab_y = matlab_result[1]
    
    print(f"Python Y: {python_y:.6f}")
    print(f"MATLAB Y: {matlab_y:.6f}")
    print(f"Difference: {abs(python_y - matlab_y):.6f}")
    print(f"Percent error: {100 * abs(python_y - matlab_y) / matlab_y:.2f}%")

if __name__ == "__main__":
    main()