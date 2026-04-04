#!/usr/bin/env python3
"""
Debug Y-coordinate calculation discrepancy between Python and MATLAB.
"""

import numpy as np
from utils.funs import tand, cosd, sind, acosd, cotd
import inputs

def debug_y_calculation():
    """Compare Y calculation step by step."""
    
    # Setup test parameters
    inputs.WEDGENUM = 3
    inputs.N = [1.0, 0.5, 1.5]
    inputs.STARTPHIX = [5.0, 8.0, 3.0]
    inputs.STARTPHIY = [0.0, 0.0, 0.0]
    inputs.STARTTHETAY = 5.0
    inputs.RY = 0.0
    inputs.int_dist = [6.0, 6.0, 6.0, 6.0]
    inputs.ref_ind = [1.0, 1.2, 1.3, 1.4]
    
    time_val = 0.5
    
    print("="*80)
    print("Y-COORDINATE CALCULATION DEBUG")
    print("="*80)
    
    # Calculate gamma and modified phi_y
    gamma = []
    phiy = inputs.STARTPHIY.copy()
    
    for i in range(inputs.WEDGENUM):
        gamma_i = (360 * inputs.N[i] * time_val) % 360
        gamma.append(gamma_i)
    
    print(f"Gamma values: {gamma}")
    print(f"Initial phiy: {phiy}")
    
    # Modified phi_y calculations (should match MATLAB)
    for i in range(inputs.WEDGENUM):
        # MATLAB: n1 = [cosd(gamma_{i})*tand(phix_{i}); sind(gamma_{i})*tand(phix_{i}); -1];
        # But for Y, we look at the Y component
        n1 = np.array([
            cosd(gamma[i]) * tand(inputs.STARTPHIX[i]),
            sind(gamma[i]) * tand(inputs.STARTPHIX[i]),
            -1
        ])
        
        ny = np.array([0, 1, 0])
        
        # Calculate angle with Y axis
        phiy[i] = 90 - acosd(np.dot(n1, ny) / (np.linalg.norm(ny) * np.linalg.norm(n1)))
        
        print(f"Wedge {i+1}:")
        print(f"  n1 vector: {n1}")
        print(f"  Angle with Y: {90 - phiy[i]:.4f} degrees")
        print(f"  Updated phiy[{i}]: {phiy[i]:.4f}")
    
    print(f"\nFinal phiy values: {phiy}")
    
    # Now check if the issue is in the normal vector for Y calculations
    print("\n" + "="*80)
    print("CHECKING NORMAL VECTORS FOR Y REFRACTION")
    print("="*80)
    
    for i in range(inputs.WEDGENUM):
        # For Y calculations, the normal should be in Y direction
        # MATLAB line 171: N = [tand(phiy_{i});0;-1];
        # But this seems wrong - it should be [0; tand(phiy_{i}); -1] for Y
        
        N_matlab_style = np.array([tand(phiy[i]), 0, -1])
        N_corrected = np.array([0, tand(phiy[i]), -1])
        
        print(f"\nWedge {i+1} (phiy = {phiy[i]:.4f}):")
        print(f"  MATLAB-style N (X component): {N_matlab_style}")
        print(f"  Should be N (Y component):     {N_corrected}")
        
        # Check which one is correct based on the physics
        # The normal should be perpendicular to the wedge surface
        # For Y deflection, the normal should have Y component
    
    return phiy

def check_matlab_y_normal():
    """Verify MATLAB's Y normal vector calculation."""
    print("\n" + "="*80)
    print("MATLAB Y NORMAL VECTOR ANALYSIS")
    print("="*80)
    
    print("\nMATLAB line 171 for Y calculations:")
    print("  N = [tand(phiy_{i});0;-1];")
    print("\nThis puts tand(phiy) in the X component, not Y!")
    print("This appears to be an error in the MATLAB code.")
    print("\nCorrect normal for Y deflection should be:")
    print("  N = [0; tand(phiy_{i}); -1];")
    
    print("\nHowever, if MATLAB uses the same convention consistently,")
    print("it might still produce valid results due to symmetry.")

if __name__ == "__main__":
    phiy = debug_y_calculation()
    check_matlab_y_normal()