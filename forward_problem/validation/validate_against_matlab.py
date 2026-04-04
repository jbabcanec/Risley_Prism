#!/usr/bin/env python3
"""
Validation script to compare Python Risley prism simulation against MATLAB reference.

This script runs the Python simulation with identical parameters to the MATLAB model
and compares the final workpiece coordinates to ensure mathematical accuracy.
"""

import numpy as np
import json
import os
import sys
from datetime import datetime

# Import our Python simulation
from model import main as run_simulation
import inputs

def setup_validation_parameters():
    """Set up identical parameters for both Python and MATLAB simulations."""
    # Standard validation configuration - matches MATLAB inputs
    validation_config = {
        'WEDGENUM': 3,
        'TIMELIM': 5.0,
        'INC': 50,  # 50 time steps
        'N': [1.0, 0.5, 1.5],  # Rotation speeds
        'STARTPHIX': [5.0, 8.0, 3.0],  # Initial X angles
        'STARTPHIY': [0.0, 0.0, 0.0],  # Initial Y angles  
        'STARTTHETAX': 10.0,  # Initial laser X angle
        'STARTTHETAY': 5.0,   # Initial laser Y angle
        'RX': 0.0,    # Initial laser X position
        'RY': 0.0,    # Initial laser Y position
        'int_dist': [6.0, 6.0, 6.0, 6.0],  # Distances between interfaces
        'ref_ind': [1.0, 1.2, 1.3, 1.4],   # Refractive indices
        'plotit': 'off'
    }
    return validation_config

def update_python_inputs(config):
    """Update Python inputs module with validation configuration."""
    for param, value in config.items():
        if hasattr(inputs, param):
            setattr(inputs, param, value)
            print(f"Set {param} = {value}")
        else:
            print(f"Warning: {param} not found in inputs module")

def run_python_simulation():
    """Run Python simulation and extract workpiece coordinates."""
    print("\n" + "="*60)
    print("RUNNING PYTHON SIMULATION")
    print("="*60)
    
    # Capture simulation data
    import pickle
    from utils.saving import save_data
    
    # Run the simulation 
    run_simulation("validation_test")
    
    # Load the results
    output_files = [f for f in os.listdir("output/examples/") if "validation_test" in f]
    if not output_files:
        raise RuntimeError("Python simulation failed - no output files found")
    
    latest_file = sorted(output_files)[-1]  # Get most recent
    filepath = os.path.join("output/examples", latest_file, "simulation_data.pkl")
    
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    
    # Extract workpiece coordinates (final position from each time step)
    Laser_coords = data['Laser_coords']
    workpiece_coords = []
    
    for idx in sorted([int(k) for k in Laser_coords.keys()]):
        coords = Laser_coords[idx]
        if coords:
            final_pos = coords[-1]  # Last position (at workpiece)
            workpiece_coords.append([final_pos[0], final_pos[1], final_pos[2]])
    
    return np.array(workpiece_coords)

def generate_matlab_reference():
    """Generate MATLAB reference data using the same parameters."""
    config = setup_validation_parameters()
    
    matlab_script = f"""
% Validation script - identical to Python parameters
clear, clc
format longg

% Fixed parameters from Python
wedgenum = {config['WEDGENUM']};
timelim = {config['TIMELIM']};
inc = {config['INC']};
time = linspace(0,timelim,inc);

% Rotation speeds
"""
    
    for i, speed in enumerate(config['N']):
        matlab_script += f"N_{{{i+1}}} = {speed};\n"
    
    matlab_script += f"""
% Initial conditions
thetax_{{1}} = {config['STARTTHETAX']};
thetay_{{1}} = {config['STARTTHETAY']};
rx_{{1}} = {config['RX']};
ry_{{1}} = {config['RY']};
d = 10; % Standard diameter

% Initial coordinates
coordx{{1}} = [rx_{{1}};0;0];
coordy{{1}} = [0;ry_{{1}};0];

% Phi angles
"""
    
    for i, phi in enumerate(config['STARTPHIX']):
        matlab_script += f"startphix_{{{i+1}}} = {phi};\n"
    
    matlab_script += "\n% Distances\n"
    for i, dist in enumerate(config['int_dist']):
        if i < len(config['int_dist']) - 1:
            matlab_script += f"k{{{i+1}}} = {dist};\n"
        else:
            matlab_script += f"k{{{len(config['int_dist'])}}} = {dist}; % Distance to workpiece\n"
    
    matlab_script += "\n% Refractive indices\n"
    for i, n in enumerate(config['ref_ind']):
        matlab_script += f"n_{{{i+1}}} = {n};\n"
    
    # Add the main MATLAB simulation loop (from lines 43-231 of model.m)
    matlab_script += """
% Main simulation loop
workpiece_coords = [];

for iter=1:length(time)
    for i=1:wedgenum
        phix_{i} = startphix_{i};
    end
    
    % Gamma calculations
    for i=1:wedgenum
        gamma_{i} = mod(360*N_{i}*time(iter),360);
    end
    
    % Modified phi's
    for i=1:wedgenum
        n1 = [cosd(gamma_{i})*tand(phix_{i}); sind(gamma_{i})*tand(phix_{i}); -1];
        ny = [0; 1; 0];
        nx = [1; 0; 0];
        
        phix_{i} = 90 - acosd((dot(n1,nx))/(norm(nx)*norm(n1)));
        phiy_{i} = 90 - acosd((dot(n1,ny))/(norm(ny)*norm(n1)));
    end
    
    % Calculate cumulative distances
    sumk = 0;
    for i=1:wedgenum
        sumk = sumk + k{i};
        K{i} = sumk;
    end
    
    % X calculations
    x1 = rx_{1};
    x2 = rx_{1} + tand(thetax_{1});
    x3 = 0;
    z1 = 0;
    z2 = 1;
    z3 = k{1};
    
    if phix_{1} == 0
        x4 = 1;
        z4 = k{1};
    else
        x4 = cotd(phix_{1});
        z4 = k{1} + 1;
    end
    
    Px_{1} = ((x1*z2 - z1*x2)*(x3 - x4) - (x1 - x2)*(x3*z4 - z3*x4))/((x1 - x2)*(z3 - z4) - (z1 - z2)*(x3 - x4));
    Pz_{1} = ((x1*z2 - z1*x2)*(z3 - z4) - (z1 - z2)*(x3*z4 - z3*x4))/((x1 - x2)*(z3 - z4) - (z1 - z2)*(x3 - x4));
    
    K{wedgenum+1} = K{wedgenum} + k{wedgenum+1};
    phix_{wedgenum+1} = 0;
    
    for i=1:wedgenum
        N = [tand(phix_{i});0;-1];
        s_i = [tand(thetax_{i});0;1];
        zmeasure = [0;0;1];
        
        N = N/norm(N);
        s_i = s_i/norm(s_i);
        
        s_f = (n_{i}/n_{i+1})*(cross(N,cross(-N,s_i))) - N*((1 - ((n_{i}/n_{i+1}).^2)*dot(cross(N,s_i),cross(N,s_i))).^(1/2));
        thetax_{i+1} = ((abs(s_f(1,1)))/s_f(1,1))*acosd(dot(zmeasure,s_f)/(norm(s_f)*norm(zmeasure)));
        
        x1 = Px_{i};
        x2 = Px_{i} + tand(thetax_{i+1});
        x3 = 0;
        z1 = Pz_{i};
        z2 = Pz_{i} + 1;
        z3 = K{i+1};
        
        if phix_{i+1} == 0
            x4 = 1;
            z4 = K{i+1};
        else
            x4 = cotd(phix_{i+1});
            z4 = K{i+1} + 1;
        end
        
        Px_{i+1} = ((x1*z2 - z1*x2)*(x3 - x4) - (x1 - x2)*(x3*z4 - z3*x4))/((x1 - x2)*(z3 - z4) - (z1 - z2)*(x3 - x4));
        Pz_{i+1} = ((x1*z2 - z1*x2)*(z3 - z4) - (z1 - z2)*(x3*z4 - z3*x4))/((x1 - x2)*(z3 - z4) - (z1 - z2)*(x3 - x4));
    end
    
    % Y calculations
    y1 = ry_{1};
    y2 = ry_{1} + tand(thetay_{1});
    y3 = 0;
    z1 = 0;
    z2 = 1;
    z3 = k{1};
    
    if phiy_{1} == 0
        y4 = 1;
        z4 = k{1};
    else
        y4 = cotd(phiy_{1});
        z4 = k{1} + 1;
    end
    
    Py_{1} = ((y1*z2 - z1*y2)*(y3 - y4) - (y1 - y2)*(y3*z4 - z3*y4))/((y1 - y2)*(z3 - z4) - (z1 - z2)*(y3 - y4));
    
    phiy_{wedgenum+1} = 0;
    
    for i=1:wedgenum
        N = [tand(phiy_{i});0;-1];
        s_i = [tand(thetay_{i});0;1];
        zmeasure = [0;0;1];
        N = N/norm(N);
        s_i = s_i/norm(s_i);
        
        s_f = (n_{i}/n_{i+1})*(cross(N,cross(-N,s_i))) - N*((1 - ((n_{i}/n_{i+1}).^2)*dot(cross(N,s_i),cross(N,s_i))).^(1/2));
        thetay_{i+1} = ((abs(s_f(1,1)))/s_f(1,1))*acosd(dot(zmeasure,s_f)/(norm(s_f)*norm(zmeasure)));
        
        y1 = Py_{i};
        y2 = Py_{i} + tand(thetay_{i+1});
        y3 = 0;
        z1 = Pz_{i};
        z2 = Pz_{i} + 1;
        z3 = K{i+1};
        
        if phiy_{i+1} == 0
            y4 = 1;
            z4 = K{i+1};
        else
            y4 = cotd(phiy_{i+1});
            z4 = K{i+1} + 1;
        end
        
        Py_{i+1} = ((y1*z2 - z1*y2)*(y3 - y4) - (y1 - y2)*(y3*z4 - z3*y4))/((y1 - y2)*(z3 - z4) - (z1 - z2)*(y3 - y4));
    end
    
    % Z calculations
    for i=1:wedgenum+1
        gamma_{i} = mod(360*N_{min(i,wedgenum)}*time(iter),360);
        if i <= wedgenum
            coordz_val = K{i} + (Px_{i}*cosd(gamma_{i}) - Py_{i}*sind(gamma_{i}))*tand(phix_{i});
        else
            coordz_val = K{i};
        end
    end
    
    % Store final workpiece coordinate
    workpiece_coords = [workpiece_coords; Px_{wedgenum+1}, Py_{wedgenum+1}, K{wedgenum+1}];
end

% Save results
save('matlab_validation_results.mat', 'workpiece_coords', 'time');
fprintf('MATLAB validation complete. Results saved to matlab_validation_results.mat\\n');
fprintf('Final workpiece coordinates shape: %dx%d\\n', size(workpiece_coords));
"""
    
    # Write MATLAB script
    with open('validation_matlab.m', 'w') as f:
        f.write(matlab_script)
    
    print(f"Generated MATLAB validation script: validation_matlab.m")
    print("To run MATLAB validation:")
    print("1. Open MATLAB")
    print("2. Navigate to this directory")
    print("3. Run: validation_matlab")
    print("4. Re-run this Python script to compare results")
    
    return matlab_script

def load_matlab_results():
    """Load MATLAB results if available."""
    try:
        from scipy.io import loadmat
        data = loadmat('matlab_validation_results.mat')
        return data['workpiece_coords']
    except ImportError:
        print("Warning: scipy not available for loading .mat files")
        return None
    except FileNotFoundError:
        print("MATLAB results not found. Please run the MATLAB validation first.")
        return None

def compare_results(python_coords, matlab_coords, tolerance=1e-10):
    """Compare Python and MATLAB results with statistical analysis."""
    print("\n" + "="*60)
    print("VALIDATION RESULTS COMPARISON")
    print("="*60)
    
    if matlab_coords is None:
        print("❌ MATLAB results not available")
        return False
    
    print(f"Python coordinates shape: {python_coords.shape}")
    print(f"MATLAB coordinates shape: {matlab_coords.shape}")
    
    if python_coords.shape != matlab_coords.shape:
        print("❌ Shape mismatch between Python and MATLAB results")
        return False
    
    # Calculate differences
    differences = python_coords - matlab_coords
    abs_differences = np.abs(differences)
    max_diff = np.max(abs_differences)
    mean_diff = np.mean(abs_differences)
    std_diff = np.std(abs_differences)
    
    print(f"\nStatistical Analysis:")
    print(f"  Maximum absolute difference: {max_diff:.2e}")
    print(f"  Mean absolute difference:    {mean_diff:.2e}")
    print(f"  Std deviation of differences: {std_diff:.2e}")
    print(f"  Tolerance threshold:         {tolerance:.2e}")
    
    # Per-coordinate analysis
    coord_names = ['X', 'Y', 'Z']
    for i, name in enumerate(coord_names):
        coord_max_diff = np.max(np.abs(differences[:, i]))
        coord_mean_diff = np.mean(np.abs(differences[:, i]))
        print(f"  {name}-coordinate max diff:      {coord_max_diff:.2e}")
        print(f"  {name}-coordinate mean diff:     {coord_mean_diff:.2e}")
    
    # Validation decision
    validation_passed = max_diff < tolerance
    
    print(f"\n{'='*60}")
    if validation_passed:
        print("✅ VALIDATION PASSED!")
        print("Python implementation matches MATLAB reference within tolerance.")
    else:
        print("❌ VALIDATION FAILED!")
        print("Differences exceed tolerance threshold.")
        
        # Show first few problematic points
        print("\nFirst 5 coordinate differences:")
        print("Point | Python (X,Y,Z) | MATLAB (X,Y,Z) | Difference")
        print("-" * 70)
        for i in range(min(5, len(differences))):
            print(f"{i:5d} | {python_coords[i]} | {matlab_coords[i]} | {differences[i]}")
    
    return validation_passed

def main():
    """Main validation function."""
    print("🔬 Risley Prism Python vs MATLAB Validation")
    print("=" * 60)
    
    # Setup validation parameters
    config = setup_validation_parameters()
    print("Validation Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Update Python inputs
    update_python_inputs(config)
    
    # Check if MATLAB results exist, if not generate script
    matlab_coords = load_matlab_results()
    if matlab_coords is None:
        generate_matlab_reference()
        print("\n📝 MATLAB validation script generated.")
        print("Please run the MATLAB script and then re-run this validation.")
        return
    
    # Run Python simulation
    python_coords = run_python_simulation()
    
    # Compare results
    validation_passed = compare_results(python_coords, matlab_coords)
    
    # Save validation report
    report = {
        'timestamp': datetime.now().isoformat(),
        'config': config,
        'python_shape': python_coords.shape,
        'matlab_shape': matlab_coords.shape,
        'validation_passed': validation_passed,
        'max_difference': float(np.max(np.abs(python_coords - matlab_coords))) if matlab_coords is not None else None
    }
    
    with open('validation_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📊 Validation report saved to: validation_report.json")
    
    return validation_passed

if __name__ == "__main__":
    main()