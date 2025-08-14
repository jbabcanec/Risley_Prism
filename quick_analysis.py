"""
Quick Analysis Script for Risley Prism Simulation
Runs the simulation with minimal output and shows workpiece projection
"""

import numpy as np
import os
import pickle
from model import main
from utils.analysis import analyze_scan_pattern, calculate_scan_efficiency

def run_quick_analysis():
    print("🔬 RISLEY PRISM SIMULATION - QUICK ANALYSIS")
    print("="*60)
    
    # Temporarily disable verbose output
    import inputs
    original_printit = inputs.printit
    inputs.printit = 'off'
    
    # Run the simulation
    print("Running simulation...")
    main()
    
    # Restore original setting
    inputs.printit = original_printit
    
    # Load and analyze the results
    output_directory = "output"
    filepath = os.path.join(output_directory, "simulation_data.pkl")
    
    with open(filepath, 'rb') as file:
        data = pickle.load(file)
    
    Laser_coords = data['Laser_coords']
    
    # Perform quick analysis
    print("\n🎯 QUICK SCAN PATTERN ANALYSIS")
    print("="*50)
    
    # Extract workpiece positions
    workpiece_positions = []
    for key, coords in Laser_coords.items():
        if coords:
            final_pos = coords[-1]  # Last position (at workpiece)
            workpiece_positions.append([final_pos[0], final_pos[1]])
    
    if workpiece_positions:
        workpiece_positions = np.array(workpiece_positions)
        x_vals = workpiece_positions[:, 0]
        y_vals = workpiece_positions[:, 1]
        
        # Calculate basic statistics
        x_range = np.max(x_vals) - np.min(x_vals)
        y_range = np.max(y_vals) - np.min(y_vals)
        scan_area = x_range * y_range
        center_x, center_y = np.mean(x_vals), np.mean(y_vals)
        
        # Calculate distances from center
        distances = np.sqrt((x_vals - center_x)**2 + (y_vals - center_y)**2)
        effective_diameter = 2 * np.percentile(distances, 95)
        
        print(f"📊 SCAN STATISTICS:")
        print(f"   Scan Area: {scan_area:.2f} square units")
        print(f"   X Range: {x_range:.2f} units")
        print(f"   Y Range: {y_range:.2f} units")
        print(f"   Center: ({center_x:.2f}, {center_y:.2f})")
        print(f"   Effective Diameter (95%): {effective_diameter:.2f} units")
        print(f"   Max Displacement: {np.max(distances):.2f} units")
        print(f"   Total Positions: {len(workpiece_positions)}")
        
        print(f"\n📍 SAMPLE WORKPIECE POSITIONS (every 10th point):")
        print("   Time(s) |    X    |    Y    | Distance")
        print("   " + "-"*40)
        for i in range(0, len(workpiece_positions), 10):
            t = i * 0.1
            x, y = workpiece_positions[i]
            dist = np.sqrt(x**2 + y**2)
            print(f"   {t:6.1f}  | {x:7.2f} | {y:7.2f} | {dist:8.2f}")
        
        # Calculate scan efficiency
        efficiency = calculate_scan_efficiency(Laser_coords)
        if 'error' not in efficiency:
            print(f"\n⚡ EFFICIENCY METRICS:")
            print(f"   Path Length: {efficiency['total_path_length']:.2f} units")
            print(f"   Efficiency Ratio: {efficiency['efficiency_ratio']:.4f}")
            print(f"   Points/Area: {efficiency['points_per_unit_area']:.2f}")
    
    print("\n" + "="*60)
    print("✅ Analysis complete! Data saved to output/simulation_data.pkl")
    print("💡 Tip: Set plotit='on' in inputs.py for full 3D visualization")

if __name__ == "__main__":
    run_quick_analysis()