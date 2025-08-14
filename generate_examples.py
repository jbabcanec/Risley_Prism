#!/usr/bin/env python3
"""
Risley Prism Simulation Examples Generator

This script generates multiple example simulations with different parameter sets
to demonstrate the capabilities of the Risley prism system.
"""

import os
import sys
import shutil
from datetime import datetime

# Import the main simulation and inputs
from model import main
import inputs

def backup_original_inputs():
    """Backup the original inputs.py file."""
    # Store original values in memory instead of file
    global original_inputs
    original_inputs = {
        'WEDGENUM': inputs.WEDGENUM,
        'N': inputs.N.copy(),
        'STARTPHIX': inputs.STARTPHIX.copy(),
        'STARTPHIY': inputs.STARTPHIY.copy(),
        'int_dist': inputs.int_dist.copy(),
        'ref_ind': inputs.ref_ind.copy(),
        'TIMELIM': inputs.TIMELIM,
        'INC': inputs.INC
    }
    print("Backed up original inputs in memory")

def restore_original_inputs():
    """Restore the original inputs from memory."""
    if 'original_inputs' in globals():
        for param, value in original_inputs.items():
            update_inputs(**{param: value})
        print("Restored original inputs")

def update_inputs(**kwargs):
    """Update inputs.py with new parameter values."""
    with open("inputs.py", "r") as f:
        content = f.read()
    
    for param, value in kwargs.items():
        if isinstance(value, list):
            value_str = str(value)
        elif isinstance(value, str):
            value_str = f"'{value}'"
        else:
            value_str = str(value)
        
        # Find and replace the parameter
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if line.strip().startswith(f"{param} ="):
                lines[i] = f"{param} = {value_str}"
                break
        content = '\n'.join(lines)
    
    with open("inputs.py", "w") as f:
        f.write(content)

def generate_rosette_pattern():
    """Generate complex rosette pattern with 4 wedges at different speeds."""
    print("\n" + "="*60)
    print("GENERATING ROSETTE PATTERN (4 WEDGES)")
    print("="*60)
    
    update_inputs(
        WEDGENUM=4,
        N=[1.0, 0.7, 1.3, 0.9],  # Different rotation speeds
        STARTPHIX=[8.0, 12.0, 5.0, 15.0],  # Varied wedge angles
        STARTPHIY=[0.0, 2.0, -1.0, 3.0],   # Some Y deflection
        int_dist=[4, 5, 4, 3, 6],  # Distance between wedges + to workpiece
        ref_ind=[1.1, 1.15, 1.2, 1.25, 1.3],  # Refractive indices
        TIMELIM=15.0,
        INC=180,
        plotit='off'
    )
    
    main(example_name="rosette_4wedge")
    print("✓ Rosette pattern example generated")

def generate_counter_spiral():
    """Generate counter-rotating spiral with 5 wedges."""
    print("\n" + "="*60)
    print("GENERATING COUNTER-SPIRAL (5 WEDGES)")
    print("="*60)
    
    update_inputs(
        WEDGENUM=5,
        N=[1.2, -0.8, 1.5, -0.6, 2.0],  # Alternating rotation directions
        STARTPHIX=[6.0, 10.0, 4.0, 12.0, 8.0],  
        STARTPHIY=[1.0, -2.0, 3.0, -1.0, 2.0],   # Complex Y patterns
        int_dist=[3, 4, 3, 5, 4, 7],  
        ref_ind=[1.05, 1.1, 1.15, 1.2, 1.25, 1.3],  
        TIMELIM=20.0,
        INC=200,
        plotit='off'
    )
    
    main(example_name="counter_spiral_5wedge")
    print("✓ Counter-spiral example generated")

def generate_harmonic_pattern():
    """Generate harmonic pattern with mathematical speed ratios."""
    print("\n" + "="*60)
    print("GENERATING HARMONIC PATTERN (6 WEDGES)")
    print("="*60)
    
    update_inputs(
        WEDGENUM=6,
        N=[1.0, 1.5, 2.0, 0.5, 3.0, 0.75],  # Harmonic ratios
        STARTPHIX=[5.0, 8.0, 12.0, 15.0, 6.0, 10.0],  
        STARTPHIY=[0.0, 1.0, 0.0, -1.0, 2.0, -0.5],   
        int_dist=[2, 3, 2, 4, 3, 2, 8],  
        ref_ind=[1.08, 1.12, 1.16, 1.2, 1.24, 1.28, 1.32],  
        TIMELIM=12.0,
        INC=150,
        plotit='off'
    )
    
    main(example_name="harmonic_6wedge")
    print("✓ Harmonic pattern example generated")

def generate_chaos_pattern():
    """Generate chaotic pattern with prime number speeds."""
    print("\n" + "="*60)
    print("GENERATING CHAOS PATTERN (5 WEDGES)")
    print("="*60)
    
    update_inputs(
        WEDGENUM=5,
        N=[1.1, 1.7, 2.3, 3.1, 0.7],  # Prime-like ratios for chaos
        STARTPHIX=[7.0, 13.0, 11.0, 17.0, 19.0],  # Prime-inspired angles
        STARTPHIY=[1.1, -1.7, 2.3, -0.7, 1.3],   
        int_dist=[3, 5, 2, 4, 6, 5],  
        ref_ind=[1.07, 1.13, 1.19, 1.23, 1.29, 1.31],  
        TIMELIM=25.0,  # Long time to see complex patterns
        INC=250,
        plotit='off'
    )
    
    main(example_name="chaos_5wedge")
    print("✓ Chaos pattern example generated")

def generate_precision_dense():
    """Generate ultra-dense precision pattern with many wedges."""
    print("\n" + "="*60)
    print("GENERATING PRECISION DENSE (4 WEDGES)")
    print("="*60)
    
    update_inputs(
        WEDGENUM=4,
        N=[0.2, 0.3, 0.25, 0.35],  # Very slow for precision
        STARTPHIX=[3.0, 4.0, 5.0, 6.0],  # Small angles for tight pattern
        STARTPHIY=[0.0, 0.5, -0.5, 1.0],   
        int_dist=[6, 6, 6, 6, 10],  # Longer optical path
        ref_ind=[1.05, 1.08, 1.11, 1.14, 1.17],  
        TIMELIM=30.0,  # Very long observation
        INC=300,       # High resolution
        plotit='off'
    )
    
    main(example_name="precision_dense_4wedge")
    print("✓ Precision dense pattern example generated")

def create_readme():
    """Create a README file for the examples."""
    readme_content = f"""# Risley Prism Simulation Examples

Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Examples Overview

This folder contains multiple example simulations demonstrating complex multi-wedge Risley prism systems with diverse scan patterns:

### 1. Rosette Pattern (4 Wedges)
- **Parameters**: N=[1.0, 0.7, 1.3, 0.9], Varied angles with Y-deflection
- **Description**: Complex rosette pattern from 4 wedges at different speeds
- **Key Features**: Flower-like patterns, balanced coverage, moderate complexity

### 2. Counter-Spiral (5 Wedges)
- **Parameters**: N=[1.2, -0.8, 1.5, -0.6, 2.0], Alternating rotation directions
- **Description**: Counter-rotating spiral with 5 wedges creating complex patterns
- **Key Features**: Spiral trajectories, high coverage density, counter-rotation effects

### 3. Harmonic Pattern (6 Wedges)
- **Parameters**: N=[1.0, 1.5, 2.0, 0.5, 3.0, 0.75], Mathematical speed ratios
- **Description**: Harmonic pattern with mathematical speed relationships
- **Key Features**: Symmetric patterns, periodic behavior, maximum wedge complexity

### 4. Chaos Pattern (5 Wedges)
- **Parameters**: N=[1.1, 1.7, 2.3, 3.1, 0.7], Prime-like ratios
- **Description**: Chaotic pattern with non-integer speed ratios
- **Key Features**: Aperiodic patterns, dense coverage, pseudo-random trajectories

### 5. Precision Dense (4 Wedges)
- **Parameters**: N=[0.2, 0.3, 0.25, 0.35], Very slow precision speeds
- **Description**: Ultra-dense precision pattern with high resolution
- **Key Features**: Dense point sampling, precise positioning, long observation time

## File Structure

Each example folder contains:
- `simulation_data.pkl` - Complete simulation state data
- `workpiece_projection.png` - Clean projection visualization
- `workpiece_projection_analysis.png` - Comprehensive analysis dashboard
- `workpiece_projections.csv` - Raw coordinate data with timestamps
- `workpiece_analysis.txt` - Statistical summary

## Usage

To reproduce any example:
1. Copy the parameters from this README to your `inputs.py`
2. Run `python model.py`
3. Check the `output/examples/` folder for timestamped results

## Analysis

Compare the different examples to understand:
- Effect of rotation speed on scan patterns
- Impact of wedge angles on coverage area
- Counter-rotation vs co-rotation behavior
- Precision vs speed trade-offs
"""
    
    readme_path = os.path.join("output", "examples", "README.md")
    with open(readme_path, "w") as f:
        f.write(readme_content)
    
    print(f"✓ README created at {readme_path}")

def main_generate():
    """Main function to generate all examples."""
    print("Starting Risley Prism Examples Generation")
    print("="*60)
    
    # Backup original inputs
    backup_original_inputs()
    
    try:
        # Generate all examples with complex multi-wedge patterns
        generate_rosette_pattern()
        generate_counter_spiral()
        generate_harmonic_pattern()
        generate_chaos_pattern()
        generate_precision_dense()
        
        # Create documentation
        create_readme()
        
        print("\n" + "="*60)
        print("ALL COMPLEX MULTI-WEDGE EXAMPLES GENERATED!")
        print("="*60)
        print("Check the output/examples/ folder for all simulation results.")
        print("Each example showcases different scan patterns from 4-6 wedge systems.")
        
    finally:
        # Always restore original inputs
        restore_original_inputs()

if __name__ == "__main__":
    main_generate()