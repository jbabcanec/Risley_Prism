# Risley Prism Simulation Examples

Generated on: 2025-08-13 23:26:28

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
