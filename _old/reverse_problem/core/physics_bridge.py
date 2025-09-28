#!/usr/bin/env python3
"""
Physics Bridge - Connects real Risley prism physics to the reverse problem solver.

This module provides the interface between the actual physics simulation
(model.py) and the reverse problem training/prediction system.
"""

import sys
import os
import numpy as np
from typing import List, Dict, Tuple, Optional
import copy

# Add parent directories to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)
sys.path.insert(0, grandparent_dir)


class RisleyPhysicsEngine:
    """Bridge to real Risley prism physics simulation."""

    def __init__(self):
        """Initialize the physics engine."""
        self.original_state = None
        self._save_original_state()

    def _save_original_state(self):
        """Save the original state of inputs module."""
        try:
            import inputs
            self.original_state = {
                'WEDGENUM': inputs.WEDGENUM,
                'N': copy.deepcopy(inputs.N),
                'STARTPHIX': copy.deepcopy(inputs.STARTPHIX),
                'STARTPHIY': copy.deepcopy(inputs.STARTPHIY),
                'STARTTHETAX': inputs.STARTTHETAX,
                'STARTTHETAY': inputs.STARTTHETAY,
                'int_dist': copy.deepcopy(inputs.int_dist),
                'ref_ind': copy.deepcopy(inputs.ref_ind),
                'TIMELIM': inputs.TIMELIM,
                'INC': inputs.INC,
                'plotit': inputs.plotit,
                'printit': inputs.printit
            }
        except ImportError:
            print("Warning: Could not import inputs module for state saving")
            self.original_state = None

    def _restore_original_state(self):
        """Restore the original state of inputs module."""
        if self.original_state is not None:
            try:
                import inputs
                for key, value in self.original_state.items():
                    setattr(inputs, key, value)
            except ImportError:
                pass

    def simulate_real_pattern(self, wedge_count: int, rotation_speeds: List[float],
                             phi_x: List[float], phi_y: List[float],
                             time_duration: float = 2.0, time_points: int = 100,
                             distances: Optional[List[float]] = None,
                             refractive_indices: Optional[List[float]] = None) -> np.ndarray:
        """
        Generate pattern using real Risley prism physics.

        Args:
            wedge_count: Number of wedges in the system
            rotation_speeds: Rotation speeds in Hz for each wedge
            phi_x: X-axis wedge angles in degrees
            phi_y: Y-axis wedge angles in degrees
            time_duration: Total simulation time in seconds
            time_points: Number of time samples to generate
            distances: Optional distances between components
            refractive_indices: Optional refractive indices

        Returns:
            np.ndarray: Pattern of shape (time_points, 2) with x,y positions
        """
        try:
            import inputs
            from model import initialize, update_angles_and_vectors
            from calcs.init_coords import initialize_coordinates
            from calcs.calc_proj_coord import calc_proj_coord
            from calcs.calc_z_coord import calc_z_coord

            # Save current state
            self._save_original_state()

            # Configure inputs for our simulation
            inputs.WEDGENUM = wedge_count
            inputs.N = rotation_speeds[:wedge_count]
            inputs.STARTPHIX = phi_x[:wedge_count]
            inputs.STARTPHIY = phi_y[:wedge_count]
            inputs.STARTTHETAX = 10.0  # Default initial beam angle
            inputs.STARTTHETAY = 5.0   # Default initial beam angle
            inputs.TIMELIM = time_duration
            inputs.INC = time_points
            inputs.plotit = 'off'
            inputs.printit = 'off'

            # Set distances - MUST be wedge_count + 1 elements
            if distances is not None:
                inputs.int_dist = distances[:wedge_count + 1]
            else:
                # Distances between each component (including to workpiece)
                inputs.int_dist = [6.0] * wedge_count + [100.0]

            # Set refractive indices - MUST be wedge_count + 1 elements
            if refractive_indices is not None:
                inputs.ref_ind = refractive_indices[:wedge_count + 1]
            else:
                # Refractive indices for each medium transition
                inputs.ref_ind = [1.0] + [1.5] * wedge_count

            # Initialize system
            phix, phiy, thetax, thetay, gamma, cum_dist = initialize()
            time = np.linspace(0, time_duration, time_points)

            # Storage for workpiece positions
            workpiece_positions = []

            # Run simulation at each time point
            for idx, current_time in enumerate(time):
                # Update angles for rotating wedges
                update_angles_and_vectors(current_time, phix, phiy, gamma)

                # Initialize coordinates
                ((orig_coordx, new_coordx), (orig_coordy, new_coordy),
                 (orig_coordz), PX0, PY0, PZ_X0, PZ_Y0) = initialize_coordinates(
                    inputs.RX, inputs.RY, thetax, thetay, phix, phiy, inputs.int_dist)

                # Calculate projections through wedges
                x_coords, new_thetax = calc_proj_coord(
                    str(idx), orig_coordx, new_coordx, phix, cum_dist,
                    thetax, PX0, PZ_X0, 'x')
                y_coords, new_thetay = calc_proj_coord(
                    str(idx), orig_coordy, new_coordy, phiy, cum_dist,
                    thetay, PY0, PZ_Y0, 'y')

                # Extract final position at workpiece
                if str(idx) in x_coords and x_coords[str(idx)]:
                    # Get the last coordinate (at workpiece)
                    x_coord_list = x_coords[str(idx)]
                    y_coord_list = y_coords[str(idx)]

                    if x_coord_list and y_coord_list:
                        final_x_coord = x_coord_list[-1]
                        final_y_coord = y_coord_list[-1]

                        # Extract x value (first component if list/tuple)
                        if isinstance(final_x_coord, (list, tuple, np.ndarray)):
                            final_x = float(final_x_coord[0])
                        else:
                            final_x = float(final_x_coord)

                        # Extract y value (second component if list/tuple)
                        if isinstance(final_y_coord, (list, tuple, np.ndarray)):
                            if len(final_y_coord) > 1:
                                final_y = float(final_y_coord[1])
                            else:
                                final_y = float(final_y_coord[0])
                        else:
                            final_y = float(final_y_coord)

                        workpiece_positions.append([final_x, final_y])

                    # Update theta for next iteration
                    thetax = new_thetax
                    thetay = new_thetay

            # Convert to numpy array
            pattern = np.array(workpiece_positions)

            # Ensure we have the right number of points
            if len(pattern) != time_points:
                # Interpolate or pad as needed
                if len(pattern) > 0:
                    # Interpolate to get exact number of points
                    old_indices = np.linspace(0, 1, len(pattern))
                    new_indices = np.linspace(0, 1, time_points)

                    new_pattern = np.zeros((time_points, 2))
                    new_pattern[:, 0] = np.interp(new_indices, old_indices, pattern[:, 0])
                    new_pattern[:, 1] = np.interp(new_indices, old_indices, pattern[:, 1])
                    pattern = new_pattern
                else:
                    # No valid points, return zeros
                    pattern = np.zeros((time_points, 2))

            return pattern

        except Exception as e:
            print(f"Error in real physics simulation: {e}")
            # Fall back to simplified pattern
            return self._fallback_pattern(wedge_count, rotation_speeds, phi_x, phi_y, time_points)

        finally:
            # Always restore original state
            self._restore_original_state()

    def _fallback_pattern(self, wedge_count: int, rotation_speeds: List[float],
                         phi_x: List[float], phi_y: List[float],
                         time_points: int) -> np.ndarray:
        """Generate fallback pattern if real physics fails."""
        print("Warning: Using fallback pattern generation")

        t = np.linspace(0, 2.0, time_points)
        x_pattern = np.zeros(time_points)
        y_pattern = np.zeros(time_points)

        # Simple superposition of rotations
        for i in range(wedge_count):
            if i < len(rotation_speeds) and i < len(phi_x) and i < len(phi_y):
                freq = rotation_speeds[i]
                amp_x = np.abs(phi_x[i]) / 15.0
                amp_y = np.abs(phi_y[i]) / 15.0

                x_pattern += amp_x * np.cos(2 * np.pi * freq * t)
                y_pattern += amp_y * np.sin(2 * np.pi * freq * t)

        pattern = np.column_stack([x_pattern, y_pattern])
        return pattern

    def validate_parameters(self, params: Dict) -> bool:
        """Validate that parameters are physically realizable."""
        try:
            wedge_count = params.get('wedgenum', 0)

            # Check wedge count
            if wedge_count < 1 or wedge_count > 6:
                return False

            # Check rotation speeds
            speeds = params.get('rotation_speeds', [])
            if len(speeds) != wedge_count:
                return False
            if any(abs(s) > 10.0 for s in speeds):  # Max 10 Hz
                return False

            # Check angles
            phi_x = params.get('phi_x', [])
            phi_y = params.get('phi_y', [])
            if len(phi_x) != wedge_count or len(phi_y) != wedge_count:
                return False
            if any(abs(angle) > 45.0 for angle in phi_x + phi_y):  # Max 45 degrees
                return False

            return True

        except Exception:
            return False


def test_physics_bridge():
    """Test the physics bridge functionality."""
    print("Testing Physics Bridge")
    print("=" * 60)

    engine = RisleyPhysicsEngine()

    # Test case 1: Single wedge
    print("\nTest 1: Single wedge")
    pattern = engine.simulate_real_pattern(
        wedge_count=1,
        rotation_speeds=[1.0],
        phi_x=[10.0],
        phi_y=[5.0],
        time_points=50
    )
    print(f"  Pattern shape: {pattern.shape}")
    print(f"  X range: [{pattern[:,0].min():.3f}, {pattern[:,0].max():.3f}]")
    print(f"  Y range: [{pattern[:,1].min():.3f}, {pattern[:,1].max():.3f}]")

    # Test case 2: Two wedges
    print("\nTest 2: Two wedges")
    pattern = engine.simulate_real_pattern(
        wedge_count=2,
        rotation_speeds=[1.0, -1.5],
        phi_x=[15.0, -10.0],
        phi_y=[5.0, 8.0],
        time_points=50
    )
    print(f"  Pattern shape: {pattern.shape}")
    print(f"  X range: [{pattern[:,0].min():.3f}, {pattern[:,0].max():.3f}]")
    print(f"  Y range: [{pattern[:,1].min():.3f}, {pattern[:,1].max():.3f}]")

    # Test parameter validation
    print("\nTest 3: Parameter validation")
    valid_params = {
        'wedgenum': 2,
        'rotation_speeds': [1.0, -2.0],
        'phi_x': [10.0, -15.0],
        'phi_y': [5.0, 8.0]
    }
    print(f"  Valid params: {engine.validate_parameters(valid_params)}")

    invalid_params = {
        'wedgenum': 2,
        'rotation_speeds': [1.0],  # Wrong length
        'phi_x': [10.0, -15.0],
        'phi_y': [5.0, 8.0]
    }
    print(f"  Invalid params: {engine.validate_parameters(invalid_params)}")

    print("\n" + "=" * 60)
    print("Physics Bridge Test Complete")


if __name__ == "__main__":
    test_physics_bridge()