#!/usr/bin/env python3
"""
Advanced Constraint Handling for Risley Prism Systems

Physics-based constraints and validation for parameter optimization.
"""

import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass

@dataclass
class ConstraintViolation:
    """Record of constraint violation."""
    parameter: str
    violation_type: str
    severity: float
    value: float
    limit: float

class PhysicsConstraints:
    """Advanced physics-based constraints for Risley prism systems."""
    
    def __init__(self, wedge_count: int):
        self.wedge_count = wedge_count
        
        # Hard physical limits
        self.hard_limits = {
            'rotation_speeds': (-10.0, 10.0),  # rad/s
            'phi_x': (-30.0, 30.0),           # degrees
            'phi_y': (-30.0, 30.0),           # degrees
            'distances': (0.5, 20.0),         # mm
            'refractive_indices': (1.3, 1.8)  # glass range
        }
        
        # Soft operational limits (preferred ranges)
        self.soft_limits = {
            'rotation_speeds': (-5.0, 5.0),
            'phi_x': (-15.0, 15.0),
            'phi_y': (-15.0, 15.0),
            'distances': (1.0, 10.0),
            'refractive_indices': (1.4, 1.6)
        }
        
        # Physics-based relationships
        self.max_total_deflection = 50.0  # degrees
        self.min_separation_ratio = 0.1   # minimum distance ratio
        
    def validate_parameters(self, params: Dict) -> Tuple[bool, List[ConstraintViolation]]:
        """
        Comprehensive parameter validation.
        
        Returns:
            is_valid: Boolean indicating if all hard constraints satisfied
            violations: List of all constraint violations
        """
        violations = []
        
        # Check basic parameter bounds
        violations.extend(self._check_parameter_bounds(params))
        
        # Check physics relationships
        violations.extend(self._check_physics_relationships(params))
        
        # Check manufacturing constraints
        violations.extend(self._check_manufacturing_constraints(params))
        
        # Determine if configuration is valid (no hard violations)
        hard_violations = [v for v in violations if v.severity >= 1.0]
        is_valid = len(hard_violations) == 0
        
        return is_valid, violations
    
    def _check_parameter_bounds(self, params: Dict) -> List[ConstraintViolation]:
        """Check parameter bounds violations."""
        violations = []
        
        for param_name, limits in self.hard_limits.items():
            if param_name not in params:
                continue
                
            values = np.array(params[param_name])
            
            # Skip boundary conditions for distances and refractive indices
            if param_name == 'distances':
                check_values = values[1:] if len(values) > 1 else values
            elif param_name == 'refractive_indices':
                check_values = values[1:-1] if len(values) > 2 else values
            else:
                check_values = values
            
            # Check lower bounds
            under_violations = check_values < limits[0]
            for i, is_violation in enumerate(under_violations):
                if is_violation:
                    violations.append(ConstraintViolation(
                        parameter=param_name,
                        violation_type='lower_bound',
                        severity=1.0,  # Hard violation
                        value=float(check_values[i]),
                        limit=limits[0]
                    ))
            
            # Check upper bounds
            over_violations = check_values > limits[1]
            for i, is_violation in enumerate(over_violations):
                if is_violation:
                    violations.append(ConstraintViolation(
                        parameter=param_name,
                        violation_type='upper_bound',
                        severity=1.0,  # Hard violation
                        value=float(check_values[i]),
                        limit=limits[1]
                    ))
            
            # Check soft limits (warnings)
            if param_name in self.soft_limits:
                soft_limits = self.soft_limits[param_name]
                
                under_soft = check_values < soft_limits[0]
                for i, is_violation in enumerate(under_soft):
                    if is_violation and not (check_values[i] < limits[0]):  # Not already a hard violation
                        violations.append(ConstraintViolation(
                            parameter=param_name,
                            violation_type='soft_lower',
                            severity=0.5,  # Soft violation
                            value=float(check_values[i]),
                            limit=soft_limits[0]
                        ))
                
                over_soft = check_values > soft_limits[1]
                for i, is_violation in enumerate(over_soft):
                    if is_violation and not (check_values[i] > limits[1]):  # Not already a hard violation
                        violations.append(ConstraintViolation(
                            parameter=param_name,
                            violation_type='soft_upper',
                            severity=0.5,  # Soft violation
                            value=float(check_values[i]),
                            limit=soft_limits[1]
                        ))
        
        return violations
    
    def _check_physics_relationships(self, params: Dict) -> List[ConstraintViolation]:
        """Check physics-based relationships between parameters."""
        violations = []
        
        # Total deflection constraint
        if 'phi_x' in params and 'phi_y' in params:
            phi_x = np.array(params['phi_x'])
            phi_y = np.array(params['phi_y'])
            
            total_deflections = np.sqrt(phi_x**2 + phi_y**2)
            max_deflection = np.max(total_deflections)
            
            if max_deflection > self.max_total_deflection:
                violations.append(ConstraintViolation(
                    parameter='total_deflection',
                    violation_type='physics_limit',
                    severity=1.0,
                    value=float(max_deflection),
                    limit=self.max_total_deflection
                ))
        
        # Minimum separation constraint
        if 'distances' in params:
            distances = np.array(params['distances'])
            if len(distances) > 1:
                separations = distances[1:]
                min_separation = np.min(separations)
                max_separation = np.max(separations)
                
                if max_separation > 0 and min_separation / max_separation < self.min_separation_ratio:
                    violations.append(ConstraintViolation(
                        parameter='separation_ratio',
                        violation_type='physics_relationship',
                        severity=0.7,
                        value=float(min_separation / max_separation),
                        limit=self.min_separation_ratio
                    ))
        
        # Refractive index consistency
        if 'refractive_indices' in params:
            ri = np.array(params['refractive_indices'])
            if len(ri) > 2:
                wedge_ri = ri[1:-1]
                ri_variation = np.max(wedge_ri) - np.min(wedge_ri)
                
                # Large variations in refractive index are unusual
                if ri_variation > 0.3:
                    violations.append(ConstraintViolation(
                        parameter='refractive_index_variation',
                        violation_type='consistency',
                        severity=0.3,
                        value=float(ri_variation),
                        limit=0.3
                    ))
        
        return violations
    
    def _check_manufacturing_constraints(self, params: Dict) -> List[ConstraintViolation]:
        """Check manufacturing and practical constraints."""
        violations = []
        
        # Wedge angle precision limits
        if 'phi_x' in params and 'phi_y' in params:
            phi_x = np.array(params['phi_x'])
            phi_y = np.array(params['phi_y'])
            
            # Very small angles are hard to manufacture precisely
            min_significant_angle = 0.1  # degrees
            
            small_x = np.abs(phi_x) < min_significant_angle
            small_y = np.abs(phi_y) < min_significant_angle
            
            for i, (is_small_x, is_small_y) in enumerate(zip(small_x, small_y)):
                if is_small_x and is_small_y:
                    violations.append(ConstraintViolation(
                        parameter=f'wedge_{i}_angles',
                        violation_type='manufacturing_precision',
                        severity=0.2,
                        value=float(np.sqrt(phi_x[i]**2 + phi_y[i]**2)),
                        limit=min_significant_angle
                    ))
        
        # Rotation speed practicality
        if 'rotation_speeds' in params:
            speeds = np.array(params['rotation_speeds'])
            
            # Very high speeds may be impractical
            practical_limit = 3.0  # rad/s
            high_speeds = np.abs(speeds) > practical_limit
            
            for i, is_high in enumerate(high_speeds):
                if is_high:
                    violations.append(ConstraintViolation(
                        parameter=f'rotation_speed_{i}',
                        violation_type='practical_limit',
                        severity=0.4,
                        value=float(abs(speeds[i])),
                        limit=practical_limit
                    ))
        
        return violations
    
    def calculate_constraint_penalty(self, violations: List[ConstraintViolation]) -> float:
        """Calculate penalty for constraint violations."""
        total_penalty = 0.0
        
        for violation in violations:
            # Exponential penalty based on severity and magnitude
            magnitude = abs(violation.value - violation.limit) / abs(violation.limit)
            penalty = violation.severity * magnitude**2 * 10.0
            total_penalty += penalty
        
        return total_penalty
    
    def suggest_repairs(self, params: Dict, violations: List[ConstraintViolation]) -> Dict:
        """Suggest parameter repairs for constraint violations."""
        repaired_params = {k: np.array(v).copy() for k, v in params.items()}
        
        for violation in violations:
            if violation.severity >= 1.0:  # Only repair hard violations
                param_name = violation.parameter
                
                if param_name in repaired_params:
                    values = repaired_params[param_name]
                    
                    if violation.violation_type == 'lower_bound':
                        # Clamp to minimum value
                        repaired_params[param_name] = np.maximum(values, violation.limit)
                    elif violation.violation_type == 'upper_bound':
                        # Clamp to maximum value
                        repaired_params[param_name] = np.minimum(values, violation.limit)
        
        # Convert back to lists
        return {k: v.tolist() if hasattr(v, 'tolist') else v for k, v in repaired_params.items()}