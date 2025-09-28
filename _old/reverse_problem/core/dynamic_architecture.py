#!/usr/bin/env python3
"""
DYNAMIC ARCHITECTURE SELECTION - Adaptive AI system selection

Revolutionary adaptive system that:
- Analyzes pattern characteristics in real-time
- Selects optimal AI architecture for each problem
- Dynamically switches between neural networks, optimizers, and strategies
- Learns from past performance to improve selection
- Provides meta-learning across different problem types
"""

import numpy as np
import time
import json
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import pickle

@dataclass
class ArchitecturePerformance:
    """Tracks performance of an architecture on specific problem types."""
    architecture_name: str
    accuracy: float
    speed: float
    confidence: float
    problem_type: str
    pattern_features: Dict
    timestamp: datetime

@dataclass
class ProblemSignature:
    """Signature of a problem for architecture selection."""
    complexity: float
    frequency_content: float
    pattern_length: int
    amplitude: float
    wedge_count_hint: int
    noise_level: float
    temporal_dynamics: float

class PatternAnalyzer:
    """Advanced pattern analyzer for problem characterization."""
    
    def __init__(self):
        self.feature_extractors = {
            'statistical': self._extract_statistical_features,
            'frequency': self._extract_frequency_features,
            'geometric': self._extract_geometric_features,
            'temporal': self._extract_temporal_features,
            'physics': self._extract_physics_features
        }
    
    def analyze_pattern(self, pattern: np.ndarray) -> ProblemSignature:
        """Comprehensive pattern analysis for architecture selection."""
        
        if len(pattern) < 2:
            return ProblemSignature(
                complexity=0.5, frequency_content=0.0, pattern_length=len(pattern),
                amplitude=0.0, wedge_count_hint=1, noise_level=0.5, temporal_dynamics=0.0
            )
        
        # Extract all feature types
        all_features = {}
        for feature_type, extractor in self.feature_extractors.items():
            try:
                features = extractor(pattern)
                all_features[feature_type] = features
            except Exception as e:
                print(f"   ⚠️ Feature extraction failed for {feature_type}: {e}")
                all_features[feature_type] = {}
        
        # Combine features into problem signature
        signature = self._create_problem_signature(all_features, pattern)
        return signature
    
    def _extract_statistical_features(self, pattern: np.ndarray) -> Dict:
        """Extract statistical features from pattern."""
        features = {}
        
        # Basic statistics
        features['mean_x'] = float(np.mean(pattern[:, 0]))
        features['mean_y'] = float(np.mean(pattern[:, 1]))
        features['std_x'] = float(np.std(pattern[:, 0]))
        features['std_y'] = float(np.std(pattern[:, 1]))
        
        # Distribution properties
        features['skewness_x'] = float(self._skewness(pattern[:, 0]))
        features['skewness_y'] = float(self._skewness(pattern[:, 1]))
        features['kurtosis_x'] = float(self._kurtosis(pattern[:, 0]))
        features['kurtosis_y'] = float(self._kurtosis(pattern[:, 1]))
        
        # Range and extremes
        features['range_x'] = float(np.ptp(pattern[:, 0]))
        features['range_y'] = float(np.ptp(pattern[:, 1]))
        features['max_distance_from_origin'] = float(np.max(np.linalg.norm(pattern, axis=1)))
        
        return features
    
    def _extract_frequency_features(self, pattern: np.ndarray) -> Dict:
        """Extract frequency domain features."""
        features = {}
        
        # FFT analysis
        fft_x = np.abs(np.fft.fft(pattern[:, 0]))
        fft_y = np.abs(np.fft.fft(pattern[:, 1]))
        
        # Dominant frequencies
        features['dominant_freq_x'] = float(np.argmax(fft_x[:len(fft_x)//2]))
        features['dominant_freq_y'] = float(np.argmax(fft_y[:len(fft_y)//2]))
        
        # Frequency distribution
        freq_energy_x = np.sum(fft_x[:len(fft_x)//2]**2)
        freq_energy_y = np.sum(fft_y[:len(fft_y)//2]**2)
        features['frequency_energy_x'] = float(freq_energy_x)
        features['frequency_energy_y'] = float(freq_energy_y)
        
        # Spectral centroid (frequency "center of mass")
        freqs = np.arange(len(fft_x)//2)
        if freq_energy_x > 0:
            features['spectral_centroid_x'] = float(np.sum(freqs * fft_x[:len(fft_x)//2]**2) / freq_energy_x)
        else:
            features['spectral_centroid_x'] = 0.0
        
        if freq_energy_y > 0:
            features['spectral_centroid_y'] = float(np.sum(freqs * fft_y[:len(fft_y)//2]**2) / freq_energy_y)
        else:
            features['spectral_centroid_y'] = 0.0
        
        return features
    
    def _extract_geometric_features(self, pattern: np.ndarray) -> Dict:
        """Extract geometric features from pattern."""
        features = {}
        
        # Convex hull area (if enough points)
        try:
            if len(pattern) >= 3:
                from scipy.spatial import ConvexHull
                hull = ConvexHull(pattern)
                features['convex_hull_area'] = float(hull.volume)  # In 2D, volume is area
                features['convex_hull_ratio'] = float(len(hull.vertices) / len(pattern))
            else:
                features['convex_hull_area'] = 0.0
                features['convex_hull_ratio'] = 1.0
        except:
            features['convex_hull_area'] = 0.0
            features['convex_hull_ratio'] = 1.0
        
        # Pattern compactness
        distances_from_center = np.linalg.norm(pattern - np.mean(pattern, axis=0), axis=1)
        features['compactness'] = float(np.std(distances_from_center) / (np.mean(distances_from_center) + 1e-6))
        
        # Circularity measure
        if len(pattern) > 1:
            center = np.mean(pattern, axis=0)
            radii = np.linalg.norm(pattern - center, axis=1)
            features['circularity'] = float(1.0 / (1.0 + np.std(radii) / (np.mean(radii) + 1e-6)))
        else:
            features['circularity'] = 1.0
        
        return features
    
    def _extract_temporal_features(self, pattern: np.ndarray) -> Dict:
        """Extract temporal dynamics features."""
        features = {}
        
        if len(pattern) < 2:
            return {'velocity_mean': 0.0, 'acceleration_mean': 0.0, 'jerk_mean': 0.0}
        
        # Velocity (first derivative)
        velocity = np.diff(pattern, axis=0)
        velocity_magnitude = np.linalg.norm(velocity, axis=1)
        features['velocity_mean'] = float(np.mean(velocity_magnitude))
        features['velocity_std'] = float(np.std(velocity_magnitude))
        features['velocity_max'] = float(np.max(velocity_magnitude))
        
        # Acceleration (second derivative)
        if len(velocity) > 1:
            acceleration = np.diff(velocity, axis=0)
            acceleration_magnitude = np.linalg.norm(acceleration, axis=1)
            features['acceleration_mean'] = float(np.mean(acceleration_magnitude))
            features['acceleration_std'] = float(np.std(acceleration_magnitude))
            features['acceleration_max'] = float(np.max(acceleration_magnitude))
        else:
            features['acceleration_mean'] = 0.0
            features['acceleration_std'] = 0.0
            features['acceleration_max'] = 0.0
        
        # Jerk (third derivative)
        if len(acceleration_magnitude) > 1:
            jerk = np.diff(acceleration_magnitude)
            features['jerk_mean'] = float(np.mean(np.abs(jerk)))
            features['jerk_std'] = float(np.std(jerk))
        else:
            features['jerk_mean'] = 0.0
            features['jerk_std'] = 0.0
        
        return features
    
    def _extract_physics_features(self, pattern: np.ndarray) -> Dict:
        """Extract physics-inspired features."""
        features = {}
        
        # Energy-like measures
        kinetic_energy = np.sum(pattern**2)
        features['total_energy'] = float(kinetic_energy)
        
        # Moment of inertia (rotational properties)
        center = np.mean(pattern, axis=0)
        r_squared = np.sum((pattern - center)**2, axis=1)
        features['moment_of_inertia'] = float(np.mean(r_squared))
        
        # Angular features
        if len(pattern) > 1:
            angles = np.arctan2(pattern[:, 1] - center[1], pattern[:, 0] - center[0])
            angle_changes = np.diff(angles)
            
            # Handle angle wraparound
            angle_changes = np.mod(angle_changes + np.pi, 2*np.pi) - np.pi
            
            features['angular_velocity_mean'] = float(np.mean(np.abs(angle_changes)))
            features['angular_velocity_std'] = float(np.std(angle_changes))
            features['total_rotation'] = float(np.sum(np.abs(angle_changes)))
        else:
            features['angular_velocity_mean'] = 0.0
            features['angular_velocity_std'] = 0.0
            features['total_rotation'] = 0.0
        
        return features
    
    def _create_problem_signature(self, all_features: Dict, pattern: np.ndarray) -> ProblemSignature:
        """Create problem signature from extracted features."""
        
        # Complexity measure (combination of multiple factors)
        complexity_factors = []
        
        # Statistical complexity
        if 'statistical' in all_features:
            stat_features = all_features['statistical']
            stat_complexity = (stat_features.get('std_x', 0) + stat_features.get('std_y', 0)) / 2
            complexity_factors.append(stat_complexity)
        
        # Geometric complexity
        if 'geometric' in all_features:
            geom_features = all_features['geometric']
            geom_complexity = geom_features.get('compactness', 0.5)
            complexity_factors.append(geom_complexity)
        
        # Temporal complexity
        if 'temporal' in all_features:
            temp_features = all_features['temporal']
            temp_complexity = temp_features.get('acceleration_std', 0)
            complexity_factors.append(temp_complexity)
        
        overall_complexity = np.mean(complexity_factors) if complexity_factors else 0.5
        
        # Frequency content
        frequency_content = 0.0
        if 'frequency' in all_features:
            freq_features = all_features['frequency']
            frequency_content = (freq_features.get('frequency_energy_x', 0) + 
                               freq_features.get('frequency_energy_y', 0)) / 2
        
        # Amplitude
        amplitude = np.sqrt(np.mean(pattern**2)) if len(pattern) > 0 else 0.0
        
        # Noise level estimation (high frequency content relative to signal)
        noise_level = 0.5
        if 'frequency' in all_features:
            freq_features = all_features['frequency']
            total_energy = freq_features.get('frequency_energy_x', 0) + freq_features.get('frequency_energy_y', 0)
            if total_energy > 0:
                high_freq_energy = total_energy * 0.2  # Assume top 20% are noise frequencies
                noise_level = min(1.0, high_freq_energy / total_energy)
        
        # Temporal dynamics
        temporal_dynamics = 0.0
        if 'temporal' in all_features:
            temp_features = all_features['temporal']
            temporal_dynamics = temp_features.get('velocity_mean', 0)
        
        # Wedge count hint (based on rotational properties)
        wedge_count_hint = 3  # Default
        if 'physics' in all_features:
            phys_features = all_features['physics']
            total_rotation = phys_features.get('total_rotation', 0)
            if total_rotation > 10:
                wedge_count_hint = min(6, max(1, int(total_rotation / 2)))
        
        return ProblemSignature(
            complexity=float(np.clip(overall_complexity, 0.0, 2.0)),
            frequency_content=float(np.clip(frequency_content, 0.0, 1000.0)),
            pattern_length=len(pattern),
            amplitude=float(amplitude),
            wedge_count_hint=wedge_count_hint,
            noise_level=float(np.clip(noise_level, 0.0, 1.0)),
            temporal_dynamics=float(np.clip(temporal_dynamics, 0.0, 10.0))
        )
    
    def _skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of data."""
        if len(data) < 2:
            return 0.0
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return float(np.mean(((data - mean) / std) ** 3))
    
    def _kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis of data."""
        if len(data) < 2:
            return 0.0
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return float(np.mean(((data - mean) / std) ** 4) - 3)

class ArchitectureSelector:
    """Intelligent architecture selection based on problem characteristics."""
    
    def __init__(self):
        self.available_architectures = {
            'transformer': {
                'strengths': ['complex_patterns', 'long_sequences', 'attention_needed'],
                'weaknesses': ['simple_patterns', 'small_data'],
                'optimal_complexity': (0.5, 2.0),
                'optimal_length': (50, 200),
                'computational_cost': 'high'
            },
            'super_nn': {
                'strengths': ['medium_complexity', 'ensemble_learning', 'robust'],
                'weaknesses': ['very_simple', 'very_complex'],
                'optimal_complexity': (0.3, 1.2),
                'optimal_length': (20, 100),
                'computational_cost': 'medium'
            },
            'standard_nn': {
                'strengths': ['simple_patterns', 'fast_inference', 'baseline'],
                'weaknesses': ['complex_patterns', 'high_noise'],
                'optimal_complexity': (0.1, 0.8),
                'optimal_length': (10, 80),
                'computational_cost': 'low'
            }
        }
        
        self.optimization_algorithms = {
            'turbo': {
                'strengths': ['caching', 'gpu_acceleration', 'adaptive'],
                'optimal_complexity': (0.2, 1.5),
                'computational_cost': 'variable'
            },
            'quantum': {
                'strengths': ['global_optimization', 'complex_landscapes', 'novel'],
                'optimal_complexity': (0.8, 2.0),
                'computational_cost': 'high'
            },
            'standard_ga': {
                'strengths': ['reliable', 'simple', 'well_tested'],
                'optimal_complexity': (0.1, 1.0),
                'computational_cost': 'medium'
            }
        }
        
        self.performance_history = {}
        self.learning_enabled = True
    
    def select_architecture(self, problem_signature: ProblemSignature) -> Dict[str, str]:
        """Select optimal architecture based on problem characteristics."""
        
        # Score each neural network architecture
        nn_scores = {}
        for arch_name, arch_info in self.available_architectures.items():
            score = self._score_architecture(arch_name, arch_info, problem_signature)
            nn_scores[arch_name] = score
        
        # Score each optimization algorithm
        opt_scores = {}
        for opt_name, opt_info in self.optimization_algorithms.items():
            score = self._score_optimizer(opt_name, opt_info, problem_signature)
            opt_scores[opt_name] = score
        
        # Select best options
        best_nn = max(nn_scores.keys(), key=lambda k: nn_scores[k])
        best_optimizer = max(opt_scores.keys(), key=lambda k: opt_scores[k])
        
        # Consider historical performance if available
        if self.learning_enabled:
            best_nn, best_optimizer = self._apply_historical_learning(
                problem_signature, nn_scores, opt_scores
            )
        
        return {
            'neural_network': best_nn,
            'optimizer': best_optimizer,
            'confidence': max(nn_scores[best_nn], opt_scores[best_optimizer]),
            'alternatives': {
                'neural_networks': dict(sorted(nn_scores.items(), key=lambda x: x[1], reverse=True)[:3]),
                'optimizers': dict(sorted(opt_scores.items(), key=lambda x: x[1], reverse=True)[:2])
            }
        }
    
    def _score_architecture(self, arch_name: str, arch_info: Dict, signature: ProblemSignature) -> float:
        """Score a neural network architecture for the given problem."""
        score = 0.0
        
        # Complexity matching
        complexity_range = arch_info['optimal_complexity']
        if complexity_range[0] <= signature.complexity <= complexity_range[1]:
            score += 0.4
        else:
            # Penalty for complexity mismatch
            distance = min(abs(signature.complexity - complexity_range[0]),
                          abs(signature.complexity - complexity_range[1]))
            score += max(0, 0.4 - distance * 0.2)
        
        # Length matching
        length_range = arch_info['optimal_length']
        if length_range[0] <= signature.pattern_length <= length_range[1]:
            score += 0.3
        else:
            # Penalty for length mismatch
            if signature.pattern_length < length_range[0]:
                score += max(0, 0.3 - (length_range[0] - signature.pattern_length) * 0.01)
            else:
                score += max(0, 0.3 - (signature.pattern_length - length_range[1]) * 0.001)
        
        # Special considerations
        if signature.noise_level > 0.7 and 'robust' in arch_info['strengths']:
            score += 0.2
        
        if signature.frequency_content > 100 and 'attention_needed' in arch_info['strengths']:
            score += 0.15
        
        if signature.temporal_dynamics > 5 and 'long_sequences' in arch_info['strengths']:
            score += 0.15
        
        # Computational efficiency consideration
        if arch_info['computational_cost'] == 'low' and signature.pattern_length > 150:
            score += 0.1  # Prefer efficient models for large data
        
        return np.clip(score, 0.0, 1.0)
    
    def _score_optimizer(self, opt_name: str, opt_info: Dict, signature: ProblemSignature) -> float:
        """Score an optimization algorithm for the given problem."""
        score = 0.0
        
        # Complexity matching
        complexity_range = opt_info['optimal_complexity']
        if complexity_range[0] <= signature.complexity <= complexity_range[1]:
            score += 0.5
        else:
            distance = min(abs(signature.complexity - complexity_range[0]),
                          abs(signature.complexity - complexity_range[1]))
            score += max(0, 0.5 - distance * 0.3)
        
        # Special algorithm strengths
        if signature.complexity > 1.5 and 'global_optimization' in opt_info['strengths']:
            score += 0.3
        
        if signature.wedge_count_hint <= 2 and 'caching' in opt_info['strengths']:
            score += 0.2  # Simple problems benefit from caching
        
        if signature.amplitude > 2.0 and 'gpu_acceleration' in opt_info['strengths']:
            score += 0.2  # Large patterns benefit from GPU
        
        return np.clip(score, 0.0, 1.0)
    
    def _apply_historical_learning(self, signature: ProblemSignature, 
                                  nn_scores: Dict, opt_scores: Dict) -> Tuple[str, str]:
        """Apply historical learning to improve selection."""
        
        # Find similar past problems
        similar_problems = self._find_similar_problems(signature)
        
        if not similar_problems:
            # No historical data, use scoring
            best_nn = max(nn_scores.keys(), key=lambda k: nn_scores[k])
            best_optimizer = max(opt_scores.keys(), key=lambda k: opt_scores[k])
            return best_nn, best_optimizer
        
        # Weight scores by historical performance
        historical_nn_scores = {}
        historical_opt_scores = {}
        
        for arch_name in nn_scores.keys():
            historical_performance = self._get_historical_performance(arch_name, similar_problems)
            historical_nn_scores[arch_name] = nn_scores[arch_name] * 0.7 + historical_performance * 0.3
        
        for opt_name in opt_scores.keys():
            historical_performance = self._get_historical_performance(opt_name, similar_problems)
            historical_opt_scores[opt_name] = opt_scores[opt_name] * 0.7 + historical_performance * 0.3
        
        best_nn = max(historical_nn_scores.keys(), key=lambda k: historical_nn_scores[k])
        best_optimizer = max(historical_opt_scores.keys(), key=lambda k: historical_opt_scores[k])
        
        return best_nn, best_optimizer
    
    def _find_similar_problems(self, signature: ProblemSignature) -> List[ArchitecturePerformance]:
        """Find historically similar problems."""
        similar = []
        
        for performance in self.performance_history.values():
            # Calculate similarity score
            similarity = self._calculate_similarity(signature, performance)
            if similarity > 0.7:  # Threshold for similarity
                similar.append(performance)
        
        return similar
    
    def _calculate_similarity(self, sig1: ProblemSignature, perf: ArchitecturePerformance) -> float:
        """Calculate similarity between problem signatures."""
        
        # Extract signature from performance (simplified)
        features = perf.pattern_features
        
        # Compare key features
        complexity_diff = abs(sig1.complexity - features.get('complexity', 0.5))
        length_diff = abs(sig1.pattern_length - features.get('pattern_length', 50)) / 100
        amplitude_diff = abs(sig1.amplitude - features.get('amplitude', 1.0))
        
        # Calculate overall similarity
        similarity = 1.0 - (complexity_diff + length_diff + amplitude_diff) / 3
        return max(0.0, similarity)
    
    def _get_historical_performance(self, architecture: str, 
                                   similar_problems: List[ArchitecturePerformance]) -> float:
        """Get historical performance for architecture on similar problems."""
        relevant_performances = [p.accuracy for p in similar_problems 
                               if p.architecture_name == architecture]
        
        if not relevant_performances:
            return 0.5  # Neutral if no history
        
        return np.mean(relevant_performances)
    
    def record_performance(self, architecture: str, performance: float, 
                          problem_signature: ProblemSignature, 
                          speed: float = 1.0, confidence: float = 0.5):
        """Record performance for learning."""
        
        perf_record = ArchitecturePerformance(
            architecture_name=architecture,
            accuracy=performance,
            speed=speed,
            confidence=confidence,
            problem_type='reverse_prism',
            pattern_features={
                'complexity': problem_signature.complexity,
                'pattern_length': problem_signature.pattern_length,
                'amplitude': problem_signature.amplitude,
                'noise_level': problem_signature.noise_level
            },
            timestamp=datetime.now()
        )
        
        # Store with unique key
        key = f"{architecture}_{int(time.time() * 1000)}"
        self.performance_history[key] = perf_record

class DynamicArchitectureManager:
    """High-level manager for dynamic architecture selection."""
    
    def __init__(self):
        self.pattern_analyzer = PatternAnalyzer()
        self.architecture_selector = ArchitectureSelector()
        self.decision_cache = {}
        self.statistics = {
            'selections_made': 0,
            'cache_hits': 0,
            'architectures_used': {},
            'performance_improvements': []
        }
    
    def select_optimal_system(self, pattern: np.ndarray) -> Dict:
        """Select optimal system configuration for given pattern."""
        
        # Check cache first
        pattern_hash = self._hash_pattern(pattern)
        if pattern_hash in self.decision_cache:
            self.statistics['cache_hits'] += 1
            return self.decision_cache[pattern_hash]
        
        # Analyze pattern
        problem_signature = self.pattern_analyzer.analyze_pattern(pattern)
        
        # Select architecture
        selection = self.architecture_selector.select_architecture(problem_signature)
        
        # Create configuration
        config = {
            'neural_network': selection['neural_network'],
            'optimizer': selection['optimizer'],
            'confidence': selection['confidence'],
            'problem_signature': problem_signature,
            'alternatives': selection['alternatives'],
            'reasoning': self._generate_reasoning(problem_signature, selection),
            'expected_performance': self._estimate_performance(selection, problem_signature)
        }
        
        # Cache decision
        self.decision_cache[pattern_hash] = config
        
        # Update statistics
        self.statistics['selections_made'] += 1
        arch_key = f"{selection['neural_network']}+{selection['optimizer']}"
        self.statistics['architectures_used'][arch_key] = \
            self.statistics['architectures_used'].get(arch_key, 0) + 1
        
        return config
    
    def record_actual_performance(self, pattern: np.ndarray, architecture: str, 
                                 performance: float, execution_time: float):
        """Record actual performance for continuous learning."""
        
        problem_signature = self.pattern_analyzer.analyze_pattern(pattern)
        speed = 1.0 / execution_time if execution_time > 0 else 1.0
        
        # Record in selector for learning
        self.architecture_selector.record_performance(
            architecture, performance, problem_signature, speed
        )
        
        # Calculate improvement over baseline
        baseline_expected = 0.5  # Baseline expectation
        improvement = (performance - baseline_expected) / baseline_expected
        self.statistics['performance_improvements'].append(improvement)
    
    def _hash_pattern(self, pattern: np.ndarray) -> str:
        """Create hash for pattern caching."""
        # Simple hash based on pattern statistics
        if len(pattern) == 0:
            return "empty"
        
        stats = [
            np.mean(pattern),
            np.std(pattern),
            len(pattern),
            np.min(pattern),
            np.max(pattern)
        ]
        
        hash_string = "_".join(f"{s:.3f}" for s in stats)
        return hash_string
    
    def _generate_reasoning(self, signature: ProblemSignature, selection: Dict) -> str:
        """Generate human-readable reasoning for selection."""
        
        reasons = []
        
        # Neural network reasoning
        nn = selection['neural_network']
        if nn == 'transformer':
            if signature.complexity > 1.0:
                reasons.append("Complex pattern requires attention mechanisms")
            if signature.pattern_length > 80:
                reasons.append("Long sequence benefits from transformer architecture")
        elif nn == 'super_nn':
            reasons.append("Medium complexity pattern suits ensemble approach")
        else:
            reasons.append("Simple pattern works well with standard neural network")
        
        # Optimizer reasoning
        opt = selection['optimizer']
        if opt == 'turbo':
            reasons.append("Turbo optimizer selected for GPU acceleration and caching")
        elif opt == 'quantum':
            reasons.append("Quantum algorithms chosen for complex optimization landscape")
        else:
            reasons.append("Standard GA selected for reliable optimization")
        
        # Pattern-specific reasoning
        if signature.noise_level > 0.7:
            reasons.append("High noise level requires robust algorithms")
        
        if signature.wedge_count_hint > 4:
            reasons.append("High wedge count suggests complex multi-modal optimization")
        
        return "; ".join(reasons)
    
    def _estimate_performance(self, selection: Dict, signature: ProblemSignature) -> Dict:
        """Estimate expected performance based on selection."""
        
        # Base performance from confidence
        base_accuracy = selection['confidence'] * 0.8  # Conservative estimate
        
        # Adjust based on problem characteristics
        if signature.complexity < 0.5:
            accuracy_bonus = 0.15  # Simple problems should perform better
        elif signature.complexity > 1.5:
            accuracy_bonus = -0.1  # Complex problems are harder
        else:
            accuracy_bonus = 0.0
        
        estimated_accuracy = min(0.95, base_accuracy + accuracy_bonus)
        
        # Estimate speed based on architecture
        speed_factors = {
            'standard_nn': 1.0,
            'super_nn': 0.7,
            'transformer': 0.4
        }
        
        opt_factors = {
            'standard_ga': 1.0,
            'turbo': 1.5,  # Faster due to caching and GPU
            'quantum': 0.8  # Slower due to complexity
        }
        
        nn_speed = speed_factors.get(selection['neural_network'], 1.0)
        opt_speed = opt_factors.get(selection['optimizer'], 1.0)
        relative_speed = nn_speed * opt_speed
        
        return {
            'accuracy': float(estimated_accuracy),
            'relative_speed': float(relative_speed),
            'confidence_level': selection['confidence']
        }
    
    def get_system_statistics(self) -> Dict:
        """Get comprehensive system statistics."""
        
        # Calculate average improvements
        avg_improvement = np.mean(self.statistics['performance_improvements']) \
                         if self.statistics['performance_improvements'] else 0.0
        
        # Most used architectures
        most_used = dict(sorted(self.statistics['architectures_used'].items(), 
                               key=lambda x: x[1], reverse=True)[:5])
        
        return {
            'total_selections': self.statistics['selections_made'],
            'cache_hit_rate': self.statistics['cache_hits'] / max(1, self.statistics['selections_made']),
            'average_performance_improvement': float(avg_improvement),
            'most_used_architectures': most_used,
            'unique_patterns_analyzed': len(self.decision_cache),
            'learning_enabled': self.architecture_selector.learning_enabled,
            'historical_data_points': len(self.architecture_selector.performance_history)
        }