#!/usr/bin/env python3
"""
ADVANCED ENSEMBLE METHODS - Supercharged prediction fusion

Revolutionary ensemble techniques:
- Multi-model voting with confidence weighting
- Bayesian model averaging
- Stacking with meta-learners
- Dynamic model selection
- Uncertainty-aware fusion
"""

import numpy as np
import json
import time
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import pickle

@dataclass
class EnsemblePrediction:
    """Container for ensemble prediction results."""
    wedge_count: int
    confidence: float
    parameters: Dict
    uncertainty: float
    contributing_models: List[str]
    model_weights: Dict[str, float]
    consensus_score: float

class AdvancedEnsemble:
    """Advanced ensemble methods for supercharged predictions."""
    
    def __init__(self):
        self.models = {}
        self.model_weights = {}
        self.model_performance_history = {}
        self.prediction_cache = {}
        
        # Ensemble strategies
        self.strategies = {
            'voting': self._weighted_voting,
            'bayesian': self._bayesian_averaging,
            'stacking': self._stacking_fusion,
            'dynamic': self._dynamic_selection
        }
        
        # Meta-learner for stacking
        self.meta_learner = None
        self.meta_features = []
    
    def register_model(self, name: str, model, weight: float = 1.0):
        """Register a model in the ensemble."""
        self.models[name] = model
        self.model_weights[name] = weight
        self.model_performance_history[name] = []
        print(f"   📡 Registered model: {name} (weight: {weight:.2f})")
    
    def predict_ensemble(self, pattern: np.ndarray, strategy: str = 'dynamic') -> EnsemblePrediction:
        """Generate ensemble prediction using specified strategy."""
        
        # Get predictions from all models
        model_predictions = self._get_all_predictions(pattern)
        
        if not model_predictions:
            raise ValueError("No models available for ensemble prediction")
        
        # Apply ensemble strategy
        if strategy not in self.strategies:
            strategy = 'dynamic'
        
        return self.strategies[strategy](pattern, model_predictions)
    
    def _get_all_predictions(self, pattern: np.ndarray) -> Dict:
        """Get predictions from all registered models."""
        predictions = {}
        
        for name, model in self.models.items():
            try:
                if hasattr(model, 'predict'):
                    pred = model.predict(pattern)
                    
                    # Ensure prediction has required fields
                    if 'wedgenum' in pred:
                        predictions[name] = {
                            'wedge_count': pred['wedgenum'],
                            'parameters': pred,
                            'confidence': pred.get('prediction_confidence', {}).get('overall_confidence', 0.5),
                            'uncertainty': pred.get('prediction_confidence', {}).get('uncertainty_score', 0.5)
                        }
                else:
                    print(f"   ⚠️ Model {name} does not have predict method")
                    
            except Exception as e:
                print(f"   ⚠️ Model {name} prediction failed: {e}")
                continue
        
        return predictions
    
    def _weighted_voting(self, pattern: np.ndarray, predictions: Dict) -> EnsemblePrediction:
        """Weighted voting ensemble with confidence-based weighting."""
        
        # Calculate effective weights (base weight × confidence × historical performance)
        effective_weights = {}
        wedge_votes = {}
        
        for name, pred in predictions.items():
            base_weight = self.model_weights[name]
            confidence = pred['confidence']
            
            # Historical performance weight
            if self.model_performance_history[name]:
                hist_perf = np.mean(self.model_performance_history[name][-10:])  # Last 10 predictions
            else:
                hist_perf = 0.5
            
            effective_weight = base_weight * confidence * hist_perf
            effective_weights[name] = effective_weight
            
            # Vote for wedge count
            wedge_count = pred['wedge_count']
            if wedge_count not in wedge_votes:
                wedge_votes[wedge_count] = 0
            wedge_votes[wedge_count] += effective_weight
        
        # Select wedge count with highest weighted vote
        best_wedge_count = max(wedge_votes.keys(), key=lambda k: wedge_votes[k])
        total_votes = sum(wedge_votes.values())
        consensus_score = wedge_votes[best_wedge_count] / total_votes if total_votes > 0 else 0
        
        # Aggregate parameters from models that voted for winning wedge count
        contributing_models = [name for name, pred in predictions.items() 
                             if pred['wedge_count'] == best_wedge_count]
        
        # Weighted parameter averaging
        final_parameters = self._aggregate_parameters(
            predictions, contributing_models, effective_weights
        )
        
        # Calculate ensemble confidence and uncertainty
        ensemble_confidence = np.mean([pred['confidence'] for name, pred in predictions.items() 
                                     if name in contributing_models])
        ensemble_uncertainty = np.mean([pred['uncertainty'] for name, pred in predictions.items() 
                                      if name in contributing_models])
        
        return EnsemblePrediction(
            wedge_count=best_wedge_count,
            confidence=ensemble_confidence,
            parameters=final_parameters,
            uncertainty=ensemble_uncertainty,
            contributing_models=contributing_models,
            model_weights={name: effective_weights[name] for name in contributing_models},
            consensus_score=consensus_score
        )
    
    def _bayesian_averaging(self, pattern: np.ndarray, predictions: Dict) -> EnsemblePrediction:
        """Bayesian model averaging with uncertainty quantification."""
        
        # Prior probabilities for each model (based on historical performance)
        priors = {}
        for name in predictions.keys():
            if self.model_performance_history[name]:
                priors[name] = np.mean(self.model_performance_history[name][-20:])
            else:
                priors[name] = 1.0 / len(predictions)  # Uniform prior
        
        # Normalize priors
        prior_sum = sum(priors.values())
        priors = {name: p / prior_sum for name, p in priors.items()}
        
        # Likelihood based on model confidence
        likelihoods = {name: pred['confidence'] for name, pred in predictions.items()}
        
        # Posterior probabilities (Bayes' theorem)
        posteriors = {}
        evidence = sum(priors[name] * likelihoods[name] for name in predictions.keys())
        
        for name in predictions.keys():
            posteriors[name] = (priors[name] * likelihoods[name]) / evidence if evidence > 0 else 0
        
        # Bayesian model averaging for wedge count
        wedge_probabilities = {}
        for name, pred in predictions.items():
            wedge_count = pred['wedge_count']
            if wedge_count not in wedge_probabilities:
                wedge_probabilities[wedge_count] = 0
            wedge_probabilities[wedge_count] += posteriors[name]
        
        # Select most probable wedge count
        best_wedge_count = max(wedge_probabilities.keys(), 
                              key=lambda k: wedge_probabilities[k])
        
        # Calculate Bayesian uncertainty
        entropy = -sum(p * np.log(p + 1e-8) for p in wedge_probabilities.values())
        uncertainty = entropy / np.log(len(wedge_probabilities))  # Normalized entropy
        
        # Contributing models (those with significant posterior probability)
        threshold = 0.1
        contributing_models = [name for name, post in posteriors.items() if post > threshold]
        
        # Aggregate parameters
        final_parameters = self._aggregate_parameters(
            predictions, contributing_models, posteriors
        )
        
        # Ensemble confidence (inverse of uncertainty)
        ensemble_confidence = 1.0 - uncertainty
        
        return EnsemblePrediction(
            wedge_count=best_wedge_count,
            confidence=ensemble_confidence,
            parameters=final_parameters,
            uncertainty=uncertainty,
            contributing_models=contributing_models,
            model_weights={name: posteriors[name] for name in contributing_models},
            consensus_score=wedge_probabilities[best_wedge_count]
        )
    
    def _stacking_fusion(self, pattern: np.ndarray, predictions: Dict) -> EnsemblePrediction:
        """Stacking ensemble with meta-learner."""
        
        # Extract meta-features from base model predictions
        meta_features = []
        for name, pred in predictions.items():
            meta_features.extend([
                pred['wedge_count'],
                pred['confidence'],
                pred['uncertainty'],
                len(pred['parameters'].get('rotation_speeds', [])),
                np.mean(pred['parameters'].get('rotation_speeds', [0])),
                np.std(pred['parameters'].get('rotation_speeds', [0]))
            ])
        
        # If meta-learner is not trained, fall back to weighted voting
        if self.meta_learner is None:
            return self._weighted_voting(pattern, predictions)
        
        # Meta-learner prediction (simplified - would use trained ML model)
        # For now, use a heuristic meta-learner
        meta_prediction = self._heuristic_meta_learner(meta_features, predictions)
        
        return meta_prediction
    
    def _dynamic_selection(self, pattern: np.ndarray, predictions: Dict) -> EnsemblePrediction:
        """Dynamic model selection based on pattern characteristics."""
        
        # Analyze pattern characteristics
        pattern_features = self._extract_pattern_features(pattern)
        
        # Select best model(s) based on pattern type
        selected_models = self._select_models_for_pattern(pattern_features, predictions)
        
        # If multiple models selected, use weighted voting among them
        if len(selected_models) > 1:
            filtered_predictions = {name: predictions[name] for name in selected_models}
            return self._weighted_voting(pattern, filtered_predictions)
        
        # If single model selected, enhance its prediction
        elif len(selected_models) == 1:
            model_name = selected_models[0]
            pred = predictions[model_name]
            
            return EnsemblePrediction(
                wedge_count=pred['wedge_count'],
                confidence=pred['confidence'],
                parameters=pred['parameters'],
                uncertainty=pred['uncertainty'],
                contributing_models=[model_name],
                model_weights={model_name: 1.0},
                consensus_score=1.0
            )
        
        # Fallback to voting if no models selected
        else:
            return self._weighted_voting(pattern, predictions)
    
    def _extract_pattern_features(self, pattern: np.ndarray) -> Dict:
        """Extract features for dynamic model selection."""
        if len(pattern) < 2:
            return {'complexity': 0.5, 'frequency': 0, 'amplitude': 0}
        
        # Basic features
        complexity = np.std(np.diff(pattern, axis=0)) / (np.mean(np.abs(pattern)) + 1e-6)
        amplitude = np.sqrt(np.mean(pattern**2))
        
        # Frequency features
        fft_x = np.abs(np.fft.fft(pattern[:, 0]))
        dominant_freq = np.argmax(fft_x[:len(fft_x)//2])
        
        return {
            'complexity': float(np.clip(complexity, 0, 2)),
            'amplitude': float(amplitude),
            'frequency': float(dominant_freq),
            'length': len(pattern)
        }
    
    def _select_models_for_pattern(self, pattern_features: Dict, predictions: Dict) -> List[str]:
        """Select best models based on pattern characteristics."""
        selected = []
        
        complexity = pattern_features['complexity']
        
        # Heuristic model selection based on pattern complexity
        for name, pred in predictions.items():
            confidence = pred['confidence']
            
            # Select models with high confidence
            if confidence > 0.6:
                selected.append(name)
            
            # For complex patterns, prefer models with higher uncertainty awareness
            elif complexity > 1.0 and pred['uncertainty'] < 0.5:
                selected.append(name)
            
            # For simple patterns, prefer confident models
            elif complexity < 0.5 and confidence > 0.4:
                selected.append(name)
        
        # Ensure at least one model is selected
        if not selected:
            # Select model with highest confidence
            best_model = max(predictions.keys(), 
                           key=lambda name: predictions[name]['confidence'])
            selected.append(best_model)
        
        return selected
    
    def _aggregate_parameters(self, predictions: Dict, contributing_models: List[str], 
                            weights: Dict[str, float]) -> Dict:
        """Aggregate parameters from contributing models."""
        
        if not contributing_models:
            return {}
        
        # Get a reference parameter structure
        ref_pred = predictions[contributing_models[0]]
        result = ref_pred['parameters'].copy()
        
        # Weighted averaging for numerical parameters
        numerical_params = ['rotation_speeds', 'phi_x', 'phi_y', 'distances', 'refractive_indices']
        
        for param_name in numerical_params:
            if param_name in result:
                values = []
                param_weights = []
                
                for model_name in contributing_models:
                    pred = predictions[model_name]
                    if param_name in pred['parameters']:
                        values.append(pred['parameters'][param_name])
                        param_weights.append(weights.get(model_name, 1.0))
                
                if values:
                    # Weighted average
                    weighted_sum = np.zeros_like(values[0])
                    total_weight = 0
                    
                    for val, weight in zip(values, param_weights):
                        weighted_sum += np.array(val) * weight
                        total_weight += weight
                    
                    if total_weight > 0:
                        result[param_name] = (weighted_sum / total_weight).tolist()
        
        return result
    
    def _heuristic_meta_learner(self, meta_features: List[float], predictions: Dict) -> EnsemblePrediction:
        """Heuristic meta-learner for stacking (placeholder for ML model)."""
        
        # Simple heuristic: weight models by their confidence and consistency
        model_scores = {}
        
        for name, pred in predictions.items():
            # Score based on confidence and low uncertainty
            score = pred['confidence'] * (1.0 - pred['uncertainty'])
            model_scores[name] = score
        
        # Select top model
        best_model = max(model_scores.keys(), key=lambda k: model_scores[k])
        best_pred = predictions[best_model]
        
        return EnsemblePrediction(
            wedge_count=best_pred['wedge_count'],
            confidence=best_pred['confidence'],
            parameters=best_pred['parameters'],
            uncertainty=best_pred['uncertainty'],
            contributing_models=[best_model],
            model_weights={best_model: 1.0},
            consensus_score=model_scores[best_model]
        )
    
    def update_model_performance(self, model_name: str, performance: float):
        """Update historical performance for a model."""
        if model_name in self.model_performance_history:
            self.model_performance_history[model_name].append(performance)
            
            # Keep only recent history
            if len(self.model_performance_history[model_name]) > 100:
                self.model_performance_history[model_name] = \
                    self.model_performance_history[model_name][-100:]
    
    def get_ensemble_stats(self) -> Dict:
        """Get comprehensive ensemble statistics."""
        stats = {
            'total_models': len(self.models),
            'model_weights': self.model_weights.copy(),
            'strategies_available': list(self.strategies.keys()),
            'cache_size': len(self.prediction_cache)
        }
        
        # Performance history summary
        perf_summary = {}
        for name, history in self.model_performance_history.items():
            if history:
                perf_summary[name] = {
                    'mean_performance': float(np.mean(history)),
                    'recent_performance': float(np.mean(history[-10:])) if len(history) >= 10 else float(np.mean(history)),
                    'predictions_made': len(history),
                    'trend': 'improving' if len(history) > 1 and history[-1] > history[0] else 'stable'
                }
        
        stats['performance_summary'] = perf_summary
        return stats

class EnsembleManager:
    """High-level manager for ensemble operations."""
    
    def __init__(self):
        self.ensemble = AdvancedEnsemble()
        self.auto_register_models()
    
    def auto_register_models(self):
        """Automatically register available models."""
        try:
            from core.super_neural_network import SuperNeuralPredictor
            super_nn = SuperNeuralPredictor()
            if super_nn.load():
                self.ensemble.register_model('super_nn', super_nn, weight=1.5)
        except:
            pass
        
        try:
            from core.transformer_nn import TransformerNeuralPredictor
            transformer = TransformerNeuralPredictor()
            if transformer.load():
                self.ensemble.register_model('transformer', transformer, weight=2.0)
        except:
            pass
        
        try:
            from core.neural_network import NeuralPredictor
            standard_nn = NeuralPredictor()
            if standard_nn.load():
                self.ensemble.register_model('standard_nn', standard_nn, weight=1.0)
        except:
            pass
    
    def predict(self, pattern: np.ndarray, strategy: str = 'dynamic') -> Dict:
        """Make ensemble prediction and return in standard format."""
        try:
            ensemble_pred = self.ensemble.predict_ensemble(pattern, strategy)
            
            # Convert to standard prediction format
            result = ensemble_pred.parameters.copy()
            result.update({
                'wedgenum': ensemble_pred.wedge_count,
                'ensemble_info': {
                    'strategy': strategy,
                    'confidence': ensemble_pred.confidence,
                    'uncertainty': ensemble_pred.uncertainty,
                    'consensus_score': ensemble_pred.consensus_score,
                    'contributing_models': ensemble_pred.contributing_models,
                    'model_weights': ensemble_pred.model_weights
                }
            })
            
            return result
            
        except Exception as e:
            print(f"Ensemble prediction failed: {e}")
            return {'wedgenum': 3, 'ensemble_info': {'error': str(e)}}