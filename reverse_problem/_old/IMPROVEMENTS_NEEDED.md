# 🔧 Reverse Risley Prism System - Critical Improvements Needed

## Current Status (as of 2025-01-15)

### Performance Metrics
- **Overall Accuracy**: 30.3% (455/1500 correct)
- **Neural Network Accuracy**: 28.1% (poor wedge count prediction)
- **Processing Speed**: 0.13 samples/sec (7.8 seconds per sample)
- **Training Data**: 5,000 samples
- **Best Performing**: 5-wedge patterns (41.5% accuracy)
- **Worst Performing**: 1-wedge patterns (9.5% accuracy)

## 🚨 Critical Issues to Fix

### 1. Neural Network Architecture Problem
**Issue**: Neural network achieves excellent training loss (0.043) but terrible real-world accuracy (28.1%)

**Root Cause**: The network is learning to minimize MSE loss on parameter vectors, not actually learning the pattern-to-wedge relationship

**MUST DO**:
```python
# CURRENT (BROKEN):
# - Single network trying to predict 31 parameters simultaneously
# - MSE loss on normalized vectors
# - No pattern-specific features

# NEEDED:
# 1. Two-stage network:
#    - Stage 1: Pattern → Wedge Count (classification, not regression)
#    - Stage 2: Pattern + Wedge Count → Parameters
# 2. Custom loss function that weights wedge count accuracy heavily
# 3. Pattern feature extraction (FFT, complexity metrics, symmetry)
```

### 2. Genetic Algorithm Speed Bottleneck
**Issue**: GA takes 100% of runtime (11,716 seconds for 1500 samples)

**MUST DO**:
```python
# Reduce GA computational load:
1. Decrease population size: 30+10*wedges → 20+5*wedges
2. Reduce generations: 15+5*wedges → 10+3*wedges
3. Use NN predictions as seed population (not just one individual)
4. Implement early stopping when cost plateaus
5. Cache fitness evaluations
```

### 3. Poor Simple Pattern Performance
**Issue**: 1-wedge accuracy is 9.5% while 5-wedge is 41.5%

**MUST DO**:
```python
# Add complexity-aware processing:
1. Pre-classify pattern complexity
2. Use different models for simple vs complex patterns
3. Adjust GA parameters based on complexity
4. Add "simplicity bias" for low-complexity patterns
```

## 📋 Action Plan (Priority Order)

### Phase 1: Fix Neural Network (1-2 days)
1. **Split wedge prediction from parameter prediction**
   ```python
   class WedgeCountClassifier(nn.Module):
       # Binary classifiers for each wedge count
       # Use CrossEntropyLoss, not MSE
   
   class ParameterPredictor(nn.Module):
       # Given wedge count, predict parameters
       # Condition on wedge count explicitly
   ```

2. **Add pattern feature extraction**
   ```python
   def extract_pattern_features(pattern):
       return {
           'fft_peaks': get_frequency_peaks(pattern),
           'complexity': calculate_complexity(pattern),
           'symmetry': measure_symmetry(pattern),
           'coverage': calculate_coverage_area(pattern)
       }
   ```

3. **Implement proper data augmentation**
   ```python
   # Add noise, rotations, scaling to training patterns
   # This will help with generalization
   ```

### Phase 2: Speed Optimization (1 day)
1. **Optimize GA parameters**
   ```python
   # solver.py changes:
   pop_size = min(20 + 5 * wedges, 50)  # Cap at 50
   generations = min(10 + 3 * wedges, 30)  # Cap at 30
   ```

2. **Implement caching**
   ```python
   from functools import lru_cache
   
   @lru_cache(maxsize=10000)
   def evaluate_fitness(params_hash):
       # Cache fitness evaluations
   ```

3. **Add parallel processing**
   ```python
   # Use multiprocessing.Pool for GA evaluation
   # Currently using parallel=False
   ```

### Phase 3: Improve Accuracy (2-3 days)
1. **Collect more diverse training data**
   - Current: 5,000 samples
   - Needed: 20,000+ samples with emphasis on simple patterns

2. **Implement ensemble approach**
   ```python
   # Train multiple models:
   models = {
       'simple': WedgeClassifier(max_wedges=2),
       'medium': WedgeClassifier(max_wedges=4),
       'complex': WedgeClassifier(max_wedges=6)
   }
   ```

3. **Add confidence scoring**
   ```python
   # Return confidence with predictions
   # Use high-confidence NN predictions directly
   # Only use GA for low-confidence cases
   ```

## 🎯 Success Metrics

### Minimum Viable Performance
- Overall Accuracy: **50%** (up from 30.3%)
- Processing Speed: **1.0 samples/sec** (up from 0.13)
- 1-wedge Accuracy: **30%** (up from 9.5%)

### Target Performance
- Overall Accuracy: **70%**
- Processing Speed: **5.0 samples/sec**
- Neural Network Accuracy: **60%**

## 🚀 Quick Fixes (Do These First!)

1. **Change neural network to classification** (1 hour)
   - Replace final layer with 6-class softmax
   - Use CrossEntropyLoss instead of MSE
   - This alone should improve NN accuracy to ~50%

2. **Reduce GA populations** (30 minutes)
   - Cut population sizes in half
   - This will double the speed immediately

3. **Fix the 1-wedge underfitting** (1 hour)
   - Add special case: if pattern is very simple, force trying 1-wedge first
   - Don't let GA skip to higher wedge counts for simple patterns

## 📊 Expected Results After Fixes

With these improvements, expect:
- **Week 1**: 45% accuracy, 0.5 samples/sec
- **Week 2**: 55% accuracy, 1.0 samples/sec  
- **Month 1**: 65% accuracy, 3.0 samples/sec

## ⚠️ If These Don't Work

Consider fundamental approach change:
1. **Template Matching**: Pre-compute common patterns and match
2. **Differentiable Renderer**: Use gradient descent instead of GA
3. **Inverse Network**: Train a network on pattern pairs (learn the inverse mapping directly)

---

**Remember**: The reverse Risley prism problem is inherently difficult. Even 50% accuracy would be a significant achievement for this inverse problem.