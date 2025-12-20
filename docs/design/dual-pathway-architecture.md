# Dual-Pathway Network Architecture Design

## Brain-Inspired Design Philosophy

Based on visual neuroscience dual-stream theory:
- **Fast Pathway (Ventral-like)**: Rapid extraction of early saliency features (EOG signals)
- **Slow Pathway (Dorsal-like)**: Stable extraction of temporal motion/tracking features

## Architecture Overview

### Input Processing
- Input: `[batch, 1, 59 channels, 200 timepoints]` (1 second @ 200Hz)
- Split strategy: Configurable (default 100:100, i.e., 0.5s split point)

### Fast Pathway: EEGNet
**Purpose**: Extract early EOG and saliency signals (0-0.5s)

**Architecture**:
```
Input [B,1,59,100]
  → Temporal Conv (kernel=64, F1=8)
  → Batch Norm + ELU
  → Spatial Depthwise Conv (59×1, groups=F1, D=2)
  → Batch Norm + ELU + Dropout(0.25)
  → Separable Conv (kernel=16, F2=16)
  → Batch Norm + ELU + Dropout(0.25)
  → Average Pooling (stride=4 and stride=8)
Output [B, 16, 1, T]
```

**Key Parameters**:
- F1=8 (temporal filters)
- D=2 (depth multiplier for spatial conv)
- F2=16 (separable filters)
- Dropout=0.25

**Reference**: Lawhern et al., 2018 - EEGNet

### Slow Pathway: C-RNN (Convolutional RNN)
**Purpose**: Extract late tracking and attention signals (0.5-1.0s)

**Architecture**:
```
Input [B,1,59,100]
  → Multi-scale Temporal Conv (kernels: 32, 64, 96)
  → Concatenate → [B, 48, 59, 100]
  → Batch Norm + Spatial Conv (59×1) → [B, 48, 1, 100]
  → Batch Norm + ELU + Dropout
  → Reshape → [B, 100, 48]
  → Bidirectional LSTM (hidden=32, layers=2)
  → Dropout(0.3)
  → Conv1D projection → [B, 16, T]
Output [B, 16, 1, T]
```

**Key Parameters**:
- Multi-scale kernels: 32, 64, 96 (inspired by TSception)
- LSTM hidden: 32
- LSTM layers: 2 (bidirectional)
- Dropout: 0.3

**Inspiration**: TSception (Ding et al., 2020) + RNN for temporal modeling

### Multi-Mechanism Fusion

Inspired by TSception's multi-scale design, enhanced with multiple fusion strategies:

#### Mechanism 1: Cross-Attention (from PSAFNet)
```python
Q_fast = W_Q(fast_features)
K_slow, V_slow = W_K(slow_features), W_V(slow_features)
attention = softmax(Q_fast @ K_slow^T / sqrt(d))
fused_1 = attention @ V_slow

# Bidirectional cross-attention
Q_slow = W_Q(slow_features)
K_fast, V_fast = W_K(fast_features), W_V(fast_features)
attention = softmax(Q_slow @ K_fast^T / sqrt(d))
fused_1 += attention @ V_fast
```

#### Mechanism 2: Gated Fusion
```python
# Learn adaptive gates based on global feature statistics
gate_fast = sigmoid(W_g_f(global_avg_pool(fast_features)))
gate_slow = sigmoid(W_g_s(global_avg_pool(slow_features)))
fused_2 = gate_fast * fast_features + gate_slow * slow_features
```

#### Mechanism 3: Adaptive Weighted Fusion
```python
# Learnable weight (initialized at 0.5)
alpha = sigmoid(learnable_parameter)
fused_3 = alpha * fast_features + (1-alpha) * slow_features
```

#### Final Fusion
```python
multi_fused = Concatenate([fused_1, fused_2, fused_3])  # [B, T, 3*C]
output = TCN_layers(multi_fused) → GlobalAvgPool → FC → [B, 2]
```

### Loss Function
```
Total Loss = CE_loss + lambda_mmd * MMD_loss
```
- **CE_loss**: CrossEntropyLoss for classification
- **MMD_loss**: Maximum Mean Discrepancy for distribution alignment between pathways
- **lambda_mmd**: Weight for MMD loss (default: 1.5)

## Implementation Roadmap

### Phase 1: Fast Pathway (EEGNet)
- [x] Implement EEGNet architecture
- [x] Unit tests for forward pass
- [x] Standalone training script for validation

### Phase 2: Slow Pathway (C-RNN)
- [x] Implement multi-scale temporal convolution
- [x] Implement Bidirectional LSTM
- [x] Unit tests

### Phase 3: Multi-Mechanism Fusion
- [x] Implement cross-attention
- [x] Implement gated fusion
- [x] Implement adaptive weighting
- [x] Integration tests

### Phase 4: Full Integration
- [x] Integrate all components into DualPathwayNet
- [x] Training scripts
- [x] Cross-subject evaluation script
- [x] Comprehensive testing

### Phase 5 (Future): ST-GCN Alternative
- [ ] Implement ST-GCN for slow pathway
- [ ] Construct EEG graph adjacency matrix
- [ ] Comparative experiments: C-RNN vs ST-GCN

## Ablation Studies Plan

1. **Single Pathway vs Dual**: Test fast-only, slow-only, and dual pathways
2. **Fusion Mechanisms**: Test each mechanism independently
3. **Time Split Points**: Test 80:120, 100:100, 120:80
4. **MMD Loss**: With/without MMD loss
5. **LSTM vs GRU**: Compare RNN variants in slow pathway

## Expected Improvements Over PSAFNet

| Metric | PSAFNet Baseline | Expected Improvement |
|--------|------------------|---------------------|
| Accuracy | ~85% | +3-5% → ~88-90% |
| Hit Rate (TPR) | ~0.82 | +5-8% → ~0.87-0.90 |
| False Alarm Rate | ~0.15 | -2-3% → ~0.12-0.13 |
| Parameters | ~150k | ~200k (+33%) |

## Comparison with PSAFNet

| Aspect | PSAFNet | Dual-Pathway Network |
|--------|---------|---------------------|
| **Encoder Design** | Dual identical Phased_Encoders | Fast (EEGNet) + Slow (C-RNN) |
| **Temporal Split** | 150:150 overlapping | 100:100 non-overlapping |
| **Fusion** | Cross-attention + TCN | Multi-mechanism (3-way) |
| **Inspiration** | Phase segmentation | Fast/slow visual pathways |
| **Feature Focus** | Multi-scale temporal | Saliency + Motion tracking |

## References

1. **EEGNet**: Lawhern, V. J., et al. (2018). "EEGNet: a compact convolutional neural network for EEG-based brain–computer interfaces." Journal of Neural Engineering.
2. **TSception**: Ding, Y., et al. (2020). "TSception: A Deep Learning Framework for Emotion Detection Using EEG."
3. **PSAFNet**: Current implementation (baseline)
4. **Dual Visual Streams**: Goodale, M. A., & Milner, A. D. (1992). "Separate visual pathways for perception and action." Trends in Neurosciences.

## Notes

- All PDF references are stored locally in `docs/develop/` (not tracked by git)
- DOI links should be added to references for reproducibility
