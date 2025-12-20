# Dual-Pathway Network for EEG-based Target Detection

> **Branch**: `develop`
> **Baseline**: PSAFNet (see `main` branch)

## Overview

Brain-inspired dual-pathway architecture combining fast and slow visual processing streams for improved EEG-based target detection.

### Key Features
- **Fast Pathway (EEGNet)**: Extracts early EOG and saliency features (0-0.5s)
- **Slow Pathway (C-RNN)**: Extracts temporal tracking and attention features (0.5-1.0s)
- **Multi-Mechanism Fusion**: Three complementary fusion strategies (cross-attention + gating + adaptive weighting)

### Architecture
```
Input [B,1,59,200]
    |
    +-- Fast (0-0.5s) --> EEGNet --> [B,16,1,T1]
    |                                     |
    +-- Slow (0.5-1.0s) -> C-RNN --> [B,16,1,T2]
                                          |
                                  Multi-Mechanism Fusion
                                          |
                                   Output [B,2]
```

## Quick Start

### Installation
```bash
# Activate virtual environment
.venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Testing
```bash
# Run all unit tests
python tests/test_eegnet.py      # Fast pathway tests
python tests/test_crnn.py        # Slow pathway tests
python tests/test_fusion.py      # Fusion tests
python tests/test_integration.py # End-to-end tests
```

### Model Usage
```python
from DualPathwayNet import DualPathwayNet

# Initialize model
model = DualPathwayNet(
    num_channels=59,
    num_timepoints=200,
    split_point=100
)

# Forward pass
x = torch.randn(4, 1, 59, 200)  # [batch, 1, channels, time]
output, mmd_loss = model(x)
```

## Model Architecture Details

### Fast Pathway: EEGNet
- **Purpose**: Early saliency/EOG signal extraction
- **Architecture**:
  - Temporal Conv (kernel=64, F1=8)
  - Spatial Depthwise Conv (across 59 channels, D=2)
  - Separable Conv (kernel=16, F2=16)
  - Dropout=0.25
- **Parameters**: 2,048
- **Output**: [B, 16, 1, T//32]

### Slow Pathway: C-RNN
- **Purpose**: Late tracking/attention signal extraction
- **Architecture**:
  - Multi-scale Temporal Conv (kernels: 32, 64, 96)
  - Spatial Conv (across 59 channels)
  - Bidirectional LSTM (hidden=32, layers=2)
  - Dropout=0.3
- **Parameters**: 186,320
- **Output**: [B, 16, 1, T]

### Multi-Mechanism Fusion
- **Mechanism 1**: Bidirectional cross-attention
- **Mechanism 2**: Gated fusion (learnable gates)
- **Mechanism 3**: Adaptive weighted fusion (learnable alpha)
- **TCN**: 3 layers, dilation [1, 2, 4]
- **MMD Loss**: Distribution alignment between pathways
- **Parameters**: 11,587
- **Output**: [B, 2]

## Configuration

Edit `configs/dual_pathway_config.py` to modify hyperparameters:

```python
# Time split
split_point = 100  # Default: 0.5s @ 200Hz

# Training
learning_rate = 0.001
batch_size = 16
num_epochs = 5
mmd_weight = 1.5  # Weight for MMD loss
```

## Project Structure

```
PSAFNet/
├── DualPathwayNet.py          # Main model
├── configs/
│   └── dual_pathway_config.py # Configuration
├── modules/
│   ├── fast_pathway/
│   │   └── eegnet.py         # EEGNet implementation
│   ├── slow_pathway/
│   │   └── crnn.py           # C-RNN implementation
│   └── fusion/
│       └── multi_mechanism_fusion.py  # Fusion module
├── tests/                    # Test suite (4 test files)
└── docs/design/              # Architecture documentation
```

## Model Statistics

| Component | Parameters | Description |
|-----------|-----------|-------------|
| Fast Pathway | 2,048 | Lightweight EEGNet |
| Slow Pathway | 186,320 | C-RNN with LSTM |
| Fusion Module | 11,587 | Multi-mechanism fusion |
| **Total** | **199,955** | **~200k parameters** |

## Comparison with PSAFNet

| Aspect | PSAFNet (main) | Dual-Pathway (develop) |
|--------|---------------|----------------------|
| Encoder | Dual Phased_Encoders | EEGNet + C-RNN |
| Split | 150:150 overlap | 100:100 non-overlap |
| Fusion | Cross-att + TCN | 3-way multi-mechanism |
| Inspiration | Phase segmentation | Fast/slow pathways |
| Parameters | ~150k | ~200k |

## Git Workflow

### Switch between versions
```bash
# Use original PSAFNet
git checkout main

# Use Dual-Pathway Network
git checkout develop

# View all branches
git branch -a
```

### Current branch structure
```
main (PSAFNet - original, unchanged)
  |
  +-- develop (Dual-Pathway Network)
```

## Testing Summary

All tests passing:
- [x] EEGNet: 5/5 tests passed
- [x] C-RNN: 7/7 tests passed
- [x] Fusion: 7/7 tests passed
- [x] Integration: 6/6 tests passed

**Total**: 25/25 tests passed

## References

1. **EEGNet**: Lawhern et al., 2018
2. **TSception**: Ding et al., 2020 (multi-scale inspiration)
3. **PSAFNet**: Baseline (main branch)
4. **Dual Visual Streams**: Goodale & Milner, 1992

## Next Steps

1. **Training**: Adapt `train.py` or `cross_subject_evaluation.py` for dual-pathway model
2. **Ablation Studies**: Test individual fusion mechanisms
3. **ST-GCN**: Implement graph-based slow pathway alternative
4. **Optimization**: Hyperparameter tuning

## Notes

- All PDF papers stored in `docs/develop/` (not tracked by git)
- Main branch remains unchanged - can always revert to original PSAFNet
- Use `git checkout main` to access original code
