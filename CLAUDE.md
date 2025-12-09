# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PSAFNet (Phase Segment and Aligned Fusion Net) is a brain-inspired deep learning model for EEG-based low-quality video target detection. This is a published research project (Expert Systems with Applications, 2025) implementing a neural network for binary classification of EEG signals.

**Key Architecture Components:**
- **Phased Encoder**: Dual spatial encoders that process split time stages of EEG data using multi-scale temporal convolutions (32, 64, 96 kernel sizes), depthwise separable convolutions, and SE (Squeeze-and-Excitation) attention mechanisms
- **Dynamic Fusion**: TCN-based (Temporal Convolutional Network) fusion module with cross-attention that aligns and combines features from two temporal phases, includes MMD (Maximum Mean Discrepancy) loss for distribution alignment
- **Input Format**: `[batch_size, 1, channels, timepoints]` where channels=59 EEG leads, timepoints=200 (1 second at 200Hz)

## Environment Setup

```bash
# Activate virtual environment
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

**Dependencies:** PyTorch 2.9.1, NumPy 1.22.4, scikit-learn 1.4.2, scipy 1.7.3, torchinfo 1.8.0, tqdm 4.65.0

## Data Format

EEG data is stored in MATLAB `.mat` files with structure:
- `EEG['data']`: shape `[channels, timepoints, trials]`
- Sampling rate: 200 Hz
- Expected channels: 59 EEG leads
- Time windows extracted:
  - No-target: 1-2 seconds (label=0)
  - Target: 5-6 seconds (label=1)

## Model Configuration

All hyperparameters are centralized in `my_config.py` via the `Config` class:

**Training Parameters:**
- `num_epochs`: Training epochs (default: 5)
- `learning_rate`: Adam optimizer learning rate (default: 0.001)
- `batchsize`: Batch size (default: 16)
- `seed`: Random seed for reproducibility (default: 42)

**Architecture Parameters:**
- `stage_timepoints`: Timepoints per phase for splitting (default: 150)
- `init_conv_layers`: Initial convolution output channels (default: 12)
- `conv_depth`: Depthwise convolution depth multiplier (default: 2)
- `SE_spatial_size`: SE layer bottleneck for spatial attention (default: 2)
- `SE_channels_size`: SE layer bottleneck for channel attention (default: 1)
- `GN_groups`: GroupNorm groups (default: 3)
- `dropout_rate`: Dropout probability (default: 0.2)
- `dilation_expand`: TCN dilation expansion factor (default: 2)
- `mmd_sigma`: MMD loss kernel bandwidth (default: 1.0)
- `TCN_hidden_dim`: TCN hidden dimension (default: 24)

**To modify hyperparameters:** Edit values in `my_config.py` and access via the global `config` object.

## Training Workflow

**Cross-Subject Evaluation** (recommended approach):
```bash
python cross_subject_evaluation.py
```
- Leave-one-subject-out cross-validation across 8 subjects
- Trains on 6 subjects, validates on 1, tests on 1
- Outputs per-subject and overall metrics (accuracy, hit rate, false alarm rate)
- Model checkpoints saved as `best_model.pth` (overwritten each fold)

**Custom Training:**
```python
from data_processing import extract_segments_path, create_dataloaders
from PSAFNet import PSAFNet
from train import train, test
from my_config import config

# Load data
X, y = extract_segments_path('path/to/data.mat', window_length=1.0)

# Create dataloaders
train_loader, val_loader, test_loader = create_dataloaders(
    X, y, batch_size=config.batchsize, train_ratio=0.8, val_ratio=0.1
)

# Initialize model
model = PSAFNet(
    stage_timepoints=config.stage_timepoints,
    lead=config.num_channels,
    time=config.num_timepoints
).to(device)

# Train
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
train(model, device, train_loader, val_loader, criterion, optimizer,
      num_epochs=config.num_epochs, save_path='best_model.pth')

# Test
test(model, device, test_loader, load_path='best_model.pth')
```

## Loss Function

The total loss combines two components:
```python
loss = CE_loss + alpha * similarity_loss
```
- `CE_loss`: CrossEntropyLoss for classification
- `similarity_loss`: MMD loss between two phase features (encourages distribution alignment)
- `alpha`: Weight for MMD loss (hardcoded as 1.5 in `train.py:13`)

## Code Architecture

**Module Responsibilities:**
- `PSAFNet.py`: Model architecture (SELayer, SE_channels_Block, Phased_Encoder, TCNLayer, CrossAttention, Dynamic_Fusion, PSAFNet)
- `my_config.py`: Centralized configuration via `Config` class, imported as global `config` object
- `data_processing.py`: Data loading, segmentation, and DataLoader creation
- `train.py`: Training loop, validation, and testing functions
- `utils.py`: Utility functions (currently only `set_seed`)
- `cross_subject_evaluation.py`: Cross-subject evaluation script (main entry point)

**Data Flow:**
1. Load `.mat` file → extract time windows → reshape to `[trials, channels, timepoints]`
2. Split into train/val/test → create DataLoaders
3. Add channel dimension in training loop: `inputs.unsqueeze(1)` → `[batch, 1, channels, timepoints]`
4. Model splits timepoints into two phases → Phased_Encoder extracts features → Dynamic_Fusion aligns and classifies

## Important Implementation Details

1. **Time Splitting**: Input is split at `stage_timepoints` (default 150):
   - Phase 1: `[:, :, :, :150]` (first 150 timepoints)
   - Phase 2: `[:, :, :, 50:]` (last 150 timepoints, overlapping)

2. **Temporal Alignment**: In `Dynamic_Fusion`, features are padded asymmetrically to simulate temporal offset before fusion:
   ```python
   padding_length = round(time_points * (200 / stage_timepoints - 1))
   padded_feature1 = [feature1, zeros(padding_length)]  # Pad right
   padded_feature2 = [zeros(padding_length), feature2]  # Pad left
   ```

3. **Model Checkpointing**: Best model is saved based on validation loss during training. Always load from checkpoint for testing.

4. **Reproducibility**: Call `set_seed(config.seed)` before training to ensure reproducible results.

5. **Device Handling**: Code supports CUDA. Set `CUDA_LAUNCH_BLOCKING=1` for debugging (already set in `cross_subject_evaluation.py`).

## File Path Conventions

The `cross_subject_evaluation.py` script expects data files at:
```
D:\machine learning\EEG_video\{data_type}_data\{subject_name}.mat
```
Update `file_paths` list in `cross_subject_evaluation.py` to match your data location.

## Metrics

- **Accuracy**: Overall classification accuracy
- **Hit Rate (TPR)**: True Positive Rate = TP / (TP + FN)
- **False Alarm Rate (FPR)**: FP / (FP + TN)
- **Prediction Time**: Average inference time per sample

## Common Modifications

**To change the phase split point:**
Modify `config.stage_timepoints` in `my_config.py` (must be ≤ `num_timepoints`)

**To adjust model capacity:**
- Increase `init_conv_layers` for more feature maps
- Increase `conv_depth` for deeper depthwise convolutions
- Increase `TCN_hidden_dim` for larger fusion network

**To modify loss weighting:**
Edit `alpha` value in `train.py:13` (currently 1.5)

**To use different time windows:**
Modify `init_time` and `target_time` parameters in `extract_segments_path()` calls
