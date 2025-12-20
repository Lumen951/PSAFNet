import torch
import torch.nn as nn
from modules.fast_pathway.eegnet import EEGNet
from modules.slow_pathway.crnn import CRNN
from modules.fusion.multi_mechanism_fusion import MultiMechanismFusion


class DualPathwayNet(nn.Module):
    """
    Dual-Pathway Network for EEG-based Target Detection

    Brain-inspired architecture with:
    - Fast Pathway (EEGNet): Early saliency and EOG features
    - Slow Pathway (C-RNN): Temporal tracking and attention features
    - Multi-Mechanism Fusion: Adaptive integration of both pathways

    Args:
        num_channels (int): Number of EEG channels (default: 59)
        num_timepoints (int): Total time points in input (default: 200)
        split_point (int): Time split for fast/slow pathways (default: 100)
        output_dim (int): Number of output classes (default: 2)
    """
    def __init__(self, num_channels=59, num_timepoints=200,
                 split_point=100, output_dim=2):
        super(DualPathwayNet, self).__init__()

        self.num_channels = num_channels
        self.num_timepoints = num_timepoints
        self.split_point = split_point

        # Fast Pathway: EEGNet for early EOG signals
        self.fast_pathway = EEGNet(
            num_channels=num_channels,
            num_timepoints=split_point,
            F1=8, D=2, F2=16, dropout_rate=0.25
        )

        # Slow Pathway: C-RNN for late tracking signals
        self.slow_pathway = CRNN(
            num_channels=num_channels,
            num_timepoints=num_timepoints - split_point,
            conv_channels=16,
            rnn_hidden=32,
            rnn_layers=2,
            output_channels=16,
            dropout_rate=0.3
        )

        # Multi-Mechanism Fusion
        self.fusion = MultiMechanismFusion(
            input_dim=16,
            hidden_dim=24,
            output_dim=output_dim,
            num_tcn_layers=3,
            mmd_sigma=1.0
        )

    def forward(self, x):
        """
        Forward pass of Dual-Pathway Network

        Args:
            x: Input EEG tensor [batch, 1, channels, timepoints]

        Returns:
            output: Classification logits [batch, num_classes]
            mmd_loss: Distribution alignment loss (scalar)
        """
        # Split input at specified time point
        fast_input = x[:, :, :, :self.split_point]
        slow_input = x[:, :, :, self.split_point:]

        # Extract features from both pathways
        fast_features = self.fast_pathway(fast_input)
        slow_features = self.slow_pathway(slow_input)

        # Fuse and classify
        output, mmd_loss = self.fusion(fast_features, slow_features)

        return output, mmd_loss


if __name__ == "__main__":
    print("="*60)
    print("Dual-Pathway Network Architecture Test")
    print("="*60)

    # Initialize model
    model = DualPathwayNet(
        num_channels=59,
        num_timepoints=200,
        split_point=100,
        output_dim=2
    )

    # Test forward pass
    x = torch.randn(4, 1, 59, 200)
    output, mmd_loss = model(x)

    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"MMD loss: {mmd_loss.item():.4f}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Component breakdown
    fast_params = sum(p.numel() for p in model.fast_pathway.parameters())
    slow_params = sum(p.numel() for p in model.slow_pathway.parameters())
    fusion_params = sum(p.numel() for p in model.fusion.parameters())

    print(f"\nParameter breakdown:")
    print(f"  Fast Pathway (EEGNet): {fast_params:,}")
    print(f"  Slow Pathway (C-RNN): {slow_params:,}")
    print(f"  Fusion Module: {fusion_params:,}")

    print("\n" + "="*60)
    print("Model test completed successfully!")
    print("="*60)
