import torch
import torch.nn as nn

class EEGNet(nn.Module):
    """
    EEGNet - Fast Pathway for early saliency feature extraction

    Reference: Lawhern et al., 2018
    Adapted for dual-pathway architecture to extract early EOG and saliency signals

    Args:
        num_channels (int): Number of EEG channels (default: 59)
        num_timepoints (int): Number of time points (default: 100)
        F1 (int): Number of temporal filters (default: 8)
        D (int): Depth multiplier for spatial filtering (default: 2)
        F2 (int): Number of separable filters (default: 16)
        dropout_rate (float): Dropout probability (default: 0.25)
    """
    def __init__(self, num_channels=59, num_timepoints=100,
                 F1=8, D=2, F2=16, dropout_rate=0.25):
        super(EEGNet, self).__init__()

        self.num_channels = num_channels
        self.num_timepoints = num_timepoints

        # Block 1: Temporal Convolution
        # Captures temporal features at different timescales
        self.temporal_conv = nn.Conv2d(
            1, F1, (1, 64), padding=(0, 32), bias=False
        )
        self.batch_norm1 = nn.BatchNorm2d(F1)

        # Block 2: Spatial Depthwise Convolution
        # Models spatial relationships across EEG channels
        self.spatial_conv = nn.Conv2d(
            F1, F1 * D, (num_channels, 1), groups=F1, bias=False
        )
        self.batch_norm2 = nn.BatchNorm2d(F1 * D)
        self.elu1 = nn.ELU()
        self.avg_pool1 = nn.AvgPool2d((1, 4))
        self.dropout1 = nn.Dropout(dropout_rate)

        # Block 3: Separable Convolution
        # Efficient feature extraction with depthwise + pointwise convolution
        self.separable_conv = nn.Conv2d(
            F1 * D, F1 * D, (1, 16), padding=(0, 8),
            groups=F1 * D, bias=False
        )
        self.pointwise_conv = nn.Conv2d(
            F1 * D, F2, (1, 1), bias=False
        )
        self.batch_norm3 = nn.BatchNorm2d(F2)
        self.elu2 = nn.ELU()
        self.avg_pool2 = nn.AvgPool2d((1, 8))
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x):
        """
        Forward pass of EEGNet

        Args:
            x: Input tensor [batch, 1, channels, timepoints]

        Returns:
            Output tensor [batch, F2, 1, timepoints//32]
        """
        # Block 1: Temporal convolution
        x = self.temporal_conv(x)  # [B, F1, C, T]
        x = self.batch_norm1(x)

        # Block 2: Spatial depthwise convolution
        x = self.spatial_conv(x)  # [B, F1*D, 1, T]
        x = self.batch_norm2(x)
        x = self.elu1(x)
        x = self.avg_pool1(x)  # [B, F1*D, 1, T//4]
        x = self.dropout1(x)

        # Block 3: Separable convolution
        x = self.separable_conv(x)  # [B, F1*D, 1, T//4]
        x = self.pointwise_conv(x)  # [B, F2, 1, T//4]
        x = self.batch_norm3(x)
        x = self.elu2(x)
        x = self.avg_pool2(x)  # [B, F2, 1, T//32]
        x = self.dropout2(x)

        return x


if __name__ == "__main__":
    # Quick test
    model = EEGNet(num_channels=59, num_timepoints=100)
    x = torch.randn(4, 1, 59, 100)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
