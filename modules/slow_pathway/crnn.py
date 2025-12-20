import torch
import torch.nn as nn

class CRNN(nn.Module):
    """
    Convolutional RNN - Slow Pathway for temporal tracking features

    Combines multi-scale temporal convolutions (inspired by TSception)
    with bidirectional LSTM for robust temporal modeling of visual
    tracking and attention signals.

    Args:
        num_channels (int): Number of EEG channels (default: 59)
        num_timepoints (int): Number of time points (default: 100)
        conv_channels (int): Base convolution channels (default: 16)
        rnn_hidden (int): LSTM hidden size (default: 32)
        rnn_layers (int): Number of LSTM layers (default: 2)
        output_channels (int): Output feature channels (default: 16)
        dropout_rate (float): Dropout probability (default: 0.3)
    """
    def __init__(self, num_channels=59, num_timepoints=100,
                 conv_channels=16, rnn_hidden=32, rnn_layers=2,
                 output_channels=16, dropout_rate=0.3):
        super(CRNN, self).__init__()

        self.num_channels = num_channels
        self.num_timepoints = num_timepoints

        # Multi-scale temporal convolutions (inspired by TSception)
        # Captures features at different temporal scales
        self.temp_conv1 = nn.Conv2d(1, conv_channels, (1, 32),
                                     padding=(0, 16), bias=False)
        self.temp_conv2 = nn.Conv2d(1, conv_channels, (1, 64),
                                     padding=(0, 32), bias=False)
        self.temp_conv3 = nn.Conv2d(1, conv_channels, (1, 96),
                                     padding=(0, 48), bias=False)

        self.batch_norm1 = nn.BatchNorm2d(conv_channels * 3)

        # Spatial convolution across all EEG channels
        self.spatial_conv = nn.Conv2d(
            conv_channels * 3, conv_channels * 3,
            (num_channels, 1), bias=False
        )
        self.batch_norm2 = nn.BatchNorm2d(conv_channels * 3)
        self.elu = nn.ELU()
        self.dropout1 = nn.Dropout(dropout_rate)

        # Bidirectional LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=conv_channels * 3,
            hidden_size=rnn_hidden,
            num_layers=rnn_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout_rate if rnn_layers > 1 else 0
        )

        # Output projection to match fast pathway dimensions
        self.output_conv = nn.Conv1d(
            rnn_hidden * 2,  # bidirectional doubles the hidden size
            output_channels,
            kernel_size=1
        )
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x):
        """
        Forward pass of C-RNN

        Args:
            x: Input tensor [batch, 1, channels, timepoints]

        Returns:
            Output tensor [batch, output_channels, 1, timepoints]
        """
        batch_size = x.size(0)

        # Multi-scale temporal convolutions
        t1 = self.temp_conv1(x)  # [B, C, 59, T]
        t2 = self.temp_conv2(x)  # [B, C, 59, T]
        t3 = self.temp_conv3(x)  # [B, C, 59, T]
        x = torch.cat([t1, t2, t3], dim=1)  # [B, 3C, 59, T]
        x = self.batch_norm1(x)

        # Spatial convolution
        x = self.spatial_conv(x)  # [B, 3C, 1, T]
        x = self.batch_norm2(x)
        x = self.elu(x)
        x = self.dropout1(x)

        # Reshape for LSTM: [B, T, 3C]
        x = x.squeeze(2).permute(0, 2, 1)  # [B, T, 3C]

        # Bidirectional LSTM
        x, (h_n, c_n) = self.lstm(x)  # [B, T, 2*hidden]
        x = self.dropout2(x)

        # Output projection: [B, T, 2*hidden] -> [B, output_channels, T]
        x = x.permute(0, 2, 1)  # [B, 2*hidden, T]
        x = self.output_conv(x)  # [B, output_channels, T]

        # Add spatial dimension back for compatibility
        x = x.unsqueeze(2)  # [B, output_channels, 1, T]

        return x


if __name__ == "__main__":
    # Quick test
    model = CRNN(num_channels=59, num_timepoints=100)
    x = torch.randn(4, 1, 59, 100)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
