class DualPathwayConfig:
    """Configuration for Dual-Pathway Network"""

    # Data parameters
    num_channels = 59
    num_timepoints = 200
    fs = 200  # Sampling frequency (Hz)
    num_class = 2

    # Time split (0.5 seconds @ 200Hz)
    split_point = 100

    # Fast Pathway (EEGNet) parameters
    eegnet_F1 = 8
    eegnet_D = 2
    eegnet_F2 = 16
    eegnet_dropout = 0.25

    # Slow Pathway (C-RNN) parameters
    crnn_conv_channels = 16
    crnn_rnn_hidden = 32
    crnn_rnn_layers = 2
    crnn_output_channels = 16
    crnn_dropout = 0.3

    # Fusion parameters
    fusion_hidden_dim = 24
    fusion_tcn_layers = 3
    fusion_mmd_sigma = 1.0

    # Training parameters
    seed = 42
    num_epochs = 5
    learning_rate = 0.001
    batchsize = 16

    # Loss weights
    mmd_weight = 1.5  # Weight for MMD loss

    def print_config(self):
        print("\nDual-Pathway Network Configuration:")
        print("="*50)
        for key, value in vars(self).items():
            if not key.startswith('_') and not callable(value):
                print(f"  {key}: {value}")
        print("="*50)


config = DualPathwayConfig()
