"""
Data Visualizer Module

This module provides visualization functions for EEG data.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional, Tuple
from pathlib import Path


class DataVisualizer:
    """
    A class for visualizing EEG data with various plot types.
    """

    def __init__(self, data: np.ndarray, times: np.ndarray,
                 channel_names: Optional[List[str]] = None,
                 sampling_rate: Optional[float] = None):
        """
        Initialize the DataVisualizer.

        Args:
            data (np.ndarray): EEG data of shape (n_channels, n_samples)
            times (np.ndarray): Time array in seconds
            channel_names (List[str], optional): List of channel names
            sampling_rate (float, optional): Sampling rate in Hz
        """
        self.data = data
        self.times = times
        self.n_channels, self.n_samples = data.shape
        self.sampling_rate = sampling_rate

        if channel_names is None:
            self.channel_names = [f"Ch{i+1}" for i in range(self.n_channels)]
        else:
            self.channel_names = channel_names

        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)

    def plot_multichannel(self, channels: Optional[List[int]] = None,
                         start_time: float = 0, end_time: Optional[float] = None,
                         scale: float = 1.0, save_path: Optional[str] = None):
        """
        Plot multiple EEG channels as stacked time series.

        Args:
            channels (List[int], optional): List of channel indices to plot.
                                           If None, plots all channels.
            start_time (float): Start time in seconds
            end_time (float, optional): End time in seconds
            scale (float): Scaling factor for vertical spacing
            save_path (str, optional): Path to save the figure
        """
        if channels is None:
            channels = list(range(self.n_channels))

        # Time window
        start_idx = np.searchsorted(self.times, start_time)
        end_idx = np.searchsorted(self.times, end_time) if end_time else len(self.times)

        time_window = self.times[start_idx:end_idx]
        data_window = self.data[channels, start_idx:end_idx]

        # Create figure
        fig, ax = plt.subplots(figsize=(14, max(8, len(channels) * 0.5)))

        # Plot each channel with vertical offset
        offsets = np.arange(len(channels)) * scale * np.std(data_window)

        for i, ch_idx in enumerate(channels):
            ax.plot(time_window, data_window[i] + offsets[i],
                   label=self.channel_names[ch_idx], linewidth=0.8)

        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel('Amplitude (μV)', fontsize=12)
        ax.set_title('Multi-Channel EEG Time Series', fontsize=14, fontweight='bold')
        ax.set_yticks(offsets)
        ax.set_yticklabels([self.channel_names[ch] for ch in channels])
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to: {save_path}")

        plt.show()

    def plot_single_channel(self, channel_idx: int,
                           start_time: float = 0, end_time: Optional[float] = None,
                           save_path: Optional[str] = None):
        """
        Plot a single EEG channel.

        Args:
            channel_idx (int): Index of the channel to plot
            start_time (float): Start time in seconds
            end_time (float, optional): End time in seconds
            save_path (str, optional): Path to save the figure
        """
        start_idx = np.searchsorted(self.times, start_time)
        end_idx = np.searchsorted(self.times, end_time) if end_time else len(self.times)

        time_window = self.times[start_idx:end_idx]
        data_window = self.data[channel_idx, start_idx:end_idx]

        fig, ax = plt.subplots(figsize=(14, 4))

        ax.plot(time_window, data_window, linewidth=1.0, color='steelblue')
        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel('Amplitude (μV)', fontsize=12)
        ax.set_title(f'EEG Channel: {self.channel_names[channel_idx]}',
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to: {save_path}")

        plt.show()

    def plot_power_spectrum(self, channel_idx: int,
                           frequencies: np.ndarray, psd: np.ndarray,
                           freq_range: Tuple[float, float] = (0, 50),
                           save_path: Optional[str] = None):
        """
        Plot power spectral density for a channel.

        Args:
            channel_idx (int): Index of the channel
            frequencies (np.ndarray): Frequency array
            psd (np.ndarray): Power spectral density array
            freq_range (Tuple[float, float]): Frequency range to display
            save_path (str, optional): Path to save the figure
        """
        # Filter frequency range
        freq_mask = (frequencies >= freq_range[0]) & (frequencies <= freq_range[1])
        freq_filtered = frequencies[freq_mask]
        psd_filtered = psd[freq_mask]

        fig, ax = plt.subplots(figsize=(12, 5))

        ax.semilogy(freq_filtered, psd_filtered, linewidth=1.5, color='darkblue')
        ax.set_xlabel('Frequency (Hz)', fontsize=12)
        ax.set_ylabel('Power Spectral Density (μV²/Hz)', fontsize=12)
        ax.set_title(f'Power Spectrum - {self.channel_names[channel_idx]}',
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Add frequency band regions
        bands = {'Delta': (0.5, 4), 'Theta': (4, 8), 'Alpha': (8, 13),
                'Beta': (13, 30), 'Gamma': (30, 50)}
        colors = ['lightblue', 'lightgreen', 'lightyellow', 'lightcoral', 'plum']

        for (band_name, (low, high)), color in zip(bands.items(), colors):
            if low >= freq_range[0] and high <= freq_range[1]:
                ax.axvspan(low, high, alpha=0.2, color=color, label=band_name)

        ax.legend(loc='upper right')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to: {save_path}")

        plt.show()

    def plot_heatmap(self, start_time: float = 0, end_time: Optional[float] = None,
                    save_path: Optional[str] = None):
        """
        Plot a heatmap of all channels over time.

        Args:
            start_time (float): Start time in seconds
            end_time (float, optional): End time in seconds
            save_path (str, optional): Path to save the figure
        """
        start_idx = np.searchsorted(self.times, start_time)
        end_idx = np.searchsorted(self.times, end_time) if end_time else len(self.times)

        time_window = self.times[start_idx:end_idx]
        data_window = self.data[:, start_idx:end_idx]

        fig, ax = plt.subplots(figsize=(14, max(8, self.n_channels * 0.3)))

        # Create heatmap
        im = ax.imshow(data_window, aspect='auto', cmap='RdBu_r',
                      extent=[time_window[0], time_window[-1], 0, self.n_channels],
                      interpolation='bilinear')

        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel('Channel', fontsize=12)
        ax.set_title('EEG Heatmap (All Channels)', fontsize=14, fontweight='bold')

        # Set y-axis labels
        ax.set_yticks(np.arange(self.n_channels) + 0.5)
        ax.set_yticklabels(self.channel_names)

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Amplitude (μV)', fontsize=11)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to: {save_path}")

        plt.show()

    def plot_correlation_matrix(self, correlation_matrix: np.ndarray,
                               save_path: Optional[str] = None):
        """
        Plot correlation matrix between channels.

        Args:
            correlation_matrix (np.ndarray): Correlation matrix
            save_path (str, optional): Path to save the figure
        """
        fig, ax = plt.subplots(figsize=(12, 10))

        # Create heatmap
        sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm',
                   center=0, vmin=-1, vmax=1,
                   xticklabels=self.channel_names,
                   yticklabels=self.channel_names,
                   square=True, linewidths=0.5, cbar_kws={'label': 'Correlation'},
                   ax=ax)

        ax.set_title('Channel Correlation Matrix', fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to: {save_path}")

        plt.show()

    def plot_band_powers(self, band_powers: dict, save_path: Optional[str] = None):
        """
        Plot power in different frequency bands.

        Args:
            band_powers (dict): Dictionary of band powers
            save_path (str, optional): Path to save the figure
        """
        bands = list(band_powers.keys())
        powers = list(band_powers.values())

        fig, ax = plt.subplots(figsize=(10, 6))

        colors = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c', '#9b59b6']
        bars = ax.bar(bands, powers, color=colors[:len(bands)], alpha=0.8, edgecolor='black')

        ax.set_xlabel('Frequency Band', fontsize=12)
        ax.set_ylabel('Power (μV²)', fontsize=12)
        ax.set_title('Power Distribution Across Frequency Bands',
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2e}',
                   ha='center', va='bottom', fontsize=10)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to: {save_path}")

        plt.show()

    def plot_statistics_summary(self, stats_dict: dict, save_path: Optional[str] = None):
        """
        Plot summary statistics for all channels.

        Args:
            stats_dict (dict): Dictionary containing statistics
            save_path (str, optional): Path to save the figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Mean
        axes[0, 0].bar(range(self.n_channels), stats_dict['mean'], color='steelblue', alpha=0.7)
        axes[0, 0].set_title('Mean Amplitude', fontweight='bold')
        axes[0, 0].set_xlabel('Channel')
        axes[0, 0].set_ylabel('Mean (μV)')
        axes[0, 0].set_xticks(range(self.n_channels))
        axes[0, 0].set_xticklabels(self.channel_names, rotation=45, ha='right')
        axes[0, 0].grid(True, alpha=0.3, axis='y')

        # Standard Deviation
        axes[0, 1].bar(range(self.n_channels), stats_dict['std'], color='coral', alpha=0.7)
        axes[0, 1].set_title('Standard Deviation', fontweight='bold')
        axes[0, 1].set_xlabel('Channel')
        axes[0, 1].set_ylabel('Std (μV)')
        axes[0, 1].set_xticks(range(self.n_channels))
        axes[0, 1].set_xticklabels(self.channel_names, rotation=45, ha='right')
        axes[0, 1].grid(True, alpha=0.3, axis='y')

        # Range (Peak-to-Peak)
        axes[1, 0].bar(range(self.n_channels), stats_dict['range'], color='mediumseagreen', alpha=0.7)
        axes[1, 0].set_title('Signal Range (Peak-to-Peak)', fontweight='bold')
        axes[1, 0].set_xlabel('Channel')
        axes[1, 0].set_ylabel('Range (μV)')
        axes[1, 0].set_xticks(range(self.n_channels))
        axes[1, 0].set_xticklabels(self.channel_names, rotation=45, ha='right')
        axes[1, 0].grid(True, alpha=0.3, axis='y')

        # RMS
        axes[1, 1].bar(range(self.n_channels), stats_dict['rms'], color='mediumpurple', alpha=0.7)
        axes[1, 1].set_title('Root Mean Square (RMS)', fontweight='bold')
        axes[1, 1].set_xlabel('Channel')
        axes[1, 1].set_ylabel('RMS (μV)')
        axes[1, 1].set_xticks(range(self.n_channels))
        axes[1, 1].set_xticklabels(self.channel_names, rotation=45, ha='right')
        axes[1, 1].grid(True, alpha=0.3, axis='y')

        plt.suptitle('Statistical Summary - All Channels', fontsize=16, fontweight='bold', y=1.00)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to: {save_path}")

        plt.show()

    def create_overview_figure(self, save_path: Optional[str] = None):
        """
        Create a comprehensive overview figure with multiple subplots.

        Args:
            save_path (str, optional): Path to save the figure
        """
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

        # 1. Multi-channel time series (top, spanning both columns)
        ax1 = fig.add_subplot(gs[0, :])
        n_display = min(10, self.n_channels)
        offsets = np.arange(n_display) * np.std(self.data[:n_display])
        for i in range(n_display):
            ax1.plot(self.times, self.data[i] + offsets[i],
                    label=self.channel_names[i], linewidth=0.6)
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Amplitude (μV)')
        ax1.set_title('Multi-Channel EEG (First 10 Channels)', fontweight='bold')
        ax1.set_yticks(offsets)
        ax1.set_yticklabels(self.channel_names[:n_display])
        ax1.grid(True, alpha=0.3)

        # 2. Heatmap
        ax2 = fig.add_subplot(gs[1, 0])
        im = ax2.imshow(self.data, aspect='auto', cmap='RdBu_r', interpolation='bilinear')
        ax2.set_xlabel('Time (samples)')
        ax2.set_ylabel('Channel')
        ax2.set_title('EEG Heatmap', fontweight='bold')
        plt.colorbar(im, ax=ax2, label='Amplitude (μV)')

        # 3. Mean amplitude per channel
        ax3 = fig.add_subplot(gs[1, 1])
        mean_vals = np.mean(self.data, axis=1)
        ax3.barh(range(self.n_channels), mean_vals, color='steelblue', alpha=0.7)
        ax3.set_yticks(range(self.n_channels))
        ax3.set_yticklabels(self.channel_names, fontsize=8)
        ax3.set_xlabel('Mean Amplitude (μV)')
        ax3.set_title('Mean Amplitude per Channel', fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='x')

        # 4. Standard deviation per channel
        ax4 = fig.add_subplot(gs[2, 0])
        std_vals = np.std(self.data, axis=1)
        ax4.bar(range(self.n_channels), std_vals, color='coral', alpha=0.7)
        ax4.set_xticks(range(self.n_channels))
        ax4.set_xticklabels(self.channel_names, rotation=45, ha='right', fontsize=8)
        ax4.set_ylabel('Standard Deviation (μV)')
        ax4.set_title('Signal Variability per Channel', fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y')

        # 5. Signal range per channel
        ax5 = fig.add_subplot(gs[2, 1])
        range_vals = np.ptp(self.data, axis=1)
        ax5.bar(range(self.n_channels), range_vals, color='mediumseagreen', alpha=0.7)
        ax5.set_xticks(range(self.n_channels))
        ax5.set_xticklabels(self.channel_names, rotation=45, ha='right', fontsize=8)
        ax5.set_ylabel('Range (μV)')
        ax5.set_title('Signal Range per Channel', fontweight='bold')
        ax5.grid(True, alpha=0.3, axis='y')

        plt.suptitle('EEG Data Overview', fontsize=18, fontweight='bold', y=0.995)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to: {save_path}")

        plt.show()


if __name__ == "__main__":
    # Example usage with synthetic data
    print("Generating synthetic EEG data for demonstration...")

    n_channels = 8
    n_samples = 2000
    sampling_rate = 200.0
    duration = n_samples / sampling_rate

    # Generate synthetic data
    np.random.seed(42)
    times = np.linspace(0, duration, n_samples)
    data = np.random.randn(n_channels, n_samples) * 10

    channel_names = [f"EEG{i+1}" for i in range(n_channels)]

    # Create visualizer
    viz = DataVisualizer(data, times, channel_names, sampling_rate)

    # Create overview
    viz.create_overview_figure()
