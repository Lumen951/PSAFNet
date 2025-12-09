"""
Data Statistics Module

This module provides statistical analysis functions for EEG data.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy import stats
from scipy.signal import welch


class DataStatistics:
    """
    A class for computing statistical measures on EEG data.
    """

    def __init__(self, data: np.ndarray, sampling_rate: float,
                 channel_names: Optional[List[str]] = None):
        """
        Initialize the DataStatistics object.

        Args:
            data (np.ndarray): EEG data of shape (n_channels, n_samples)
            sampling_rate (float): Sampling rate in Hz
            channel_names (List[str], optional): List of channel names
        """
        self.data = data
        self.sampling_rate = sampling_rate
        self.n_channels, self.n_samples = data.shape

        if channel_names is None:
            self.channel_names = [f"Ch{i+1}" for i in range(self.n_channels)]
        else:
            self.channel_names = channel_names

    def compute_basic_stats(self) -> Dict:
        """
        Compute basic statistical measures for all channels.

        Returns:
            dict: Dictionary containing statistical measures
        """
        stats_dict = {
            'mean': np.mean(self.data, axis=1),
            'std': np.std(self.data, axis=1),
            'min': np.min(self.data, axis=1),
            'max': np.max(self.data, axis=1),
            'median': np.median(self.data, axis=1),
            'variance': np.var(self.data, axis=1),
            'range': np.ptp(self.data, axis=1),  # peak-to-peak
            'rms': np.sqrt(np.mean(self.data**2, axis=1))  # root mean square
        }
        return stats_dict

    def compute_channel_stats(self, channel_idx: int) -> Dict:
        """
        Compute detailed statistics for a specific channel.

        Args:
            channel_idx (int): Index of the channel

        Returns:
            dict: Dictionary containing channel statistics
        """
        channel_data = self.data[channel_idx]

        stats_dict = {
            'channel_name': self.channel_names[channel_idx],
            'mean': np.mean(channel_data),
            'std': np.std(channel_data),
            'min': np.min(channel_data),
            'max': np.max(channel_data),
            'median': np.median(channel_data),
            'variance': np.var(channel_data),
            'range': np.ptp(channel_data),
            'rms': np.sqrt(np.mean(channel_data**2)),
            'skewness': stats.skew(channel_data),
            'kurtosis': stats.kurtosis(channel_data),
            'percentile_25': np.percentile(channel_data, 25),
            'percentile_75': np.percentile(channel_data, 75),
            'iqr': stats.iqr(channel_data)
        }
        return stats_dict

    def compute_power_spectrum(self, channel_idx: int,
                              nperseg: int = 256) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute power spectral density for a channel using Welch's method.

        Args:
            channel_idx (int): Index of the channel
            nperseg (int): Length of each segment for Welch's method

        Returns:
            Tuple[np.ndarray, np.ndarray]: (frequencies, power spectral density)
        """
        channel_data = self.data[channel_idx]
        frequencies, psd = welch(channel_data, fs=self.sampling_rate,
                                nperseg=nperseg, scaling='density')
        return frequencies, psd

    def compute_band_power(self, channel_idx: int,
                          freq_bands: Optional[Dict[str, Tuple[float, float]]] = None) -> Dict:
        """
        Compute power in different frequency bands.

        Args:
            channel_idx (int): Index of the channel
            freq_bands (dict, optional): Dictionary of frequency bands
                                        Default: Delta, Theta, Alpha, Beta, Gamma

        Returns:
            dict: Dictionary containing power in each frequency band
        """
        if freq_bands is None:
            freq_bands = {
                'Delta': (0.5, 4),
                'Theta': (4, 8),
                'Alpha': (8, 13),
                'Beta': (13, 30),
                'Gamma': (30, 50)
            }

        frequencies, psd = self.compute_power_spectrum(channel_idx)

        band_powers = {}
        for band_name, (low_freq, high_freq) in freq_bands.items():
            idx_band = np.logical_and(frequencies >= low_freq, frequencies <= high_freq)
            band_powers[band_name] = np.trapz(psd[idx_band], frequencies[idx_band])

        return band_powers

    def detect_artifacts(self, threshold_std: float = 5.0) -> Dict:
        """
        Detect potential artifacts in the data based on amplitude threshold.

        Args:
            threshold_std (float): Number of standard deviations for threshold

        Returns:
            dict: Dictionary containing artifact information for each channel
        """
        artifacts = {}

        for i, ch_name in enumerate(self.channel_names):
            channel_data = self.data[i]
            mean = np.mean(channel_data)
            std = np.std(channel_data)
            threshold = threshold_std * std

            # Find samples exceeding threshold
            artifact_indices = np.where(np.abs(channel_data - mean) > threshold)[0]

            artifacts[ch_name] = {
                'n_artifacts': len(artifact_indices),
                'artifact_percentage': (len(artifact_indices) / self.n_samples) * 100,
                'artifact_indices': artifact_indices
            }

        return artifacts

    def compute_correlation_matrix(self) -> np.ndarray:
        """
        Compute correlation matrix between all channels.

        Returns:
            np.ndarray: Correlation matrix of shape (n_channels, n_channels)
        """
        return np.corrcoef(self.data)

    def compute_signal_quality(self) -> Dict:
        """
        Compute signal quality metrics for all channels.

        Returns:
            dict: Dictionary containing quality metrics
        """
        quality_metrics = {}

        for i, ch_name in enumerate(self.channel_names):
            channel_data = self.data[i]

            # Signal-to-noise ratio (simplified)
            signal_power = np.mean(channel_data**2)
            noise_estimate = np.var(np.diff(channel_data))
            snr = 10 * np.log10(signal_power / noise_estimate) if noise_estimate > 0 else np.inf

            # Zero-crossing rate
            zero_crossings = np.sum(np.diff(np.sign(channel_data)) != 0)
            zcr = zero_crossings / self.n_samples

            quality_metrics[ch_name] = {
                'snr_db': snr,
                'zero_crossing_rate': zcr,
                'dynamic_range': np.ptp(channel_data)
            }

        return quality_metrics

    def generate_summary_report(self) -> str:
        """
        Generate a comprehensive summary report of the data.

        Returns:
            str: Formatted summary report
        """
        basic_stats = self.compute_basic_stats()
        artifacts = self.detect_artifacts()

        report = []
        report.append("=" * 80)
        report.append("EEG DATA STATISTICAL SUMMARY")
        report.append("=" * 80)
        report.append(f"\nData Shape: {self.n_channels} channels × {self.n_samples} samples")
        report.append(f"Sampling Rate: {self.sampling_rate} Hz")
        report.append(f"Duration: {self.n_samples / self.sampling_rate:.2f} seconds")

        report.append("\n" + "-" * 80)
        report.append("OVERALL STATISTICS")
        report.append("-" * 80)
        report.append(f"Mean (across all channels): {np.mean(basic_stats['mean']):.4f}")
        report.append(f"Std (across all channels): {np.mean(basic_stats['std']):.4f}")
        report.append(f"Global Min: {np.min(basic_stats['min']):.4f}")
        report.append(f"Global Max: {np.max(basic_stats['max']):.4f}")

        report.append("\n" + "-" * 80)
        report.append("PER-CHANNEL STATISTICS")
        report.append("-" * 80)
        report.append(f"{'Channel':<15} {'Mean':<12} {'Std':<12} {'Min':<12} {'Max':<12}")
        report.append("-" * 80)

        for i, ch_name in enumerate(self.channel_names):
            report.append(
                f"{ch_name:<15} "
                f"{basic_stats['mean'][i]:<12.4f} "
                f"{basic_stats['std'][i]:<12.4f} "
                f"{basic_stats['min'][i]:<12.4f} "
                f"{basic_stats['max'][i]:<12.4f}"
            )

        report.append("\n" + "-" * 80)
        report.append("ARTIFACT DETECTION (>5 std)")
        report.append("-" * 80)

        total_artifacts = sum(art['n_artifacts'] for art in artifacts.values())
        report.append(f"Total artifacts detected: {total_artifacts}")

        channels_with_artifacts = [ch for ch, art in artifacts.items()
                                  if art['n_artifacts'] > 0]
        if channels_with_artifacts:
            report.append(f"\nChannels with artifacts ({len(channels_with_artifacts)}):")
            for ch_name in channels_with_artifacts:
                art = artifacts[ch_name]
                report.append(f"  {ch_name}: {art['n_artifacts']} "
                            f"({art['artifact_percentage']:.2f}%)")
        else:
            report.append("\nNo artifacts detected in any channel.")

        report.append("\n" + "=" * 80)

        return "\n".join(report)

    def print_summary(self):
        """Print the summary report."""
        print(self.generate_summary_report())

    def save_summary(self, output_path: str):
        """
        Save the summary report to a text file.

        Args:
            output_path (str): Path to save the report
        """
        report = self.generate_summary_report()
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"Summary report saved to: {output_path}")


def analyze_eeg_data(data: np.ndarray, sampling_rate: float,
                    channel_names: Optional[List[str]] = None) -> DataStatistics:
    """
    Convenience function to create a DataStatistics object.

    Args:
        data (np.ndarray): EEG data of shape (n_channels, n_samples)
        sampling_rate (float): Sampling rate in Hz
        channel_names (List[str], optional): List of channel names

    Returns:
        DataStatistics: DataStatistics object
    """
    return DataStatistics(data, sampling_rate, channel_names)


if __name__ == "__main__":
    # Example usage with synthetic data
    print("Generating synthetic EEG data for demonstration...")

    n_channels = 5
    n_samples = 1000
    sampling_rate = 200.0

    # Generate synthetic data
    np.random.seed(42)
    data = np.random.randn(n_channels, n_samples) * 10

    channel_names = [f"EEG{i+1}" for i in range(n_channels)]

    # Create statistics object
    stats = DataStatistics(data, sampling_rate, channel_names)

    # Print summary
    stats.print_summary()
