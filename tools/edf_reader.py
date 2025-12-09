"""
EDF Reader Module

This module provides functionality to read and parse EDF (European Data Format)
files containing EEG data.
"""

import numpy as np
import mne
from pathlib import Path
from typing import Dict, List, Tuple, Optional


class EDFReader:
    """
    A class for reading and parsing EDF files containing EEG data.

    Attributes:
        file_path (Path): Path to the EDF file
        raw (mne.io.Raw): MNE Raw object containing the EEG data
        info (dict): Dictionary containing metadata about the recording
    """

    def __init__(self, file_path: str):
        """
        Initialize the EDFReader with a file path.

        Args:
            file_path (str): Path to the EDF file

        Raises:
            FileNotFoundError: If the file does not exist
            ValueError: If the file is not a valid EDF file
        """
        self.file_path = Path(file_path)

        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if self.file_path.suffix.lower() != '.edf':
            raise ValueError(f"File must be an EDF file, got: {self.file_path.suffix}")

        # Load the EDF file using MNE
        self.raw = mne.io.read_raw_edf(str(self.file_path), preload=True, verbose=False)
        self.info = self._extract_info()

    def _extract_info(self) -> Dict:
        """
        Extract metadata information from the EDF file.

        Returns:
            dict: Dictionary containing metadata
        """
        info = {
            'n_channels': self.raw.info['nchan'],
            'channel_names': self.raw.ch_names,
            'sampling_rate': self.raw.info['sfreq'],
            'n_samples': len(self.raw.times),
            'duration': self.raw.times[-1],
            'subject_info': self.raw.info.get('subject_info', {}),
            'meas_date': self.raw.info.get('meas_date', None)
        }
        return info

    def get_data(self, channels: Optional[List[str]] = None,
                 start: float = 0, stop: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get EEG data from the file.

        Args:
            channels (List[str], optional): List of channel names to extract.
                                           If None, all channels are returned.
            start (float): Start time in seconds (default: 0)
            stop (float, optional): Stop time in seconds. If None, reads until end.

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - data: EEG data array of shape (n_channels, n_samples)
                - times: Time array in seconds
        """
        # Convert time to samples
        start_sample = int(start * self.info['sampling_rate'])
        stop_sample = int(stop * self.info['sampling_rate']) if stop is not None else None

        # Get data
        if channels is not None:
            data, times = self.raw.get_data(picks=channels,
                                           start=start_sample,
                                           stop=stop_sample,
                                           return_times=True)
        else:
            data, times = self.raw.get_data(start=start_sample,
                                           stop=stop_sample,
                                           return_times=True)

        return data, times

    def get_channel_data(self, channel_name: str,
                        start: float = 0, stop: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get data for a specific channel.

        Args:
            channel_name (str): Name of the channel
            start (float): Start time in seconds (default: 0)
            stop (float, optional): Stop time in seconds. If None, reads until end.

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - data: 1D array of channel data
                - times: Time array in seconds
        """
        if channel_name not in self.info['channel_names']:
            raise ValueError(f"Channel '{channel_name}' not found. "
                           f"Available channels: {self.info['channel_names']}")

        data, times = self.get_data(channels=[channel_name], start=start, stop=stop)
        return data[0], times

    def get_info(self) -> Dict:
        """
        Get metadata information about the recording.

        Returns:
            dict: Dictionary containing metadata
        """
        return self.info.copy()

    def get_channel_names(self) -> List[str]:
        """
        Get list of all channel names.

        Returns:
            List[str]: List of channel names
        """
        return self.info['channel_names'].copy()

    def get_sampling_rate(self) -> float:
        """
        Get the sampling rate of the recording.

        Returns:
            float: Sampling rate in Hz
        """
        return self.info['sampling_rate']

    def get_duration(self) -> float:
        """
        Get the total duration of the recording.

        Returns:
            float: Duration in seconds
        """
        return self.info['duration']

    def get_n_channels(self) -> int:
        """
        Get the number of channels.

        Returns:
            int: Number of channels
        """
        return self.info['n_channels']

    def print_info(self):
        """
        Print a summary of the recording information.
        """
        print(f"\n{'='*60}")
        print(f"EDF File Information: {self.file_path.name}")
        print(f"{'='*60}")
        print(f"Number of channels: {self.info['n_channels']}")
        print(f"Sampling rate: {self.info['sampling_rate']} Hz")
        print(f"Duration: {self.info['duration']:.2f} seconds")
        print(f"Total samples: {self.info['n_samples']}")
        print(f"\nChannel names:")
        for i, ch in enumerate(self.info['channel_names'], 1):
            print(f"  {i:2d}. {ch}")
        print(f"{'='*60}\n")

    def export_to_numpy(self, output_path: str,
                       channels: Optional[List[str]] = None,
                       start: float = 0, stop: Optional[float] = None):
        """
        Export EEG data to a NumPy file (.npz format).

        Args:
            output_path (str): Path to save the .npz file
            channels (List[str], optional): List of channel names to export
            start (float): Start time in seconds
            stop (float, optional): Stop time in seconds
        """
        data, times = self.get_data(channels=channels, start=start, stop=stop)

        channel_names = channels if channels is not None else self.info['channel_names']

        np.savez(output_path,
                data=data,
                times=times,
                channel_names=channel_names,
                sampling_rate=self.info['sampling_rate'])

        print(f"Data exported to: {output_path}")

    def __repr__(self) -> str:
        """String representation of the EDFReader object."""
        return (f"EDFReader(file='{self.file_path.name}', "
                f"channels={self.info['n_channels']}, "
                f"duration={self.info['duration']:.2f}s)")


def load_edf_file(file_path: str) -> EDFReader:
    """
    Convenience function to load an EDF file.

    Args:
        file_path (str): Path to the EDF file

    Returns:
        EDFReader: EDFReader object
    """
    return EDFReader(file_path)


if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        reader = EDFReader(file_path)
        reader.print_info()
    else:
        print("Usage: python edf_reader.py <path_to_edf_file>")
