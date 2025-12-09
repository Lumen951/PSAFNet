"""
Batch Processor Module

This module provides batch processing utilities for multiple EEG files.
"""

import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
from tqdm import tqdm
import pandas as pd

try:
    from .edf_reader import EDFReader
    from .data_statistics import DataStatistics
except ImportError:
    from edf_reader import EDFReader
    from data_statistics import DataStatistics


class BatchProcessor:
    """
    A class for batch processing multiple EEG files.
    """

    def __init__(self, data_dir: str, file_pattern: str = "*.edf"):
        """
        Initialize the BatchProcessor.

        Args:
            data_dir (str): Directory containing EEG files
            file_pattern (str): File pattern to match (default: "*.edf")
        """
        self.data_dir = Path(data_dir)

        if not self.data_dir.exists():
            raise FileNotFoundError(f"Directory not found: {data_dir}")

        # Find all matching files
        self.file_paths = sorted(list(self.data_dir.glob(file_pattern)))

        if not self.file_paths:
            raise ValueError(f"No files found matching pattern '{file_pattern}' "
                           f"in directory '{data_dir}'")

        print(f"Found {len(self.file_paths)} EDF files in {data_dir}")

    def get_file_list(self) -> List[Path]:
        """
        Get list of all files to be processed.

        Returns:
            List[Path]: List of file paths
        """
        return self.file_paths.copy()

    def generate_overview_report(self, output_path: Optional[str] = None) -> pd.DataFrame:
        """
        Generate an overview report for all files.

        Args:
            output_path (str, optional): Path to save the report CSV

        Returns:
            pd.DataFrame: DataFrame containing overview information
        """
        print("\nGenerating overview report...")

        records = []

        for file_path in tqdm(self.file_paths, desc="Processing files"):
            try:
                reader = EDFReader(str(file_path))
                info = reader.get_info()

                record = {
                    'filename': file_path.name,
                    'n_channels': info['n_channels'],
                    'sampling_rate': info['sampling_rate'],
                    'duration_sec': info['duration'],
                    'n_samples': info['n_samples'],
                    'file_size_mb': file_path.stat().st_size / (1024 * 1024)
                }

                records.append(record)

            except Exception as e:
                print(f"\nError processing {file_path.name}: {e}")
                records.append({
                    'filename': file_path.name,
                    'error': str(e)
                })

        df = pd.DataFrame(records)

        if output_path:
            df.to_csv(output_path, index=False)
            print(f"\nOverview report saved to: {output_path}")

        return df

    def generate_statistics_report(self, output_dir: str,
                                   start_time: float = 0,
                                   duration: Optional[float] = None):
        """
        Generate detailed statistics reports for all files.

        Args:
            output_dir (str): Directory to save individual reports
            start_time (float): Start time for analysis (seconds)
            duration (float, optional): Duration to analyze (seconds)
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        print(f"\nGenerating statistics reports in: {output_dir}")

        for file_path in tqdm(self.file_paths, desc="Processing files"):
            try:
                # Read data
                reader = EDFReader(str(file_path))
                end_time = start_time + duration if duration else None
                data, times = reader.get_data(start=start_time, stop=end_time)

                # Compute statistics
                stats = DataStatistics(data, reader.get_sampling_rate(),
                                     reader.get_channel_names())

                # Save report
                report_path = output_path / f"{file_path.stem}_statistics.txt"
                stats.save_summary(str(report_path))

            except Exception as e:
                print(f"\nError processing {file_path.name}: {e}")

        print(f"\nStatistics reports saved to: {output_dir}")

    def compare_datasets(self, dir1: str, dir2: str,
                        output_path: Optional[str] = None) -> pd.DataFrame:
        """
        Compare two datasets (e.g., raw vs cleaned data).

        Args:
            dir1 (str): First directory (e.g., raw data)
            dir2 (str): Second directory (e.g., cleaned data)
            output_path (str, optional): Path to save comparison CSV

        Returns:
            pd.DataFrame: Comparison results
        """
        print(f"\nComparing datasets:")
        print(f"  Dataset 1: {dir1}")
        print(f"  Dataset 2: {dir2}")

        dir1_path = Path(dir1)
        dir2_path = Path(dir2)

        files1 = {f.name: f for f in dir1_path.glob("*.edf")}
        files2 = {f.name: f for f in dir2_path.glob("*.edf")}

        common_files = set(files1.keys()) & set(files2.keys())
        only_in_dir1 = set(files1.keys()) - set(files2.keys())
        only_in_dir2 = set(files2.keys()) - set(files1.keys())

        print(f"\nCommon files: {len(common_files)}")
        print(f"Only in dataset 1: {len(only_in_dir1)}")
        print(f"Only in dataset 2: {len(only_in_dir2)}")

        if only_in_dir1:
            print(f"  Files only in dataset 1: {sorted(only_in_dir1)}")
        if only_in_dir2:
            print(f"  Files only in dataset 2: {sorted(only_in_dir2)}")

        # Compare common files
        records = []

        for filename in tqdm(sorted(common_files), desc="Comparing files"):
            try:
                reader1 = EDFReader(str(files1[filename]))
                reader2 = EDFReader(str(files2[filename]))

                data1, _ = reader1.get_data()
                data2, _ = reader2.get_data()

                record = {
                    'filename': filename,
                    'dataset1_channels': reader1.get_n_channels(),
                    'dataset2_channels': reader2.get_n_channels(),
                    'dataset1_duration': reader1.get_duration(),
                    'dataset2_duration': reader2.get_duration(),
                    'dataset1_mean': np.mean(data1),
                    'dataset2_mean': np.mean(data2),
                    'dataset1_std': np.std(data1),
                    'dataset2_std': np.std(data2),
                    'mean_difference': np.mean(data1) - np.mean(data2),
                    'std_difference': np.std(data1) - np.std(data2)
                }

                records.append(record)

            except Exception as e:
                print(f"\nError comparing {filename}: {e}")

        df = pd.DataFrame(records)

        if output_path:
            df.to_csv(output_path, index=False)
            print(f"\nComparison report saved to: {output_path}")

        return df

    def extract_time_windows(self, output_dir: str,
                            windows: List[tuple],
                            window_names: Optional[List[str]] = None):
        """
        Extract specific time windows from all files.

        Args:
            output_dir (str): Directory to save extracted windows
            windows (List[tuple]): List of (start_time, end_time) tuples in seconds
            window_names (List[str], optional): Names for each window
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        if window_names is None:
            window_names = [f"window_{i+1}" for i in range(len(windows))]

        print(f"\nExtracting time windows to: {output_dir}")
        print(f"Windows: {list(zip(window_names, windows))}")

        for file_path in tqdm(self.file_paths, desc="Processing files"):
            try:
                reader = EDFReader(str(file_path))

                for window_name, (start_time, end_time) in zip(window_names, windows):
                    data, times = reader.get_data(start=start_time, stop=end_time)

                    # Save as numpy file
                    save_path = output_path / f"{file_path.stem}_{window_name}.npz"
                    np.savez(save_path,
                            data=data,
                            times=times,
                            channel_names=reader.get_channel_names(),
                            sampling_rate=reader.get_sampling_rate(),
                            original_file=file_path.name,
                            window_start=start_time,
                            window_end=end_time)

            except Exception as e:
                print(f"\nError processing {file_path.name}: {e}")

        print(f"\nTime windows extracted to: {output_dir}")

    def compute_summary_statistics(self) -> Dict:
        """
        Compute summary statistics across all files.

        Returns:
            dict: Dictionary containing summary statistics
        """
        print("\nComputing summary statistics across all files...")

        all_means = []
        all_stds = []
        all_durations = []
        all_n_channels = []
        all_sampling_rates = []

        for file_path in tqdm(self.file_paths, desc="Processing files"):
            try:
                reader = EDFReader(str(file_path))
                data, _ = reader.get_data()

                all_means.append(np.mean(data))
                all_stds.append(np.std(data))
                all_durations.append(reader.get_duration())
                all_n_channels.append(reader.get_n_channels())
                all_sampling_rates.append(reader.get_sampling_rate())

            except Exception as e:
                print(f"\nError processing {file_path.name}: {e}")

        summary = {
            'n_files': len(self.file_paths),
            'mean_amplitude': {
                'mean': np.mean(all_means),
                'std': np.std(all_means),
                'min': np.min(all_means),
                'max': np.max(all_means)
            },
            'std_amplitude': {
                'mean': np.mean(all_stds),
                'std': np.std(all_stds),
                'min': np.min(all_stds),
                'max': np.max(all_stds)
            },
            'duration': {
                'mean': np.mean(all_durations),
                'std': np.std(all_durations),
                'min': np.min(all_durations),
                'max': np.max(all_durations)
            },
            'n_channels': {
                'mean': np.mean(all_n_channels),
                'unique': list(set(all_n_channels))
            },
            'sampling_rate': {
                'mean': np.mean(all_sampling_rates),
                'unique': list(set(all_sampling_rates))
            }
        }

        return summary

    def print_summary_statistics(self):
        """Print summary statistics across all files."""
        summary = self.compute_summary_statistics()

        print("\n" + "=" * 80)
        print("BATCH PROCESSING SUMMARY")
        print("=" * 80)
        print(f"\nTotal files processed: {summary['n_files']}")

        print("\n" + "-" * 80)
        print("AMPLITUDE STATISTICS (across all files)")
        print("-" * 80)
        print(f"Mean amplitude:")
        print(f"  Average: {summary['mean_amplitude']['mean']:.4f} μV")
        print(f"  Std Dev: {summary['mean_amplitude']['std']:.4f} μV")
        print(f"  Range: [{summary['mean_amplitude']['min']:.4f}, "
              f"{summary['mean_amplitude']['max']:.4f}] μV")

        print(f"\nStandard deviation:")
        print(f"  Average: {summary['std_amplitude']['mean']:.4f} μV")
        print(f"  Std Dev: {summary['std_amplitude']['std']:.4f} μV")
        print(f"  Range: [{summary['std_amplitude']['min']:.4f}, "
              f"{summary['std_amplitude']['max']:.4f}] μV")

        print("\n" + "-" * 80)
        print("RECORDING PARAMETERS")
        print("-" * 80)
        print(f"Duration:")
        print(f"  Average: {summary['duration']['mean']:.2f} seconds")
        print(f"  Std Dev: {summary['duration']['std']:.2f} seconds")
        print(f"  Range: [{summary['duration']['min']:.2f}, "
              f"{summary['duration']['max']:.2f}] seconds")

        print(f"\nNumber of channels:")
        print(f"  Average: {summary['n_channels']['mean']:.1f}")
        print(f"  Unique values: {summary['n_channels']['unique']}")

        print(f"\nSampling rate:")
        print(f"  Average: {summary['sampling_rate']['mean']:.1f} Hz")
        print(f"  Unique values: {summary['sampling_rate']['unique']}")

        print("\n" + "=" * 80)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
        processor = BatchProcessor(data_dir)
        processor.print_summary_statistics()
    else:
        print("Usage: python batch_processor.py <data_directory>")
