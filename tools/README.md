# EEG Data Tools

A comprehensive toolkit for reading, analyzing, and visualizing EEG data in EDF format for the PSAFNet project.

## Features

- **EDF File Reading**: Load and parse EDF (European Data Format) files
- **Statistical Analysis**: Compute comprehensive statistics on EEG signals
- **Data Visualization**: Generate various plots including time series, heatmaps, and power spectra
- **Batch Processing**: Process multiple files efficiently
- **Dataset Comparison**: Compare raw vs cleaned datasets
- **Interactive Mode**: User-friendly menu-driven interface

## Installation

### 1. Install Dependencies

```bash
cd tools
pip install -r requirements.txt
```

### 2. Verify Installation

```bash
python main_viewer.py --help
```

## Quick Start

### Interactive Mode (Recommended for Beginners)

```bash
python main_viewer.py --interactive
```

This launches an interactive menu where you can:
- Browse and select EEG files
- View statistics
- Generate various plots
- Export data

### View a Single File

```bash
# Show file information and statistics
python main_viewer.py --file ../data/EEG/P01.edf --stats

# Generate overview visualization
python main_viewer.py --file ../data/EEG/P01.edf --plot --plot-type overview

# Show everything (stats + plots)
python main_viewer.py --file ../data/EEG/P01.edf --all

# Analyze specific time window (0-10 seconds)
python main_viewer.py --file ../data/EEG/P01.edf --start 0 --end 10 --all
```

### Batch Processing

```bash
# Generate overview report for all files
python main_viewer.py --batch --data-dir ../data/EEG/ --overview --output reports/

# Generate detailed statistics for all files
python main_viewer.py --batch --data-dir ../data/EEG/ --statistics --output reports/

# Show summary statistics across all files
python main_viewer.py --batch --data-dir ../data/EEG/ --summary
```

### Compare Datasets

```bash
# Compare raw vs cleaned data
python main_viewer.py --compare \
    --dir1 "../data/EEG/" \
    --dir2 "../data/EEG_clean data/" \
    --output reports/
```

## Module Documentation

### 1. EDFReader (`edf_reader.py`)

Read and parse EDF files.

```python
from edf_reader import EDFReader

# Load file
reader = EDFReader('data/P01.edf')

# Get file information
reader.print_info()

# Get data for specific time window
data, times = reader.get_data(start=0, stop=10)  # 0-10 seconds

# Get single channel data
channel_data, times = reader.get_channel_data('EEG Fp1')

# Export to NumPy
reader.export_to_numpy('output.npz')
```

**Key Methods:**
- `get_data(channels, start, stop)`: Get EEG data
- `get_channel_data(channel_name, start, stop)`: Get single channel
- `get_info()`: Get metadata
- `print_info()`: Print file information
- `export_to_numpy(output_path)`: Export to .npz format

### 2. DataStatistics (`data_statistics.py`)

Compute statistical measures on EEG data.

```python
from data_statistics import DataStatistics

# Create statistics object
stats = DataStatistics(data, sampling_rate, channel_names)

# Compute basic statistics
basic_stats = stats.compute_basic_stats()  # mean, std, min, max, etc.

# Compute channel-specific statistics
channel_stats = stats.compute_channel_stats(channel_idx=0)

# Compute power spectrum
frequencies, psd = stats.compute_power_spectrum(channel_idx=0)

# Compute band power (Delta, Theta, Alpha, Beta, Gamma)
band_powers = stats.compute_band_power(channel_idx=0)

# Detect artifacts
artifacts = stats.detect_artifacts(threshold_std=5.0)

# Compute correlation matrix
corr_matrix = stats.compute_correlation_matrix()

# Print summary report
stats.print_summary()

# Save report to file
stats.save_summary('report.txt')
```

**Key Methods:**
- `compute_basic_stats()`: Mean, std, min, max, median, variance, range, RMS
- `compute_channel_stats(channel_idx)`: Detailed stats for one channel
- `compute_power_spectrum(channel_idx)`: Power spectral density
- `compute_band_power(channel_idx)`: Power in frequency bands
- `detect_artifacts(threshold_std)`: Artifact detection
- `compute_correlation_matrix()`: Channel correlations
- `generate_summary_report()`: Comprehensive text report

### 3. DataVisualizer (`data_visualizer.py`)

Visualize EEG data with various plot types.

```python
from data_visualizer import DataVisualizer

# Create visualizer
viz = DataVisualizer(data, times, channel_names, sampling_rate)

# Plot multiple channels
viz.plot_multichannel(channels=[0, 1, 2], start_time=0, end_time=10)

# Plot single channel
viz.plot_single_channel(channel_idx=0, start_time=0, end_time=10)

# Plot heatmap
viz.plot_heatmap(start_time=0, end_time=10)

# Plot power spectrum
frequencies, psd = stats.compute_power_spectrum(0)
viz.plot_power_spectrum(0, frequencies, psd)

# Plot band powers
band_powers = stats.compute_band_power(0)
viz.plot_band_powers(band_powers)

# Plot correlation matrix
corr_matrix = stats.compute_correlation_matrix()
viz.plot_correlation_matrix(corr_matrix)

# Plot statistics summary
basic_stats = stats.compute_basic_stats()
viz.plot_statistics_summary(basic_stats)

# Create comprehensive overview
viz.create_overview_figure(save_path='overview.png')
```

**Key Methods:**
- `plot_multichannel()`: Stacked time series of multiple channels
- `plot_single_channel()`: Single channel waveform
- `plot_heatmap()`: Channel × time heatmap
- `plot_power_spectrum()`: Frequency spectrum
- `plot_band_powers()`: Bar chart of frequency band powers
- `plot_correlation_matrix()`: Channel correlation heatmap
- `plot_statistics_summary()`: Statistical overview
- `create_overview_figure()`: Comprehensive multi-panel figure

All plot methods support `save_path` parameter to save figures.

### 4. BatchProcessor (`batch_processor.py`)

Process multiple EEG files efficiently.

```python
from batch_processor import BatchProcessor

# Initialize processor
processor = BatchProcessor('data/EEG/')

# Get file list
files = processor.get_file_list()

# Generate overview report
df = processor.generate_overview_report(output_path='overview.csv')

# Generate statistics for all files
processor.generate_statistics_report(output_dir='reports/')

# Compare two datasets
comparison = processor.compare_datasets(
    'data/EEG/',
    'data/EEG_clean data/',
    output_path='comparison.csv'
)

# Extract time windows from all files
windows = [(1.0, 2.0), (5.0, 6.0)]  # No-target and target windows
window_names = ['no_target', 'target']
processor.extract_time_windows('extracted/', windows, window_names)

# Compute summary statistics
summary = processor.compute_summary_statistics()
processor.print_summary_statistics()
```

**Key Methods:**
- `get_file_list()`: List all files
- `generate_overview_report()`: Create CSV with file metadata
- `generate_statistics_report()`: Generate stats for all files
- `compare_datasets()`: Compare two directories
- `extract_time_windows()`: Extract specific time windows
- `compute_summary_statistics()`: Aggregate statistics
- `print_summary_statistics()`: Print summary

## Command-Line Reference

### Main Viewer Options

```
usage: main_viewer.py [-h] [--file FILE | --batch | --compare | --interactive]
                      [--start START] [--end END] [--stats] [--plot]
                      [--plot-type {overview,multichannel,heatmap,channel}]
                      [--channel CHANNEL] [--export EXPORT] [--all]
                      [--data-dir DATA_DIR] [--overview] [--statistics]
                      [--summary] [--dir1 DIR1] [--dir2 DIR2]
                      [--output OUTPUT]

Mode Selection:
  --file FILE           Single EDF file to view
  --batch               Batch processing mode
  --compare             Compare two datasets
  --interactive         Interactive mode

Single File Options:
  --start START         Start time in seconds
  --end END             End time in seconds
  --stats               Show statistics
  --plot                Generate plots
  --plot-type TYPE      Plot type: overview, multichannel, heatmap, channel
  --channel CHANNEL     Channel index for single channel plot
  --export EXPORT       Export data to NumPy file (.npz)
  --all                 Show all (stats + plots)

Batch Processing Options:
  --data-dir DATA_DIR   Directory containing EDF files
  --overview            Generate overview report
  --statistics          Generate statistics reports
  --summary             Show summary statistics

Comparison Options:
  --dir1 DIR1           First directory for comparison
  --dir2 DIR2           Second directory for comparison

Output Options:
  --output OUTPUT       Output directory for results
```

## Usage Examples

### Example 1: Quick Data Inspection

```bash
# View file info and basic statistics
python main_viewer.py --file ../data/EEG/P01.edf --stats
```

### Example 2: Generate Comprehensive Report

```bash
# Create output directory
mkdir -p results/P01

# Generate all visualizations and statistics
python main_viewer.py \
    --file ../data/EEG/P01.edf \
    --all \
    --output results/P01/
```

### Example 3: Analyze Specific Time Window

```bash
# Analyze the target detection window (5-6 seconds)
python main_viewer.py \
    --file ../data/EEG/P01.edf \
    --start 5 \
    --end 6 \
    --all \
    --output results/P01_target_window/
```

### Example 4: Plot Single Channel

```bash
# Plot channel 0 (first EEG channel)
python main_viewer.py \
    --file ../data/EEG/P01.edf \
    --plot \
    --plot-type channel \
    --channel 0
```

### Example 5: Batch Process All Subjects

```bash
# Create reports directory
mkdir -p reports

# Generate overview for all subjects
python main_viewer.py \
    --batch \
    --data-dir ../data/EEG/ \
    --overview \
    --summary \
    --output reports/
```

### Example 6: Compare Raw and Cleaned Data

```bash
# Compare datasets
python main_viewer.py \
    --compare \
    --dir1 "../data/EEG/" \
    --dir2 "../data/EEG_clean data/" \
    --output reports/
```

### Example 7: Export Data for Further Analysis

```bash
# Export specific time window to NumPy format
python main_viewer.py \
    --file ../data/EEG/P01.edf \
    --start 5 \
    --end 6 \
    --export P01_target_window.npz
```

## Python API Examples

### Example: Custom Analysis Pipeline

```python
from edf_reader import EDFReader
from data_statistics import DataStatistics
from data_visualizer import DataVisualizer

# Load data
reader = EDFReader('data/P01.edf')
data, times = reader.get_data(start=5, stop=6)  # Target window

# Compute statistics
stats = DataStatistics(data, reader.get_sampling_rate(),
                      reader.get_channel_names())

# Get band powers for all channels
for i, ch_name in enumerate(reader.get_channel_names()):
    band_powers = stats.compute_band_power(i)
    print(f"\n{ch_name}:")
    for band, power in band_powers.items():
        print(f"  {band}: {power:.4e}")

# Visualize
viz = DataVisualizer(data, times, reader.get_channel_names(),
                    reader.get_sampling_rate())
viz.create_overview_figure(save_path='analysis.png')
```

### Example: Batch Extract Time Windows

```python
from batch_processor import BatchProcessor

# Initialize processor
processor = BatchProcessor('data/EEG/')

# Define time windows (matching PSAFNet paper)
windows = [
    (1.0, 2.0),  # No-target window
    (5.0, 6.0)   # Target window
]
window_names = ['no_target', 'target']

# Extract windows from all files
processor.extract_time_windows(
    output_dir='extracted_windows/',
    windows=windows,
    window_names=window_names
)
```

## Output Files

### Statistics Report (`.txt`)

Text file containing:
- Data shape and duration
- Overall statistics
- Per-channel statistics (mean, std, min, max)
- Artifact detection results

### Overview Report (`.csv`)

CSV file with columns:
- `filename`: File name
- `n_channels`: Number of channels
- `sampling_rate`: Sampling rate (Hz)
- `duration_sec`: Duration (seconds)
- `n_samples`: Total samples
- `file_size_mb`: File size (MB)

### Comparison Report (`.csv`)

CSV file comparing two datasets:
- `filename`: File name
- `dataset1_channels`, `dataset2_channels`: Channel counts
- `dataset1_duration`, `dataset2_duration`: Durations
- `dataset1_mean`, `dataset2_mean`: Mean amplitudes
- `dataset1_std`, `dataset2_std`: Standard deviations
- `mean_difference`, `std_difference`: Differences

### Exported NumPy Files (`.npz`)

NumPy archive containing:
- `data`: EEG data array (n_channels × n_samples)
- `times`: Time array (seconds)
- `channel_names`: List of channel names
- `sampling_rate`: Sampling rate (Hz)

Load with:
```python
import numpy as np
data = np.load('output.npz')
eeg_data = data['data']
times = data['times']
```

## Troubleshooting

### Issue: "No module named 'mne'"

**Solution:** Install dependencies
```bash
pip install -r requirements.txt
```

### Issue: "File not found"

**Solution:** Use absolute paths or ensure you're in the correct directory
```bash
# Use absolute path
python main_viewer.py --file "D:/University/Junior/1st/code/PSAFNet/data/EEG/P01.edf"

# Or navigate to tools directory first
cd tools
python main_viewer.py --file ../data/EEG/P01.edf
```

### Issue: Plots not displaying

**Solution:** Ensure matplotlib backend is configured correctly
```python
import matplotlib
matplotlib.use('TkAgg')  # or 'Qt5Agg'
import matplotlib.pyplot as plt
```

### Issue: Memory error with large files

**Solution:** Process specific time windows instead of entire file
```bash
python main_viewer.py --file data.edf --start 0 --end 10
```

## Project Structure

```
tools/
├── __init__.py              # Package initialization
├── edf_reader.py            # EDF file reading module
├── data_statistics.py       # Statistical analysis module
├── data_visualizer.py       # Visualization module
├── batch_processor.py       # Batch processing utilities
├── main_viewer.py           # Main CLI entry point
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## Integration with PSAFNet

These tools are designed to work with the PSAFNet project:

```python
# Extract time windows for PSAFNet training
from batch_processor import BatchProcessor

processor = BatchProcessor('data/EEG/')
processor.extract_time_windows(
    output_dir='psafnet_data/',
    windows=[(1.0, 2.0), (5.0, 6.0)],
    window_names=['no_target', 'target']
)

# Load extracted data
import numpy as np
data = np.load('psafnet_data/P01_no_target.npz')
X_no_target = data['data']  # Shape: (n_channels, n_samples)

data = np.load('psafnet_data/P01_target.npz')
X_target = data['data']
```

## Contributing

When adding new features:
1. Follow existing code style
2. Add docstrings to all functions
3. Update this README
4. Test with sample data

## License

This tool is part of the PSAFNet project.

## Contact

For issues or questions, please refer to the main PSAFNet repository.

---

**Last Updated:** 2025-12-09
