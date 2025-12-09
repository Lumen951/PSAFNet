"""
PSAFNet EEG Data Tools

This package provides tools for reading, analyzing, and visualizing EEG data
in EDF format for the PSAFNet project.

Modules:
    - edf_reader: EDF file reading and parsing
    - data_statistics: Statistical analysis of EEG data
    - data_visualizer: Visualization tools for EEG signals
    - batch_processor: Batch processing utilities
    - main_viewer: Main command-line interface
"""

__version__ = "1.0.0"
__author__ = "PSAFNet Team"

try:
    from .edf_reader import EDFReader
    from .data_statistics import DataStatistics
    from .data_visualizer import DataVisualizer
    from .batch_processor import BatchProcessor
except ImportError:
    from edf_reader import EDFReader
    from data_statistics import DataStatistics
    from data_visualizer import DataVisualizer
    from batch_processor import BatchProcessor

__all__ = [
    'EDFReader',
    'DataStatistics',
    'DataVisualizer',
    'BatchProcessor'
]
