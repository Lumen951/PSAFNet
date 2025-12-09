"""
Main Viewer - Command Line Interface for EEG Data Tools

This is the main entry point for the EEG data viewing and analysis tools.
"""

import argparse
import sys
from pathlib import Path

from edf_reader import EDFReader
from data_statistics import DataStatistics
from data_visualizer import DataVisualizer
from batch_processor import BatchProcessor


def view_single_file(args):
    """View and analyze a single EEG file."""
    print(f"\n{'='*80}")
    print(f"Loading EEG file: {args.file}")
    print(f"{'='*80}\n")

    try:
        # Load file
        reader = EDFReader(args.file)
        reader.print_info()

        # Get data
        start_time = args.start if args.start else 0
        end_time = args.end if args.end else None
        data, times = reader.get_data(start=start_time, stop=end_time)

        # Statistics
        if args.stats or args.all:
            print("\nComputing statistics...")
            stats = DataStatistics(data, reader.get_sampling_rate(),
                                 reader.get_channel_names())
            stats.print_summary()

            if args.output:
                stats_path = Path(args.output) / f"{Path(args.file).stem}_statistics.txt"
                stats.save_summary(str(stats_path))

        # Visualization
        if args.plot or args.all:
            print("\nGenerating visualizations...")
            viz = DataVisualizer(data, times, reader.get_channel_names(),
                               reader.get_sampling_rate())

            if args.plot_type == 'overview' or args.all:
                save_path = None
                if args.output:
                    save_path = str(Path(args.output) / f"{Path(args.file).stem}_overview.png")
                viz.create_overview_figure(save_path=save_path)

            elif args.plot_type == 'multichannel':
                save_path = None
                if args.output:
                    save_path = str(Path(args.output) / f"{Path(args.file).stem}_multichannel.png")
                viz.plot_multichannel(save_path=save_path)

            elif args.plot_type == 'heatmap':
                save_path = None
                if args.output:
                    save_path = str(Path(args.output) / f"{Path(args.file).stem}_heatmap.png")
                viz.plot_heatmap(save_path=save_path)

            elif args.plot_type == 'channel':
                if args.channel is None:
                    print("Error: --channel must be specified for channel plot")
                    return
                save_path = None
                if args.output:
                    save_path = str(Path(args.output) / f"{Path(args.file).stem}_ch{args.channel}.png")
                viz.plot_single_channel(args.channel, save_path=save_path)

        # Export
        if args.export:
            export_path = args.export
            reader.export_to_numpy(export_path, start=start_time, stop=end_time)

        print("\nProcessing complete!")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


def batch_process(args):
    """Batch process multiple EEG files."""
    print(f"\n{'='*80}")
    print(f"Batch Processing: {args.data_dir}")
    print(f"{'='*80}\n")

    try:
        processor = BatchProcessor(args.data_dir)

        if args.overview:
            output_path = None
            if args.output:
                output_path = str(Path(args.output) / "overview_report.csv")
            df = processor.generate_overview_report(output_path=output_path)
            print("\nOverview Report:")
            print(df.to_string())

        if args.statistics:
            output_dir = args.output if args.output else "statistics_reports"
            processor.generate_statistics_report(output_dir)

        if args.summary:
            processor.print_summary_statistics()

        print("\nBatch processing complete!")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


def compare_datasets(args):
    """Compare two datasets."""
    print(f"\n{'='*80}")
    print(f"Comparing Datasets")
    print(f"{'='*80}\n")

    try:
        processor = BatchProcessor(args.dir1)

        output_path = None
        if args.output:
            output_path = str(Path(args.output) / "comparison_report.csv")

        df = processor.compare_datasets(args.dir1, args.dir2, output_path=output_path)

        print("\nComparison Results:")
        print(df.to_string())

        print("\nComparison complete!")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


def interactive_mode():
    """Interactive mode for exploring EEG data."""
    print("\n" + "="*80)
    print("EEG Data Viewer - Interactive Mode")
    print("="*80)

    # Get data directory
    data_dir = input("\nEnter data directory path: ").strip()
    if not Path(data_dir).exists():
        print(f"Error: Directory not found: {data_dir}")
        return

    try:
        processor = BatchProcessor(data_dir)
        files = processor.get_file_list()

        print(f"\nFound {len(files)} files:")
        for i, file_path in enumerate(files, 1):
            print(f"  {i}. {file_path.name}")

        # Select file
        while True:
            try:
                choice = input(f"\nSelect file (1-{len(files)}) or 'q' to quit: ").strip()
                if choice.lower() == 'q':
                    return

                file_idx = int(choice) - 1
                if 0 <= file_idx < len(files):
                    break
                else:
                    print(f"Please enter a number between 1 and {len(files)}")
            except ValueError:
                print("Invalid input. Please enter a number.")

        selected_file = files[file_idx]
        print(f"\nLoading: {selected_file.name}")

        # Load file
        reader = EDFReader(str(selected_file))
        reader.print_info()

        # Interactive menu
        while True:
            print("\n" + "-"*80)
            print("Options:")
            print("  1. Show statistics")
            print("  2. Plot overview")
            print("  3. Plot multichannel")
            print("  4. Plot heatmap")
            print("  5. Plot single channel")
            print("  6. Export to NumPy")
            print("  7. Select another file")
            print("  q. Quit")
            print("-"*80)

            option = input("\nSelect option: ").strip()

            if option == 'q':
                break
            elif option == '7':
                interactive_mode()
                return

            data, times = reader.get_data()

            if option == '1':
                stats = DataStatistics(data, reader.get_sampling_rate(),
                                     reader.get_channel_names())
                stats.print_summary()

            elif option == '2':
                viz = DataVisualizer(data, times, reader.get_channel_names(),
                                   reader.get_sampling_rate())
                viz.create_overview_figure()

            elif option == '3':
                viz = DataVisualizer(data, times, reader.get_channel_names(),
                                   reader.get_sampling_rate())
                viz.plot_multichannel()

            elif option == '4':
                viz = DataVisualizer(data, times, reader.get_channel_names(),
                                   reader.get_sampling_rate())
                viz.plot_heatmap()

            elif option == '5':
                print(f"\nAvailable channels:")
                for i, ch in enumerate(reader.get_channel_names()):
                    print(f"  {i}. {ch}")
                try:
                    ch_idx = int(input("Enter channel index: ").strip())
                    viz = DataVisualizer(data, times, reader.get_channel_names(),
                                       reader.get_sampling_rate())
                    viz.plot_single_channel(ch_idx)
                except (ValueError, IndexError):
                    print("Invalid channel index")

            elif option == '6':
                output_path = input("Enter output path (.npz): ").strip()
                reader.export_to_numpy(output_path)

            else:
                print("Invalid option")

    except Exception as e:
        print(f"Error: {e}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="EEG Data Viewer and Analysis Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # View single file with overview
  python main_viewer.py --file data/P01.edf --plot --plot-type overview

  # View file with statistics and save output
  python main_viewer.py --file data/P01.edf --stats --output results/

  # View specific time window
  python main_viewer.py --file data/P01.edf --start 0 --end 10 --plot

  # Batch process directory
  python main_viewer.py --batch --data-dir data/EEG/ --overview --output reports/

  # Compare two datasets
  python main_viewer.py --compare --dir1 data/EEG/ --dir2 "data/EEG_clean data/" --output reports/

  # Interactive mode
  python main_viewer.py --interactive
        """
    )

    # Mode selection
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument('--file', type=str, help='Single EDF file to view')
    mode_group.add_argument('--batch', action='store_true', help='Batch processing mode')
    mode_group.add_argument('--compare', action='store_true', help='Compare two datasets')
    mode_group.add_argument('--interactive', action='store_true', help='Interactive mode')

    # Single file options
    parser.add_argument('--start', type=float, help='Start time in seconds')
    parser.add_argument('--end', type=float, help='End time in seconds')
    parser.add_argument('--stats', action='store_true', help='Show statistics')
    parser.add_argument('--plot', action='store_true', help='Generate plots')
    parser.add_argument('--plot-type', type=str, default='overview',
                       choices=['overview', 'multichannel', 'heatmap', 'channel'],
                       help='Type of plot to generate')
    parser.add_argument('--channel', type=int, help='Channel index for single channel plot')
    parser.add_argument('--export', type=str, help='Export data to NumPy file (.npz)')
    parser.add_argument('--all', action='store_true', help='Show all (stats + plots)')

    # Batch processing options
    parser.add_argument('--data-dir', type=str, help='Directory containing EDF files')
    parser.add_argument('--overview', action='store_true', help='Generate overview report')
    parser.add_argument('--statistics', action='store_true', help='Generate statistics reports')
    parser.add_argument('--summary', action='store_true', help='Show summary statistics')

    # Comparison options
    parser.add_argument('--dir1', type=str, help='First directory for comparison')
    parser.add_argument('--dir2', type=str, help='Second directory for comparison')

    # Output options
    parser.add_argument('--output', type=str, help='Output directory for results')

    args = parser.parse_args()

    # If no arguments, show help
    if len(sys.argv) == 1:
        parser.print_help()
        print("\nTip: Use --interactive for an interactive menu-driven interface")
        sys.exit(0)

    # Route to appropriate function
    if args.interactive:
        interactive_mode()
    elif args.file:
        view_single_file(args)
    elif args.batch:
        if not args.data_dir:
            print("Error: --data-dir is required for batch processing")
            sys.exit(1)
        batch_process(args)
    elif args.compare:
        if not args.dir1 or not args.dir2:
            print("Error: --dir1 and --dir2 are required for comparison")
            sys.exit(1)
        compare_datasets(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
