"""
Unified mmWave radar simulation script.
Supports SMPL meshes, HY-Motion mesh artifacts, and joint sequences.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import yaml
from easydict import EasyDict as edict
import os
import glob

from rich import box
from rich.console import Console
from rich.panel import Panel

from config_expand import expand_env_vars
from simulation_base import SimulationPipeline
from data_loaders import SMPLDataLoader, HunyuanWoodenDataLoader, JointsDataLoader
from utils.logger import log_message

_rich_console = Console()


def print_config(config: edict, title: str = "Simulation Configuration"):
    """Print configuration in a formatted way."""
    print()
    print("=" * 60)
    log_message("CONFIG", f"{title:^60}", color="cyan", attrs=("bold",))
    print("=" * 60)
    print()

    def print_section(section_name: str, items: dict, indent: int = 0):
        """Print a configuration section."""
        prefix = "  " * indent
        log_message("CONFIG", f"{prefix}[{section_name}]", color="yellow", attrs=("bold",))
        for key, value in items.items():
            if isinstance(value, dict):
                print_section(key, value, indent + 1)
            else:
                log_message("CONFIG", f"{prefix}  {key}: {value}", color="green")
        print()

    # Organize config into sections
    sections = {
        'Data Source': {},
        'Motion Parameters': {},
        'Simulation': {},
        'Camera Settings': {},
        'Radar': {},
        'Processing': {},
        'Output': {}
    }

    # Categorize config items
    for key, value in config.items():
        if key in ['data_source', 'data_root', 'model_type', 'data_pattern', 'smpl_model_path', 'hunyuan_wooden_model_dir']:
            sections['Data Source'][key] = value
        elif key in ['fps', 'interp_n', 'mesh_sample_rate', 'wooden_batch_size', 'wooden_smooth_sigma']:
            sections['Motion Parameters'][key] = value
        elif key in ['simulation_method', 'num_rotations']:
            sections['Simulation'][key] = value
        elif key in ['camera_position', 'camera_orientation', 'simulation_bounds']:
            sections['Camera Settings'][key] = value
        elif key in ['radar_config', 'num_antenna']:
            sections['Radar'][key] = value
        elif key in ['start', 'end', 'skip_exist', 'verbose', 'auto_yes']:
            sections['Processing'][key] = value
        elif key in ['output_dir']:
            sections['Output'][key] = value

    # Print each section
    for section_name, items in sections.items():
        if items:  # Only print non-empty sections
            print_section(section_name, items)

    print("=" * 60)
    print()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Unified mmWave Radar Simulation Framework',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # SMPL simulation with rotation augmentation
  python simulator/run_simulation.py --data_source smpl \\
      --data_root /path/to/HumanML3D --output_dir /path/to/output \\
      --method reflection --num_rotations 8

  # Joints simulation
  python simulator/run_simulation.py --data_source joints \\
      --data_pattern '/path/to/joints/*.npy' --output_dir /path/to/output \\
      --fps 30 --interp_n 50

  # Using configuration file
  python simulator/run_simulation.py \\
      --config simulator/configs/simulation/simulation_smpl.yaml
        """
    )

    # Configuration file
    parser.add_argument(
        '--config',
        type=str,
        help='Path to YAML configuration file (overrides other arguments)'
    )

    # Data source selection
    parser.add_argument(
        '--data_source',
        type=str,
        choices=['smpl', 'hunyuan_wooden', 'wooden', 'joints'],
        help='Type of motion data source'
    )

    # Data paths
    parser.add_argument(
        '--data_pattern',
        type=str,
        help='Glob pattern to match data files (e.g., "/path/**/*.npy" or "/path/level1/*/smpl/*.npy")'
    )
    parser.add_argument(
        '--data_root',
        type=str,
        help='[Deprecated] Root directory for SMPL or Joints data (use --data_pattern instead)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        help='Output directory for udoppler data'
    )

    # Motion parameters
    parser.add_argument(
        '--fps',
        type=int,
        default=20,
        help='Target frame rate for simulation'
    )
    parser.add_argument(
        '--interp_n',
        type=int,
        default=50,
        help='Number of interpolation points between joints'
    )
    parser.add_argument(
        '--mesh_sample_rate',
        type=int,
        default=50,
        help='Mesh sampling rate (for SMPL)'
    )
    parser.add_argument(
        '--model_type',
        type=str,
        default='smpl',
        help='Model type subdirectory name (for SMPL data source)'
    )

    # Simulation parameters
    parser.add_argument(
        '-m', '--method',
        type=str,
        default='points',
        choices=['reflection', 'points', 'pytorch', 'triangle_reflection'],
        help='Simulation method'
    )
    parser.add_argument(
        '--num_rotations',
        type=int,
        default=1,
        help='Number of rotations for data augmentation (1-8)'
    )

    # Processing parameters
    parser.add_argument(
        '-s', '--start',
        type=int,
        default=None,
        help='Start index of files to process (with --config: overrides YAML; default: from YAML or 0)'
    )
    parser.add_argument(
        '-e', '--end',
        type=int,
        default=None,
        help='End index of files to process, -1 for all (with --config: overrides YAML; default: from YAML or -1)'
    )
    parser.add_argument(
        '--skip_exist',
        action='store_true',
        help='Skip existing output files'
    )

    # Radar parameters
    parser.add_argument(
        '--num_antenna',
        type=int,
        default=1,
        help='Number of radar antennas'
    )
    parser.add_argument(
        '--radar_config',
        type=str,
        default='configs.radar.iwr1843_udoppler',
        help='Radar configuration module'
    )

    # Other parameters
    parser.add_argument(
        '--verbose',
        action='store_true',
        default=True,
        help='Print detailed information'
    )
    parser.add_argument(
        '--yes', '-y',
        action='store_true',
        help='Skip confirmation prompt (auto-yes)'
    )
    parser.add_argument(
        '--save_config',
        type=str,
        help='Save current configuration to YAML file'
    )
    parser.add_argument(
        '--no-rich-progress',
        action='store_true',
        help='Disable Rich progress bar (use plain [PROGRESS] log lines instead)'
    )

    return parser.parse_args()


def load_config_from_file(config_path: str) -> edict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config = expand_env_vars(config)
    return edict(config)


def save_config_to_file(config: edict, config_path: str):
    """Save configuration to YAML file."""
    # Convert edict to regular dict for YAML serialization
    config_dict = dict(config)
    with open(config_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
    log_message("CONFIG", f"Configuration saved to: {config_path}", color="green")


def validate_config(config: edict) -> bool:
    """Validate configuration."""
    errors = []

    # Check data source
    if 'data_source' not in config:
        errors.append("'data_source' is required (smpl, hunyuan_wooden, or joints)")

    # Check data paths
    data_source = config.get('data_source')
    has_pattern = 'data_pattern' in config and config.data_pattern
    has_root = 'data_root' in config and config.data_root

    if data_source in {'smpl', 'hunyuan_wooden', 'wooden', 'joints'}:
        if not has_pattern and not has_root:
            errors.append(f"Either 'data_pattern' or 'data_root' is required for {data_source} data source")

    # Check output directory
    if 'output_dir' not in config:
        errors.append("'output_dir' is required")

    # Print errors if any
    if errors:
        log_message("ERROR", "Configuration Errors:", color="red", attrs=("bold",))
        for error in errors:
            log_message("ERROR", f"- {error}", color="red")
        return False

    return True


def create_data_loader(config: edict):
    """Create appropriate data loader based on configuration."""
    data_source = config.data_source

    if data_source == 'smpl':
        return SMPLDataLoader(config)
    elif data_source in {'hunyuan_wooden', 'wooden'}:
        return HunyuanWoodenDataLoader(config)
    elif data_source == 'joints':
        return JointsDataLoader(config)
    else:
        raise ValueError(f"Unknown data source: {data_source}")


class FilteredDataLoader:
    """Wrapper class that filters out existing files from data loader."""

    def __init__(self, original_loader, output_dir):
        self.original_loader = original_loader
        self.output_dir = output_dir
        self._filtered_files = None

    def get_file_list(self):
        """Get filtered file list excluding files with existing outputs."""
        if self._filtered_files is None:
            all_files = self.original_loader.get_file_list()
            self._filtered_files = []

            for data_file in all_files:
                # Get base filename (without extension)
                base_name = os.path.splitext(os.path.basename(data_file))[0]

                # Check for corresponding output files (.npz or .npy)
                output_pattern_npz = os.path.join(self.output_dir, f"{base_name}*.npz")
                output_pattern_npy = os.path.join(self.output_dir, f"{base_name}*.npy")
                if not glob.glob(output_pattern_npz) and not glob.glob(output_pattern_npy):
                    self._filtered_files.append(data_file)

        return self._filtered_files

    def load_motion(self, file_path: str, *, quiet: bool = False):
        """Delegate to original loader."""
        return self.original_loader.load_motion(file_path, quiet=quiet)

    def get_output_name(self, input_path: str):
        """Delegate to original loader."""
        return self.original_loader.get_output_name(input_path)


def filter_existing_files(data_loader, config):
    """Filter out existing files from data loader and return new loader."""
    log_message("INFO", "Checking for existing mmwave data...", color="yellow")

    # Create filtered loader
    filtered_loader = FilteredDataLoader(data_loader, config.output_dir)

    # Get filtered file list
    filtered_files = filtered_loader.get_file_list()

    if len(filtered_files) == 0:
        log_message("SUCCESS", "All mmwave files already exist. Skipping simulation.", color="green")
        return None

    log_message("INFO", f"Updated file list: {len(filtered_files)} files to process", color="cyan")
    return filtered_loader


def main():
    """Main function."""
    args = parse_args()

    # Load configuration
    if args.config:
        # Load from file
        config = load_config_from_file(args.config)
        log_message("CONFIG", f"Loaded configuration from: {args.config}", color="cyan")

        # Override with command line --yes flag
        if args.yes:
            config.auto_yes = True

        # CLI slice overrides YAML (required for run_multi_gpu.py --start/--end)
        if args.start is not None:
            config.start = args.start
        elif 'start' not in config:
            config.start = 0
        if args.end is not None:
            config.end = args.end
        elif 'end' not in config:
            config.end = -1

        # Auto-enable file filtering for HumanML3D mmwave simulation
        if 'HumanML3D' in args.config or 'mmwave' in args.config:
            log_message("INFO", "Auto-enabled file filtering for HumanML3D mmwave simulation", color="yellow")

        # Ensure camera settings exist in config
        if 'camera_position' not in config:
            log_message("ERROR", "Simulation config must contain camera_position setting", color="red")
            return
    else:
        # Create from command line arguments
        config = edict({
            'data_source': args.data_source,
            'data_pattern': args.data_pattern,
            'data_root': args.data_root,
            'bvh_dir': args.bvh_dir,
            'output_dir': args.output_dir,
            'fps': args.fps,
            'interp_n': args.interp_n,
            'mesh_sample_rate': args.mesh_sample_rate,
            'model_type': args.model_type,
            'simulation_method': args.method,
            'num_rotations': args.num_rotations,
            'start': args.start if args.start is not None else 0,
            'end': args.end if args.end is not None else -1,
            'skip_exist': args.skip_exist,
            'num_antenna': args.num_antenna,
            'radar_config': args.radar_config,
            'verbose': args.verbose,
            'auto_yes': args.yes,
        })

        log_message("ERROR", "Camera settings must be provided via configuration file", color="red")
        log_message(
            "ERROR",
            "Please create a config file with camera_position, camera_orientation, and simulation_bounds",
            color="red",
        )
        return

    # Save configuration if requested
    if args.save_config:
        save_config_to_file(config, args.save_config)

    # Validate configuration
    if not validate_config(config):
        return

    _rich_console.print(
        Panel.fit(
            f"[bold cyan]mmWave simulation[/bold cyan]\n[dim]{args.config}[/dim]\n"
            f"[dim]File loop: Rich bar + ETA on stderr (disable: --no-rich-progress)[/dim]",
            border_style="cyan",
            box=box.ROUNDED,
        )
    )

    # Print configuration
    print_config(config)

    # Ask for confirmation (skip in non-interactive mode or with --yes flag)
    if config.get('verbose', True) and not config.get('auto_yes', False):
        try:
            log_message("PROMPT", "Proceed with simulation? [Y/n]: ", color="yellow", end="")
            response = input()
            if response.lower() in ['n', 'no']:
                log_message("INFO", "Simulation cancelled.", color="red")
                return
        except EOFError:
            # Non-interactive mode, proceed automatically
            log_message("INFO", "Non-interactive mode detected, proceeding...", color="yellow")
            pass

    # Create data loader
    try:
        data_loader = create_data_loader(config)
    except Exception as e:
        log_message("ERROR", f"Error creating data loader: {e}", color="red")
        return

    # Filter existing files (for HumanML3D mmwave simulation)
    if 'HumanML3D' in args.config or 'mmwave' in args.config:
        data_loader = filter_existing_files(data_loader, config)
        if data_loader is None:
            return  # All files already exist, simulation skipped

    # Create and run pipeline
    try:
        pipeline = SimulationPipeline(
            data_loader=data_loader,
            config=config,
            simulation_method=config.simulation_method,
            num_rotations=config.num_rotations,
            verbose=config.verbose,
            rich_progress=not args.no_rich_progress,
        )

        pipeline.run(start_idx=config.start, end_idx=config.end)

    except Exception as e:
        log_message("ERROR", f"Error running simulation: {e}", color="red")
        import traceback
        traceback.print_exc()
        return


if __name__ == '__main__':
    main()
