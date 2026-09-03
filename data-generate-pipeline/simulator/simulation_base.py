"""
Base framework for mmWave radar simulation.
Provides a unified interface for different data sources and simulation methods.
"""

import os
import gc
import json
from pathlib import Path

import torch
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional
from easydict import EasyDict as edict
import importlib

import sys

sys.path.insert(
    0,
    os.environ.get("MMEXPERT_GEN_ROOT") or str(Path(__file__).resolve().parent.parent),
)
from utils.logger import log_message

from fmcw_radar import FMCWRadar
from mm_simulator import mmSimulator
from generate_udoppler import (
    get_raw_radar_frame,
    get_udoppler,
    get_udoppler_real_style,
    get_multi_view_data,
    get_multi_view_data_real_stable_v1,
    print_radar_spectrum_info,
)


class DataLoader(ABC):
    """Abstract base class for loading motion data from different sources."""

    def __init__(self, config: edict):
        self.config = config

    @abstractmethod
    def get_file_list(self) -> List[str]:
        """Get list of files to process."""
        pass

    @abstractmethod
    def load_motion(self, file_path: str, *, quiet: bool = False) -> Tuple[object, Dict]:
        """
        Load motion data from file.

        Args:
            file_path: Path to motion file
            quiet: If True, do not print loader-specific progress lines (dashboard mode)

        Returns:
            mesh_manager: Initialized mesh manager object
            metadata: Dictionary containing motion metadata (fps, num_frames, etc.)
        """
        pass

    @abstractmethod
    def get_output_name(self, input_path: str) -> str:
        """Generate output filename from input path."""
        pass


class SimulationPipeline:
    """Main simulation pipeline that orchestrates the entire process."""

    def __init__(
        self,
        data_loader: DataLoader,
        config: edict,
        simulation_method: str = 'points',
        num_rotations: int = 1,
        verbose: bool = True,
        rich_progress: bool = True,
    ):
        """
        Initialize simulation pipeline.

        Args:
            data_loader: Data loader instance
            config: Configuration dictionary
            simulation_method: Simulation method ('points', 'reflection', 'pytorch', 'triangle_reflection')
            num_rotations: Number of rotations to simulate (for data augmentation)
            verbose: Whether to print detailed information
            rich_progress: Use Rich dashboard (files + per-sequence) with ETA on stderr
        """
        self.data_loader = data_loader
        self.config = config
        self.simulation_method = simulation_method
        self.num_rotations = num_rotations
        self.verbose = verbose
        self.rich_progress = rich_progress
        # Set in run() while Rich dual progress is active; read in process/simulate
        self._rich_progress_ctx: Optional[dict] = None

        # Initialize radar
        self._init_radar()

        # Load mesh configuration
        self._load_mesh_config()

        # Create output directory
        os.makedirs(self.config.output_dir, exist_ok=True)

        # Optional path cache. This caches geometry/path propagation results,
        # not raw radar cubes, so downstream cube/three-view processing can be
        # changed without rerunning mesh/path simulation.
        self.path_cache = self._normalize_path_cache_config()

    def _normalize_path_cache_config(self) -> edict:
        raw = self.config.get('path_cache', {})
        if raw is None:
            raw = {}
        if not isinstance(raw, dict):
            raise ValueError("path_cache must be a mapping")

        enabled = bool(raw.get('enabled', False))
        mode = raw.get('mode', 'save_and_process')
        if mode not in {'save_only', 'process_only', 'save_and_process'}:
            raise ValueError(
                "path_cache.mode must be one of save_only/process_only/save_and_process"
            )
        cache_dir = raw.get('dir') or os.path.join(self.config.output_dir, '_path_cache')
        overwrite = bool(raw.get('overwrite', False))
        return edict({
            'enabled': enabled,
            'mode': mode,
            'dir': cache_dir,
            'overwrite': overwrite,
        })

    def _path_cache_path(self, output_name: str, rot_suffix: str = "") -> str:
        filename = f"{output_name}{rot_suffix}.pathdicts.npz"
        return os.path.join(self.path_cache.dir, filename)

    @staticmethod
    def _idx_to_numpy(idx):
        if idx is None:
            return np.array([], dtype=np.int64)
        if torch.is_tensor(idx):
            return idx.cpu().numpy().astype(np.int64, copy=False)
        return np.asarray(idx, dtype=np.int64)

    @staticmethod
    def _pack_ragged_float_arrays(arrays: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Pack per-frame arrays whose first dimension may vary.

        The simulator can return a different number of valid paths for each
        mesh frame.  Store those arrays as a flat value buffer plus offsets,
        while keeping the per-path trailing shape in metadata.  This avoids
        object arrays/pickle and keeps the cache portable.
        """
        offsets = [0]
        values = []
        trailing_shape = None
        for arr in arrays:
            arr = np.asarray(arr, dtype=np.float32)
            if arr.ndim == 0:
                raise ValueError("Path cache arrays must have at least one dimension")
            if trailing_shape is None:
                trailing_shape = arr.shape[1:]
            elif tuple(arr.shape[1:]) != tuple(trailing_shape):
                raise ValueError(
                    "Path cache arrays have inconsistent trailing shapes: "
                    f"{arr.shape[1:]} vs {trailing_shape}"
                )
            values.append(arr.reshape((arr.shape[0],) + tuple(trailing_shape)))
            offsets.append(offsets[-1] + arr.shape[0])

        if trailing_shape is None:
            trailing_shape = ()
        if values:
            flat = np.concatenate(values, axis=0).astype(np.float32, copy=False)
        else:
            flat = np.empty((0,) + tuple(trailing_shape), dtype=np.float32)
        return (
            flat,
            np.asarray(offsets, dtype=np.int64),
            np.asarray(trailing_shape, dtype=np.int64),
        )

    @staticmethod
    def _unpack_ragged_float_arrays(
        values: np.ndarray,
        offsets: np.ndarray,
        trailing_shape: np.ndarray,
    ) -> List[np.ndarray]:
        shape_tail = tuple(int(x) for x in trailing_shape.tolist())
        arrays = []
        for i in range(len(offsets) - 1):
            start = int(offsets[i])
            end = int(offsets[i + 1])
            arrays.append(values[start:end].reshape((end - start,) + shape_tail))
        return arrays

    def _save_path_cache(
        self,
        cache_path: str,
        path_dicts: List[Dict],
        *,
        metadata: Dict,
        output_name: str,
        rot_suffix: str,
        rotation_angle: float,
    ) -> None:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        if os.path.exists(cache_path) and not self.path_cache.overwrite:
            return

        a_arrays = [np.asarray(p['a'], dtype=np.float32) for p in path_dicts]
        tau_arrays = [np.asarray(p['tau'], dtype=np.float32) for p in path_dicts]

        a_lengths = np.asarray([arr.shape[0] for arr in a_arrays], dtype=np.int64)
        tau_lengths = np.asarray([arr.shape[0] for arr in tau_arrays], dtype=np.int64)
        if not np.array_equal(a_lengths, tau_lengths):
            raise ValueError("Path cache a/tau path counts do not match")

        a_values, a_offsets, a_shape_tail = self._pack_ragged_float_arrays(a_arrays)
        tau_values, tau_offsets, tau_shape_tail = self._pack_ragged_float_arrays(tau_arrays)

        idx_arrays = [self._idx_to_numpy(p.get('idx')) for p in path_dicts]
        idx_lengths = np.asarray([len(x) for x in idx_arrays], dtype=np.int64)
        if idx_arrays and len(set(idx_lengths.tolist())) == 1:
            idx = np.stack(idx_arrays, axis=0).astype(np.int64, copy=False)
            idx_offsets = np.array([], dtype=np.int64)
            idx_values = np.array([], dtype=np.int64)
            idx_format = 'dense'
        else:
            offsets = [0]
            values = []
            for arr in idx_arrays:
                values.append(arr)
                offsets.append(offsets[-1] + len(arr))
            idx = np.array([], dtype=np.int64)
            idx_offsets = np.asarray(offsets, dtype=np.int64)
            idx_values = (
                np.concatenate(values).astype(np.int64, copy=False)
                if values else np.array([], dtype=np.int64)
            )
            idx_format = 'ragged'

        meta = {
            'format': 'mmexpert_pathdicts_v2',
            'output_name': output_name,
            'rot_suffix': rot_suffix,
            'rotation_angle': float(rotation_angle),
            'num_path_frames': int(len(path_dicts)),
            'num_radar_frames': int(max(0, len(path_dicts) - 1)),
            'num_antenna': int(self.config.get('num_antenna', 1)),
            'radar_config': str(self.config.radar_config),
            'simulation_method': str(self.simulation_method),
            'camera_position': list(self.camera_position),
            'camera_orientation': list(self.camera_orientation),
            'metadata': {
                k: v for k, v in metadata.items()
                if isinstance(v, (str, int, float, bool, type(None)))
            },
            'idx_format': idx_format,
            'path_array_format': 'ragged_offsets',
        }

        np.savez_compressed(
            cache_path,
            a_values=a_values,
            a_offsets=a_offsets,
            a_shape_tail=a_shape_tail,
            tau_values=tau_values,
            tau_offsets=tau_offsets,
            tau_shape_tail=tau_shape_tail,
            path_lengths=a_lengths,
            idx=idx,
            idx_offsets=idx_offsets,
            idx_values=idx_values,
            idx_lengths=idx_lengths,
            meta=np.array(json.dumps(meta, ensure_ascii=False)),
        )

    def _load_path_cache(self, cache_path: str) -> List[Dict]:
        with np.load(cache_path, allow_pickle=False) as data:
            meta = json.loads(str(data['meta']))
            if meta.get('path_array_format') == 'ragged_offsets':
                a_arrays = self._unpack_ragged_float_arrays(
                    data['a_values'].astype(np.float32, copy=False),
                    data['a_offsets'],
                    data['a_shape_tail'],
                )
                tau_arrays = self._unpack_ragged_float_arrays(
                    data['tau_values'].astype(np.float32, copy=False),
                    data['tau_offsets'],
                    data['tau_shape_tail'],
                )
            else:
                # Backward compatibility with early dense-only caches.
                a = data['a'].astype(np.float32, copy=False)
                tau = data['tau'].astype(np.float32, copy=False)
                a_arrays = [a[i] for i in range(a.shape[0])]
                tau_arrays = [tau[i] for i in range(tau.shape[0])]
            idx_format = meta.get('idx_format', 'dense')
            if idx_format == 'dense':
                idx_dense = data['idx']
                idx_arrays = [idx_dense[i] for i in range(idx_dense.shape[0])]
            else:
                offsets = data['idx_offsets']
                values = data['idx_values']
                idx_arrays = [
                    values[offsets[i]:offsets[i + 1]]
                    for i in range(len(offsets) - 1)
                ]

        return [
            {'a': a_arrays[i], 'tau': tau_arrays[i], 'idx': idx_arrays[i]}
            for i in range(len(a_arrays))
        ]

    def _init_radar(self):
        """Initialize FMCW radar."""
        radar_cfg = importlib.import_module(self.config.radar_config)
        self.radar_cfg = radar_cfg
        self.radar = FMCWRadar(radar_cfg)
        if self.verbose:
            log_message('RADAR', f'Initialized radar: {self.config.radar_config}', color='cyan')

    def _load_mesh_config(self):
        """Load mesh configuration from embedded settings."""
        # Use embedded configuration instead of separate mesh config file
        if 'simulation_bounds' in self.config:
            bounds = self.config.simulation_bounds
            self.bounds_x = bounds.get('x', [-5.0, 5.0])
            self.bounds_y = bounds.get('y', [-5.0, 5.0])
            self.bounds_z_max = bounds.get('z_max', 1.0)
        else:
            # Default bounds
            self.bounds_x = [-5.0, 5.0]
            self.bounds_y = [-5.0, 5.0]
            self.bounds_z_max = 1.0

        # Store camera settings from config for access by other components
        self.camera_position = self.config.get('camera_position', [-5.6, 0.0, 1.0])
        self.camera_orientation = self.config.get('camera_orientation', [0.0, 0.0, 0.0])

        if self.verbose:
            log_message('CAMERA', 'Loaded camera settings from config', color='cyan')
            log_message('CAMERA', f'Camera position: {self.camera_position}', color='cyan')
            log_message('CAMERA', f'Camera orientation: {self.camera_orientation}', color='cyan')

    def _check_mesh_position(self, verts: torch.Tensor) -> bool:
        """Check if mesh is within valid bounds (read from config)."""
        if torch.min(verts[:, 0]) <= self.bounds_x[0]: return False
        if torch.max(verts[:, 0]) >= self.bounds_x[1]: return False
        if torch.min(verts[:, 1]) <= self.bounds_y[0]: return False
        if torch.max(verts[:, 1]) >= self.bounds_y[1]: return False
        if torch.min(verts[:, 2]) > self.bounds_z_max: return False
        return True

    def _compute_paths(self, simulator: mmSimulator) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """Compute radar paths using specified method."""
        if self.simulation_method == 'points':
            a, tau, _, idx = simulator.compute_paths_using_points()
        elif self.simulation_method == 'reflection':
            a, tau, _, idx = simulator.compute_paths_with_reflection()
        elif self.simulation_method == 'pytorch':
            a, tau, _, idx = simulator.compute_paths_pytorch()
        elif self.simulation_method == 'triangle_reflection':
            a, tau, _, idx = simulator.compute_paths_triangle_centers_with_reflection()
        else:
            raise ValueError(f"Unknown simulation method: {self.simulation_method}")

        return a, tau, idx

    def _paths_to_output(self, path_dicts: List[Dict]) -> Dict[str, np.ndarray]:
        num_frame = len(path_dicts) - 1
        if num_frame <= 0:
            return {}

        ctx = self._rich_progress_ctx
        n_p = len(path_dicts)
        n_r = num_frame
        total_seq = n_p + n_r
        if ctx:
            rot = ctx.get("rot_suffix") or ""
            label = ctx.get("file_label", "")
            prefix = f"{label}" + (f" · rot {rot}" if rot else "")
            ctx["progress"].update(
                ctx["task_seq"],
                completed=n_p,
                total=max(1, total_seq),
                description=f"Seq │ {prefix} │ radar 0/{n_r}",
            )

        radar_frames = torch.zeros(
            num_frame, 128, self.config.num_antenna, 256,
            dtype=torch.complex64
        ).cuda()

        for i in range(num_frame):
            radar_frame, indices = get_raw_radar_frame(
                path_dicts[i],
                path_dicts[i + 1],
                fps=50, base_time=0
            )
            if radar_frame.ndim == 4:
                radar_frames[i] = radar_frame.sum(dim=1)
            elif radar_frame.ndim == 3:
                radar_frames[i] = radar_frame
            else:
                raise ValueError(f"Unexpected radar_frame shape: {radar_frame.shape}")
            if ctx:
                rot = ctx.get("rot_suffix") or ""
                label = ctx.get("file_label", "")
                prefix = f"{label}" + (f" · rot {rot}" if rot else "")
                done = n_p + i + 1
                ctx["progress"].update(
                    ctx["task_seq"],
                    completed=min(done, total_seq),
                    total=total_seq,
                    description=f"Seq │ {prefix} │ radar {i + 1}/{n_r}",
                )

        output_mode = self.config.get('output_mode', 'full')
        if output_mode == 'udoppler':
            udoppler_preset = self.config.get('udoppler_preset', 'legacy')
            if udoppler_preset == 'legacy':
                udoppler = get_udoppler(radar_frames.cuda())
            elif udoppler_preset == 'real_style_v1':
                cfg = self.config.get('udoppler_config', {})
                udoppler = get_udoppler_real_style(
                    radar_frames.cuda(),
                    range_start=int(cfg.get('range_start', 16)),
                    range_end=int(cfg.get('range_end', 128)),
                    num_doppler_bins=int(cfg.get('num_doppler_bins', 128)),
                    target_range_bins=int(cfg.get('target_range_bins', 8)),
                    interpolate_center=bool(cfg.get('interpolate_center', True)),
                )
            else:
                raise ValueError(
                    "udoppler_preset must be one of legacy/real_style_v1, "
                    f"got {udoppler_preset!r}"
                )
            return {'doppler_time': udoppler.cpu().numpy()}

        three_view_preset = self.config.get('three_view_preset', 'legacy')
        if three_view_preset in {'real_stable_v1', 'synthetic_stable_v1'}:
            cfg = self.config.get('three_view_config', {})
            view_noise_seed = cfg.get('view_noise_seed', None)
            if view_noise_seed is not None:
                seed_key = ""
                if ctx:
                    seed_key = f"{ctx.get('file_label', '')}{ctx.get('rot_suffix', '')}"
                seed_offset = sum(ord(ch) for ch in seed_key)
                view_noise_seed = int(view_noise_seed) + seed_offset
            return get_multi_view_data_real_stable_v1(
                radar_frames.cuda(),
                range_start=int(cfg.get('range_start', 0)),
                num_range_bins=int(cfg.get('num_range_bins', 128)),
                num_doppler_bins=int(cfg.get('num_doppler_bins', 128)),
                num_angle_bins=int(cfg.get('num_angle_bins', 128)),
                diagonal_loading=float(cfg.get('diagonal_loading', 0.01)),
                antenna_spacing_lambda=float(cfg.get('antenna_spacing_lambda', 0.5)),
                steering_sign=float(cfg.get('steering_sign', -1.0)),
                normalize_views=bool(cfg.get('normalize_views', True)),
                doppler_lognorm=bool(cfg.get('doppler_lognorm', True)),
                doppler_threshold=cfg.get('doppler_threshold', None),
                view_noise_preset=cfg.get('view_noise_preset', None),
                view_noise_seed=view_noise_seed,
            )
        if three_view_preset != 'legacy':
            raise ValueError(
                "three_view_preset must be one of legacy/real_stable_v1/"
                f"synthetic_stable_v1, got {three_view_preset!r}"
            )

        return get_multi_view_data(
            radar_frames.cuda(),
            num_range_bins=256,
            num_doppler_bins=128,
            num_angle_bins=128
        )

    def _simulate_single_motion(
        self,
        mesh_manager: object,
        metadata: Dict,
        rotation_angle: float = 0.0,
        *,
        cache_path: str | None = None,
        output_name: str = "",
        rot_suffix: str = "",
    ) -> Optional[np.ndarray]:
        """
        Simulate radar echoes for a single motion sequence.

        Args:
            mesh_manager: Mesh manager with loaded motion
            metadata: Motion metadata
            rotation_angle: Rotation angle for data augmentation (radians)

        Returns:
            udoppler: Udoppler spectrum array, or None if failed
        """
        if self.path_cache.enabled and self.path_cache.mode == 'process_only':
            if cache_path is None or not os.path.exists(cache_path):
                raise FileNotFoundError(f"Path cache not found: {cache_path}")
            path_dicts = self._load_path_cache(cache_path)
            return self._paths_to_output(path_dicts)

        # Apply rotation if needed
        if rotation_angle != 0.0:
            rot_mat = np.array([
                [np.cos(rotation_angle), -np.sin(rotation_angle), 0],
                [np.sin(rotation_angle), np.cos(rotation_angle), 0],
                [0, 0, 1]
            ])
            mesh_manager.vertices = np.dot(mesh_manager.vertices, rot_mat.T)

        # Initialize simulator with number of RX antennas
        num_rx = self.config.get('num_antenna', 1)
        simulator = mmSimulator(self.radar_cfg, self.radar, mesh_manager=mesh_manager, num_rx=num_rx)

        # Configure simulator based on metadata
        if 'v_num' in metadata:
            simulator.v_num = metadata['v_num']
        if 'f_num' in metadata:
            simulator.f_num = metadata['f_num']
        if 'mesh_sample_rate' in metadata:
            simulator.mesh_sample_rate = metadata['mesh_sample_rate']

        # Create mesh config from embedded settings
        mesh_cfg = {
            'camera_position': self.camera_position,
            'camera_orientation': self.camera_orientation
        }
        simulator.init(mesh_cfg=mesh_cfg)

        # Update faces if available
        if 'faces' in metadata and metadata['faces'] is not None:
            simulator.update_mesh_faces(metadata['faces'])

        # Generate radar echoes for each frame
        path_dicts = []
        ctx = self._rich_progress_ctx
        n_mesh = mesh_manager.num_frame
        if ctx:
            rot = ctx.get("rot_suffix") or ""
            label = ctx.get("file_label", "")
            prefix = f"{label}" + (f" · rot {rot}" if rot else "")
            ctx["progress"].update(
                ctx["task_seq"],
                completed=0,
                total=max(1, n_mesh),
                description=f"Seq │ {prefix} │ paths 0/{n_mesh}",
            )

        for id_frame in range(n_mesh):
            # Update mesh at current time step
            simulator.mesh_at_time_step(id_frame)

            # Check if mesh is within bounds
            if not self._check_mesh_position(simulator.vertex_positions):
                if self.verbose and not ctx:
                    log_message('WARNING', f'Frame {id_frame} out of bounds, stopping', color='yellow')
                if ctx:
                    rot = ctx.get("rot_suffix") or ""
                    label = ctx.get("file_label", "")
                    prefix = f"{label}" + (f" · rot {rot}" if rot else "")
                    ctx["progress"].update(
                        ctx["task_seq"],
                        description=f"Seq │ {prefix} │ stopped · frame {id_frame} out of bounds",
                    )
                break

            # Compute paths
            a, tau, idx = self._compute_paths(simulator)

            path_dicts.append({
                'a': np.array(a.cpu().numpy(), dtype=np.float32),
                'tau': np.array(tau.cpu().numpy(), dtype=np.float32),
                'idx': idx
            })
            if ctx:
                rot = ctx.get("rot_suffix") or ""
                label = ctx.get("file_label", "")
                prefix = f"{label}" + (f" · rot {rot}" if rot else "")
                done = len(path_dicts)
                ctx["progress"].update(
                    ctx["task_seq"],
                    completed=done,
                    total=max(done, n_mesh),
                    description=f"Seq │ {prefix} │ paths {done}/{n_mesh}",
                )

        # Check if we have enough frames
        num_frame = len(path_dicts) - 1
        if num_frame <= 0:
            if self.verbose and not ctx:
                log_message('ERROR', 'Not enough valid frames', color='red')
            if ctx:
                rot = ctx.get("rot_suffix") or ""
                label = ctx.get("file_label", "")
                prefix = f"{label}" + (f" · rot {rot}" if rot else "")
                ctx["progress"].update(
                    ctx["task_seq"],
                    completed=1,
                    total=1,
                    description=f"Seq │ {prefix} │ insufficient frames",
                )
            return None

        if self.path_cache.enabled and cache_path is not None:
            self._save_path_cache(
                cache_path,
                path_dicts,
                metadata=metadata,
                output_name=output_name,
                rot_suffix=rot_suffix,
                rotation_angle=rotation_angle,
            )
            if self.path_cache.mode == 'save_only':
                return {'path_cache': np.array([cache_path])}

        return self._paths_to_output(path_dicts)

    def process_single_file_with_stats(self, file_path: str) -> dict:
        """
        Process a single file with detailed statistics.

        Args:
            file_path: Path to input file

        Returns:
            result: Dictionary with 'success', 'skipped', and 'reason' keys
        """
        result = {'success': False, 'skipped': False, 'reason': None}
        output_name = self.data_loader.get_output_name(file_path)
        ctx = self._rich_progress_ctx
        dash = ctx is not None

        try:
            # Fast skip before loading SMPL/motion data.  The previous flow
            # loaded the motion first and only then skipped each rotation,
            # which is very slow when resuming a large run with many existing
            # outputs.
            rotation_suffixes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'][:self.num_rotations]
            output_mode = self.config.get('output_mode', 'full')
            output_ext = '.npy' if output_mode == 'udoppler' else '.npz'
            if self.config.get('skip_exist', False):
                expected_paths = []
                for suffix in rotation_suffixes:
                    rot_suffix = suffix if self.num_rotations > 1 else ""
                    if self.path_cache.enabled and self.path_cache.mode == 'save_only':
                        expected_paths.append(self._path_cache_path(output_name, rot_suffix))
                    else:
                        output_file = f'{output_name}{rot_suffix}{output_ext}'
                        expected_paths.append(os.path.join(self.config.output_dir, output_file))
                if expected_paths and all(os.path.exists(p) for p in expected_paths):
                    result['skipped'] = True
                    result['reason'] = 'All outputs already exist'
                    if ctx:
                        label = ctx.get("file_label", output_name)
                        ctx["progress"].update(
                            ctx["task_seq"],
                            completed=1,
                            total=1,
                            description=f"Seq │ {label} │ fast-skip existing output",
                        )
                    elif self.verbose:
                        log_message('PROCESS', f'Fast-skip existing: {output_name}', color='yellow')
                    return result

            # Pure path-cache processing should not load SMPL/motion data.
            # This is the intended second phase of the path-first workflow:
            # pathdicts -> in-memory radar cube -> stable three-view NPZ.
            if self.path_cache.enabled and self.path_cache.mode == 'process_only':
                for suffix in rotation_suffixes:
                    rot_suffix = suffix if self.num_rotations > 1 else ""
                    output_file = f'{output_name}{rot_suffix}{output_ext}'
                    output_path = os.path.join(self.config.output_dir, output_file)
                    cache_path = self._path_cache_path(output_name, rot_suffix)

                    if self.config.get('skip_exist', False) and os.path.exists(output_path):
                        if ctx:
                            label = ctx.get("file_label", output_name)
                            ctx["progress"].update(
                                ctx["task_seq"],
                                completed=1,
                                total=1,
                                description=f"Seq │ {label} │ skip · output exists",
                            )
                        continue

                    if ctx:
                        ctx["rot_suffix"] = suffix if self.num_rotations > 1 else ""

                    sim_result = self._simulate_single_motion(
                        None,
                        {},
                        0.0,
                        cache_path=cache_path,
                        output_name=output_name,
                        rot_suffix=rot_suffix,
                    )
                    if sim_result is None:
                        continue

                    if output_mode == 'udoppler':
                        np.save(output_path, sim_result['doppler_time'])
                    else:
                        np.savez_compressed(output_path, **sim_result)

                    if ctx:
                        label = ctx.get("file_label", output_name)
                        out_short = os.path.basename(output_path)
                        if output_mode == 'udoppler':
                            dt = sim_result["doppler_time"].shape
                            ctx["progress"].update(
                                ctx["task_seq"],
                                description=f"Seq │ {label} │ saved {out_short} · D{dt[1]}",
                            )
                        else:
                            rt = sim_result["range_time"].shape
                            dt = sim_result["doppler_time"].shape
                            az = sim_result["azimuth_time"].shape
                            ctx["progress"].update(
                                ctx["task_seq"],
                                description=(
                                    f"Seq │ {label} │ saved {out_short} · "
                                    f"R{rt[1]} D{dt[1]} A{az[1]}"
                                ),
                            )

                gc.collect()
                torch.cuda.empty_cache()
                result['success'] = True
                result['reason'] = 'Successfully processed from path cache'
                return result

            # Load motion data
            if self.verbose and not dash:
                log_message('PROCESS', f'Processing: {output_name}', color='cyan')
            if ctx:
                label = ctx.get("file_label", output_name)
                ctx["progress"].update(
                    ctx["task_seq"],
                    completed=0,
                    total=1,
                    description=f"Seq │ {label} │ loading SMPL meshes…",
                )

            mesh_manager, metadata = self.data_loader.load_motion(file_path, quiet=dash)

            if self.verbose and not dash:
                log_message('PROCESS', f'Loaded motion: {metadata.get("num_frames", "?")} frames', color='green')
            if ctx:
                label = ctx.get("file_label", output_name)
                nf = metadata.get("num_frames", getattr(mesh_manager, "num_frame", "?"))
                ctx["progress"].update(
                    ctx["task_seq"],
                    completed=0,
                    total=1,
                    description=f"Seq │ {label} │ ready · {nf} frames",
                )

            # Process with rotations if needed
            rotation_angles = [2 * np.pi * i / self.num_rotations for i in range(self.num_rotations)]

            for rot_idx, (angle, suffix) in enumerate(zip(rotation_angles, rotation_suffixes)):
                # Generate output path
                rot_suffix = suffix if self.num_rotations > 1 else ""
                output_file = f'{output_name}{rot_suffix}{output_ext}'
                output_path = os.path.join(self.config.output_dir, output_file)
                cache_path = (
                    self._path_cache_path(output_name, rot_suffix)
                    if self.path_cache.enabled else None
                )
                exists_path = (
                    cache_path
                    if self.path_cache.enabled and self.path_cache.mode == 'save_only'
                    else output_path
                )

                # Skip if exists
                if self.config.get('skip_exist', False) and os.path.exists(exists_path):
                    if self.verbose and not dash:
                        log_message('PROCESS', f'Skipping rotation {suffix} (exists)', color='yellow')
                    if ctx:
                        label = ctx.get("file_label", output_name)
                        ctx["progress"].update(
                            ctx["task_seq"],
                            completed=1,
                            total=1,
                            description=f"Seq │ {label} │ skip · rot {suffix} exists",
                        )
                    continue

                # Reload motion for each rotation (to avoid accumulating rotations)
                if rot_idx > 0:
                    if ctx:
                        label = ctx.get("file_label", output_name)
                        ctx["progress"].update(
                            ctx["task_seq"],
                            completed=0,
                            total=1,
                            description=f"Seq │ {label} │ reload for rot {suffix}…",
                        )
                    mesh_manager, metadata = self.data_loader.load_motion(file_path, quiet=dash)

                # Simulate
                if self.verbose and self.num_rotations > 1 and not dash:
                    log_message('PROCESS', f'Processing rotation {suffix} ({np.degrees(angle):.1f}°)', color='cyan')

                if ctx:
                    ctx["rot_suffix"] = suffix if self.num_rotations > 1 else ""

                sim_result = self._simulate_single_motion(
                    mesh_manager,
                    metadata,
                    angle,
                    cache_path=cache_path,
                    output_name=output_name,
                    rot_suffix=rot_suffix,
                )

                if sim_result is None:
                    continue

                if self.path_cache.enabled and self.path_cache.mode == 'save_only':
                    if ctx:
                        label = ctx.get("file_label", output_name)
                        cache_short = os.path.basename(cache_path)
                        ctx["progress"].update(
                            ctx["task_seq"],
                            description=f"Seq │ {label} │ saved {cache_short}",
                        )
                    elif self.verbose:
                        log_message('PROCESS', f'Saved path cache: {cache_path}', color='green')
                    continue

                # Save result based on output_mode
                if output_mode == 'udoppler':
                    # udoppler mode: save doppler_time as a single .npy
                    np.save(output_path, sim_result['doppler_time'])
                else:
                    # full mode: save all views as compressed .npz
                    np.savez_compressed(output_path, **sim_result)

                if ctx:
                    label = ctx.get("file_label", output_name)
                    out_short = os.path.basename(output_path)
                    if output_mode == 'udoppler':
                        dt = sim_result["doppler_time"].shape
                        ctx["progress"].update(
                            ctx["task_seq"],
                            description=f"Seq │ {label} │ saved {out_short} · D{dt[1]}",
                        )
                    else:
                        rt = sim_result["range_time"].shape
                        dt = sim_result["doppler_time"].shape
                        az = sim_result["azimuth_time"].shape
                        ctx["progress"].update(
                            ctx["task_seq"],
                            description=(
                                f"Seq │ {label} │ saved {out_short} · "
                                f"R{rt[1]} D{dt[1]} A{az[1]}"
                            ),
                        )
                elif self.verbose:
                    if output_mode == 'udoppler':
                        log_message('PROCESS', f'Saved: {output_path}, doppler={sim_result["doppler_time"].shape}', color='green')
                    else:
                        shapes_str = f"range={sim_result['range_time'].shape}, " \
                                    f"doppler={sim_result['doppler_time'].shape}, " \
                                    f"azimuth={sim_result['azimuth_time'].shape}"
                        log_message('PROCESS', f'Saved: {output_path}, {shapes_str}', color='green')

            # Clean up
            gc.collect()
            torch.cuda.empty_cache()

            result['success'] = True
            result['reason'] = 'Successfully processed'
            return result

        except Exception as e:
            result['reason'] = f'Processing error: {str(e)}'
            log_message('ERROR', f'Error processing {output_name}: {e}', color='red')
            import traceback
            traceback.print_exc()
            return result

    def run(self, start_idx: int = 0, end_idx: int = -1):
        """
        Run the simulation pipeline.

        Args:
            start_idx: Start index of files to process
            end_idx: End index of files to process (-1 for all)
        """
        # Get file list
        file_list = self.data_loader.get_file_list()

        if end_idx == -1:
            end_idx = len(file_list)

        file_list = file_list[start_idx:end_idx]
        n_files = len(file_list)
        quiet_dash_header = self.rich_progress and n_files > 0

        if not quiet_dash_header:
            log_message('SIM', '=== Starting Simulation Pipeline ===', color='cyan', attrs=('bold',))
            log_message('SIM', f'Data source: {self.data_loader.__class__.__name__}', color='cyan')
            log_message('SIM', f'Simulation method: {self.simulation_method}', color='cyan')
            log_message('SIM', f'Number of rotations: {self.num_rotations}', color='cyan')
            log_message('SIM', f'Total files: {n_files}', color='green')
            log_message('SIM', f'Output directory: {self.config.output_dir}', color='cyan')

        # Print radar spectrum parameters (only once)
        if self.verbose and not quiet_dash_header:
            print_radar_spectrum_info(
                radar_cfg=self.radar_cfg,
                num_antennas=self.config.get('num_antenna', 1),
                num_range_bins=256,  # Full range bins
                num_doppler_bins=128,
                num_angle_bins=128
            )

        # Process files with detailed statistics
        success_count = 0
        failed_count = 0
        skipped_count = 0
        failure_reasons = {}

        def _one_file(i: int, file_path: str) -> tuple:
            result = self.process_single_file_with_stats(file_path)
            if result['success']:
                sc, fj, fk = 1, 0, 0
            elif result['skipped']:
                sc, fj, fk = 0, 1, 0
            else:
                sc, fj, fk = 0, 0, 1
            return result, sc, fj, fk

        if self.rich_progress and n_files > 0:
            from rich import box
            from rich.console import Console, Group
            from rich.panel import Panel
            from rich.progress import (
                BarColumn,
                MofNCompleteColumn,
                Progress,
                SpinnerColumn,
                TextColumn,
                TimeElapsedColumn,
                TimeRemainingColumn,
            )

            prog_console = Console(stderr=True)
            prog_columns = (
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(bar_width=32),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
            )
            prog_console.print(
                Panel.fit(
                    Group(
                        "[bold]Simulation dashboard[/bold]",
                        "[dim]Files: all inputs · Seq: load / paths / radar / saved[/dim]",
                        f"[dim]{self.data_loader.__class__.__name__} · {self.simulation_method} · "
                        f"rots={self.num_rotations} · files={n_files}[/dim]",
                        f"[dim]out:[/dim] {self.config.output_dir}",
                    ),
                    border_style="bright_blue",
                    box=box.ROUNDED,
                )
            )
            with Progress(
                *prog_columns,
                console=prog_console,
                transient=False,
            ) as progress:
                task_files = progress.add_task("", total=n_files)
                task_seq = progress.add_task("", total=1)
                for i, file_path in enumerate(file_list):
                    base = os.path.basename(file_path)
                    progress.update(
                        task_files,
                        description=f"Files │ {i + 1}/{n_files} │ {base} │ …",
                    )
                    progress.update(
                        task_seq,
                        completed=0,
                        total=1,
                        description=f"Seq │ {base} │ waiting…",
                    )
                    self._rich_progress_ctx = {
                        "progress": progress,
                        "task_files": task_files,
                        "task_seq": task_seq,
                        "file_label": base,
                        "rot_suffix": "",
                    }
                    try:
                        result, sc, sj, sf = _one_file(i, file_path)
                    finally:
                        self._rich_progress_ctx = None
                        progress.update(
                            task_seq,
                            completed=1,
                            total=1,
                            description="Seq │ — │ idle",
                        )
                    success_count += sc
                    skipped_count += sj
                    failed_count += sf
                    if sf:
                        reason = result.get('reason', 'Unknown error')
                        failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
                    done = i + 1
                    if result['success']:
                        tag = "ok"
                    elif result['skipped']:
                        tag = "skip"
                    else:
                        tag = "FAIL"
                    progress.update(
                        task_files,
                        description=f"Files │ {done}/{n_files} │ {base} │ {tag}",
                    )
                    progress.advance(task_files)
        else:
            for i, file_path in enumerate(file_list):
                result, sc, sj, sf = _one_file(i, file_path)
                success_count += sc
                skipped_count += sj
                failed_count += sf
                if sf:
                    reason = result.get('reason', 'Unknown error')
                    failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
                done = i + 1
                base = os.path.basename(file_path)
                pct = 100.0 * done / n_files if n_files else 0.0
                if result['success']:
                    status, st_color = "ok", "green"
                elif result['skipped']:
                    status, st_color = "skipped", "yellow"
                else:
                    status, st_color = "FAIL", "red"
                log_message(
                    "PROGRESS",
                    f"{done}/{n_files} ({pct:.2f}%) {base} [{status}]",
                    color=st_color,
                )

        print()
        log_message('SUMMARY', '=== Simulation Summary ===', color='cyan', attrs=('bold',))
        log_message('SUMMARY', f'Total files: {len(file_list)}', color='cyan')
        log_message('SUMMARY', f'Successfully processed: {success_count}', color='green', attrs=('bold',))
        log_message('SUMMARY', f'Failed: {failed_count}', color='red')
        log_message('SUMMARY', f'Skipped (already exists): {skipped_count}', color='yellow')

        # Print failure details if there were failures
        if failed_count > 0:
            print()
            log_message('SUMMARY', 'Failure Details:', color='red', attrs=('bold',))
            for reason, count in failure_reasons.items():
                log_message('SUMMARY', f'{reason}: {count}', color='red')

        # Print success rate
        success_rate = (success_count / len(file_list)) * 100 if len(file_list) > 0 else 0
        print()
        color = 'green' if success_rate >= 90 else 'yellow' if success_rate >= 50 else 'red'
        log_message('SUMMARY', f'Success rate: {success_rate:.1f}%', color=color)

    def process_single_file(self, file_path: str) -> bool:
        """
        Legacy method for backward compatibility.

        Args:
            file_path: Path to input file

        Returns:
            success: Whether processing was successful
        """
        result = self.process_single_file_with_stats(file_path)
        return result['success']
