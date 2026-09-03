"""
Data loaders for different motion data sources.
"""

import os
import glob
from typing import Dict, List, Tuple
from easydict import EasyDict as edict

from simulation_base import DataLoader
from mesh_manager import SMPLMeshManager, HunyuanWoodenMeshManager, EllipseMeshManager
from utils.logger import log_message


class SMPLDataLoader(DataLoader):
    """Data loader for SMPL motion data."""

    def __init__(self, config: edict):
        super().__init__(config)
        self.data_pattern = config.get('data_pattern', None)
        self.data_root = config.get('data_root', None)  # Backward compatibility
        self.model_type = config.get('model_type', 'smpl')
        self.mesh_sample_rate = config.get('mesh_sample_rate', 50)
        # One BodyModel per process; reloading weights every file exhausts VRAM and stalls CUDA.
        self._smpl_mesh_manager: SMPLMeshManager | None = None

    def get_file_list(self) -> List[str]:
        """Get list of SMPL files using glob pattern or data_root."""
        if self.data_pattern:
            # Use glob pattern
            files = sorted(glob.glob(self.data_pattern))
        elif self.data_root:
            # Backward compatibility: use data_root
            smpl_dir = os.path.join(self.data_root, self.model_type)
            file_list = sorted(os.listdir(smpl_dir))
            files = [os.path.join(smpl_dir, f) for f in file_list if f.endswith('.npy')]
        else:
            raise ValueError("Either 'data_pattern' or 'data_root' must be specified")

        return [f for f in files if f.endswith('.npy')]

    def load_motion(self, file_path: str, *, quiet: bool = False) -> Tuple[SMPLMeshManager, Dict]:
        """Load SMPL motion data."""
        if not quiet:
            log_message(
                'SMPL',
                f'Loading sequence meshes: {os.path.basename(file_path)}',
                color='cyan',
            )
        manager = SMPLMeshManager(config=self.config)
        manager.load_sequence_meshes(file_path)

        metadata = {
            'num_frames': manager.num_frame,
            'v_num': 6890,
            'f_num': 13776,
            'faces': manager.smpl.f,
            'mesh_sample_rate': self.mesh_sample_rate,
        }

        return manager, metadata

    def get_output_name(self, input_path: str) -> str:
        """Generate output filename."""
        return os.path.basename(input_path)[:-4]  # Remove .npy


class HunyuanWoodenDataLoader(DataLoader):
    """Data loader for HY-Motion wooden mesh NPZ artifacts."""

    def __init__(self, config: edict):
        super().__init__(config)
        self.data_pattern = config.get('data_pattern', None)
        self.data_root = config.get('data_root', None)
        self.fps = config.get('fps', 30)
        self.mesh_sample_rate = config.get('mesh_sample_rate', 24)
        self.strip_seed_suffix = bool(config.get('strip_hunyuan_seed_suffix', True))

    def get_file_list(self) -> List[str]:
        """Get HY-Motion wooden NPZ files using glob pattern or data_root."""
        if self.data_pattern:
            files = sorted(glob.glob(self.data_pattern))
        elif self.data_root:
            files = sorted(glob.glob(os.path.join(self.data_root, '*.npz')))
        else:
            raise ValueError("Either 'data_pattern' or 'data_root' must be specified")
        return [f for f in files if f.endswith('.npz')]

    def load_motion(self, file_path: str, *, quiet: bool = False) -> Tuple[HunyuanWoodenMeshManager, Dict]:
        """Load HY-Motion wooden mesh vertices from poses/trans NPZ."""
        if not quiet:
            log_message(
                'WOODEN',
                f'Loading HY-Motion wooden mesh: {os.path.basename(file_path)}',
                color='cyan',
            )
        manager = HunyuanWoodenMeshManager(fps=self.fps, config=self.config)
        manager.load_sequence_meshes(file_path)

        metadata = {
            'num_frames': manager.num_frame,
            'v_num': manager.vertices.shape[1],
            'f_num': manager.faces.shape[0],
            'faces': manager.faces,
            'mesh_sample_rate': self.mesh_sample_rate,
            'mesh_schema': 'hunyuan_wooden_smplx_npz',
        }
        return manager, metadata

    def get_output_name(self, input_path: str) -> str:
        """Generate output filename, optionally removing HY-Motion seed suffix."""
        stem = os.path.basename(input_path)[:-4]
        if self.strip_seed_suffix and stem.endswith('_000'):
            stem = stem[:-4]
        return stem


class JointsDataLoader(DataLoader):
    """Data loader for joints motion data."""

    def __init__(self, config: edict):
        super().__init__(config)
        self.data_pattern = config.get('data_pattern', None)
        self.data_root = config.get('data_root', None)  # Backward compatibility
        self.fps = config.get('fps', 20)
        self.interp_n = config.get('interp_n', 50)

    def get_file_list(self) -> List[str]:
        """Get list of joints files using glob pattern or data_root."""
        if self.data_pattern:
            # Use glob pattern
            files = sorted(glob.glob(self.data_pattern))
        elif self.data_root:
            # Backward compatibility: use data_root/joints
            joints_dir = os.path.join(self.data_root, 'joints')
            files = sorted(glob.glob(os.path.join(joints_dir, '*.npy')))
        else:
            raise ValueError("Either 'data_pattern' or 'data_root' must be specified")

        return files

    def load_motion(self, file_path: str, *, quiet: bool = False) -> Tuple[EllipseMeshManager, Dict]:
        """Load joints motion data."""
        manager = EllipseMeshManager(fps=self.fps, config=self.config)
        manager.num_insect = self.interp_n
        manager.load_motion_sequence(file_path)

        metadata = {
            'num_frames': manager.num_frame,
            'v_num': manager.vertices.shape[1],
            'f_num': 0,  # No faces for point cloud
            'faces': None,
        }

        return manager, metadata

    def get_output_name(self, input_path: str) -> str:
        """Generate output filename."""
        return os.path.basename(input_path)[:-4]
