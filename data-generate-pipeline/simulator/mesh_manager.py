import numpy as np
import os
import os.path as osp
import torch

import sys; sys.path.append(osp.dirname(__file__))
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d


class SMPLMeshManager:
    def __init__(self, config=None):
        try:
            from human_body_prior.body_model.body_model import BodyModel
        except ImportError as exc:
            raise ImportError(
                "SMPL simulation requires the human_body_prior package. "
                "Install it separately before using data_source=smpl."
            ) from exc

        configured_model = config.get('smpl_model_path') if config else None
        model_path = configured_model or os.environ.get('SMPL_MODEL_PATH')
        if not model_path:
            raise ValueError(
                "Set smpl_model_path in the simulation config or export "
                "SMPL_MODEL_PATH to a licensed SMPL-H neutral model.npz file."
            )
        if not osp.isfile(model_path):
            raise FileNotFoundError(f"SMPL model file not found: {model_path}")

        device_name = str(config.get('device', 'cuda')) if config else 'cuda'
        self.device = torch.device(device_name)
        self.smpl = BodyModel(bm_fname=str(model_path), num_betas=16).to(self.device)
        self.adaptive_xy_fit = None
        if config and hasattr(config, 'adaptive_xy_fit'):
            fit_cfg = config.adaptive_xy_fit
            if getattr(fit_cfg, 'enabled', False):
                self.adaptive_xy_fit = fit_cfg

    def _apply_adaptive_xy_fit_smpl(self, vertices, trans_np):
        """Fit root XY trajectory into configured bounds while preserving body shape.

        Args:
            vertices: (T, V, 3) numpy array of vertex positions.
            trans_np: (T, 2) numpy array of root XY translation.

        Returns:
            vertices with compressed/translated XY, same shape.
        """
        fit_cfg = self.adaptive_xy_fit
        x_bounds = np.asarray(getattr(fit_cfg, 'x'), dtype=np.float64)
        y_bounds = np.asarray(getattr(fit_cfg, 'y'), dtype=np.float64)
        margin = float(getattr(fit_cfg, 'margin', 0.0))
        preserve_aspect = bool(getattr(fit_cfg, 'preserve_aspect', True))

        x_min, x_max = float(x_bounds[0] + margin), float(x_bounds[1] - margin)
        y_min, y_max = float(y_bounds[0] + margin), float(y_bounds[1] - margin)
        if not (x_max > x_min and y_max > y_min):
            raise ValueError(
                f"Invalid adaptive_xy_fit bounds after margin: "
                f"x=[{x_min}, {x_max}], y=[{y_min}, {y_max}]"
            )

        root_xy = trans_np.copy()  # (T, 2)
        local_xy = vertices[:, :, :2] - root_xy[:, None, :]  # (T, V, 2)
        local_min = local_xy.reshape(-1, 2).min(axis=0)
        local_max = local_xy.reshape(-1, 2).max(axis=0)

        allowed_min = np.array([x_min, y_min], dtype=np.float64) - local_min
        allowed_max = np.array([x_max, y_max], dtype=np.float64) - local_max
        if np.any(allowed_max <= allowed_min):
            raise ValueError(
                "adaptive_xy_fit bounds are too small for the body extents: "
                f"allowed_min={allowed_min.tolist()}, allowed_max={allowed_max.tolist()}, "
                f"local_min={local_min.tolist()}, local_max={local_max.tolist()}"
            )

        old_min = root_xy.min(axis=0)
        old_max = root_xy.max(axis=0)
        old_center = 0.5 * (old_min + old_max)
        old_span = np.maximum(old_max - old_min, 1e-6)

        target_center = 0.5 * (allowed_min + allowed_max)
        target_span = allowed_max - allowed_min

        scale_xy = target_span / old_span
        if preserve_aspect:
            scale = min(1.0, float(scale_xy.min()))
            scale_vec = np.array([scale, scale], dtype=np.float64)
        else:
            scale_vec = np.minimum(1.0, scale_xy)

        fitted_root_xy = target_center + (root_xy - old_center) * scale_vec

        # Ensure fitted root stays within bounds
        fitted_min = fitted_root_xy.min(axis=0)
        fitted_max = fitted_root_xy.max(axis=0)
        shift = np.zeros(2, dtype=np.float64)
        if fitted_min[0] < allowed_min[0]:
            shift[0] += allowed_min[0] - fitted_min[0]
        if fitted_max[0] > allowed_max[0]:
            shift[0] += allowed_max[0] - fitted_max[0]
        if fitted_min[1] < allowed_min[1]:
            shift[1] += allowed_min[1] - fitted_min[1]
        if fitted_max[1] > allowed_max[1]:
            shift[1] += allowed_max[1] - fitted_max[1]
        fitted_root_xy += shift

        vertices = vertices.copy()
        vertices[:, :, :2] = fitted_root_xy[:, None, :] + local_xy
        return vertices

    def load_sequence_meshes(self, smpl_path=None):
        self.vertices = self.load_meshes(smpl_path, device=self.device)
        self.num_frame = self.vertices.shape[0]

    def mesh_at_time_step(self, frame, chirp=0):
        vertices_curr = self.vertices[frame]
        # use trimesh to compute normals
        return vertices_curr, None

    def load_meshes(self, smpl_path, device):
        motion = np.load(smpl_path, allow_pickle=True).item()
        trans = torch.tensor(motion['bdata_trans'], device=self.device)
        root_orient = torch.tensor(motion['bdata_poses'][:, :3], device=self.device).float()
        pose_body = torch.tensor(motion['bdata_poses'][:, 3:66], device=self.device).float()
        pose_hand = torch.tensor(motion['bdata_poses'][:, 66:], device=self.device).float()
        num_frame = pose_body.shape[0]

        output = self.smpl(
            root_orient=root_orient,
            pose_body=pose_body,
            pose_hand=pose_hand,
            betas=torch.zeros(num_frame, 16, device=self.device),
            trans=trans,
        )
        vertices = output.v
        vertices = vertices @ torch.tensor([[0, -1, 0], [1, 0, 0], [0, 0, 1]], device=device, dtype=torch.float64)
        # put on the floor
        # floor_height = vertices[:, :, 2].min(dim=1).values
        # vertices -= torch.stack([torch.zeros_like(floor_height), torch.zeros_like(floor_height), floor_height], dim=1).unsqueeze(1)

        # perform interpolation
        vertices_np = vertices.cpu().numpy()  # (num_frame, 6890, 3) — already rotated
        trans_np = trans.cpu().numpy()        # (num_frame, 3) — NOT rotated yet!

        # Apply the same rotation to trans so it matches vertices coordinate system
        rot = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float64)
        trans_np = trans_np @ rot  # (num_frame, 3) now in same coords as vertices

        # optional per-sample XY trajectory compression (before interpolation)
        if self.adaptive_xy_fit is not None:
            vertices_np = self._apply_adaptive_xy_fit_smpl(
                vertices_np, trans_np[:, :2]
            )

        vertices = vertices_np.reshape(-1, 6890 * 3)
        num_frames = int(len(vertices) * 50 / 20) # 20 fps -> 50 fps
        interpolater = interp1d(np.arange(len(vertices)), vertices, axis=0)
        vertices = interpolater(np.linspace(0, len(vertices) - 1, num_frames))
        vertices = vertices.reshape(-1, 6890, 3)
        vertices = gaussian_filter1d(vertices, sigma=1., axis=0)
        return vertices


class HunyuanWoodenMeshManager:
    """Load HY-Motion wooden-mesh NPZ files as animated mesh vertices."""

    def __init__(self, fps=30, config=None):
        self.fps = fps
        self.config = config
        self.adaptive_xy_fit = None
        if config and hasattr(config, 'adaptive_xy_fit'):
            fit_cfg = config.adaptive_xy_fit
            if getattr(fit_cfg, 'enabled', False):
                self.adaptive_xy_fit = fit_cfg

        self.scale = config.get('scale', 1.0) if config else 1.0
        self.batch_size = int(config.get('wooden_batch_size', 8)) if config else 8
        self.smooth_sigma = float(config.get('wooden_smooth_sigma', 1.0)) if config else 1.0

        default_model_dir = osp.abspath(
            osp.join(
                osp.dirname(__file__),
                "..",
                "backends",
                "hymotion",
                "scripts",
                "gradio",
                "static",
                "assets",
                "dump_wooden",
            )
        )
        self.model_dir = str(config.get('hunyuan_wooden_model_dir', default_model_dir)) if config else default_model_dir
        self._wooden_mesh = None
        self.faces = None

    @staticmethod
    def _coordinate_rotation():
        return np.array([[0, -1, 0], [0, 0, 1], [-1, 0, 0]], dtype=np.float64)

    def _load_wooden_model(self):
        if self._wooden_mesh is not None:
            return self._wooden_mesh

        hymotion_root = osp.abspath(
            osp.join(osp.dirname(__file__), "..", "backends", "hymotion")
        )
        if hymotion_root not in sys.path:
            sys.path.insert(0, hymotion_root)
        from hymotion.pipeline.body_model import WoodenMesh

        self._wooden_mesh = WoodenMesh(self.model_dir).eval()
        self.faces = np.asarray(self._wooden_mesh.faces, dtype=np.int64)
        return self._wooden_mesh

    def _apply_adaptive_xy_fit_wooden(self, vertices, root_xy):
        fit_cfg = self.adaptive_xy_fit
        if fit_cfg is None:
            return vertices

        x_bounds = np.asarray(getattr(fit_cfg, 'x'), dtype=np.float64)
        y_bounds = np.asarray(getattr(fit_cfg, 'y'), dtype=np.float64)
        margin = float(getattr(fit_cfg, 'margin', 0.0))
        preserve_aspect = bool(getattr(fit_cfg, 'preserve_aspect', True))

        x_min, x_max = float(x_bounds[0] + margin), float(x_bounds[1] - margin)
        y_min, y_max = float(y_bounds[0] + margin), float(y_bounds[1] - margin)
        if not (x_max > x_min and y_max > y_min):
            raise ValueError(
                f"Invalid adaptive_xy_fit bounds after margin: "
                f"x=[{x_min}, {x_max}], y=[{y_min}, {y_max}]"
            )

        local_xy = vertices[:, :, :2] - root_xy[:, None, :]
        local_min = local_xy.reshape(-1, 2).min(axis=0)
        local_max = local_xy.reshape(-1, 2).max(axis=0)

        allowed_min = np.array([x_min, y_min], dtype=np.float64) - local_min
        allowed_max = np.array([x_max, y_max], dtype=np.float64) - local_max
        if np.any(allowed_max <= allowed_min):
            raise ValueError(
                "adaptive_xy_fit bounds are too small for the wooden mesh extents: "
                f"allowed_min={allowed_min.tolist()}, allowed_max={allowed_max.tolist()}, "
                f"local_min={local_min.tolist()}, local_max={local_max.tolist()}"
            )

        old_min = root_xy.min(axis=0)
        old_max = root_xy.max(axis=0)
        old_center = 0.5 * (old_min + old_max)
        old_span = np.maximum(old_max - old_min, 1e-6)

        target_center = 0.5 * (allowed_min + allowed_max)
        target_span = allowed_max - allowed_min
        scale_xy = target_span / old_span
        if preserve_aspect:
            scale = min(1.0, float(scale_xy.min()))
            scale_vec = np.array([scale, scale], dtype=np.float64)
        else:
            scale_vec = np.minimum(1.0, scale_xy)

        fitted_root_xy = target_center + (root_xy - old_center) * scale_vec
        fitted_min = fitted_root_xy.min(axis=0)
        fitted_max = fitted_root_xy.max(axis=0)
        shift = np.zeros(2, dtype=np.float64)
        if fitted_min[0] < allowed_min[0]:
            shift[0] += allowed_min[0] - fitted_min[0]
        if fitted_max[0] > allowed_max[0]:
            shift[0] += allowed_max[0] - fitted_max[0]
        if fitted_min[1] < allowed_min[1]:
            shift[1] += allowed_min[1] - fitted_min[1]
        if fitted_max[1] > allowed_max[1]:
            shift[1] += allowed_max[1] - fitted_max[1]
        fitted_root_xy += shift

        vertices = vertices.copy()
        vertices[:, :, :2] = fitted_root_xy[:, None, :] + local_xy
        return vertices

    def load_sequence_meshes(self, wooden_npz_path):
        data = np.load(wooden_npz_path, allow_pickle=False)
        if "poses" not in data or "trans" not in data:
            raise ValueError(
                f"HY-Motion wooden NPZ must contain poses/trans: {wooden_npz_path}"
            )

        poses = np.asarray(data["poses"], dtype=np.float32)
        trans = np.asarray(data["trans"], dtype=np.float32)
        if poses.ndim == 3:
            poses = poses.reshape(-1, poses.shape[-1])
        if trans.ndim == 3:
            trans = trans.reshape(-1, trans.shape[-1])
        if poses.shape[0] != trans.shape[0]:
            raise ValueError(
                f"poses/trans frame mismatch for {wooden_npz_path}: "
                f"{poses.shape} vs {trans.shape}"
            )
        if poses.shape[1] % 3 != 0:
            raise ValueError(f"poses should be flattened axis-angle joints: {poses.shape}")

        wooden = self._load_wooden_model()
        vertices = []
        with torch.no_grad():
            for start in range(0, poses.shape[0], self.batch_size):
                end = min(poses.shape[0], start + self.batch_size)
                result = wooden(
                    {
                        "poses": torch.tensor(poses[start:end], dtype=torch.float32),
                        "trans": torch.tensor(trans[start:end], dtype=torch.float32),
                    }
                )
                vertices.append(result["vertices"].detach().cpu().numpy())
        vertices = np.concatenate(vertices, axis=0).astype(np.float32, copy=False)

        rot = self._coordinate_rotation()
        vertices = vertices @ rot
        trans_rot = trans @ rot

        if self.scale != 1.0:
            vertices[:, :, :2] = vertices[:, :, :2] * self.scale
            trans_rot[:, :2] = trans_rot[:, :2] * self.scale

        vertices = self._apply_adaptive_xy_fit_wooden(vertices, trans_rot[:, :2])

        floor_height = vertices[0, :, 2].min()
        vertices[:, :, 2] -= floor_height

        flat = vertices.reshape(vertices.shape[0], -1)
        num_frames = int(len(flat) * 50 / self.fps)
        interpolater = interp1d(np.arange(len(flat)), flat, axis=0)
        flat = interpolater(np.linspace(0, len(flat) - 1, num_frames))
        vertices = flat.reshape(num_frames, vertices.shape[1], 3)
        if self.smooth_sigma > 0:
            vertices = gaussian_filter1d(vertices, sigma=self.smooth_sigma, axis=0)

        self.vertices = vertices.astype(np.float32, copy=False)
        self.num_frame = self.vertices.shape[0]
        self.num_vertices = self.vertices.shape[1]

    def mesh_at_time_step(self, frame, chirp=0):
        vertices_curr = self.vertices[frame]
        return vertices_curr, None


class EllipseMeshManager:
    def __init__(self, fps=20, config=None):
        self.joint_schema = "body24"
        self._configure_body24_skeleton()

        # XY scale factor for movement scaling (from config)
        if config and hasattr(config, 'scale'):
            self.scale = config.scale
        else:
            self.scale = 1.0  # Default: no scaling

        # Optional per-sample root-trajectory fit. This compresses/translates
        # the global XY path into a configured room window without shrinking
        # the local body pose around the root joint.
        self.adaptive_xy_fit = None
        if config and hasattr(config, 'adaptive_xy_fit'):
            fit_cfg = config.adaptive_xy_fit
            if getattr(fit_cfg, 'enabled', False):
                self.adaptive_xy_fit = fit_cfg

        self.num_insect = 50
        self.fps = fps

    def _configure_body24_skeleton(self):
        """Use the supported 24-joint body skeleton."""
        self.joint_schema = "body24"
        self.num_joints = 24
        self.UNIT_TO_METER = 0.021
        self.bodyparts = [
            [0, 1, 2, 3, 4, 5, 6],     # Chain 0: main spine/torso
            [3, 7, 8, 9, 10],          # Chain 1: one arm from spine joint 3
            [3, 11, 12, 13, 14],       # Chain 2: other arm from spine joint 3
            [0, 15, 16, 17, 18, 19],   # Chain 3: one leg from root joint 0
            [15, 20, 21, 22, 23]       # Chain 4: other leg from joint 15
        ]
        self._set_bones_from_bodyparts()

    def _configure_hunyuan_motion_body_skeleton(self):
        """Use HY-Motion wooden-skeleton body joints without fingers."""
        self.joint_schema = "hunyuan_motion_body22"
        self.num_joints = 22
        self.UNIT_TO_METER = 1.0

        model_dir = osp.abspath(
            osp.join(
                osp.dirname(__file__),
                "..",
                "backends",
                "hymotion",
                "scripts",
                "gradio",
                "static",
                "assets",
                "dump_wooden",
            )
        )
        kintree_path = osp.join(model_dir, "kintree.bin")
        parents = self._default_hunyuan_motion_parents()[:self.num_joints]
        if osp.exists(kintree_path) and not self._looks_like_lfs_pointer(kintree_path):
            with open(kintree_path, "rb") as f:
                loaded = np.frombuffer(f.read(), dtype=np.int32)
            loaded = loaded[:self.num_joints]
            if len(loaded) == self.num_joints and loaded.max(initial=0) < self.num_joints:
                parents = loaded

        self.bodyparts = self._bodyparts_from_parents(parents)
        self.bones = [
            [int(parent), int(idx)]
            for idx, parent in enumerate(parents)
            if idx != 0 and parent >= 0
        ]

    def _configure_skeleton_for_joint_count(self, joint_count):
        if joint_count == self.num_joints:
            return
        if joint_count == 24:
            self._configure_body24_skeleton()
            return
        if joint_count == 22:
            self._configure_hunyuan_motion_body_skeleton()
            return
        if joint_count == 52:
            self._configure_hunyuan_motion_body_skeleton()
            return
        raise ValueError(
            f"Motion data has {joint_count} joints, but only 24-joint and "
            "HY-Motion 22-joint body schemas are supported"
        )

    def _set_bones_from_bodyparts(self):
        self.bones = []
        for part in self.bodyparts:
            for i in range(len(part) - 1):
                self.bones.append([part[i], part[i + 1]])

    @staticmethod
    def _bodyparts_from_parents(parents):
        children = {idx: [] for idx in range(len(parents))}
        root = 0
        for idx, parent in enumerate(parents):
            if parent < 0:
                root = idx
            elif idx != parent:
                children[int(parent)].append(idx)

        bodyparts = []

        def walk(node, path):
            if not children[node]:
                if len(path) > 1:
                    bodyparts.append(path)
                return
            for child in children[node]:
                walk(child, path + [child])

        walk(root, [root])
        return bodyparts

    @staticmethod
    def _looks_like_lfs_pointer(path):
        with open(path, "rb") as f:
            head = f.read(64)
        return head.startswith(b"version https://git-lfs.github.com/spec")

    @staticmethod
    def _default_hunyuan_motion_parents():
        return np.array(
            [
                -1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                9, 9, 12, 13, 14, 16, 17, 18, 19,
                20, 22, 23, 20, 25, 26, 20, 28, 29,
                20, 31, 32, 20, 34, 35,
                21, 37, 38, 21, 40, 41, 21, 43, 44,
                21, 46, 47, 21, 49, 50,
            ],
            dtype=np.int32,
        )

    def _apply_adaptive_xy_fit(self, motion):
        """Fit root XY trajectory into configured bounds while preserving local pose."""
        fit_cfg = self.adaptive_xy_fit
        if fit_cfg is None:
            return motion

        x_bounds = np.asarray(getattr(fit_cfg, 'x'), dtype=np.float64)
        y_bounds = np.asarray(getattr(fit_cfg, 'y'), dtype=np.float64)
        margin = float(getattr(fit_cfg, 'margin', 0.0))
        preserve_aspect = bool(getattr(fit_cfg, 'preserve_aspect', True))

        x_min, x_max = float(x_bounds[0] + margin), float(x_bounds[1] - margin)
        y_min, y_max = float(y_bounds[0] + margin), float(y_bounds[1] - margin)
        if not (x_max > x_min and y_max > y_min):
            raise ValueError(
                f"Invalid adaptive_xy_fit bounds after margin: "
                f"x=[{x_min}, {x_max}], y=[{y_min}, {y_max}]"
            )

        root_xy = motion[:, 0, :2].copy()
        local_xy = motion[:, :, :2] - root_xy[:, None, :]
        local_min = local_xy.reshape(-1, 2).min(axis=0)
        local_max = local_xy.reshape(-1, 2).max(axis=0)

        # The simulator validates every body point, not just the root.
        # Therefore fit the root into the smaller range that keeps all
        # root-relative local body offsets inside the requested room window.
        allowed_min = np.array([x_min, y_min], dtype=np.float64) - local_min
        allowed_max = np.array([x_max, y_max], dtype=np.float64) - local_max
        if np.any(allowed_max <= allowed_min):
            raise ValueError(
                "adaptive_xy_fit bounds are too small for the local body extents: "
                f"allowed_min={allowed_min.tolist()}, allowed_max={allowed_max.tolist()}, "
                f"local_min={local_min.tolist()}, local_max={local_max.tolist()}"
            )

        old_min = root_xy.min(axis=0)
        old_max = root_xy.max(axis=0)
        old_center = 0.5 * (old_min + old_max)
        old_span = np.maximum(old_max - old_min, 1e-6)

        target_center = 0.5 * (allowed_min + allowed_max)
        target_span = allowed_max - allowed_min

        scale_xy = target_span / old_span
        if preserve_aspect:
            scale = min(1.0, float(scale_xy.min()))
            scale_vec = np.array([scale, scale], dtype=np.float64)
        else:
            scale_vec = np.minimum(1.0, scale_xy)

        fitted_root_xy = target_center + (root_xy - old_center) * scale_vec

        # If the clip was already inside the box but not centered, translate it
        # just enough to be inside. If it was compressed, the centered version
        # is already inside by construction.
        fitted_min = fitted_root_xy.min(axis=0)
        fitted_max = fitted_root_xy.max(axis=0)
        shift = np.zeros(2, dtype=np.float64)
        if fitted_min[0] < allowed_min[0]:
            shift[0] += allowed_min[0] - fitted_min[0]
        if fitted_max[0] > allowed_max[0]:
            shift[0] += allowed_max[0] - fitted_max[0]
        if fitted_min[1] < allowed_min[1]:
            shift[1] += allowed_min[1] - fitted_min[1]
        if fitted_max[1] > allowed_max[1]:
            shift[1] += allowed_max[1] - fitted_max[1]
        fitted_root_xy += shift

        motion = motion.copy()
        motion[:, :, :2] = fitted_root_xy[:, None, :] + local_xy
        return motion

    def mesh_at_time_step(self, frame, chirp=0):
        vertices_curr = self.vertices[frame]
        return vertices_curr, None

    def get_joints_at_time_step(self, frame):
        """
        Get joint positions at a specific time step.

        Args:
            frame: Frame index

        Returns:
            joints: (num_joints, 3) array of joint positions
        """
        if not hasattr(self, 'interpolated_motion'):
            raise ValueError("Motion data not loaded. Call load_motion_sequence first.")

        # Return interpolated joint positions (already transformed)
        return self.interpolated_motion[frame].copy()

    def get_bone_point_counts(self):
        """
        Get the number of interpolated points for each bone.

        Returns:
            point_counts: List of integers, one for each bone
        """
        if not hasattr(self, 'interpolated_motion'):
            raise ValueError("Motion data not loaded. Call load_motion_sequence first.")

        point_counts = []
        for idx1, idx2 in self.bones:
            # Calculate bone length for each frame using interpolated motion
            point1 = self.interpolated_motion[:, idx1]
            point2 = self.interpolated_motion[:, idx2]
            bone_vectors = point2 - point1
            bone_lengths = np.linalg.norm(bone_vectors, axis=1)

            # Calculate number of points based on bone length (points per unit length)
            avg_bone_length = np.mean(bone_lengths)
            points_per_unit = self.num_insect
            num_points = max(2, int(avg_bone_length * points_per_unit))

            point_counts.append(num_points)

        return point_counts

    def load_motion_sequence(self, motion_path):
        motion = np.load(motion_path)

        # Verify motion data shape
        if len(motion.shape) == 3:
            input_joint_count = motion.shape[1]
            self._configure_skeleton_for_joint_count(input_joint_count)
            if self.joint_schema == "hunyuan_motion_body22" and input_joint_count > self.num_joints:
                motion = motion[:, :self.num_joints, :]

        # Store original motion data for joint extraction
        self.motion = motion.copy()

        # Apply transformations to motion data BEFORE interpolation
        # 1. Coordinate transformation
        motion = motion @ np.array([[0, -1, 0], [0, 0, 1], [-1, 0, 0]])

        # 2. Unit conversion (scale human body size)
        motion = motion * self.UNIT_TO_METER

        # 3. XY scaling (scale movement range without affecting body size)
        if self.scale != 1.0:
            motion[:, :, :2] = motion[:, :, :2] * self.scale

        # 3b. Optional room-window fit for the global root trajectory.
        motion = self._apply_adaptive_xy_fit(motion)

        # 4. Move to floor (using initial root position)
        init_root = np.min(motion[0, :, 2])
        motion[:, :, 2] -= init_root

        # Store transformed motion for joint extraction
        self.transformed_motion = motion.copy()

        # perform interpolation AFTER all transformations
        motion = motion.reshape(-1, self.num_joints * 3)
        num_frames = int(len(motion) * 50 / self.fps) # ? fps -> 50 fps
        interpolater = interp1d(np.arange(len(motion)), motion, axis=0)
        motion = interpolater(np.linspace(0, len(motion) - 1, num_frames))
        motion = gaussian_filter1d(motion, sigma=2., axis=0)
        motion = motion.reshape(-1, self.num_joints, 3)

        # Store interpolated motion data for joint extraction
        self.interpolated_motion = motion.copy()

        bodypoints = []
        # insect body bones
        for idx1, idx2 in self.bones:
            point1 = motion[:, idx1][:, None]
            point2 = motion[:, idx2][:, None]

            # Calculate bone length for each frame
            bone_vectors = point2 - point1
            bone_lengths = np.linalg.norm(bone_vectors, axis=2)  # Shape: (num_frames, 1)

            # Calculate number of points based on bone length (points per unit length)
            # Use average bone length across all frames to determine point count
            avg_bone_length = np.mean(bone_lengths)
            points_per_unit = self.num_insect  # This becomes points per unit length instead of total points
            num_points = max(2, int(avg_bone_length * points_per_unit))

            # Generate interpolated points
            bonepoints = np.linspace(0, 1, num_points).reshape(-1, 1) * (point2 - point1) + point1
            # bonepoints = bonepoints[:, 1:-1]  # Exclude endpoints
            bodypoints.append(bonepoints)

        bodypoints = np.concatenate(bodypoints, axis=1)

        self.vertices = bodypoints
        self.num_frame = bodypoints.shape[0]
        # print(f"Loaded motion sequence: {self.num_frame} frames, {len(self.bones)} bones, {bodypoints.shape[1]} body points")
