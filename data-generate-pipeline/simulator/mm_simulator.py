import os
import numpy as np
import torch

# Constants
PI = np.pi
SPEED_OF_LIGHT = 299792458.0

# Data types
rdtype = np.float32
cdtype = np.complex64


class mmSimulator:
    def __init__(self, radar_cfg, radar, mesh_manager=None, num_rx=1):
        # Set CUDA device if not already specified (defer to instance creation)
        if "CUDA_VISIBLE_DEVICES" not in os.environ:
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"

        self.radar = radar
        self.frequency = radar_cfg.start_freq
        self.mesh_manager = mesh_manager
        self.mesh_sample_rate = 10

        # Configurable number of antennas
        self.num_tx = 1
        self.num_rx = num_rx  # Support multiple RX antennas

        self.f_num = 20908
        self.v_num = 10475

        # Mesh data
        self.vertex_positions = None
        self.faces = None

        # Reflection environment configuration
        # Defaults can be overridden in init(mesh_cfg=...)
        self.use_random_reflections = True
        self.reflection_count = 6
        # Axis-aligned bounding box for random points (meters)
        self.reflection_bounds = {
            'x': (-2.0, 2.0),
            'y': (-2.0, 2.0),
            'z': (0.0, 3.0),  # z>=0 (above ground)
        }
        # Coefficient range (dimensionless, 0~1)
        self.reflection_coeff_range = (0.2, 0.5)

        # Placeholders until init() populates actual arrays
        self.reflection_points = torch.empty(0, 3, dtype=torch.float32)
        self.reflection_coefficients = torch.empty(0, dtype=torch.float32)
        self.num_reflection_points = 0

    def _generate_random_reflections(self, count: int) -> None:
        """Generate sparse random reflection points and coefficients within bounds.
        This replaces fixed reflection configuration when use_random_reflections is True.
        """
        x_min, x_max = self.reflection_bounds['x']
        y_min, y_max = self.reflection_bounds['y']
        z_min, z_max = self.reflection_bounds['z']

        # Uniform random sampling within the AABB
        xs = np.random.uniform(x_min, x_max, size=(count, 1)).astype(np.float32)
        ys = np.random.uniform(y_min, y_max, size=(count, 1)).astype(np.float32)
        zs = np.random.uniform(z_min, z_max, size=(count, 1)).astype(np.float32)
        pts = np.concatenate([xs, ys, zs], axis=1)

        # Optional: avoid points too close to the radar height to reduce degenerate paths
        # (light heuristic; can be tuned/removed)
        # Keep as-is to stay simple/fast.

        c_min, c_max = self.reflection_coeff_range
        coeffs = np.random.uniform(c_min, c_max, size=(count,)).astype(np.float32)

        self.reflection_points = torch.tensor(pts, dtype=torch.float32)
        self.reflection_coefficients = torch.tensor(coeffs, dtype=torch.float32)
        self.num_reflection_points = int(self.reflection_points.shape[0])

    def init(self, mesh_cfg=None):
        if mesh_cfg is None:
            self.camera_position = np.array([-3.6, -0.6, 1.48], dtype=rdtype)
            self.camera_orientation = np.array([0.0, 0.151, 0.0], dtype=rdtype)
            self.tx_position = np.array([-3.6, -0.6, 1.28], dtype=rdtype)
            self.radar_orientation = np.array([0, 0, 0], dtype=rdtype)
        else:
            self.camera_position = np.array(mesh_cfg['camera_position'], dtype=rdtype)
            self.camera_orientation = np.array(mesh_cfg['camera_orientation'], dtype=rdtype) / 180 * PI
            self.tx_position = self.camera_position #+ np.array([-0.06, .0, -0.1], dtype=rdtype)
            self.radar_orientation = self.camera_orientation

            # Optional overrides for reflection generation
            # Toggle: use_random_reflections (default True)
            if 'use_random_reflections' in mesh_cfg:
                self.use_random_reflections = bool(mesh_cfg['use_random_reflections'])
            if 'reflection_count' in mesh_cfg:
                self.reflection_count = int(mesh_cfg['reflection_count'])
            if 'reflection_bounds' in mesh_cfg:
                rb = mesh_cfg['reflection_bounds']
                # Only overwrite keys provided
                for axis in ['x', 'y', 'z']:
                    if axis in rb and isinstance(rb[axis], (list, tuple)) and len(rb[axis]) == 2:
                        self.reflection_bounds[axis] = (float(rb[axis][0]), float(rb[axis][1]))
            if 'reflection_coeff_range' in mesh_cfg and isinstance(mesh_cfg['reflection_coeff_range'], (list, tuple)) and len(mesh_cfg['reflection_coeff_range']) == 2:
                self.reflection_coeff_range = (float(mesh_cfg['reflection_coeff_range'][0]), float(mesh_cfg['reflection_coeff_range'][1]))

        d = (SPEED_OF_LIGHT / self.frequency) / 2     # wave_length / 2 (half wavelength)

        # Create RX antenna array based on num_rx
        if self.num_rx == 1:
            # Single antenna at origin
            rx_ref_array = np.array([[0, 0, 0]], dtype=rdtype)
        else:
            # Multiple antennas in horizontal line (Y-axis), spaced by d (half wavelength)
            # Center the array around origin
            rx_ref_array = np.zeros((self.num_rx, 3), dtype=rdtype)
            for i in range(self.num_rx):
                # Horizontal spacing along Y-axis
                rx_ref_array[i] = [0, (i - (self.num_rx - 1) / 2) * d, 0]

        # Apply rotation if needed
        if np.any(self.radar_orientation != 0):
            Rz = np.array([
                [np.cos(self.radar_orientation[0]), -np.sin(self.radar_orientation[0]), 0],
                [np.sin(self.radar_orientation[0]), np.cos(self.radar_orientation[0]), 0],
                [0, 0, 1]], dtype=rdtype)
            Ry = np.array([
                [np.cos(self.radar_orientation[1]), 0, np.sin(self.radar_orientation[1])],
                [0, 1, 0],
                [-np.sin(self.radar_orientation[1]), 0, np.cos(self.radar_orientation[1])]], dtype=rdtype)
            rx_ref_array = rx_ref_array @ Ry.T @ Rz.T

        # Final RX positions relative to TX
        self.rx_positions = self.tx_position[None, :] + rx_ref_array

        # Populate reflection config: always random
        self._generate_random_reflections(self.reflection_count)

    def set_mesh_data(self, vertex_positions, faces):
        """Set mesh vertex positions and faces"""
        self.vertex_positions = torch.tensor(vertex_positions, dtype=torch.float32)
        self.faces = torch.tensor(faces, dtype=torch.long)

    def compute_paths_with_reflection(self, consider_energy: bool = True):
        """
        Compute direct and reflected paths using fixed reflection points.
        Returns:
            a:   [num_points, 1 + num_reflection_points, num_rx]
            tau: [num_points, 1 + num_reflection_points, num_rx]
            paths: None
            all_path_index: [num_points]
        """
        points = self.vertex_positions.reshape(self.v_num, 3)
        points = points[::self.mesh_sample_rate]  # [P,3]

        tx_pos = torch.tensor(self.tx_position, dtype=torch.float32)           # [3]
        rx_pos = torch.tensor(self.rx_positions, dtype=torch.float32)          # [RX,3]

        num_points = points.shape[0]
        num_rx = rx_pos.shape[0]
        num_paths = 1 + self.num_reflection_points

        tau = torch.zeros((num_points, num_paths, num_rx), dtype=torch.float32)
        a = torch.zeros((num_points, num_paths, num_rx), dtype=torch.float32)

        # Direct path: TX -> P -> RX
        d_tx_to_p = torch.norm(points - tx_pos, dim=-1).unsqueeze(-1)                 # [P,1]
        d_p_to_rx = torch.norm(points.unsqueeze(1) - rx_pos.unsqueeze(0), dim=-1)     # [P,RX]
        d_total_direct = d_tx_to_p + d_p_to_rx                                        # [P,RX]
        tau[:, 0, :] = d_total_direct / SPEED_OF_LIGHT
        if consider_energy:
            a[:, 0, :] = 1.0 / (d_total_direct + 1e-9)
        else:
            a[:, 0, :] = 1.0

        # Reflected paths: TX -> P -> R -> RX
        refl_pts = self.reflection_points                                            # [R,3]
        coeffs = self.reflection_coefficients.view(-1, 1)                            # [R,1]
        d_p_to_r = torch.cdist(points, refl_pts, p=2).unsqueeze(-1)                  # [P,R,1]
        d_r_to_rx = torch.cdist(refl_pts, rx_pos, p=2).unsqueeze(0)                  # [1,R,RX]
        d_total_reflect = d_tx_to_p.unsqueeze(-1) + d_p_to_r + d_r_to_rx             # [P,R,RX]
        tau[:, 1:, :] = d_total_reflect / SPEED_OF_LIGHT
        if consider_energy:
            a[:, 1:, :] = coeffs.view(1, -1, 1) / (d_total_reflect + 1e-9)
        else:
            a[:, 1:, :] = 1.0

        all_path_index = torch.arange(num_points, dtype=torch.long)
        return a, tau, None, all_path_index

    def compute_visible_triangles(self, tx_pos, hit_points, normals):
        """
        Compute which triangles are visible from transmitter
        Returns mask of visible triangles
        """
        # Calculate view direction from hit points to transmitter
        view_direction = tx_pos - hit_points

        # Calculate dot product between normals and view direction
        dot_product = torch.sum(normals * view_direction, dim=-1)

        # Triangle is visible if dot_product > 0 (facing towards transmitter)
        visible_mask = dot_product > 0

        return visible_mask

    def compute_paths_pytorch(self, consider_energy=True):
        """
        Compute paths using PyTorch ray tracing with visibility check
        a: [num_visible_triangles, num_rx]
        tau: [num_visible_triangles, num_rx]
        paths: None (for compatibility with ref.py)
        all_path_index: [num_visible_triangles] - indices of visible triangles
        """
        assert self.vertex_positions is not None, "Mesh data is not initialized."

        # Get mesh tracking points (triangle centers)
        hit_points = self.generate_mesh_tracking_points()

        # Convert to torch tensors
        tx_pos = torch.tensor(self.tx_position, dtype=torch.float32)
        rx_pos = torch.tensor(self.rx_positions, dtype=torch.float32)
        hit_points = torch.tensor(hit_points, dtype=torch.float32)

        # Calculate normal vectors for each triangle
        normals = self.compute_mesh_triangle_normals()

        # Compute visible triangles
        visible_mask = self.compute_visible_triangles(tx_pos, hit_points, normals)

        # Get indices of visible triangles
        all_path_index = torch.where(visible_mask)[0]

        # Filter to only visible triangles
        visible_hit_points = hit_points[visible_mask]
        visible_normals = normals[visible_mask]

        # Calculate distances from tx to visible hit points
        tx_distances = torch.norm(visible_hit_points - tx_pos, dim=1)

        # Calculate distances from visible hit points to each rx
        rx_distances = torch.norm(visible_hit_points.unsqueeze(1) - rx_pos.unsqueeze(0), dim=2)

        # Total distances
        total_distances = tx_distances.unsqueeze(1) + rx_distances

        # Convert to time delays - shape: [num_visible_triangles, num_rx]
        tau = total_distances / SPEED_OF_LIGHT

        if consider_energy:
            # Calculate amplitudes with energy consideration
            # Get triangle areas for visible triangles only
            areas = self.compute_mesh_triangle_area()
            visible_areas = areas[visible_mask]

            # Calculate incident vectors from tx to visible hit points
            incident_vectors = visible_hit_points - tx_pos
            incident_vectors = incident_vectors / torch.norm(incident_vectors, dim=1, keepdim=True)

            # Calculate cos_theta as in reference implementation
            cos_theta = torch.sum(incident_vectors * visible_normals, dim=-1) / torch.norm(visible_hit_points - tx_pos, dim=1)
            cos_theta = torch.abs(cos_theta)  # Take absolute value as in reference

            # Calculate amplitude scaling
            amplitude_scale = cos_theta * visible_areas
            amplitude_scale = amplitude_scale.unsqueeze(1)  # Shape: [num_visible_triangles, 1]
            amplitude_scale = amplitude_scale.expand(-1, self.num_rx)  # Shape: [num_visible_triangles, num_rx]

            # Apply energy loss according to distance
            amplitude_scale = amplitude_scale / total_distances

            # Create amplitude tensor - same shape as tau
            a = amplitude_scale
        else:
            # Calculate amplitude without energy loss - same shape as tau
            a = torch.ones_like(tau)

        return a, tau, None, all_path_index

    def compute_paths_using_points(self):
        """
        Compute path delay using distances. No physics considered.
        a: [num_points, num_rx]
        tau: [num_points, num_rx]
        paths: None (for compatibility with ref.py)
        all_path_index: [num_points] - indices of all sampled points
        """
        points = self.vertex_positions.reshape(self.v_num, 3)
        points = points[::self.mesh_sample_rate]

        tx_pos = torch.tensor(self.tx_position, dtype=torch.float32)
        rx_pos = torch.tensor(self.rx_positions, dtype=torch.float32)

        distance1 = torch.norm(points - tx_pos, dim=-1).unsqueeze(1)  # [num_points, 1]
        distance2 = torch.norm(points.unsqueeze(1) - rx_pos.unsqueeze(0), dim=-1)  # [num_points, num_rx]
        distances = distance1 + distance2  # [num_points, num_rx]
        tau = distances / SPEED_OF_LIGHT
        a = torch.ones_like(tau)

        # Create indices for all sampled points
        all_path_index = torch.arange(len(points), dtype=torch.long)

        return a, tau, None, all_path_index

    def compute_paths_triangle_centers_with_reflection(self, consider_energy: bool = True):
        """
        Compute paths using triangle centers + random environment reflection points.
        Combines the precision of triangle centers with environmental reflections.

        Returns:
            a:   [num_triangles, 1 + num_reflection_points, num_rx]
            tau: [num_triangles, 1 + num_reflection_points, num_rx]
            paths: None
            all_path_index: [num_triangles] - indices of triangle centers
        """
        # Get triangle centers as hit points
        hit_points = self.generate_mesh_tracking_points()
        hit_points = torch.tensor(hit_points, dtype=torch.float32)  # [num_triangles, 3]

        tx_pos = torch.tensor(self.tx_position, dtype=torch.float32)           # [3]
        rx_pos = torch.tensor(self.rx_positions, dtype=torch.float32)          # [RX,3]

        num_triangles = hit_points.shape[0]
        num_rx = rx_pos.shape[0]
        num_paths = 1 + self.num_reflection_points

        tau = torch.zeros((num_triangles, num_paths, num_rx), dtype=torch.float32)
        a = torch.zeros((num_triangles, num_paths, num_rx), dtype=torch.float32)

        # Direct path: TX -> Triangle Center -> RX
        d_tx_to_center = torch.norm(hit_points - tx_pos, dim=-1).unsqueeze(-1)                 # [T,1]
        d_center_to_rx = torch.norm(hit_points.unsqueeze(1) - rx_pos.unsqueeze(0), dim=-1)     # [T,RX]
        d_total_direct = d_tx_to_center + d_center_to_rx                                        # [T,RX]
        tau[:, 0, :] = d_total_direct / SPEED_OF_LIGHT

        if consider_energy:
            # Get triangle areas for amplitude scaling
            areas = self.compute_mesh_triangle_area()
            # Calculate incident angle effect (simplified)
            incident_vectors = hit_points - tx_pos
            incident_vectors = incident_vectors / torch.norm(incident_vectors, dim=1, keepdim=True)
            # Use triangle normals for more accurate incident angle
            normals = self.compute_mesh_triangle_normals()
            cos_theta = torch.abs(torch.sum(incident_vectors * normals, dim=-1))  # [T]
            # Amplitude includes area, incident angle, and distance attenuation
            amplitude_scale = cos_theta * areas  # [T]
            a[:, 0, :] = amplitude_scale.unsqueeze(-1) / (d_total_direct + 1e-9)
        else:
            a[:, 0, :] = 1.0 / (d_total_direct + 1e-9)

        # Reflected paths: TX -> Triangle Center -> Random Reflection Point -> RX
        refl_pts = self.reflection_points                                            # [R,3]
        coeffs = self.reflection_coefficients.view(-1, 1)                            # [R,1]
        d_center_to_r = torch.cdist(hit_points, refl_pts, p=2).unsqueeze(-1)         # [T,R,1]
        d_r_to_rx = torch.cdist(refl_pts, rx_pos, p=2).unsqueeze(0)                 # [1,R,RX]
        d_total_reflect = d_tx_to_center.unsqueeze(-1) + d_center_to_r + d_r_to_rx   # [T,R,RX]
        tau[:, 1:, :] = d_total_reflect / SPEED_OF_LIGHT

        if consider_energy:
            # For reflected paths, apply reflection coefficient and distance attenuation
            # Include triangle area and incident angle effects
            amplitude_scale = cos_theta * areas  # [T]
            a[:, 1:, :] = coeffs.view(1, -1, 1) * amplitude_scale.unsqueeze(-1).unsqueeze(-1) / (d_total_reflect + 1e-9)
        else:
            a[:, 1:, :] = coeffs.view(1, -1, 1) / (d_total_reflect + 1e-9)

        all_path_index = torch.arange(num_triangles, dtype=torch.long)
        return a, tau, None, all_path_index

    def generate_mesh_tracking_points(self):
        """Generate tracking points from mesh triangles"""
        if self.vertex_positions is None or self.faces is None:
            # Fallback to using vertex positions if no mesh data
            points = self.vertex_positions.reshape(self.v_num, 3)
            return points[::self.mesh_sample_rate].numpy()

        # Calculate triangle centers
        triangle_vpositions = self.vertex_positions[self.faces]
        triangle_cpositions = torch.mean(triangle_vpositions, dim=1)

        return triangle_cpositions[::self.mesh_sample_rate].numpy()   # downsample

    def compute_mesh_triangle_area(self):
        """Compute area of each triangle in the mesh"""
        if self.vertex_positions is None or self.faces is None:
            return torch.ones(self.f_num // self.mesh_sample_rate, dtype=torch.float32)

        # Calculate triangle areas
        triangle_vpositions = self.vertex_positions[self.faces]
        v1 = triangle_vpositions[:, 1] - triangle_vpositions[:, 0]
        v2 = triangle_vpositions[:, 2] - triangle_vpositions[:, 0]
        cross = torch.cross(v1, v2, dim=1)
        area = 0.5 * torch.norm(cross, dim=1)

        return area[::self.mesh_sample_rate]   # downsample

    def compute_mesh_triangle_normals(self):
        """Compute normal vectors for each triangle in the mesh"""
        if self.vertex_positions is None or self.faces is None:
            return torch.zeros(self.f_num // self.mesh_sample_rate, 3, dtype=torch.float32)

        # Calculate triangle normals
        triangle_vpositions = self.vertex_positions[self.faces]
        v1 = triangle_vpositions[:, 1] - triangle_vpositions[:, 0]
        v2 = triangle_vpositions[:, 2] - triangle_vpositions[:, 0]
        cross = torch.cross(v1, v2, dim=1)
        denom = torch.clamp(torch.norm(cross, dim=1, keepdim=True), min=1e-8)
        normals = cross / denom

        return normals[::self.mesh_sample_rate]   # downsample

    def mesh_at_time_step(self, frame, chirp=0):
        assert 0 <= chirp < 64
        vertex_positions, _ = self.mesh_manager.mesh_at_time_step(frame, chirp)
        self.vertex_positions = torch.tensor(vertex_positions, dtype=torch.float32).reshape(self.v_num, 3)

    def update_mesh_faces(self, faces):
        if torch.is_tensor(faces):
            self.faces = faces.detach().cpu().long()
        else:
            self.faces = torch.tensor(faces, dtype=torch.long)
