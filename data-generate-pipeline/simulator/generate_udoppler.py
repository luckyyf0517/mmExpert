import sys

import torch
import numpy as np


from utils.logger import log_message


def print_radar_spectrum_info(radar_cfg, num_antennas=8, num_range_bins=128,
                              num_doppler_bins=128, num_angle_bins=128):
    """
    Print radar spectrum parameters (range, doppler, angle).
    Call once during initialization to show measurement specifications.

    Args:
        radar_cfg: Radar configuration module
        num_antennas: Number of receiving antennas
        num_range_bins: Number of range bins to extract
        num_doppler_bins: Number of doppler bins to extract
        num_angle_bins: Number of angle bins for DOA
    """
    # Read parameters from radar config
    light_speed = radar_cfg.light_speed
    start_freq = radar_cfg.start_freq
    freq_slope = radar_cfg.freq_slope
    adc_sample_rate = radar_cfg.adc_sample_rate
    num_samples = radar_cfg.num_adc_samples
    num_chirps = radar_cfg.num_chirps
    ramp_end_time = radar_cfg.ramp_end_time

    # Calculate resolutions
    bandwidth = freq_slope * num_samples / adc_sample_rate
    range_resolution = light_speed / (2 * bandwidth)
    max_range_used = num_range_bins * range_resolution

    chirp_last_period = ramp_end_time * num_chirps * 2
    wave_length = light_speed / start_freq
    doppler_resolution = wave_length / (2 * chirp_last_period)
    max_doppler = num_doppler_bins / 2 * doppler_resolution

    # Angle resolution (for antenna array)
    if num_antennas > 1:
        antenna_spacing = wave_length / 2  # Half wavelength
        angle_resolution = np.rad2deg(wave_length / (num_antennas * antenna_spacing))
        max_angle = np.rad2deg(np.arcsin(wave_length / (2 * antenna_spacing)))
    else:
        antenna_spacing = 0.0
        angle_resolution = 0
        max_angle = 0

    print()
    print("=" * 60)
    log_message("RADAR", "Radar Spectrum Parameters", color="cyan", attrs=("bold",))
    print("=" * 60)
    print()
    log_message("RADAR", "[Range Spectrum]", color="yellow", attrs=("bold",))
    log_message("RADAR", f"Total ADC samples: {num_samples}", color="green")
    log_message("RADAR", f"Range bins: {num_range_bins}", color="green")
    log_message("RADAR", f"Range resolution: {range_resolution:.4f} m", color="green")
    log_message("RADAR", f"Max range: {max_range_used:.2f} m", color="green")
    log_message("RADAR", f"Range coverage: [0, {max_range_used:.2f}] m", color="green")
    print()
    log_message("RADAR", "[Doppler Spectrum]", color="yellow", attrs=("bold",))
    log_message("RADAR", f"Doppler bins: {num_doppler_bins}", color="green")
    log_message("RADAR", f"Doppler resolution: {doppler_resolution:.4f} m/s", color="green")
    log_message("RADAR", f"Max doppler: ±{max_doppler:.2f} m/s", color="green")
    log_message("RADAR", f"Velocity coverage: [{-max_doppler:.2f}, {max_doppler:.2f}] m/s", color="green")
    print()
    log_message("RADAR", "[Azimuth Spectrum (DOA - MVDR Capon)]", color="yellow", attrs=("bold",))
    log_message("RADAR", f"Number of antennas: {num_antennas}", color="green")
    if num_antennas > 1:
        log_message("RADAR", f"Antenna spacing: λ/2 = {antenna_spacing*1000:.4f} mm", color="green")
    else:
        log_message("RADAR", "Antenna spacing: N/A (single antenna)", color="green")
    log_message("RADAR", f"Angle bins: {num_angle_bins}", color="green")
    log_message("RADAR", "DOA method: MVDR Capon Beamforming", color="green")
    if num_antennas > 1:
        log_message("RADAR", f"Angle resolution: {angle_resolution:.2f}° (Rayleigh)", color="green")
        log_message("RADAR", "MVDR resolution: Higher than FFT-based methods", color="green")
        log_message("RADAR", f"Max angle: ±{max_angle:.2f}°", color="green")
        log_message("RADAR", f"Angular coverage: [{-max_angle:.2f}°, {max_angle:.2f}°]", color="green")
    else:
        log_message("RADAR", "Single antenna - no DOA estimation", color="green")
    print()
    print("=" * 60)
    print()
    sys.stdout.flush()
    sys.stderr.flush()


def get_raw_radar_frame(path_data, next_data, fps=30, base_time=0.0, device='cuda'):
    start_freq = 77e9
    freq_slope = 70e12
    adc_sample_rate = 5.209e6
    idle_time = 20.985e-6
    adc_start_time = idle_time + 7e-6
    ts = adc_start_time + torch.arange(256, device=device) / adc_sample_rate
    fs = ts * freq_slope
    frame_period = 2e-2

    a_base, tau_base, idx_base = path_data.values()
    a_next, tau_next, idx_next = next_data.values()

    if idx_base is not None and idx_next is not None:
        intersect = np.intersect1d(idx_base, idx_next)
        mask_base = np.isin(idx_base, intersect)
        mask_next = np.isin(idx_next, intersect)
        a_base = a_base[mask_base]
        tau_base = tau_base[mask_base]
        a_next = a_next[mask_next]
        tau_next = tau_next[mask_next]
    else:
        intersect = None
    assert a_base.shape == a_next.shape
    a_velocity = (a_next - a_base) / (1 / fps)
    tau_velocity = (tau_next - tau_base) / (1 / fps)

    a_base = torch.tensor(a_base, dtype=torch.float32, device=device)
    tau_base = torch.tensor(tau_base, dtype=torch.float32, device=device)
    a_velocity = torch.tensor(a_velocity, dtype=torch.float32, device=device)
    tau_velocity = torch.tensor(tau_velocity, dtype=torch.float32, device=device)

    # fullfill the batch using chirps
    assert 0.0 <= base_time < 1 / fps
    time_steps = (torch.arange(128) / 128 * frame_period + base_time).to(device)
    # Support both 2D [P, RX] and 3D [P, PATH, RX]
    if a_base.ndim == 2:
        a_chirp = a_base + a_velocity * time_steps.reshape((128, 1, 1))
        tau_chirp = tau_base + tau_velocity * time_steps.reshape((128, 1, 1))
        frequencies = fs.reshape(1, 1, 1, -1)
        a_chirp = a_chirp.unsqueeze(-1)
        tau_chirp = tau_chirp.unsqueeze(-1)
        ft = 2 * np.pi * frequencies * tau_chirp
        phase = 2 * np.pi * start_freq * tau_chirp
        radar_frame = (a_chirp * torch.exp(1j * (ft + phase)))
        return radar_frame, intersect
    elif a_base.ndim == 3:
        a_chirp = a_base + a_velocity * time_steps.reshape((128, 1, 1, 1))
        tau_chirp = tau_base + tau_velocity * time_steps.reshape((128, 1, 1, 1))
        frequencies = fs.reshape(1, 1, 1, 1, -1)
        ft = 2 * np.pi * frequencies * tau_chirp.unsqueeze(-1)
        phase = 2 * np.pi * start_freq * tau_chirp.unsqueeze(-1)
        radar_frame_paths = (a_chirp.unsqueeze(-1) * torch.exp(1j * (ft + phase)))
        radar_frame = radar_frame_paths.sum(2)  # sum over PATH
        return radar_frame, intersect
    else:
        raise ValueError('Unsupported shape for a/tau: expected 2D or 3D')


def get_udoppler(radar_frame):
    """
    Legacy function for backward compatibility.
    Returns only udoppler spectrum (summed over range and antenna).
    """
    assert len(radar_frame.shape) == 4
    radar_frame = radar_frame * torch.hann_window(256).reshape(1, 1, 1, -1).cuda()
    range_fft = torch.fft.fft(radar_frame, axis=3)
    range_fft = range_fft * torch.hann_window(128).reshape(1, -1, 1, 1).cuda()
    doppler_fft = torch.fft.fft(range_fft, axis=1)
    doppler_fft = torch.fft.fftshift(doppler_fft, axis=1)
    return torch.abs(doppler_fft).sum(-1).sum(-1)


def get_udoppler_real_style(
    radar_frame,
    *,
    range_start=16,
    range_end=128,
    num_doppler_bins=128,
    target_range_bins=8,
    interpolate_center=True,
):
    """Real-style uDoppler projection for raw radar cubes.

    The projection applies an ADC Hann window, a fixed range slice,
    chirp-mean MTI, range and Doppler FFTs, per-frame target range-window
    selection, optional center-bin interpolation, and RX/range summation.
    """
    if len(radar_frame.shape) != 4:
        raise ValueError(f"Expected radar_frame shape (T, chirps, RX, ADC), got {tuple(radar_frame.shape)}")

    device = radar_frame.device
    num_adc = radar_frame.shape[-1]
    num_chirps = radar_frame.shape[1]
    range_start = int(range_start)
    range_end = int(range_end)
    num_doppler_bins = int(num_doppler_bins)
    target_range_bins = int(target_range_bins)
    if range_start < 0 or range_end <= range_start or range_end > num_adc:
        raise ValueError(f"Invalid range slice [{range_start},{range_end}) for ADC={num_adc}")
    if target_range_bins <= 0 or target_range_bins % 2 != 0:
        raise ValueError(f"target_range_bins must be a positive even integer, got {target_range_bins}")

    adc_win = torch.hann_window(num_adc, device=device).reshape(1, 1, 1, -1)
    data = radar_frame * adc_win
    data = data[..., range_start:range_end]
    data = data - data.mean(dim=1, keepdim=True)
    data = torch.fft.fft(data, dim=3)
    chirp_win = torch.hann_window(num_chirps, device=device).reshape(1, -1, 1, 1)
    data = data * chirp_win
    data = torch.fft.fft(data, n=num_doppler_bins, dim=1)
    data = torch.fft.fftshift(data, dim=1)
    power = torch.abs(data)

    doppler_spec = power.sum(dim=(2, 3))
    doppler_center = doppler_spec.argmax(dim=1)

    half = target_range_bins // 2
    selected = []
    max_center = power.shape[3] - half
    min_center = target_range_bins
    for i in range(power.shape[0]):
        rangespec = power[i, int(doppler_center[i].item())].sum(dim=0)
        range_center = int(rangespec.argmax().item())
        range_center = max(min_center, min(range_center, max_center))
        selected.append(power[i, :, :, range_center - half:range_center + half])

    target = torch.stack(selected, dim=0)
    center_bin = num_doppler_bins // 2
    if interpolate_center and 0 < center_bin < num_doppler_bins - 1:
        target[:, center_bin] = (target[:, center_bin - 1] + target[:, center_bin + 1]) / 2.0

    return target.sum(dim=-1).sum(dim=-1)


def get_multi_view_data(radar_frame, num_range_bins=256, num_doppler_bins=128, num_angle_bins=128):
    """
    Generate multi-view radar data: Range-Time, Doppler-Time, and Azimuth-Time.

    Args:
        radar_frame: Radar frame data, shape (T, num_chirps, num_antennas, num_samples)
                    T: number of frames
                    num_chirps: typically 128
                    num_antennas: number of receiving antennas (e.g., 8)
                    num_samples: number of ADC samples (e.g., 256)
        num_range_bins: Number of range bins to extract (default: 256, full range)
        num_doppler_bins: Number of doppler bins to extract (default: 128)
        num_angle_bins: Number of angle bins for DOA estimation (default: 128)

    Returns:
        data_dict: Dictionary containing:
                  'range_time': Shape (256, T) - Range-Time spectrum
                  'doppler_time': Shape (128, T) - Doppler-Time spectrum (Udoppler)
                  'azimuth_time': Shape (128, T) - Azimuth-Time spectrum (DOA)
    """
    device = radar_frame.device
    T = radar_frame.shape[0]
    num_chirps = radar_frame.shape[1]
    num_antennas = radar_frame.shape[2]
    num_samples = radar_frame.shape[3]

    # Apply window and perform Range FFT
    radar_frame = radar_frame * torch.hann_window(num_samples, device=device).reshape(1, 1, 1, -1)
    range_fft = torch.fft.fft(radar_frame, n=num_samples, dim=3)

    # Extract first num_range_bins (use full 256 bins for range)
    range_fft_full = range_fft[:, :, :, :num_range_bins]  # (T, num_chirps, num_antennas, num_range_bins)

    # 1. Range-Time: Average over chirps and antennas, take magnitude
    range_time = torch.abs(range_fft_full).mean(dim=1).mean(dim=1)  # (T, num_range_bins)
    range_time = range_time.T  # (num_range_bins, T) -> (256, T)

    # 2. Doppler-Time (Udoppler): Apply Doppler FFT
    range_fft_windowed = range_fft_full * torch.hann_window(num_chirps, device=device).reshape(1, -1, 1, 1)
    doppler_fft = torch.fft.fft(range_fft_windowed, n=num_doppler_bins, dim=1)
    doppler_fft = torch.fft.fftshift(doppler_fft, dim=1)  # (T, num_doppler_bins, num_antennas, num_range_bins)

    # Average over antennas and range bins, take magnitude
    doppler_time = torch.abs(doppler_fft).mean(dim=2).mean(dim=2)  # (T, num_doppler_bins)
    doppler_time = doppler_time.T  # (num_doppler_bins, T) -> (128, T)

    # 3. Azimuth-Time (DOA): Use MVDR Capon Beamforming
    # Average over chirps first to get coherent signal
    range_fft_avg_chirp = range_fft_full.mean(dim=1)  # (T, num_antennas, num_range_bins)

    # Find range bins with significant energy for focused MVDR processing
    range_profile = torch.abs(range_fft_full).mean(dim=1).mean(dim=1)  # (T, num_range_bins)

    # MVDR processing for each time frame
    azimuth_time_list = []
    for t in range(T):
        # Get signal data for this time frame: (num_antennas, num_range_bins)
        signal_data = range_fft_avg_chirp[t]

        # Find active range bins (>10% of max energy)
        energy_dist = range_profile[t]
        threshold = energy_dist.max() * 0.1
        active_mask = energy_dist > threshold

        if active_mask.sum() > 0:
            # Use only active range bins for covariance estimation
            active_signal = signal_data[:, active_mask]  # (num_antennas, num_active_bins)

            # Compute spatial covariance matrix: R = (1/M) * X * X^H
            # X: (num_antennas, num_active_bins), R: (num_antennas, num_antennas)
            R = torch.matmul(active_signal, active_signal.conj().T) / active_signal.shape[1]

            # Diagonal loading for numerical stability
            # Use adaptive loading based on trace
            loading = 0.01 * torch.trace(R).real / num_antennas
            R = R + loading * torch.eye(num_antennas, device=device, dtype=R.dtype)

            # Compute steering vectors for all angles
            # Angle range: [-90°, 90°]
            angles = torch.linspace(-np.pi/2, np.pi/2, num_angle_bins, device=device)

            # Steering vector: a(θ) = [1, e^{-jπsinθ}, e^{-j2πsinθ}, ..., e^{-j(M-1)πsinθ}]
            # For half-wavelength spacing, the phase difference is π*sin(θ)
            antenna_indices = torch.arange(num_antennas, device=device).reshape(-1, 1)  # (M, 1)
            phase_shifts = -np.pi * antenna_indices * torch.sin(angles).reshape(1, -1)  # (M, num_angle_bins)
            steering_vectors = torch.exp(1j * phase_shifts)  # (M, num_angle_bins)

            # Compute R^{-1} using Cholesky decomposition for stability
            try:
                R_inv = torch.linalg.inv(R)
            except:
                # Fallback: increase diagonal loading
                R = R + 0.1 * torch.trace(R).real / num_antennas * torch.eye(num_antennas, device=device, dtype=R.dtype)
                R_inv = torch.linalg.inv(R)

            # MVDR spectrum: P(θ) = 1 / (a^H * R^{-1} * a)
            # Vectorized computation for all angles
            # R_inv_a: (M, num_angle_bins)
            R_inv_a = torch.matmul(R_inv, steering_vectors)

            # a^H * R^{-1} * a: (num_angle_bins,)
            denominator = (steering_vectors.conj() * R_inv_a).sum(dim=0).real

            # MVDR power spectrum (avoid division by zero)
            mvdr_spectrum = 1.0 / (denominator + 1e-10)

            # Keep raw MVDR power (no normalization here)
            azimuth_time_list.append(mvdr_spectrum)
        else:
            # No active signal, output zeros
            azimuth_time_list.append(torch.zeros(num_angle_bins, device=device))

    azimuth_time = torch.stack(azimuth_time_list, dim=1)  # (num_angle_bins, T) -> (128, T)

    # Return as dictionary for flexible access
    data_dict = {
        'range_time': range_time.cpu().numpy(),      # (256, T)
        'doppler_time': doppler_time.cpu().numpy(),  # (128, T)
        'azimuth_time': azimuth_time.cpu().numpy(),  # (128, T)
    }

    return data_dict


def _torch_per_frame_minmax(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Per-column min-max normalize a (bins, T) tensor to [0, 1].

    Keep this aligned with the NumPy real-data dataloader normalization:
    divide by (max - min) when the range is positive, otherwise output zeros.
    Do not add a fixed epsilon to the denominator because MVDR azimuth values
    can have very small dynamic ranges; adding 1e-6 suppresses the normalized
    maximum below 1 for otherwise valid frames.
    """
    x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    lo = x.min(dim=0, keepdim=True).values
    hi = x.max(dim=0, keepdim=True).values
    denom = hi - lo
    return torch.where(denom > 0, (x - lo) / denom, torch.zeros_like(x))


def add_stable_v1_view_noise(
    views,
    *,
    preset=None,
    seed=None,
    clip=True,
):
    """Apply deterministic lightweight synthetic view-domain noise.

    This is used after cube-level stable-v1 preprocessing to make synthetic
    three-view data less unrealistically clean while keeping the signal
    processing itself unchanged. The available noise preset is
    ``floor_speckle_v1``:

    - range:   weak range-colored floor + small Gaussian speckle
    - doppler: weak diffuse floor + small Gaussian speckle
    - azimuth: weak diffuse floor + small Gaussian speckle

    Args:
        views: dict with range_time/doppler_time/azimuth_time in [0, 1].
        preset: None/``clean`` to disable, or ``floor_speckle_v1``.
        seed: deterministic RNG seed.
        clip: clip outputs to [0, 1].
    """
    if preset in {None, '', 'none', 'clean'}:
        return {k: np.asarray(v, dtype=np.float32) for k, v in views.items()}
    if preset != 'floor_speckle_v1':
        raise ValueError(f"Unknown stable-v1 view noise preset: {preset!r}")

    rng = np.random.default_rng(seed)
    cfg = {
        'range_time': (0.020, 0.010),
        'doppler_time': (0.030, 0.014),
        'azimuth_time': (0.022, 0.010),
    }
    out = {}
    for key, arr in views.items():
        x = np.asarray(arr, dtype=np.float32)
        floor_strength, speckle_strength = cfg[key]
        y = x.copy()
        if key == 'range_time':
            n_bins, n_t = y.shape
            r = np.linspace(0.0, 1.0, n_bins, dtype=np.float32)
            profile = (0.45 + 0.55 * np.exp(-2.5 * r)).reshape(n_bins, 1)
            jitter = rng.normal(1.0, 0.10, size=(1, n_t)).astype(np.float32)
            y = y + floor_strength * profile * jitter
        else:
            y = y + floor_strength * rng.uniform(0.6, 1.0, size=y.shape).astype(np.float32)
        y = y + rng.normal(0.0, speckle_strength, size=y.shape).astype(np.float32)
        if clip:
            y = np.clip(y, 0.0, 1.0)
        out[key] = y.astype(np.float32, copy=False)
    return out


def get_multi_view_data_real_stable_v1(
    radar_frame,
    range_start=0,
    num_range_bins=128,
    num_doppler_bins=128,
    num_angle_bins=128,
    diagonal_loading=0.01,
    antenna_spacing_lambda=0.5,
    steering_sign=-1.0,
    normalize_views=True,
    doppler_lognorm=True,
    doppler_threshold=None,
    view_noise_preset=None,
    view_noise_seed=None,
):
    """Cube-level stable three-view preset aligned with real preprocessing v1.

    This is intended for future synthetic processing from in-memory radar cube
    or pathdicts.  It deliberately operates before projection/aggregation:

        radar cube -> ADC Hann -> range FFT -> ROI [range_start, range_start+128)
        -> chirp-mean MTI -> range/doppler/azimuth views

    The default view transforms are:
    - range_time:   ROI+MTI, linear per-frame minmax
    - doppler_time: ROI+MTI, Doppler FFT, log1p + per-frame minmax
    - azimuth_time: ROI+MTI, MVDR over chirp/range snapshots, per-frame minmax

    Args:
        radar_frame: complex tensor, shape (T, chirps, RX, ADC)
        range_start: first range bin after range FFT.
        num_range_bins: number of range bins to keep; current standard is 128.
        normalize_views: if False, return unnormalized power views after the
            cube-level ROI/MTI/MVDR processing.
        doppler_threshold: optional threshold after lognorm+per-frame minmax.

    Returns:
        dict with `range_time`, `doppler_time`, `azimuth_time` as numpy arrays.
    """
    if len(radar_frame.shape) != 4:
        raise ValueError(f"Expected radar_frame shape (T, chirps, RX, ADC), got {tuple(radar_frame.shape)}")

    device = radar_frame.device
    T, num_chirps, num_antennas, num_samples = radar_frame.shape
    range_start = int(range_start)
    num_range_bins = int(num_range_bins)
    if range_start < 0 or num_range_bins <= 0 or range_start + num_range_bins > num_samples:
        raise ValueError(
            f"Invalid range ROI [{range_start},{range_start + num_range_bins}) "
            f"for ADC samples={num_samples}"
        )

    # Shared real/synthetic stable front-end.
    adc_win = torch.hann_window(num_samples, device=device).reshape(1, 1, 1, -1)
    range_fft = torch.fft.fft(radar_frame * adc_win, n=num_samples, dim=3)
    range_fft = range_fft[:, :, :, range_start: range_start + num_range_bins]
    range_fft_mti = range_fft - range_fft.mean(dim=1, keepdim=True)

    # 1. Range-Time from clutter-removed range FFT.
    range_time = torch.abs(range_fft_mti).mean(dim=1).mean(dim=1).T  # (range, T)

    # 2. Doppler-Time from clutter-removed range FFT.
    chirp_win = torch.hann_window(num_chirps, device=device).reshape(1, -1, 1, 1)
    doppler_fft = torch.fft.fft(range_fft_mti * chirp_win, n=num_doppler_bins, dim=1)
    doppler_fft = torch.fft.fftshift(doppler_fft, dim=1)
    doppler_time = torch.abs(doppler_fft).mean(dim=2).mean(dim=2).T  # (doppler, T)

    # 3. Azimuth-Time: MVDR over chirp/range snapshots.
    angles = torch.linspace(-np.pi / 2, np.pi / 2, num_angle_bins, device=device)
    antenna_indices = torch.arange(num_antennas, device=device).reshape(-1, 1)
    phase = (
        float(steering_sign)
        * 2.0
        * np.pi
        * float(antenna_spacing_lambda)
        * antenna_indices
        * torch.sin(angles).reshape(1, -1)
    )
    steering_vectors = torch.exp(1j * phase)  # (RX, angle)
    eye = torch.eye(num_antennas, device=device, dtype=range_fft_mti.dtype)
    azimuth_cols = []
    for t in range(T):
        # X: (RX, chirps * range)
        X = range_fft_mti[t].permute(1, 0, 2).reshape(num_antennas, -1)
        if X.shape[1] == 0:
            azimuth_cols.append(torch.zeros(num_angle_bins, device=device, dtype=torch.float32))
            continue
        R = torch.matmul(X, X.conj().T) / X.shape[1]
        trace = torch.trace(R).real
        if (not torch.isfinite(trace)) or float(trace.item()) <= 1e-12:
            azimuth_cols.append(torch.zeros(num_angle_bins, device=device, dtype=torch.float32))
            continue
        loading = float(diagonal_loading) * trace / num_antennas
        R = R + loading * eye
        try:
            R_inv = torch.linalg.inv(R)
        except Exception:
            R = R + 0.1 * trace / num_antennas * eye
            try:
                R_inv = torch.linalg.inv(R)
            except Exception:
                R_inv = torch.linalg.pinv(R)

        R_inv_a = torch.matmul(R_inv, steering_vectors)
        denom = (steering_vectors.conj() * R_inv_a).sum(dim=0).real
        azimuth_cols.append((1.0 / (denom + 1e-10)).float())

    azimuth_time = torch.stack(azimuth_cols, dim=1)  # (angle, T)

    if normalize_views:
        range_time = _torch_per_frame_minmax(range_time)
        if doppler_lognorm:
            doppler_time = torch.log1p(torch.clamp(doppler_time, min=0.0))
        doppler_time = _torch_per_frame_minmax(doppler_time)
        if doppler_threshold is not None:
            doppler_time = doppler_time.clone()
            doppler_time[doppler_time < float(doppler_threshold)] = 0.0
        azimuth_time = _torch_per_frame_minmax(azimuth_time)

    views = {
        'range_time': range_time.detach().cpu().numpy(),
        'doppler_time': doppler_time.detach().cpu().numpy(),
        'azimuth_time': azimuth_time.detach().cpu().numpy(),
    }

    if view_noise_preset not in {None, '', 'none', 'clean'}:
        views = add_stable_v1_view_noise(
            views,
            preset=view_noise_preset,
            seed=view_noise_seed,
        )

    return views
