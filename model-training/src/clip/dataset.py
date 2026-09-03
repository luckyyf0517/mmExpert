import os
import json
import random
import numpy as np
from copy import deepcopy
from easydict import EasyDict as edict
from src.logger import log_message

# Constants for radar data processing
DEFAULT_EPSILON = 1e-6

# Default configuration for radar dataset
DEFAULT_RADAR_OPT = {
    'max_motion_length': 496,
    'min_motion_len': 96,
    'unit_length': 16,
    'normalize': 'per_frame',  # 'none', 'per_frame', 'global', or 'log'
    'random_scale': False,
    'thresholding': False,
    # Set True for already-normalized three-view NPZs. The dataloader will
    # still crop/pad and create the mask, but it will skip a second
    # normalization, threshold, and random-scaling pass.
    'radar_npz_preprocessed': False,
    # Processing mode for single-view uDoppler NPY files.
    #
    # - synthetic_p995_gamma065:
    #     current raw synthetic mesh uDoppler model-input transform:
    #     log1p -> global minmax -> p99.5 stretch -> gamma 0.65.
    # - legacy_prelogged_p995_gamma065:
    #     compatibility path for older HumanML3D uDoppler NPYs that already look
    #     log/compressed.  It skips log1p and applies the same tail transform:
    #     global minmax -> p99.5 stretch -> gamma 0.65.  This is equivalent to
    #     treating the legacy file as pre-log-space data and then joining the
    #     current model-input pipeline after its log step.
    # - none/raw/direct:
    #     crop/pad only; useful for diagnostics, not the current training target.
    'udoppler_preprocess': 'legacy',
    'udoppler_preprocess_percentile': 99.5,
    'udoppler_preprocess_gamma': 0.65,
    # Optional post-threshold enhancement for real uDoppler modes.  Values < 1
    # brighten only nonzero motion responses after the fixed threshold has
    # removed background, so zeroed background stays zero.
    'udoppler_post_threshold_gamma': None,
    'raw': False,
    'text2doppler': False,
    'random_rotate': False,
    'separate': False,
    # Train-time view dropout.  Doppler should usually stay at 0.0 and serve as
    # the stable anchor; range/azimuth can be randomly zeroed for robustness.
    'range_drop_prob': 0.0,
    'doppler_drop_prob': 0.0,
    'azimuth_drop_prob': 0.0,
    # Optional raw real cube preprocessing. Synthetic/data-generated NPZ files
    # keep the historical path. Raw real cubes (T, chirps, RX, ADC) can use the
    # stable preprocessing preset below.
    'real_preprocess': 'auto',  # 'auto', 'stable_v1', or 'none'
    'real_range_start': 0,
    'real_range_bins': 128,
    'real_doppler_bins': 128,
    'real_angle_bins': 128,
    'real_diagonal_loading': 0.01,
    'real_antenna_spacing_lambda': 0.5,
    'real_steering_sign': -1.0,
    'real_rx_order': '',
    # Real-view display/input transforms.
    # range/azimuth: linear per-frame; doppler: log1p + per-frame.
    'real_view_transform': True,
    'real_doppler_threshold': None,
    # Current uDoppler reset should not silently fall back across data sources.
    # When disabled, doppler-only legacy HumanML3D splits resolve only to
    # sibling udoppler NPY files; missing samples are skipped by the dataset.
    'allow_radar_fallback': True,
    'skip_missing_radar': False,
    'udoppler_legacy_suffix': 'A',
}


# Radar data processing functions (for DATA_FORMAT.md .npz files)
def _normalize_per_frame(data):
    """Normalize each time frame to [0, 1] for visualization."""
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    normalized = data.copy().astype(np.float32, copy=False)
    num_bins, T = data.shape

    for t in range(T):
        frame = normalized[:, t]
        frame_min = frame.min()
        frame_max = frame.max()
        if frame_max > frame_min:
            normalized[:, t] = (frame - frame_min) / (frame_max - frame_min)
        else:
            normalized[:, t] = 0.0

    return normalized


def _normalize_global(data):
    """Global normalization across all frames."""
    dmin, dmax = data.min(), data.max()
    if dmax > dmin:
        return (data - dmin) / (dmax - dmin)
    return data


def _normalize_log(data):
    """Log-scale normalization."""
    return np.log10(data + 1e-10)


def _as_complex_cube(arr):
    """Accept complex cubes or real I/Q cubes with last dim size 2."""
    if np.iscomplexobj(arr):
        return arr.astype(np.complex64, copy=False)
    if arr.ndim >= 1 and arr.shape[-1] == 2:
        return (arr[..., 0] + 1j * arr[..., 1]).astype(np.complex64)
    raise ValueError(
        f"Input is not complex and does not look like I/Q last-dim=2; "
        f"shape={arr.shape}, dtype={arr.dtype}"
    )


def _parse_rx_order(rx_order, num_rx):
    if not rx_order:
        return None
    order = [int(x) for x in str(rx_order).split(',')]
    if len(order) != num_rx or sorted(order) != list(range(num_rx)):
        raise ValueError(f"Invalid real_rx_order={rx_order!r} for {num_rx} RX")
    return order


def _compute_mvdr_azimuth_time(
    range_fft_mti,
    *,
    angle_bins,
    diagonal_loading,
    antenna_spacing_lambda,
    steering_sign,
    rx_order='',
):
    """MVDR over chirp/range snapshots for real azimuth-time.

    Args:
        range_fft_mti: complex array (T, chirps, RX, range_bins)

    Returns:
        azimuth_time: float32 array (angle_bins, T)
    """
    T, _num_chirps, num_rx, _num_range = range_fft_mti.shape
    order = _parse_rx_order(rx_order, num_rx)
    if order is not None:
        range_fft_mti = range_fft_mti[:, :, order, :]

    angles = np.linspace(-np.pi / 2, np.pi / 2, angle_bins, dtype=np.float32)
    rx = np.arange(num_rx, dtype=np.float32).reshape(-1, 1)
    phase = (
        float(steering_sign)
        * 2.0
        * np.pi
        * float(antenna_spacing_lambda)
        * rx
        * np.sin(angles).reshape(1, -1)
    )
    steering = np.exp(1j * phase).astype(np.complex64)
    eye = np.eye(num_rx, dtype=np.complex64)

    cols = []
    for t in range(T):
        X = np.transpose(range_fft_mti[t], (1, 0, 2)).reshape(num_rx, -1)
        if X.shape[1] == 0 or not np.isfinite(X).all():
            cols.append(np.zeros(angle_bins, dtype=np.float32))
            continue

        R = (X @ X.conj().T) / X.shape[1]
        loading = float(diagonal_loading) * np.trace(R).real / num_rx
        R = R + loading * eye
        try:
            R_inv = np.linalg.inv(R)
        except np.linalg.LinAlgError:
            R = R + (0.1 * np.trace(R).real / num_rx) * eye
            R_inv = np.linalg.inv(R)

        R_inv_a = R_inv @ steering
        denom = np.real(np.sum(steering.conj() * R_inv_a, axis=0))
        power = 1.0 / (denom + 1e-10)
        cols.append(np.asarray(power, dtype=np.float32))

    return np.stack(cols, axis=1).astype(np.float32)


def compute_real_stable_three_views(cube, opt):
    """Stable preprocessing preset for raw real-data radar cubes.

    Input cube shape is (T, chirps=128, RX=4, ADC=256).  Output follows the
    model three-view convention:

    - range_time:   (128, T), linear per-frame normalized
    - doppler_time: (128, T), log1p + per-frame normalized
    - azimuth_time: (128, T), linear per-frame normalized

    This implements the real-cube preprocessing used by the data loader.
    """
    cube = _as_complex_cube(cube)
    if cube.ndim != 4:
        raise ValueError(f"Expected raw real cube (T, chirps, RX, ADC), got {cube.shape}")

    T, num_chirps, _num_rx, num_adc = cube.shape
    range_start = int(opt.get('real_range_start', 0))
    range_bins = int(opt.get('real_range_bins', 128))
    doppler_bins = int(opt.get('real_doppler_bins', 128))
    angle_bins = int(opt.get('real_angle_bins', 128))
    if range_start < 0 or range_bins <= 0 or range_start + range_bins > num_adc:
        raise ValueError(
            f"Invalid real range ROI [{range_start},{range_start + range_bins}) "
            f"for ADC={num_adc}"
        )

    adc_win = np.hanning(num_adc).astype(np.float32).reshape(1, 1, 1, -1)
    range_fft = np.fft.fft(cube * adc_win, n=num_adc, axis=3)
    range_fft = range_fft[:, :, :, range_start: range_start + range_bins]

    # Per-frame chirp-mean MTI in range-FFT domain.
    range_fft_mti = range_fft - range_fft.mean(axis=1, keepdims=True)

    # Range-Time: motion range response from clutter-removed cube.
    range_time = np.abs(range_fft_mti).mean(axis=1).mean(axis=1).T.astype(np.float32)

    # Doppler-Time: Doppler FFT from clutter-removed range FFT, then aggregate
    # over RX/range.  Lognorm was visually selected for real data.
    chirp_win = np.hanning(num_chirps).astype(np.float32).reshape(1, -1, 1, 1)
    doppler_fft = np.fft.fft(range_fft_mti * chirp_win, n=doppler_bins, axis=1)
    doppler_fft = np.fft.fftshift(doppler_fft, axes=1)
    doppler_time = np.abs(doppler_fft).mean(axis=2).mean(axis=2).T.astype(np.float32)

    # Azimuth-Time: MVDR over chirp/range snapshots from clutter-removed cube.
    azimuth_time = _compute_mvdr_azimuth_time(
        range_fft_mti,
        angle_bins=angle_bins,
        diagonal_loading=float(opt.get('real_diagonal_loading', 0.01)),
        antenna_spacing_lambda=float(opt.get('real_antenna_spacing_lambda', 0.5)),
        steering_sign=float(opt.get('real_steering_sign', -1.0)),
        rx_order=opt.get('real_rx_order', ''),
    )

    views = {
        'range_time': range_time,
        'doppler_time': doppler_time,
        'azimuth_time': azimuth_time,
    }

    if bool(opt.get('real_view_transform', True)):
        views['range_time'] = _normalize_per_frame(views['range_time'])
        views['doppler_time'] = _normalize_per_frame(np.log1p(np.maximum(views['doppler_time'], 0.0)))
        threshold = opt.get('real_doppler_threshold', None)
        if threshold is not None:
            views['doppler_time'] = views['doppler_time'].copy()
            views['doppler_time'][views['doppler_time'] < float(threshold)] = 0.0
        views['azimuth_time'] = _normalize_per_frame(views['azimuth_time'])

    return {k: v.astype(np.float32, copy=False) for k, v in views.items()}


def _apply_random_scale(radar_view, opt):
    if not opt.get('random_scale'):
        return radar_view
    max_val = radar_view.max()
    if max_val <= 0:
        return radar_view
    random_shift = random.random() * 0.01
    return np.log10(radar_view / max_val + random_shift + DEFAULT_EPSILON)


def _apply_thresholding(radar_view, opt):
    if opt.get('thresholding'):
        radar_view = radar_view.copy()
        radar_view[radar_view < 0.1] = 0
    return radar_view


def _apply_radar_normalization(radar_view, opt):
    """Apply normalization based on strategy for a single radar view."""
    normalize_type = opt.get('normalize', 'none')

    if normalize_type == 'per_frame':
        return _normalize_per_frame(radar_view)
    elif normalize_type == 'global':
        return _normalize_global(radar_view)
    elif normalize_type == 'log':
        return _normalize_log(radar_view)
    else:  # 'none'
        return radar_view


def _crop_radar_view(radar_view, opt):
    """Crop radar view to unit length boundaries."""
    T = radar_view.shape[1]
    T = (T // opt.unit_length) * opt.unit_length
    idx = random.randint(0, radar_view.shape[1] - T)
    radar_view = radar_view[:, idx: idx + T]
    return radar_view, T


def _pad_radar_view(radar_view, T, max_motion_length):
    """Pad radar view to max_motion_length and create mask."""
    num_bins, current_T = radar_view.shape
    mask = np.ones((max_motion_length,), dtype=np.float32)

    if T < max_motion_length:
        padding = np.zeros((num_bins, max_motion_length - T))
        radar_view = np.concatenate([radar_view, padding], axis=1)
        mask[T:] = 0
    else:
        radar_view = radar_view[:, :max_motion_length]
        T = max_motion_length

    return radar_view, mask, T


def process_radar_view(radar_view, opt):
    """Process a single radar view with transformations and normalizations."""
    # Crop to unit length boundaries
    radar_view, T = _crop_radar_view(radar_view, opt)

    # Optional scale jitter in log domain
    radar_view = _apply_random_scale(radar_view, opt)

    # Apply normalization
    radar_view = _apply_radar_normalization(radar_view, opt)

    # Sparsify weak responses after normalization
    radar_view = _apply_thresholding(radar_view, opt)

    # Pad and create mask
    radar_view, mask, T = _pad_radar_view(radar_view, T, opt.max_motion_length)

    return radar_view, mask, T


def process_preprocessed_radar_view(radar_view, opt):
    """Crop/pad an already-preprocessed radar view without value transforms."""
    radar_view, T = _crop_radar_view(radar_view, opt)
    radar_view, mask, T = _pad_radar_view(radar_view, T, opt.max_motion_length)
    return radar_view, mask, T


def process_legacy_udoppler(motion, opt):
    """Process doppler-only .npy inputs.

    Input motion is expected as (T, 128). Output keeps the same layout so we can
    transpose once at the call site to satisfy the CLIP radar encoder's [D, T].
    """
    if opt.get('downsample') is not None and opt.downsample:
        downsample_start = random.randint(0, opt.downsample - 1)
        motion = motion[downsample_start::opt.downsample]
        max_motion_length = opt.max_motion_length // opt.downsample
    else:
        max_motion_length = opt.max_motion_length

    m_length = motion.shape[0]
    m_length = (m_length // opt.unit_length) * opt.unit_length
    idx = random.randint(0, motion.shape[0] - m_length)
    motion = motion[idx: idx + m_length]

    if opt.get('separate'):
        coin = random.choice([0, 0, 0, 1, 2])
        if coin == 0:
            weights = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        elif coin == 1:
            weights = np.array([0.0, 1.0, 0.0, 1.0, 1.0])
        else:
            weights = np.array([0.0, 1.0, 0.0, 1.0, 0.0])
        motion = np.einsum('ijk,j->ik', motion, weights)

    preprocess_mode = opt.get('udoppler_preprocess', 'legacy')
    direct_modes = {'none', 'raw', 'direct'}
    prelogged_modes = {
        'legacy_prelogged_p995_gamma065',
        'prelogged_p995_gamma065',
        'legacy_logspace_p995_gamma065',
    }
    raw_synthetic_modes = {
        'synthetic_p995_gamma065',
        'log1p_global_pclip_gamma',
    }
    # Real-data gamma modes: log10 → min-max → gamma>1 → thr0.1.
    # An optional post-threshold gamma can then brighten retained nonzero
    # motion responses while keeping thresholded background at zero.
    #
    # Gamma > 1 dampens low values (noise) while preserving high values
    # (signal), widening the separability so a single fixed threshold 0.1
    # works across different signal-to-noise regimes.
    #
    #   legacy_real_gamma20: gamma=2.0 for higher-SNR data
    #   real_gamma15:        gamma=1.5 for lower-SNR data
    real_gamma_modes = {
        'legacy_real_gamma20',
        'real_gamma15',
    }

    def _global_pclip_gamma(x):
        x = (x - x.min()) / (x.max() - x.min() + 1e-9)
        percentile = float(opt.get('udoppler_preprocess_percentile', 99.5))
        gamma = float(opt.get('udoppler_preprocess_gamma', 0.65))
        hi = float(np.percentile(x, percentile))
        if hi <= 1e-9:
            return np.zeros_like(x, dtype=np.float32)
        x = np.clip(x / (hi + 1e-9), 0.0, 1.0)
        if gamma != 1.0:
            x = np.power(x, gamma)
        return x.astype(np.float32, copy=False)

    def _post_threshold_gamma(x):
        gamma = opt.get('udoppler_post_threshold_gamma', None)
        if gamma in (None, '', 1, 1.0):
            return x.astype(np.float32, copy=False)
        gamma = float(gamma)
        if gamma <= 0:
            raise ValueError(f"udoppler_post_threshold_gamma must be > 0, got {gamma}")
        y = x.astype(np.float32, copy=True)
        nonzero = y > 0
        if np.any(nonzero):
            y[nonzero] = np.power(y[nonzero], gamma)
        return y

    if preprocess_mode in direct_modes or preprocess_mode in prelogged_modes:
        pass
    elif opt.get('logged'):
        motion = np.power(10, motion)

    if preprocess_mode in direct_modes:
        pass
    elif preprocess_mode in {None, '', 'legacy'}:
        if opt.get('text2doppler'):
            motion = 20 * np.log10(motion / motion.max() + 1e-6)
            motion = np.clip(motion, motion.max() - 60, motion.max())
        if opt.get('raw'):
            motion = np.log10(motion + 1e-6)
        if opt.get('random_scale'):
            random_shift = random.random() * 0.01
            motion = np.log10(motion / motion.max() + random_shift + 1e-6)
        if opt.get('log_norm'):
            motion = np.log10(motion / motion.max() + 1e-6)
        if opt.get('ignore_energy'):
            motion = (
                motion - motion.min(axis=1, keepdims=True)
            ) / (
                motion.max(axis=1, keepdims=True) - motion.min(axis=1, keepdims=True) + 1e-6
            )

        motion = (motion - motion.min()) / (motion.max() - motion.min() + 1e-6)
        if opt.get('thresholding'):
            motion[motion < 0.1] = 0
    elif preprocess_mode in raw_synthetic_modes:
        motion = np.log1p(np.maximum(motion, 0.0))
        motion = _global_pclip_gamma(motion)
    elif preprocess_mode in prelogged_modes:
        motion = _global_pclip_gamma(motion)
    elif preprocess_mode in real_gamma_modes:
        # Real-data gamma mode: log10 → min-max → gamma>1 → thr0.1, followed
        # by optional post-threshold nonzero-only enhancement.
        #
        # The gamma value is encoded in the mode name:
        #   legacy_real_gamma20: gamma=2.0
        #   real_gamma15:        gamma=1.5
        #
        # Gamma > 1 widens the gap between noise floor and signal peaks,
        # making the fixed threshold 0.1 effective across both SNR regimes.
        _gamma = float(preprocess_mode.split('gamma')[-1]) / 10.0
        motion = np.log10(np.maximum(motion, 0) + 1e-6)
        motion = (motion - motion.min()) / (motion.max() - motion.min() + 1e-6)
        motion = np.power(motion, _gamma)
        motion[motion < 0.1] = 0.0
        motion = _post_threshold_gamma(motion)
    else:
        raise ValueError(f"Unknown udoppler_preprocess mode: {preprocess_mode!r}")

    mask = np.ones((max_motion_length,), dtype=np.float32)
    if m_length < max_motion_length:
        motion = np.concatenate([
            motion,
            np.zeros((max_motion_length - m_length, motion.shape[1]))
        ], axis=0)
        mask[m_length:] = 0
    else:
        motion = motion[:max_motion_length]
        m_length = max_motion_length

    if opt.get('add_noise'):
        noise = np.random.normal(0, 0.001, motion.shape)
        noise[motion > 0] = 0
        motion = motion + noise

    return motion.astype(np.float32), mask.astype(np.float32), int(m_length)


# ---------------------------------------------------------------------------
# On-the-fly static background overlay for training
# ---------------------------------------------------------------------------

DEFAULT_STATIC_OVERLAY_PRESET = {
    "mode": "linear_add",
    "static_gain": 1.0,
    "motion_gain": 1.0,
    "view_gains": {"range_time": 1.0, "doppler_time": 1.0, "azimuth_time": 0.8},
}


def mix_static_overlay(
    static_views: dict[str, np.ndarray],
    synth_views: dict[str, np.ndarray],
    preset: dict | None = None,
) -> dict[str, np.ndarray]:
    """On-the-fly static background overlay: synth + static_bg → mixed.

    Args:
        static_views: dict with 'range_time'/'doppler_time'/'azimuth_time' (H, T_static).
        synth_views:  dict with same keys (H, T_synth), the loaded synthetic NPZ.
        preset:       optional override for DEFAULT_STATIC_OVERLAY_PRESET.

    Returns:
        mixed dict with same keys, float32.
    """
    if preset is None:
        preset = DEFAULT_STATIC_OVERLAY_PRESET

    mixed = {}
    for key in ("range_time", "doppler_time", "azimuth_time"):
        static = static_views[key]
        synth = synth_views[key]
        target_t = synth.shape[1]
        # time-align static to synthetic
        static_crop = static[:, :target_t]

        # auto-scale motion residual
        motion_base = float(np.quantile(synth, 0.50))
        residual = np.clip(synth - motion_base, 0.0, None)
        static_q50 = float(np.quantile(static_crop, 0.50))
        static_q95 = float(np.quantile(static_crop, 0.95))
        target_dynamic = max(
            static_q95 - static_q50,
            0.15 * max(static_q95 - static_q50, 1e-6),
            1e-6,
        )
        motion_q95 = max(float(np.quantile(residual, 0.95)), 1e-6)
        auto_scale = target_dynamic / motion_q95
        scaled_motion = residual * auto_scale

        vg = float(preset["view_gains"].get(key, 1.0))
        sg = float(preset["static_gain"])
        mg = float(preset["motion_gain"])

        if preset.get("mode") == "power_add":
            mixed[key] = np.sqrt(
                np.square(sg * static_crop) + np.square(mg * vg * scaled_motion)
            )
        else:  # linear_add
            mixed[key] = sg * static_crop + mg * vg * scaled_motion

    return {k: v.astype(np.float32, copy=False) for k, v in mixed.items()}


def resolve_radar_path(data_dict, opt, *, require_exists=False):
    """Resolve radar path across CLIP/LLM datasets with one shared policy.

    Supported cases:
    - multi-view generated data from `mmwave*.npz`;
    - raw real cube data from direct `.npy` paths;
    - legacy single-view data from `udoppler/*.npy`.

    The resolver intentionally preserves the historical CLIP preference:
    single-view configs prefer legacy `.npy` when available, while all-view
    configs prefer three-view `.npz` and then raw real cube `.npy`.
    """
    motion_folder = data_dict['filefolder']
    motion_index = data_dict['fileindex']
    radar_views = opt.get('radar_views', 'all')
    if radar_views == 'udoppler':
        radar_views = 'doppler'
    mmwave_postfix = opt.get('mmwave_postfix', '')
    if mmwave_postfix and os.path.basename(motion_folder) == 'mmwave':
        mmwave_folder = motion_folder + mmwave_postfix
    else:
        mmwave_folder = motion_folder.replace('udoppler', 'mmwave' + mmwave_postfix)
    udoppler_folder = motion_folder
    if os.path.basename(udoppler_folder) == 'mmwave':
        udoppler_folder = os.path.join(os.path.dirname(udoppler_folder), 'udoppler')
    else:
        udoppler_folder = udoppler_folder.replace('mmwave', 'udoppler')
    radar_root_prefix = opt.get('radar_root_prefix', '')
    if radar_root_prefix and mmwave_postfix and not os.path.isabs(mmwave_folder):
        rel_parts = mmwave_folder.split('/')
        if len(rel_parts) >= 3 and rel_parts[0] == 'dataset' and rel_parts[1] == 'HumanML3D':
            mmwave_folder = os.path.join(radar_root_prefix, rel_parts[-1])
        else:
            mmwave_folder = os.path.join(radar_root_prefix, mmwave_folder)

    legacy_suffix = str(opt.get('udoppler_legacy_suffix', 'A'))

    if radar_views == 'all':
        candidates = [
            os.path.join(mmwave_folder, f'{motion_index}.npz'),
            os.path.join(motion_folder, f'{motion_index}.npz'),
            os.path.join(motion_folder, f'{motion_index}.npy'),
            os.path.join(mmwave_folder, f'{motion_index}.npy'),
        ]
    else:
        if os.path.basename(motion_folder) == 'mmwave':
            candidates = [
                os.path.join(udoppler_folder, f'{motion_index}.npy'),
                os.path.join(udoppler_folder, f'{motion_index}{legacy_suffix}.npy'),
            ]
        else:
            candidates = [
                os.path.join(motion_folder, f'{motion_index}.npy'),
                os.path.join(motion_folder, f'{motion_index}{legacy_suffix}.npy'),
                os.path.join(udoppler_folder, f'{motion_index}.npy'),
                os.path.join(udoppler_folder, f'{motion_index}{legacy_suffix}.npy'),
            ]
        if bool(opt.get('allow_radar_fallback', True)):
            candidates += [
                os.path.join(mmwave_folder, f'{motion_index}.npz'),
                os.path.join(motion_folder, f'{motion_index}.npz'),
                os.path.join(mmwave_folder, f'{motion_index}.npy'),
            ]

    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate

    if require_exists:
        raise FileNotFoundError(f"Radar file not found; tried: {candidates}")
    return candidates[0]


def load_radar_data(npz_file_path, opt):
    """Load radar data from NPZ (multi-view) or NPY (single-view udoppler) file."""
    try:
        if npz_file_path.endswith('.npy'):
            raw = np.load(npz_file_path)
            if raw.ndim == 4:
                real_preprocess = opt.get('real_preprocess', 'auto')
                if real_preprocess in {'auto', 'stable_v1'}:
                    real_views = compute_real_stable_three_views(raw, opt)
                    # The stable preset already applies the per-view
                    # transforms.  Reuse process_radar_view only for temporal
                    # crop/pad/mask, avoiding accidental second normalization
                    # or thresholding unless real_view_transform is disabled.
                    real_opt = deepcopy(opt)
                    if bool(opt.get('real_view_transform', True)):
                        real_opt.normalize = 'none'
                        real_opt.thresholding = False
                        real_opt.random_scale = False
                    range_processed, range_mask, range_T = process_radar_view(real_views['range_time'], real_opt)
                    doppler_processed, _doppler_mask, _doppler_T = process_radar_view(real_views['doppler_time'], real_opt)
                    azimuth_processed, _azimuth_mask, _azimuth_T = process_radar_view(real_views['azimuth_time'], real_opt)
                    return {
                        'range_time': range_processed.astype(np.float32),
                        'doppler_time': doppler_processed.astype(np.float32),
                        'azimuth_time': azimuth_processed.astype(np.float32),
                        'mask': range_mask.astype(np.float32),
                        'T': range_T,
                    }
                raise ValueError(
                    f"Raw real cube {npz_file_path} requires opt.real_preprocess='stable_v1' "
                    f"or 'auto', got {real_preprocess!r}"
                )

            if raw.ndim != 2:
                raise ValueError(f"Expected 2D udoppler array or 4D real cube, got shape {raw.shape}")
            if raw.shape[1] != 128:
                raise ValueError(
                    f"Expected udoppler stored as (T, 128), got shape {raw.shape}"
                )
            doppler_processed, doppler_mask, doppler_T = process_legacy_udoppler(raw, opt)
            return {
                'doppler_time': doppler_processed.T.astype(np.float32),
                'mask': doppler_mask.astype(np.float32),
                'T': doppler_T,
            }

        data = np.load(npz_file_path)
        radar_views = opt.get('radar_views', 'all')
        if radar_views == 'udoppler':
            radar_views = 'doppler'

        # For uDoppler-only NPZs, accept either canonical `udoppler` or
        # historical `doppler_time` without requiring range/azimuth keys.
        view_processor = (
            process_preprocessed_radar_view
            if bool(opt.get('radar_npz_preprocessed', False))
            else process_radar_view
        )
        if radar_views == 'doppler' and ('udoppler' in data.files or 'doppler_time' in data.files):
            doppler_key = 'udoppler' if 'udoppler' in data.files else 'doppler_time'
            doppler_time = data[doppler_key]
            doppler_processed, doppler_mask, doppler_T = view_processor(doppler_time, opt)
            return {
                'doppler_time': doppler_processed.astype(np.float32),
                'mask': doppler_mask.astype(np.float32),
                'T': doppler_T,
            }

        # Extract three-view data.  Legacy NPZs may be raw/unnormalized, while
        # stable-v1-C NPZs are already normalized and noise-augmented.
        range_time = data['range_time']      # stable-v1: (128, T), legacy: (256, T)
        doppler_time = data['doppler_time']  # (128, T)
        azimuth_time = data['azimuth_time']  # (128, T)

        range_processed, range_mask, range_T = view_processor(range_time, opt)
        doppler_processed, doppler_mask, doppler_T = view_processor(doppler_time, opt)
        azimuth_processed, azimuth_mask, azimuth_T = view_processor(azimuth_time, opt)

        # Return as dictionary with three separate views
        return {
            'range_time': range_processed.astype(np.float32),    # (256, T_processed)
            'doppler_time': doppler_processed.astype(np.float32),  # (128, T_processed)
            'azimuth_time': azimuth_processed.astype(np.float32),  # (128, T_processed)
            'mask': range_mask.astype(np.float32),  # All views should have same T after processing
            'T': range_T
        }

    except Exception as e:
        log_message("ERROR", f"Error loading radar data from {npz_file_path}: {e}", color="red")
        return None


class Text2DopplerDatasetV2():
    """Dataset for radar data processing during training and validation."""

    # Constants
    ROTATE_POSTFIXES = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    PERSON_SYNONYMS = ['person', 'man', 'human']

    def __init__(self, opt, split_file, data_scale=1.0):
        merged_opt = deepcopy(DEFAULT_RADAR_OPT)
        merged_opt.update(dict(deepcopy(opt)))
        self.opt = edict(merged_opt)
        split_file_name_lower = os.path.basename(str(split_file)).lower()
        if 'real' in str(split_file).lower() or split_file_name_lower in {'real.json', 'real0.json', 'real1.json'}:
            self.opt.separate = False
            self.opt.random_rotate = False
            self.opt.random_scale = False
            self.opt.text2doppler = False
            self.opt.raw = True
        self.pointer = 0
        self.max_motion_length = opt.max_motion_length
        self.min_motion_len = opt.min_motion_len
        self.data_list = self._load_data(split_file, data_scale)
        self.mmwave_postfix = opt.get('mmwave_postfix', '')
        self.is_train = 'train' in os.path.basename(str(split_file)).lower()

    def _load_data(self, split_file, data_scale):
        """Load and scale dataset."""
        data_dict = json.load(open(split_file, 'r'))

        data_list = list(data_dict.values())
        data_list_ = []
        for data_item in data_list:
            extra_label = data_item.get('label')
            for caption in data_item['captions']:
                data_item_ = deepcopy(data_item)
                data_item_['caption'] = caption
                del data_item_['captions']
                if extra_label is not None:
                    data_item_['label'] = extra_label
                data_list_.append(data_item_)

        if bool(self.opt.get('skip_missing_radar', False)):
            kept = []
            skipped = 0
            for data_item in data_list_:
                radar_path = resolve_radar_path(data_item, self.opt, require_exists=False)
                if os.path.exists(radar_path):
                    kept.append(data_item)
                else:
                    skipped += 1
            if skipped:
                log_message(
                    "DATASET",
                    f"Skipped {skipped} caption items with missing radar files from {split_file}",
                    color="yellow",
                )
            data_list_ = kept

        data_amount = int(len(data_list_) * data_scale)
        data_list_ = random.sample(data_list_, data_amount)
        log_message("DATASET", f"Total number of data {len(data_list_)} (scale factor: {data_scale}) from {split_file}", color="blue")
        return data_list_

    def _get_radar_path(self, data_dict):
        """Resolve radar path across legacy single-view NPY and newer mmwave NPZ layouts.

        Supported cases:
        - multi-view training from `mmwave/*.npz`
        - doppler-only (or other single-view configs) backed by `mmwave/*.npz`
        - legacy single-view training from `udoppler/*.npy`
        """
        return resolve_radar_path(data_dict, self.opt, require_exists=False)

    def _process_caption(self, data_dict):
        """Normalize caption text for training."""
        caption = data_dict['caption']
        return caption.lower()

    def _create_item_dict(self, data_dict, motion_path, radar_data):
        """Create item dictionary with all required fields."""
        radar_views = self.opt.get('radar_views', 'all')

        # Map radar_views config to actual view keys
        view_key_map = {
            'range_time': 'input_wave_range',
            'doppler_time': 'input_wave_doppler',
            'azimuth_time': 'input_wave_azimuth',
        }
        view_selection = {
            'all': ['range_time', 'doppler_time', 'azimuth_time'],
            'doppler': ['doppler_time'],
            'udoppler': ['doppler_time'],
            'range': ['range_time'],
            'azimuth': ['azimuth_time'],
        }

        if radar_views not in view_selection:
            raise ValueError(f"Unknown radar_views configuration: {radar_views}")

        selected_views = view_selection[radar_views]

        # Filter radar_data dict to only include selected views (plus mask/T)
        filtered_radar_data = {k: v for k, v in radar_data.items() if k in selected_views or k in ('mask', 'T')}

        # Build item dict - only include selected view keys (no None values)
        item_dict = {
            'filename': motion_path,
            'radar_data': filtered_radar_data,
            'caption': self._process_caption(data_dict),
        }

        for view_key in selected_views:
            item_dict[view_key_map[view_key]] = radar_data[view_key]

        # Store the configuration for model reference
        item_dict['radar_views_config'] = radar_views

        if 'label' in data_dict:
            item_dict['motion_label'] = data_dict['label']

        if 'classid' in data_dict:
            item_dict.update({
                'classid': data_dict['classid'],
                'classname': data_dict['classname'],
            })
        else:
            item_dict.update({
                'classid': -1,
                'classname': 'null'
            })

        return item_dict

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data_dict = self.data_list[idx]
        radar_path = self._get_radar_path(data_dict)

        # Load radar data from NPZ file
        radar_data = load_radar_data(radar_path, self.opt)
        if radar_data is None:
            # Raise error if radar data loading fails instead of using dummy data
            raise ValueError(f"Failed to load radar data from {radar_path}. Radar data file is missing or corrupted.")
        if self.is_train:
            for view_key, prob_key in (
                ("range_time", "range_drop_prob"),
                ("doppler_time", "doppler_drop_prob"),
                ("azimuth_time", "azimuth_drop_prob"),
            ):
                prob = float(self.opt.get(prob_key, 0.0))
                if prob > 0 and random.random() < prob and view_key in radar_data:
                    radar_data[view_key] = np.zeros_like(radar_data[view_key])
        return self._create_item_dict(data_dict, radar_path, radar_data)
