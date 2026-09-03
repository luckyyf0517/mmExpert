#!/usr/bin/env python3
"""
Batch-extract uDoppler (T, 128) from raw ADC cubes (T, 128, 4, 256).

Processing chain:
  ADC Hann -> slice [16,128) -> chirp-mean MTI -> range FFT -> chirp Hann
  -> Doppler FFT -> fftshift -> abs -> per-frame target range selection
  -> bin-64 interpolation -> sum over range/RX -> (T, 128) float32

Usage:
  python extract_udoppler_from_raw.py \
      --input-dir /path/to/raw-cubes \
      --output-dir /path/to/udoppler \
      [--reflect /path/to/background.npy]
"""

import argparse
import numpy as np
from pathlib import Path
import sys


def extract_udoppler(
    cube: np.ndarray,
    *,
    range_start: int = 16,
    range_end: int = 128,
    num_doppler_bins: int = 128,
    target_range_bins: int = 8,
    interpolate_center: bool = True,
) -> np.ndarray:
    """Extract uDoppler from a raw ADC cube.

    Args:
        cube: complex64, shape (T, chirps=128, RX, ADC=256)
        range_start: ADC bin to start range slice
        range_end: ADC bin to end range slice
        num_doppler_bins: number of Doppler bins (128)
        target_range_bins: width of per-frame range selection window (even)
        interpolate_center: whether to replace bin 64 with (bin63+bin65)/2

    Returns:
        float32 array of shape (T, 128)
    """
    if cube.ndim != 4:
        raise ValueError(f"Expected 4D cube, got shape {cube.shape}")

    T, num_chirps, num_rx, num_adc = cube.shape

    # 1. ADC Hann window
    adc_win = np.hanning(num_adc).reshape(1, 1, 1, -1)
    data = cube * adc_win

    # 2. Fixed ADC/range slice [range_start, range_end)
    data = data[..., range_start:range_end]

    # 3. Chirp-mean MTI (static suppression)
    data = data - data.mean(axis=1, keepdims=True)

    # 4. Range FFT (over ADC axis)
    data = np.fft.fft(data, axis=3)

    # 5. Chirp Hann window
    chirp_win = np.hanning(num_chirps).reshape(1, -1, 1, 1)
    data = data * chirp_win

    # 6. Doppler FFT (over chirp axis)
    data = np.fft.fft(data, n=num_doppler_bins, axis=1)

    # 7. fftshift along Doppler axis
    data = np.fft.fftshift(data, axes=1)

    # 8. Magnitude (power spectrum)
    power = np.abs(data)  # (T, 128, RX, range_bins)

    # 9. Per-frame target range selection
    doppler_spec = power.sum(axis=(2, 3))        # (T, 128)
    doppler_center = doppler_spec.argmax(axis=1) # (T,)

    half = target_range_bins // 2
    max_center = power.shape[3] - half
    min_center = target_range_bins  # = 8

    selected = []
    for i in range(T):
        rangespec = power[i, int(doppler_center[i])].sum(axis=0)  # sum over RX
        range_center = int(rangespec.argmax())
        range_center = max(min_center, min(range_center, max_center))
        selected.append(power[i, :, :, range_center - half:range_center + half])

    target = np.stack(selected, axis=0)  # (T, 128, RX, 8)

    # 10. Replace Doppler bin 64 by average of bins 63 and 65
    center_bin = num_doppler_bins // 2  # = 64
    if interpolate_center and 0 < center_bin < num_doppler_bins - 1:
        target[:, center_bin] = (target[:, center_bin - 1] + target[:, center_bin + 1]) / 2.0

    # 11. Sum over selected range bins and RX → (T, 128)
    udoppler = target.sum(axis=-1).sum(axis=-1).astype(np.float32)

    return udoppler


def process_directory(input_dir: Path, output_dir: Path, reflect_path: Path | None = None):
    """Process all .npy files in input_dir and save uDoppler to output_dir."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    npy_files = sorted(input_dir.glob("*.npy"))

    # Load reflect / background if provided
    reflect_udoppler = None
    if reflect_path is not None:
        reflect_path = Path(reflect_path)
        if reflect_path.exists():
            print(f"Loading reflect / background: {reflect_path}")
            reflect_data = np.load(reflect_path)
            reflect_udoppler = extract_udoppler(reflect_data)
            reflect_out = output_dir / "reflect_udoppler.npy"
            np.save(reflect_out, reflect_udoppler)
            print(f"  -> saved {reflect_out}  shape={reflect_udoppler.shape}")
        else:
            print(f"Warning: reflect file not found: {reflect_path}", file=sys.stderr)

    for fp in npy_files:
        if fp.name == "reflect.npy":
            continue  # handled above

        print(f"Processing {fp.name} ...", end=" ", flush=True)
        try:
            cube = np.load(str(fp))
        except Exception as e:
            print(f"ERROR loading: {e}", file=sys.stderr)
            continue

        if cube.ndim != 4:
            print(f"SKIP (ndim={cube.ndim}, expected 4D)", file=sys.stderr)
            continue

        udoppler = extract_udoppler(cube)

        # Optional: subtract background reflect
        if reflect_udoppler is not None:
            udoppler = np.maximum(udoppler - reflect_udoppler, 0)

        out_path = output_dir / f"{fp.stem}_udoppler.npy"
        np.save(out_path, udoppler)
        print(f"done -> {out_path}  shape={udoppler.shape}  "
              f"range=[{udoppler.min():.2f}, {udoppler.max():.2f}]")

    print(f"\nAll done. {len(npy_files)} files processed -> {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Extract uDoppler from raw ADC cubes")
    parser.add_argument("--input-dir", type=str, required=True,
                        help="Directory containing raw .npy cubes")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for uDoppler .npy files")
    parser.add_argument("--reflect", type=str, default=None,
                        help="Path to reflect/background capture (optional)")
    args = parser.parse_args()

    process_directory(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        reflect_path=args.reflect,
    )


if __name__ == "__main__":
    main()
