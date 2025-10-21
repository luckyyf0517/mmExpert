# Tools Directory

This directory contains utility tools for the mmExpert project.

## view_data.py

### Description
Data visualization script that loads and displays dataset samples using the project's data interface. Shows the actual data that would be fed to the model during training, including all preprocessing steps, and optionally saves visualizations as image files.

### Features
- Uses project's data interface for authentic data loading
- Displays detailed tensor statistics (shape, dtype, min/max/mean/std)
- Shows text processing information and captions
- Verifies dataset preprocessing pipeline
- Supports both training and validation data
- **NEW**: Saves spectrum visualizations as high-quality PNG images

### Usage

```bash
# Basic usage with data config
python tools/view_data.py --data-config config/data/humanml3d.yaml

# With custom number of samples
python tools/view_data.py --data-config config/data/humanml3d.yaml --num-samples 5

# With both data and model configs (for complete setup test)
python tools/view_data.py \
  --data-config config/data/humanml3d.yaml \
  --model-config config/model/siglip.yaml \
  --num-samples 2

# With image saving enabled
python tools/view_data.py \
  --data-config config/data/humanml3d.yaml \
  --num-samples 2 \
  --save-images \
  --output-dir /path/to/output

# Example for analyzing data before training
python tools/view_data.py \
  --data-config config/data/humanml3d.yaml \
  --num-samples 1 \
  --save-images \
  --output-dir ./data_preview
```

### Command Line Arguments
- `--data-config` (required): Path to data configuration file
- `--num-samples` (optional): Number of samples to display (default: 3)
- `--model-config` (optional): Path to model configuration file
- `--save-images` (optional): Save spectrum images and summary plots
- `--output-dir` (optional): Output directory for saved images (default: tmp/preview)

### Image Output Files

When `--save-images` is enabled, the following files are generated for each sample:

1. **Individual Spectrum Images**:
   - `sample_XXX_range_spectrum.png` - Range-time spectrum (raw + normalized views)
   - `sample_XXX_doppler_spectrum.png` - Doppler-time spectrum (raw + normalized views)
   - `sample_XXX_azimuth_spectrum.png` - Azimuth-time spectrum (raw + normalized views)

2. **Summary Plot**:
   - `sample_XXX_summary.png` - Comprehensive overview with all three views and statistics

Each spectrum image includes:
- Raw data visualization with viridis colormap
- Normalized data visualization with jet colormap
- Statistical information (shape, min/max/mean values)
- Proper axis labels and colorbars

The summary plot provides:
- All three radar views side by side
- Detailed statistics for each view
- Text caption associated with the sample
- Clean publication-ready layout

### Output Directory Structure
```
tmp/preview/
├── sample_001_range_spectrum.png
├── sample_001_doppler_spectrum.png
├── sample_001_azimuth_spectrum.png
├── sample_001_summary.png
├── sample_002_range_spectrum.png
└── ...
```

### Example Output
```
[SUCCESS] Data interface setup completed

[DATASET] Dataset Information:
  Training samples:   12540
  Validation samples: 1390
  Train batch size:  64
  Val batch size:    64

[IMAGES] Saving images for sample 1...
[SUCCESS] Saved spectrum image: sample_001_range_spectrum.png
[SUCCESS] Saved spectrum image: sample_001_doppler_spectrum.png
[SUCCESS] Saved spectrum image: sample_001_azimuth_spectrum.png
[SUCCESS] Saved summary: sample_001_summary.png

--- Sample 1 Wave Embedding Details ---
  [INPUT_WAVE_RANGE]
    Shape:      [64, 256, 496]
    Data type:  torch.float32
    Min value:  0.000000
    Max value:  0.043627
    Mean value: 0.000037
    Std value:  0.000303

[CAPTION] Text Caption:
    Caption 1: "A person walks slowly across the room while waving their hand"
```

### Use Cases
- **Before Training**: Verify data loading and preprocessing pipeline
- **Debug Data Issues**: Check tensor shapes, values, and normalization
- **Configuration Testing**: Test different data configs and their effects
- **Data Analysis**: Generate visualizations for analysis and reporting
- **Pipeline Verification**: Ensure data interface works correctly
- **Documentation**: Create visual examples for papers and presentations

### Image Visualization Features
- **Dual View**: Each spectrum shows both raw and normalized data
- **Statistical Overlays**: Shape and value statistics displayed on images
- **High Resolution**: 150 DPI output suitable for publications
- **Color Optimization**: Scientific colormaps (viridis, jet, plasma)
- **Comprehensive Layout**: All views organized in grid format

This tool provides comprehensive visibility into your data pipeline, ensuring that what you see matches what the model receives during training, with professional-quality visualizations for analysis and documentation.