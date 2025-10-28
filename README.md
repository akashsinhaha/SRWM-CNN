# DeepOSWSRM: Deep Feature Collaborative CNN for Water Super-Resolution Mapping

Implementation of "Super-resolution water body mapping with a feature collaborative CNN model by fusing Sentinel-1 and Sentinel-2 images" (Yin et al., 2024)

## Overview

DeepOSWSRM is a deep learning method that generates high-resolution water body maps by fusing Sentinel-1 (SAR) and Sentinel-2 (optical) imagery. The method addresses two main challenges:
1. **Mixed pixel problem**: Traditional classification struggles with pixels containing both water and land
2. **Cloud interference**: Optical imagery is often obscured by clouds

### Key Features

- **Multi-sensor fusion**: Combines SAR (cloud-penetrating) and optical (detailed) data
- **Super-resolution mapping**: Generates maps finer than input resolution
- **Configurable scale factors**: Supports 2x, 4x, and 6x super-resolution
- **Cloud robustness**: Leverages Sentinel-1 to map water under cloud cover
- **Free alternative to PlanetScope**: Uses Landsat-8/9 for training reference data

## Architecture

The model consists of two main modules:

1. **Water Fraction Unmixing Module**
   - Pseudo-Siamese CNN for extracting features from Sentinel-1 and Sentinel-2
   - Stacked residual blocks for deep feature learning
   - Estimates coarse-resolution water fraction (0-1)

2. **Super-Resolution Mapping Module**
   - Encoder-decoder architecture with skip connections
   - Multi-scale feature fusion
   - Spatial-channel attention mechanism
   - Generates fine-resolution binary water maps

## Installation

### Option 1: Google Colab (Recommended for beginners)

```python
# Install dependencies
!pip install -q earthengine-api geemap rasterio albumentations tensorboard

# Authenticate with Google Earth Engine
import ee
ee.Authenticate()
ee.Initialize()

# Clone or upload the code files
```

### Option 2: Local Installation

```bash
# Clone repository
git clone <repository-url>
cd deeposwsrm

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Authenticate with Google Earth Engine
earthengine authenticate
```

## Quick Start

### 1. Download Data

```python
# Run data download script
python data_download.py
```

This will:
- Download Sentinel-1 (10m SAR imagery)
- Download Sentinel-2 (10m optical imagery)
- Download Landsat-8/9 or Sentinel-2 as reference (free alternative to PlanetScope)
- Generate water masks using NDWI
- Save metadata for training

**Customize training sites** by editing `prepare_sample_training_sites()` in `data_download.py`:

```python
sites = [
    {
        'name': 'Your_Site_Name',
        'roi': ee.Geometry.Rectangle([lon_min, lat_min, lon_max, lat_max]),
        'start_date': '2023-06-01',
        'end_date': '2023-08-31'
    },
    # Add more sites...
]
```

### 2. Train the Model

```bash
# Train with default parameters (scale factor 4)
python train.py --data_dir ./deeposwsrm_data --epochs 100

# Train with scale factor 2
python train.py --data_dir ./deeposwsrm_data --scale_factor 2 --epochs 100

# Train with scale factor 6
python train.py --data_dir ./deeposwsrm_data --scale_factor 6 --epochs 100 --batch_size 4

# Train with custom parameters
python train.py \
    --data_dir ./deeposwsrm_data \
    --scale_factor 4 \
    --batch_size 8 \
    --epochs 100 \
    --learning_rate 1e-4 \
    --base_channels 64 \
    --patch_size 64
```

#### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--scale_factor` | 4 | Super-resolution scale factor (2, 4, or 6) |
| `--batch_size` | 8 | Batch size for training |
| `--epochs` | 100 | Number of training epochs |
| `--learning_rate` | 1e-4 | Initial learning rate |
| `--base_channels` | 64 | Base number of channels in network |
| `--patch_size` | 64 | Patch size for coarse resolution |
| `--lambda_weight` | 1.0 | Weight for SRM loss in combined loss |
| `--eta` | -0.5 | Eta parameter for adaptive loss |

### 3. Monitor Training

```bash
# Launch TensorBoard
tensorboard --logdir ./outputs
```

Open browser at `http://localhost:6006` to view:
- Training/validation loss curves
- Accuracy and IoU metrics
- Learning rate schedule

### 4. Generate Water Maps

```bash
# Generate water map from trained model
python inference.py \
    --checkpoint ./outputs/scale4_<timestamp>/checkpoints/best.pth \
    --sentinel1 path/to/sentinel1.tif \
    --sentinel2 path/to/sentinel2.tif \
    --output path/to/output_water_map.tif \
    --visualize
```

This will generate:
- `output_water_map.tif`: Binary water map at fine resolution
- `output_water_map_fraction.tif`: Water fraction at coarse resolution
- `output_water_map_probability.tif`: Water probability at fine resolution
- `output_water_map_visualization.png`: Visualization (if --visualize flag used)

## Usage in Google Colab

```python
# 1. Install and setup
!pip install -q earthengine-api geemap rasterio albumentations tensorboard

import ee
ee.Authenticate()
ee.Initialize()

# 2. Upload code files to Colab
# Or clone from GitHub

# 3. Download data
!python data_download.py

# 4. Train model
!python train.py \
    --data_dir ./deeposwsrm_data \
    --scale_factor 4 \
    --epochs 50 \
    --batch_size 4

# 5. Run inference
!python inference.py \
    --checkpoint ./outputs/scale4_<timestamp>/checkpoints/best.pth \
    --sentinel1 ./test_data/sentinel1.tif \
    --sentinel2 ./test_data/sentinel2.tif \
    --output ./results/water_map.tif \
    --visualize

# 6. Display results
from IPython.display import Image
Image('./results/water_map_visualization.png')
```

## Model Architecture Details

### Loss Function

The combined loss function consists of:

1. **Fraction Loss (L_frac)**: MSE loss for water fraction estimation
   ```
   L_frac = MSE(predicted_fraction, target_fraction)
   ```

2. **SRM Loss (L_SRM)**: Adaptive fraction-based cross-entropy loss
   ```
   L_SRM = -Σ [exp(η*f) * (m*log(p) + (1-m)*log(1-p))]
   ```
   where η=-0.5, f is water fraction, m is target, p is prediction

3. **Total Loss**:
   ```
   L_total = L_frac + λ * L_SRM
   ```
   where λ=1.0 (configurable)

### Network Components

- **Stacked Residual CNN**: 5 residual blocks for feature extraction
- **Spatial-Channel Attention**: Enhances important features
- **Multi-scale Feature Fusion**: Combines features from multiple decoder levels
- **Skip Connections**: Preserves spatial information

## Data Format

### Input Data Structure
```
deeposwsrm_data/
├── metadata.json
├── Site1/
│   ├── Site1_S1.tif          # Sentinel-1 (2 bands: VV, VH)
│   ├── Site1_S2.tif          # Sentinel-2 (4 bands: B, G, R, NIR)
│   ├── Site1_reference.tif   # Reference image
│   └── Site1_water_mask.tif  # Binary water mask
├── Site2/
│   └── ...
└── Site3/
    └── ...
```

### Image Specifications

| Data Type | Resolution | Bands | Format |
|-----------|-----------|-------|--------|
| Sentinel-1 | 10m | VV, VH | Float32 |
| Sentinel-2 | 10m | Blue, Green, Red, NIR | Float32 |
| Reference | 10m or finer | RGB + NIR | Float32 |
| Water Mask | Scale × 10m | Binary | Uint8 |

## Advanced Usage

### Custom Dataset

```python
from dataset import DeepOSWSRMDataset
from torch.utils.data import DataLoader

# Create custom dataset
dataset = DeepOSWSRMDataset(
    data_dir='./my_data',
    patch_size=64,
    scale_factor=4,
    cloud_coverage_range=(0.3, 0.8),
    augment=True
)

# Create dataloader
dataloader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=True,
    num_workers=4
)
```

### Model Fine-tuning

```python
from deeposwsrm_model import DeepOSWSRM
import torch

# Load pretrained model
checkpoint = torch.load('path/to/checkpoint.pth')
model = DeepOSWSRM(scale_factor=4)
model.load_state_dict(checkpoint['model_state_dict'])

# Freeze unmixing module, fine-tune SRM module only
for param in model.unmixing.parameters():
    param.requires_grad = False

# Continue training
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-5
)
```

### Batch Processing

```python
from inference import WaterMapper
from pathlib import Path

# Load model
mapper = WaterMapper('path/to/checkpoint.pth')

# Process multiple sites
input_dir = Path('./input_images')
output_dir = Path('./output_maps')

for s1_file in input_dir.glob('*_S1.tif'):
    site_name = s1_file.stem.replace('_S1', '')
    s2_file = input_dir / f'{site_name}_S2.tif'
    output_file = output_dir / f'{site_name}_water_map.tif'
    
    if s2_file.exists():
        mapper.predict_from_files(
            sentinel1_path=str(s1_file),
            sentinel2_path=str(s2_file),
            output_path=str(output_file)
        )
```

## Troubleshooting

### Google Earth Engine Authentication

If you encounter authentication errors:

```bash
# In terminal
earthengine authenticate

# Or in Python
import ee
ee.Authenticate()
```

### CUDA Out of Memory

Reduce batch size or patch size:

```bash
python train.py --batch_size 4 --patch_size 32
```

### Low Accuracy

1. **Increase training data**: Download more diverse sites
2. **Adjust loss weights**: Try `--lambda_weight 2.0`
3. **Longer training**: Use `--epochs 200`
4. **Learning rate**: Try `--learning_rate 5e-5`

### Data Quality Issues

Ensure:
- Sentinel-1 and Sentinel-2 have same extent and resolution
- Images are from similar dates (< 7 days apart)
- Cloud-free Sentinel-2 images for training
- Reference data has good water/land contrast

## Performance Benchmarks

Based on paper results:

| Scale Factor | Overall Accuracy | MIOU | Training Time (100 epochs) |
|--------------|-----------------|------|---------------------------|
| 2x | 0.989 | 0.975 | ~4 hours (V100) |
| 4x | 0.987 | 0.971 | ~6 hours (V100) |
| 6x | 0.985 | 0.965 | ~8 hours (V100) |

## Citation

If you use this implementation, please cite:

```bibtex
@article{yin2024deeposwsrm,
  title={Super-resolution water body mapping with a feature collaborative CNN model by fusing Sentinel-1 and Sentinel-2 images},
  author={Yin, Zhixiang and Wu, Penghai and Li, Xinyan and Hao, Zhen and Ma, Xiaoshuang and Fan, Ruirui and Liu, Chun and Ling, Feng},
  journal={International Journal of Applied Earth Observation and Geoinformation},
  volume={134},
  pages={104176},
  year={2024},
  publisher={Elsevier}
}
```

## License

This implementation is provided for research purposes. Please refer to the original paper for methodology details.

## Contact

For questions or issues, please open an issue on GitHub or contact the authors of the original paper.

## Acknowledgments

- Original paper authors: Zhixiang Yin, Penghai Wu, et al.
- Google Earth Engine for satellite data access
- PyTorch team for the deep learning framework
