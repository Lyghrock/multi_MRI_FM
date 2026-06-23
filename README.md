# Multi-Modal Brain Foundation Model

A PyTorch implementation of a multi-modal brain foundation model using structural MRI (sMRI), functional MRI (fMRI), and diffusion MRI (dMRI) data.

## Features

- **Foundation Model**: Shared encoder architecture for multi-modal brain MRI
- **Training Methods**: Support for MAE, JEPA, and DINO training paradigms
- **Distributed Training**: Full support for torchrun-based DDP training
- **Flexible Architecture**: Modular design with separate encoders and fusion

## Training Methods

This project supports three self-supervised training methods:

| Method | Description | Key Features |
|--------|-------------|--------------|
| **MAE** | Masked Autoencoder | Modality-specific decoders, 75% masking |
| **JEPA** | Joint Embedding Predictive Architecture | MLP/Transformer/MoE predictor, EMA, Intra/Inter masking |
| **DINO** | Self-Distillation with No Labels | Student-Teacher, Multi-Crop, Center mechanism |

### JEPA Options

**Predictor Types**:
- `mlp`: Lightweight MLP predictor
- `transformer`: Transformer-based predictor for capturing token relationships
- `moe`: Mixture-of-Experts predictor (recommended for richer representations)

**Masking Strategies** (Ablation-supported):
- `intra`: Intra-modality masking (preserve cross-modal correspondence)
- `inter`: Inter-modality masking (force cross-modal prediction)
- `both`: Combine both strategies

### Which to Choose?

- **MAE**: Good for learning reconstructive representations
- **JEPA**: Best for semantic representations, avoids pixel-level reconstruction
- **DINO**: Simple and stable, no decoder/predictor needed

## Atlas Dependencies

### Schaefer Atlas (Configurable)

This project supports flexible **Schaefer atlas parcellation** with configurable ROI granularity.

**Available Atlas Options**:

| Atlas | ROI | File | Use Case |
|-------|-----|------|----------|
| Schaefer-200 | 200 | Schaefer2018_200Parcels_7Networks_order_FSLMNI152_2mm.nii.gz | Fast experiments, baseline |
| Schaefer-400 | 400 | Schaefer2018_400Parcels_7Networks_order_FSLMNI152_2mm.nii.gz | Fine-grained analysis |
| Schaefer-600 | 600 | Schaefer2018_600Parcels_7Networks_order_FSLMNI152_2mm.nii.gz | High resolution |

**Configuration**:
```yaml
# In config file
atlas:
  n_rois: 400  # Change to 200, 400, or 600
```

**Download**:
```bash
# Schaefer-200 (default)
wget https://raw.githubusercontent.com/ThomasYeoLab/CBIG/master/stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/Parcellations/MNI/Schaefer2018_200Parcels_7Networks_order_FSLMNI152_2mm.nii.gz

# Schaefer-400
wget https://raw.githubusercontent.com/ThomasYeoLab/CBIG/master/stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/Parcellations/MNI/Schaefer2018_400Parcels_7Networks_order_FSLMNI152_2mm.nii.gz

# Schaefer-600
wget https://raw.githubusercontent.com/ThomasYeoLab/CBIG/master/stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/Parcellations/MNI/Schaefer2018_600Parcels_7Networks_order_FSLMNI152_2mm.nii.gz
```

Or use the built-in download function:
```python
from data.atlas import download_schaefer_atlas
atlas_path = download_schaefer_atlas(n_rois=400, n_networks=7, output_dir='./atlas')
```

**What is it?**: A 3D NIfTI image where each voxel contains an ROI index. It defines brain regions based on functional networks.

## Project Structure

```
multi_MRI/
|-- model/                   # Foundation model
|   |-- backbone/            # Transformer blocks + positional encoding
|   |-- encoders/            # sMRI / fMRI / dMRI encoders
|   |-- fusion/              # Latent hub fusion layers
|   `-- foundation_model.py  # Main multi-modal encoder
|
|-- baseline/
|   |-- MAE/                 # MAE wrappers, loss, decoders, trainer
|   |-- JEPA/                # JEPA wrappers, masking, predictor, trainer
|   `-- DINO/                # DINO wrappers, loss, trainer
|
|-- data/
|   |-- atlas.py             # Schaefer atlas helpers
|   |-- brain_dataset.py     # Dataset definitions
|   |-- data_loader.py       # DataLoader utilities
|   `-- preprocess.py        # Preprocessing entry point
|
|-- utils/                   # Config, checkpoint, logging, metrics, batching
|-- config/
|   `-- default.yaml         # Default training configuration
|-- planning/                # Design notes and implementation plans
|-- requirements.txt
`-- train.py                 # Training entry point
```

## Installation

```bash
pip install -r requirements.txt
```

### Requirements

```
torch>=2.0.0
numpy>=1.24.0
scipy>=1.10.0
nibabel>=5.0.0     # For NIfTI file handling
pyyaml>=6.0
tensorboard>=2.13.0
tqdm>=4.65.0
```

## Quick Start

### 1. Download Atlas

```bash
wget https://raw.githubusercontent.com/ThomasYeoLab/CBIG/master/stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/Parcellations/MNI/Schaefer2018_200Parcels_7Networks_order_FSLMNI152_2mm.nii.gz -O Schaefer2018_200Parcels_7Networks_order_FSLMNI152_2mm.nii.gz
```

### 2. Preprocess Data

Output files are saved as `.pt` (float16) with patches for efficient torch loading.

```bash
# Sequential (single process)
python -m data.preprocess \
    --input_dir /path/to/raw/data/train \
    --output_dir ./processed_data \
    --mode metric \
    --atlas_path ./Schaefer2018_200Parcels_7Networks_order_FSLMNI152_2mm.nii.gz

# Parallel (multi-process, recommended for large datasets)
python -m data.preprocess \
    --input_dir /path/to/raw/data/train \
    --output_dir ./processed_data \
    --mode metric \
    --num_workers 4
```

**Output format (metric mode with patches):**
```
processed_data/
└── metric/
    └── train/
        # sMRI
        ├── sub-001_sMRI_patches.pt   # (N, 32, 32, 32) voxel patches
        ├── sub-001_sMRI_roi.pt       # (200, 3) ROI features

        # fMRI
        ├── sub-001_fMRI_time_patches.pt  # (N_time, T_patch, 200) time patches
        ├── sub-001_fMRI_fc_patches.pt    # (N_fc, P, P) FC patches

        # dMRI
        ├── sub-001_dMRI_FA_patches.pt    # (N, 32, 32, 32) FA patches
        ├── sub-001_dMRI_MD_patches.pt    # (N, 32, 32, 32) MD patches
        └── sub-001_dMRI_SC_placeholder.pt # (200, 200) SC placeholder
```

Note: Data preprocessing is CPU-bound (I/O + numpy), so use Python multiprocessing (`--num_workers`) rather than torchrun.

### 3. Train

```bash
# MAE
python train.py --config config/default.yaml --exp_type MAE-METRIC
torchrun --nproc_per_node=4 train.py --config config/default.yaml --exp_type MAE-METRIC

# JEPA
torchrun --nproc_per_node=4 train.py --config config/default.yaml --exp_type JEPA-METRIC

# DINO
torchrun --nproc_per_node=4 train.py --config config/default.yaml --exp_type DINO-METRIC
```

## Experiment Types

Six experiment configurations supported:

| Experiment | Training | Data Type | Data Path |
|------------|----------|-----------|-----------|
| `MAE-METRIC` | MAE | Atlas-based (ROI) | `data/metric/` |
| `MAE-RAW` | MAE | Voxel-based | `data/raw/` |
| `JEPA-METRIC` | JEPA | Atlas-based (ROI) | `data/metric/` |
| `JEPA-RAW` | JEPA | Voxel-based | `data/raw/` |
| `DINO-METRIC` | DINO | Atlas-based (ROI) | `data/metric/` |
| `DINO-RAW` | DINO | Voxel-based | `data/raw/` |

## Data Format

### Preprocessed Data (.pt, float16)

After running preprocessing, data is saved as:

```
processed_data/
├── metric/
│   └── train/                    # Or val/, test/
│       ├── sub-001_sMRI.pt      # (200, 3) ROI features [mean, vol, max]
│       ├── sub-001_fMRI.pt      # (T, 200) time-series
│       ├── sub-001_FC.pt        # (200, 200) functional connectivity
│       └── sub-001_dMRI.pt      # (200, 200) diffusion/FA
└── raw/
    └── train/
        └── ...
```

### Raw Data (Input)

```
raw_data/
├── train/                       # Or val/, test/
│   ├── sub-001_T1w.nii.gz      # sMRI
│   ├── sub-001_bold.nii.gz     # fMRI
│   └── sub-001_FA.nii.gz       # dMRI
└── ...
```

### Training Data Path

When training, specify the parent directory containing `metric/` and `raw/`:

```bash
torchrun --nproc_per_node=4 train.py \
    --data_root ./processed_data \
    --exp_type MAE-METRIC
```

This will automatically look for `./processed_data/metric/train/`.

## Output Structure

Checkpoints and logs are saved by experiment type and seed:

```
outputs/
├── MAE-METRIC/
│   └── seed_42/
│       ├── checkpoints/
│       │   ├── checkpoint_epoch_0001.pt
│       │   └── best_model.pt
│       └── logs/
│           ├── train.log
│           └── tensorboard/
├── MAE-RAW/
│   └── seed_42/
└── ...
```

## Architecture

```
Input: sMRI + fMRI + dMRI
         │
         ▼
┌─────────────────────────────────────┐
│  Modality Encoders (Transformer)    │
│  ├─ sMRI Encoder                   │
│  ├─ fMRI Encoder                   │
│  └─ dMRI Encoder                   │
└─────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│  Shared Latent Hub                  │
│  (Cross-Attention Fusion)            │
└─────────────────────────────────────┘
         │
         ▼
    H ∈ R^(B×64×d)
         │
         ▼
┌─────────────────────────────────────┐
│  Training Head (Select One)          │
│  ├─ MAE: Modality Decoders        │
│  ├─ JEPA: Predictor + EMA         │
│  │     (MLP / Transformer / MoE)   │
│  └─ DINO: Student-Teacher          │
└─────────────────────────────────────┘
```

### Training Methods

- **MAE**: Masked Autoencoder with modality-specific decoders
- **JEPA**: Joint Embedding Predictive Architecture
  - Predictor: MLP / Transformer / MoE
  - Masking: Intra / Inter / Both
- **DINO**: Self-Distillation with Student-Teacher architecture

## License

No `LICENSE` file is currently present in this repository. Add one before
redistributing the code or claiming a specific open-source license.
