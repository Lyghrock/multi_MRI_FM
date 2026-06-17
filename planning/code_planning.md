# Code Implementation Planning

> **目标**: Multi-Modal Brain Foundation Model 代码实现规划
> **运行方式**: torchrun 分布式训练 (DDP)

---

## 1. 项目结构

```
multi_MRI/
│
├── README.md                    # 项目说明
├── requirements.txt             # 依赖
├── config/                      # 配置文件
│   ├── default.yaml            # 默认配置
│   ├── mae_base.yaml           # MAE 配置
│   ├── jepa_base.yaml          # JEPA 配置
│   ├── dino_base.yaml           # DINO 配置
│   └── train_local.yaml        # 本地调试
│
├── data/                        # 数据处理
│   ├── atlas.py                # Schaefer-{N} atlas
│   ├── brain_dataset.py         # Dataset
│   ├── data_loader.py           # DataLoader
│   └── preprocess*.py           # 预处理
│
├── model/                       # Foundation Model
│   ├── backbone/               # Transformer, PE
│   ├── encoders/               # sMRI/fMRI/dMRI encoders
│   ├── fusion/                 # Latent Hub
│   └── foundation_model.py      # 主模型
│
├── baseline/                    # Training Methods
│   ├── MAE/                   # MAE
│   ├── JEPA/                  # JEPA
│   └── DINO/                  # DINO
│
├── utils/                      # 工具函数
├── train.py                    # 训练入口
└── scripts/                    # 运行脚本
```

---

## 2. Atlas 配置

### 2.1 可选 Atlas 颗粒度

```yaml
# config/default.yaml
atlas:
  name: 'Schaefer'           # 固定 Schaefer 系列
  n_rois: ${ATLAS_N_ROIS}   # 可选: 200, 400, 600
  n_networks: 7
  space: 'MNI152_2mm'
```

### 2.2 Atlas 下载

| Atlas | 文件名 | ROI |
|-------|--------|-----|
| Schaefer-200 | Schaefer2018_200Parcels_7Networks_order_FSLMNI152_2mm.nii.gz | 200 |
| Schaefer-400 | Schaefer2018_400Parcels_7Networks_order_FSLMNI152_2mm.nii.gz | 400 |
| Schaefer-600 | Schaefer2018_600Parcels_7Networks_order_FSLMNI152_2mm.nii.gz | 600 |

---

## 3. MAE Training

### 3.1 模块结构

```
baseline/MAE/
├── __init__.py
├── decoders/
│   ├── __init__.py
│   ├── smri_decoder.py      # sMRI: H → Linear(256) → Linear(3)
│   ├── fmri_decoder.py       # fMRI: H → Linear(256) → Linear(d)
│   └── dmri_decoder.py       # dMRI: H → Linear(512) → Linear(N²)
├── mae_model.py              # MultiModalMAE
├── mae_loss.py               # MSE + VICReg
└── mae_trainer.py            # 训练循环
```

### 3.2 核心组件

| 组件 | 结构 | 说明 |
|------|------|------|
| **sMRI Decoder** | MLP | H → Linear(d, 256) → Linear(256, N_ROI×3) |
| **fMRI Decoder** | MLP | H → Linear(d, 256) → Linear(256, d) |
| **dMRI Decoder** | MLP | H → Linear(d, 512) → Linear(512, N²_upper) |
| **Masking** | High-Ratio | 固定 75% |
| **Loss** | Multi-Task MSE + VICReg | Σλ_i·L_i + λ_vicreg·L_vicreg |

### 3.3 配置示例

```yaml
# config/mae_base.yaml
mae:
  mask_ratio: 0.75
  lambda_sMRI: 1.0
  lambda_fMRI: 1.0
  lambda_dMRI: 1.0
  lambda_vicreg: 0.5
```

---

## 4. JEPA Training

### 4.1 模块结构

```
baseline/JEPA/
├── __init__.py
├── predictor/
│   ├── __init__.py
│   ├── mlp_predictor.py         # MLP: Linear(d, 4d) → GELU → Linear(4d, d)
│   ├── transformer_predictor.py  # Transformer: 2 layers, 4 heads
│   └── moe_predictor.py         # MoE: Mixture-of-Experts
├── masking.py                   # Curriculum Masking (Intra/Inter 可选)
├── target_encoder.py            # EMA Target Encoder
├── jepa_model.py                # MultiModalJEPA
├── jepa_loss.py                # Soft-L1 + VICReg
└── jepa_trainer.py             # 训练循环
```

### 4.2 Intra-Mask vs Inter-Mask

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Masking 策略对比                                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Intra-Mask (模态内掩码):                                               │
│  ───────────────────────────────────────────────────────────────────  │
│  • 在单个模态的表示内进行掩码                                           │
│  • 每个模态独立 masking                                                │
│  • 保留跨模态对应关系                                                  │
│  • 适合: 同一受试者的多模态数据                                       │
│                                                                         │
│  Inter-Mask (跨模态掩码):                                               │
│  ───────────────────────────────────────────────────────────────────  │
│  • 随机丢弃整个模态的全部表示                                          │
│  • 强迫模型学习跨模态预测                                              │
│  • 需要更强的表示能力                                                  │
│  • Ablation: 对比 Intra vs Inter 效果差异                              │
│                                                                         │
│  配置选项:                                                             │
│  ───────────────────────────────────────────────────────────────────  │
│  jepa:                                                                 │
│    mask_type: 'intra'       # 'intra', 'inter', 'both'             │
│    intra_mask_ratio: 0.3     # 模态内掩码比例                         │
│    inter_mask_ratio: 0.2     # 跨模态掩码比例                         │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4.3 Predictor 类型

#### 4.3.1 MLP Predictor

```
结构: Linear(d, 4d) → GELU → Linear(4d, d)
轻量，适合快速实验
```

#### 4.3.2 Transformer Predictor

```
结构: Transformer Encoder (2 layers, 4 heads)
可捕获 token 间关系
```

#### 4.3.3 MoE Predictor (推荐)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    MoE Predictor - Mixture of Experts                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  核心思想:                                                            │
│  ───────────────────────────────────────────────────────────────────  │
│  • 多个 Expert 网络，每个擅长处理不同类型的表示                        │
│  • Router 网络学习选择哪个 Expert                                      │
│  • 避免单个 Predictor 的表示瓶颈                                      │
│                                                                         │
│  结构:                                                                │
│  ───────────────────────────────────────────────────────────────────  │
│  H → Router → 选择 Top-K Experts → 加权输出                           │
│                                                                         │
│  配置:                                                                │
│  ───────────────────────────────────────────────────────────────────  │
│  n_experts: 8              # Expert 数量                              │
│  top_k: 2                  # 每次选择 2 个 Expert                    │
│  d_expert: 4d              # 每个 Expert 隐藏维度                    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4.4 核心组件

| 组件 | 选项 | 说明 |
|------|------|------|
| **Predictor** | MLP / Transformer / MoE | 预测网络 |
| **Target Encoder** | EMA | β=0.99→0.999 |
| **Masking** | Intra / Inter / Both | Curriculum 0%→30% |
| **Loss** | Soft-L1 + VICReg | λ_vicreg=0.25→0.5 |

### 4.5 配置示例

```yaml
# config/jepa_base.yaml
jepa:
  predictor_type: 'moe'        # 'mlp', 'transformer', 'moe'

  # MoE 配置
  n_experts: 8
  top_k: 2
  expert_hidden_dim: 1024

  # EMA
  ema_beta: 0.99
  ema_update_every: 1

  # Masking
  mask_type: 'both'           # 'intra', 'inter', 'both'
  intra_mask_ratio: 0.3        # 模态内掩码
  inter_mask_ratio: 0.2        # 跨模态掩码

  # Curriculum
  mask_warmup: 0.0            # Stage 1: 0%
  mask_alignment: 0.15         # Stage 2: 0%→15%
  mask_refinement: 0.30        # Stage 3: 30%

  # Loss
  lambda_vicreg: 0.5
```

---

## 5. DINO Training

### 5.1 模块结构

```
baseline/DINO/
├── __init__.py
├── multi_crop.py              # Multi-Crop Transform
├── center.py                  # Center 机制
├── sam.py                     # SAM Optimizer
├── dino_model.py              # MultiModalDINO
├── dino_loss.py               # Cross-Entropy
└── dino_trainer.py            # 训练循环
```

### 5.2 核心组件

| 组件 | 说明 |
|------|------|
| **Student-Teacher** | Foundation Model 共享 backbone |
| **EMA** | β=0.996，更慢的更新 |
| **Multi-Crop** | 2 Global + 8 Local |
| **Center** | 防止 collapse |
| **Loss** | Cross-Entropy |

### 5.3 配置示例

```yaml
# config/dino_base.yaml
dino:
  ema_beta: 0.996
  center_momentum: 0.9

  # Multi-Crop
  n_crops_global: 2
  n_crops_local: 8
  size_global: 200
  size_local: 64

  # Temperature
  temperature_student: 0.1
  temperature_teacher: 0.04

  # SAM (可选)
  use_sam: true
  sam_rho: 0.05
```

---

## 6. 共用模块

### 6.1 Foundation Model

```
model/
├── backbone/
│   ├── transformer.py           # Transformer Encoder
│   └── positional_encoding.py   # 3D/2D Sinusoidal, Anatomical
├── encoders/
│   ├── sMRI_encoder.py         # ROI + Voxel Patch encoder
│   ├── fMRI_encoder.py         # Temporal + FC encoder
│   └── dMRI_encoder.py         # Voxel Patch + SC encoder
├── fusion/
│   └── latent_hub.py           # Cross-Attention Fusion
└── foundation_model.py          # 完整模型
```

### 6.2 Data Processing

```
data/
├── atlas.py                    # Schaefer-{N} 处理
├── brain_dataset.py             # BrainMRIDataset
├── data_loader.py               # DDP DataLoader
└── preprocess*.py              # 各模态预处理
```

### 6.3 Utils

```
utils/
├── distributed.py              # DDP 工具
├── checkpoint.py               # 断点保存/加载
├── logger.py                  # TensorBoard 日志
├── metrics.py                 # 评估指标
└── config_utils.py            # 配置加载
```

---

## 7. 训练入口

### 7.1 统一入口 (train.py)

```python
# train.py - 根据 exp_type 选择训练方法
import argparse
parser.add_argument('--exp_type', type=str,
    choices=['MAE-METRIC', 'MAE-RAW', 'JEPA-METRIC', 'JEPA-RAW', 'DINO-METRIC', 'DINO-RAW'])
```

### 7.2 运行命令

```bash
# MAE
torchrun --nproc_per_node=4 train.py --config config/mae_base.yaml --exp_type MAE-METRIC

# JEPA
torchrun --nproc_per_node=4 train.py --config config/jepa_base.yaml --exp_type JEPA-METRIC

# DINO
torchrun --nproc_per_node=4 train.py --config config/dino_base.yaml --exp_type DINO-METRIC
```

---

## 8. 依赖

```
torch>=2.0.0
torchvision>=0.15.0
nibabel>=5.0.0
nilearn>=0.10.0
scipy>=1.10.0
numpy>=1.24.0
pyyaml>=6.0
tensorboard>=2.13.0
```

---

## 9. 实现顺序

```
Phase 1: 基础设施
├── model/ 模块 (Foundation Model)
└── data/ 模块 (Dataset, Atlas)

Phase 2: Training Methods
├── baseline/MAE/ (MAE 训练)
├── baseline/JEPA/ (JEPA 训练)
└── baseline/DINO/ (DINO 训练)

Phase 3: 整合
├── train.py (统一入口)
└── 配置文件
```

---

## 10. Ablation 实验设计

### 10.1 Atlas 粒度

| 实验 | Atlas | 说明 |
|------|-------|------|
| Atlas-200 | Schaefer-200 | 基准 |
| Atlas-400 | Schaefer-400 | 精细分区 |
| Atlas-600 | Schaefer-600 | 极细粒度 |

### 10.2 Masking 类型 (JEPA)

| 实验 | Mask Type | 说明 |
|------|-----------|------|
| Mask-Intra | Intra | 模态内掩码 |
| Mask-Inter | Inter | 跨模态掩码 |
| Mask-Both | Both | 两者结合 |

### 10.3 Predictor 类型 (JEPA)

| 实验 | Predictor | 说明 |
|------|----------|------|
| Pred-MLP | MLP | 轻量基准 |
| Pred-Trans | Transformer | 捕获序列关系 |
| Pred-MoE | MoE | 多专家混合 |

---

**文档版本**: v2.0
**更新日期**: 2026-05-23
**项目**: Multi-Modal Brain Foundation Model
