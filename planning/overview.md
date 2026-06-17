# Multi-Modal Brain Foundation Model

> **架构**: Transformer-based Encoder + Unified Schaefer Atlas + Simplified Latent Hub Fusion
> **设计原则**: Foundation Model 固定结构，JEPA/MAE/DINO 是 Training Guide，用于训练这个 Foundation Model

---

## 1. 核心概念

### 1.1 Foundation Model vs Training Guide

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Foundation Model vs Training Guide                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Foundation Model (固定结构)                                           │
│  ───────────────────────────────────────────────────────────────────  │
│                                                                         │
│  • Data: Schaefer-{N} + MNI152 (N 可选: 200/400/600)             │
│  • Encoder: Transformer (固定)                                        │
│  • Fusion: Latent Hub (固定)                                          │
│  • Output: Brain Semantic Representation H                            │
│                                                                         │
│  → 这是真正用于接上下游任务的部分                                    │
│  → 输出的是模型认为的"好表示"                                       │
│                                                                         │
│  ───────────────────────────────────────────────────────────────────  │
│                                                                         │
│  Training Guide (可替换)                                              │
│  ───────────────────────────────────────────────────────────────────  │
│                                                                         │
│  • JEPA: Predictor + EMA Target + Soft-L1                          │
│  • MAE: 模态专用 Decoder + MSE                                      │
│  • DINO: Student-Teacher Self-Distillation                          │
│                                                                         │
│  → 这是用来训练 Foundation Model 的方法                              │
│  → 训练完后可以移除                                                  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 1.2 整体架构

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Complete Architecture                                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │  FOUNDATION MODEL (固定)                                         │ │
│  │  ──────────────────────────────────────────────────────────── │ │
│  │                                                                   │ │
│  │  Input: sMRI, fMRI, dMRI                                       │ │
│  │       │                                                         │ │
│  │       ▼                                                         │ │
│  │  Modality Encoders (Transformer)                                 │ │
│  │       │                                                         │ │
│  │       ▼                                                         │ │
│  │  Shared Latent Hub                                              │ │
│  │       │                                                         │ │
│  │       ▼                                                         │ │
│  │  H ∈ R^(B×64×d)  (Shared Brain Representation)               │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                                │                                        │
│                                ▼                                        │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │  TRAINING GUIDE (可替换)                                       │ │
│  │  ──────────────────────────────────────────────────────────── │ │
│  │                                                                   │ │
│  │  Option 1: JEPA                                                │ │
│  │  ├─ Predictor: MLP / Transformer / MoE                      │ │
│  │  ├─ Target Encoder: EMA(foundation)                          │ │
│  │  ├─ Loss: Soft-L1 + VICReg                                  │ │
│  │  └─ Masking: Intra / Inter / Both (Curriculum 0%→30%)     │ │
│  │                                                                   │ │
│  │  Option 2: MAE                                                 │ │
│  │  ├─ Decoder: sMRI/fMRI/dMRI (简单 MLP)                       │ │
│  │  ├─ Loss: MSE + VICReg                                       │ │
│  │  └─ Masking: High-Ratio (75%)                               │ │
│  │                                                                   │ │
│  │  Option 3: DINO                                                │ │
│  │  ├─ Student-Teacher: Self-Distillation                       │ │
│  │  ├─ EMA Teacher: β=0.996                                     │ │
│  │  ├─ Loss: Cross-Entropy                                      │ │
│  │  └─ Multi-Crop: 2 Global + 8 Local                          │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Foundation Model（固定）

### 2.1 Atlas 选择

| Atlas | ROI 数量 | 适用场景 |
|-------|---------|----------|
| **Schaefer-200** | 200 | 快速实验、基准 |
| **Schaefer-400** | 400 | 平衡分辨率与成本 |
| **Schaefer-600** | 600 | 高分辨率需求 |

### 2.2 Data Preprocessing

| 模态 | 输入 | 输出 | Shape |
|------|------|------|-------|
| **sMRI** | Raw nii.gz | VBM GM + ROI features | (182,218,182), (N_ROI,3) |
| **fMRI** | Raw nii.gz | Time-series + dFC | (T,N_ROI), (W,N_ROI,N_ROI) |
| **dMRI** | Raw nii.gz | FA + SC matrix | (182,218,182), (N_ROI,N_ROI) |

### 2.3 Tokenization

| 输入 | Token 类型 | N tokens | PE 类型 |
|------|-----------|----------|---------|
| VBM/FA volumes | Voxel Patches | ~15K | 3D Sinusoidal |
| ROI features | ROI Tokens | N_ROI | Anatomical Coord |
| ROI time-series | Temporal ROI | N_windows | Time + ROI |
| dFC/SC matrices | FC/SC Patches | N_ROI² | 2D Anatomical |

### 2.4 Encoder + Fusion

```
Encoder: Transformer (d_model=256, n_heads=8, n_layers=4)
Fusion: Cross-Attention Latent Hub (64 slots)
Output: H ∈ R^(B×64×d)
```

---

## 3. Training Guide 设计

### 3.1 JEPA Training Guide

| Component | 选项 | 说明 |
|----------|------|------|
| **Predictor** | MLP / Transformer / MoE | 预测网络 |
| **Target Encoder** | EMA | β=0.99→0.999 |
| **Masking** | Intra / Inter / Both | Curriculum 0%→30% |
| **Loss** | Soft-L1 + VICReg | λ_vicreg=0.25→0.5 |

**Masking 策略**:
- **Intra-Mask**: 模态内掩码，保留跨模态对应
- **Inter-Mask**: 跨模态掩码，强迫跨模态预测
- **Both**: 两者结合

**MoE Predictor**: Mixture-of-Experts，多专家混合表示

### 3.2 MAE Training Guide

| Component | 内容 | 说明 |
|----------|------|------|
| **sMRI Decoder** | MLP | H → Linear(256) → Linear(3) |
| **fMRI Decoder** | MLP | H → Linear(256) → Linear(d) |
| **dMRI Decoder** | MLP | H → Linear(512) → Linear(N²) |
| **Masking** | High-Ratio | 固定 75% |
| **Loss** | Multi-Task MSE + VICReg | Σλ_i·L_i |

### 3.3 DINO Training Guide

| Component | 内容 | 说明 |
|----------|------|------|
| **Architecture** | Student-Teacher | Foundation Model 作为 backbone |
| **Teacher** | EMA | β=0.996，更慢的 EMA |
| **Student** | 有梯度更新 | 与 Teacher 共享架构 |
| **Multi-Crop** | 2 Global + 8 Local | Global 给 Teacher，Local 给 Student |
| **Loss** | Cross-Entropy | Student 匹配 Teacher 的 soft label |
| **Center** | 可学习参数 | 防止 collapse |

### 3.4 JEPA vs MAE vs DINO 对比

| | JEPA | MAE | DINO |
|---|---|---|---|
| **预测目标** | latent representation | 原始输入 | probability distribution |
| **Target 来源** | EMA encoder output | 原始像素 | EMA teacher softmax |
| **额外网络** | Predictor (MLP/Trans/MoE) | Decoders (模态专用) | 无 |
| **Masking** | Intra/Inter/Both (0→30%) | High-Ratio (75%) | Multi-Crop (Global+Local) |
| **Loss** | Soft-L1 + VICReg | MSE + VICReg | Cross-Entropy |
| **EMA** | 有 | 无 | 有 |

---

## 4. Ablation 实验设计

### 4.1 Atlas 粒度

| 实验 | Atlas | 说明 |
|------|-------|------|
| Atlas-200 | Schaefer-200 | 基准 |
| Atlas-400 | Schaefer-400 | 精细分区 |
| Atlas-600 | Schaefer-600 | 极细粒度 |

### 4.2 Masking 类型 (JEPA)

| 实验 | Mask Type | 说明 |
|------|-----------|------|
| Mask-Intra | Intra | 模态内掩码 |
| Mask-Inter | Inter | 跨模态掩码 |
| Mask-Both | Both | 两者结合 |

### 4.3 Predictor 类型 (JEPA)

| 实验 | Predictor | 说明 |
|------|----------|------|
| Pred-MLP | MLP | 轻量基准 |
| Pred-Trans | Transformer | 捕获序列关系 |
| Pred-MoE | MoE | 多专家混合 |

---

## 5. 文档结构

```
multi_MRI/planning/
│
├── 01_data_preprocessing.md        ← 预处理流程（Atlas 可选）
├── 02_encoder_fusion.md            ← Encoder + Fusion
├── 03_JEPA_training_design.md       ← JEPA Training Guide
│   ├── Predictor (MLP/Trans/MoE)
│   ├── Target Encoder (EMA)
│   ├── Masking (Intra/Inter/Both)
│   └── Loss Function
│
├── 04_MAE_training_design.md        ← MAE Training Guide
├── 05_DINO_training_design.md       ← DINO Training Guide
├── code_planning.md                 ← 代码实现规划（合并）
└── overview.md                      ← 本文件
```

---

**文档版本**: v8.0
**创建日期**: 2026-05-07
**更新日期**: 2026-05-23
**项目**: Multi-Modal Brain Foundation Model
