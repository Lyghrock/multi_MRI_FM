# 1️⃣ 数据预处理与表征设计（Data Preprocessing）

> **文档目标**：定义 sMRI、fMRI、dMRI 三种 MRI 模态的预处理流程与输出表征
> **核心输出**：统一到 Schaefer atlas + MNI space，为后续 Transformer encoder 提供标准输入

---

## Atlas 选择配置

### 可选 Atlas 颗粒度

本项目支持灵活的 Atlas 选择，可在配置中自选分区颗粒度：

| Atlas | ROI 数量 | Networks | 适用场景 |
|-------|---------|----------|----------|
| **Schaefer-200** | 200 | 7 | 快速实验、对比研究 |
| **Schaefer-400** | 400 | 7 | 精细分区、高分辨率需求 |
| **Schaefer-600** | 600 | 7 | 极细粒度、详细分析 |

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Atlas 选择建议                                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Schaefer-200 (默认):                                                   │
│  ───────────────────────────────────────────────────────────────────  │
│  • 快速实验/消融实验                                                    │
│  • 计算资源有限                                                        │
│  • 验证概念可行性                                                      │
│                                                                         │
│  Schaefer-400:                                                         │
│  ───────────────────────────────────────────────────────────────────  │
│  • 平衡分辨率与计算成本                                                │
│  • 适合下游任务需要更精细分区                                          │
│  • Ablation 实验                                                      │
│                                                                         │
│  Schaefer-600:                                                         │
│  ───────────────────────────────────────────────────────────────────  │
│  • 高分辨率需求                                                        │
│  • 长期训练资源充足                                                    │
│  • 研究 ROI 粒度对表示的影响                                          │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Atlas 配置示例

```yaml
# config 里面自选
atlas:
  name: 'Schaefer'           # 固定 Schaefer 系列
  n_rois: 400                # 可选: 200, 400, 600
  n_networks: 7              # 7 或 17 networks
  space: 'MNI152_2mm'       # 固定 MNI152 2mm

# 对应的 Atlas 下载
# Schaefer-200:
#   Schaefer2018_200Parcels_7Networks_order_FSLMNI152_2mm.nii.gz
# Schaefer-400:
#   Schaefer2018_400Parcels_7Networks_order_FSLMNI152_2mm.nii.gz
# Schaefer-600:
#   Schaefer2018_600Parcels_7Networks_order_FSLMNI152_2mm.nii.gz
```

---

## 预处理四阶段

| 阶段 | 定义 | 处理内容 | 输出 |
|------|------|----------|------|
| **Raw** | 原始采集数据 | nii.gz 格式 | 存档基准 |
| **Cleaned** | 清洗后数据 | 运动校正、去噪、配准到 MNI | 统一空间 |
| **Atlas-parceled** | Atlas 划分 | Schaefer-{N} parcellation | ROI masks |
| **Metric-extracted** | 特征提取 | 各模态专用 metric | 标准输入 |

**关键约束**：所有模态在 `Cleaned` 阶段完成后必须对齐到 **MNI152 2mm** 空间。

**Atlas 约束**：统一使用 Schaefer 系列 atlas (200/400/600 可选)，确保 ROI 语义、索引、拓扑一致。

---

## 1.1 sMRI 处理流程（结构磁共振）

### 处理流水线

```
Raw sMRI (.nii.gz)
        │
        ▼
STAGE 1: CLEANED
───────────────────────────────────────────────────────────────────────
所有清洗完成，配准到 MNI152 2mm

Step 1.1: 偏置场校正 (N4BiasFieldCorrection)
Step 1.2: 脑提取 (ANTs brain extraction)
Step 1.3: 非线性配准到 MNI152 2mm (ANTs)

输出: smri_cleaned.nii.gz ∈ R^(182, 218, 182) @ 2mm MNI
        │
        ▼
STAGE 2: VBM ANALYSIS
───────────────────────────────────────────────────────────────────────
产生 voxel-level gray matter map

Step 2.1: 组织分割 (SPM New Segment / ANTs Atropos)
          ├─ Gray Matter (GM) probability map
          ├─ White Matter (WM) probability map
          └─ CSF probability map

Step 2.2: 调制 (Modulation) - 保留绝对密度

输出:
  smri_gm.nii.gz ∈ R^(182, 218, 182) @ 2mm MNI
  shape: (182, 218, 182) = 6.9M voxels
        │
        ▼
STAGE 3: ROI METRIC EXTRACTION
───────────────────────────────────────────────────────────────────────
基于 Schaefer-{N} 提取 ROI-level 特征

Step 3.1: Schaefer-{N} parcellation in MNI space
          └─ 加载 Schaefer2018_{N}Parcels_{M}Networks_order.nii.gz

Step 3.2: ROI-level 特征提取
          对每个 ROI 聚合 voxel values:
          ├─ mean intensity
          ├─ GM volume (sum of GM probability)
          └─ max intensity

输出:
  smri_roi_features ∈ R^(B × N_ROI × 3)
  维度说明: (batch, ROI, features=[mean, volume, max])
```

### sMRI 最终输出

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    sMRI 输出规格                                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  输出1: VBM Gray Matter Volume                                        │
│  ───────────────────────────────────────────────────────────────────  │
│  用途: voxel-level structural representation (用于 voxel-patch branch)│
│  shape: (182, 218, 182) @ 2mm MNI                                   │
│  type: float32, range ~ [0, 1] (probability)                         │
│                                                                         │
│  输出2: ROI-level Structural Features                                 │
│  ───────────────────────────────────────────────────────────────────  │
│  用途: ROI-level structural representation (用于 ROI-token branch)   │
│  shape: (N_ROI, 3)                                                  │
│  features: [GM_mean, GM_volume, GM_max]                              │
│  type: float32                                                        │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 1.2 fMRI 处理流程（功能磁共振）

### 处理流水线

```
Raw fMRI (.nii.gz, 4D time-series)
        │
        ▼
STAGE 1: CLEANED
───────────────────────────────────────────────────────────────────────
时间校正 + 配准到 MNI152 2mm

Step 1.1: Slice-time correction (AFNI 3dTshift)
Step 1.2: Motion correction (MCFLIRT / 3dvolreg)
          └─ 计算 motion parameters: FMC_realignment.par
Step 1.3: Despiking (AFNI 3dDespike)
Step 1.4: 配准到 MNI152 2mm (ANTs / FLIRT)
Step 1.5: Spatial smoothing (FWHM=6mm, Gaussian)
Step 1.6: Temporal filtering (0.01-0.08 Hz Butterworth)
Step 1.7: Global signal regression (可选)
Step 1.8: WM/CSF covariate regression

输出: fmri_cleaned.nii.gz ∈ R^(182, 218, 182, T) @ 2mm MNI
        │
        ▼
STAGE 2: ROI TIME-SERIES EXTRACTION
───────────────────────────────────────────────────────────────────────
基于 Schaefer-{N} 提取 ROI 时间序列

Step 2.1: Schaefer-{N} parcellation in MNI space

Step 2.2: Extract mean time-series per ROI
          └─ 对每个 ROI mask内的 voxels 取平均

输出:
  fmri_roi_timeseries ∈ R^(B × T × N_ROI)
  维度: (batch, time_points, ROI_Schaefer{N})
        │
        ▼
STAGE 3: DYNAMIC FUNCTIONAL CONNECTIVITY
───────────────────────────────────────────────────────────────────────
计算滑动窗口动态功能连接矩阵

Step 3.1: Sliding Window
          ├─ Window length: 30-60 TR (典型 ~30-40s)
          ├─ Window step: 1 TR (或 half-overlap)
          └─ Number of windows: W = floor((T - window_len) / step) + 1

Step 3.2: Window-wise Correlation
          对每个窗口内的 ROI time-series 计算 Pearson correlation
          └─ Fisher z-transform 归一化

输出:
  fmri_dfc ∈ R^(B × W × N_ROI × N_ROI)
  维度: (batch, windows, ROI, ROI)
  type: symmetric matrix, diagonal=1
```

### fMRI 最终输出

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    fMRI 输出规格                                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  输出1: ROI BOLD Time-Series                                          │
│  ───────────────────────────────────────────────────────────────────  │
│  用途: primary functional signal (用于 temporal ROI-token branch)       │
│  shape: (T, N_ROI)                                                    │
│  T: number of time-points (取决于扫描协议)                            │
│  type: float32, z-scored                                             │
│                                                                         │
│  输出2: Dynamic Functional Connectivity (dFC)                         │
│  ───────────────────────────────────────────────────────────────────  │
│  用途: temporal graph representation (用于 dFC graph-token branch)    │
│  shape: (W, N_ROI, N_ROI)                                           │
│  W: number of sliding windows                                        │
│  type: float32, Fisher-z transformed                                  │
│  注意: 使用 dFC 而非 static FC，保留时间动态性                        │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 1.3 dMRI 处理流程（扩散磁共振）

### 处理流水线

```
Raw dMRI (.nii.gz, 4D with multiple b-values/shells)
        │
        ▼
STAGE 1: CLEANED & PREPROCESSED
───────────────────────────────────────────────────────────────────────
扩散预处理完成

Step 1.1: Denoising (MRtrix3 dwidenoise)
Step 1.2: Gibbs ringing removal (MRtrix3 mrdegibbs)
Step 1.3: Eddy current correction (MRtrix3 dwifslpreproc)
          └─ 包括 motion correction
Step 1.4: Bias field correction (ANTs N4)
Step 1.5: Brain extraction
Step 1.6: Multi-shell to MNI registration (ANTs)

输出: dwi_preprocessed.nii.gz ∈ R^(X, Y, Z, N_shells) @ 2mm MNI
        │
        ▼
STAGE 2: TENSOR FITTING & FA MAP
───────────────────────────────────────────────────────────────────────
计算 Fractional Anisotropy (FA) map

Step 2.1: DTI tensor fitting (MRtrix3 dwi2tensor)
          └─ Fit: S = S0 * exp(-b*g'D*g)

Step 2.2: Compute FA map (MRtrix3 tensor2metric)
          └─ FA = sqrt(0.5 * ((λ1-λ2)² + (λ2-λ3)² + (λ3-λ1)²) / (λ1²+λ2²+λ3²))

Step 2.3: 配准 FA to MNI152 2mm (ANTs)

输出:
  dmri_fa.nii.gz ∈ R^(182, 218, 182) @ 2mm MNI
  shape: (182, 218, 182) = 6.9M voxels
  type: float32, range [0, 1]
        │
        ▼
STAGE 3: TRACTOGRAPHY & SC MATRIX
───────────────────────────────────────────────────────────────────────
白质纤维追踪 + 结构连接矩阵构建

Step 3.1: Response function estimation
          └─ MRtrix3 dwi2response (tournier algorithm)

Step 3.2: Fiber Orientation Distribution (FOD)
          └─ MRtrix3 dwi2fod msmt_csd

Step 3.3: Global Tracking (iFOD2)
          ├─ 10M streamlines (可调整)
          ├─ Angular threshold: 45°
          ├─ Step size: 0.5mm
          └─ Length constraints: 10-250mm

Step 3.4: SIFT2 Filtering
          └─ MRtrix3 tckmap + sift2 -term_ratio 0.1
          └─ 减少假阳性 streamline，提高定量准确性

Step 3.5: Parcellation with Schaefer-{N}
          └─ Assign each streamline endpoint to ROI

Step 3.6: Streamline Count Normalization
          ├─ 行归一化: divide by row sum
          └─ Log transform: log(1 + count)

Step 3.7: Symmetrize + Zero diagonal

输出:
  dmri_sc ∈ R^(B × N_ROI × N_ROI)
  维度: (batch, ROI_Schaefer{N}, ROI_Schaefer{N})
  type: symmetric matrix, diagonal=0, normalized log-count
```

### dMRI 最终输出

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    dMRI 输出规格                                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  输出1: FA Map (Fractional Anisotropy)                               │
│  ───────────────────────────────────────────────────────────────────  │
│  用途: voxel-level structural connectivity signal (用于 voxel-patch)  │
│  shape: (182, 218, 182) @ 2mm MNI                                   │
│  type: float32, range [0, 1]                                         │
│  说明: FA 反映水分子扩散各向异性，与白质完整性高度相关                 │
│                                                                         │
│  输出2: Structural Connectivity Matrix                                │
│  ───────────────────────────────────────────────────────────────────  │
│  用途: structural graph representation (用于 SC graph-token branch)   │
│  shape: (N_ROI, N_ROI)                                                │
│  type: symmetric, diagonal=0, log-normalized streamline count        │
│  说明: 统一 Schaefer-{N} parcellation，与其他模态一致                │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 1.4 Patch 设计与 Scale 平衡

### 为什么需要 Patch？

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    为什么要 Patch？                                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  原始数据维度差异巨大：                                                 │
│  ────────────────────────────────────────────────────────────────────  │
│                                                                         │
│  sMRI:  VBM (182,218,182) → 需要 patch 成小立方体                    │
│  fMRI:  Time-series (T, N_ROI)  → 需要 time-patch                   │
│  dMRI:  FA (182,218,182) + SC (N_ROI,N_ROI)  → voxel-patch + SC-patch  │
│                                                                         │
│  ────────────────────────────────────────────────────────────────────  │
│                                                                         │
│  Patch 的好处：                                                        │
│  ✓ 增加数据多样性（数据增强）                                          │
│  ✓ 控制模型输入维度一致                                                │
│  ✓ 减少显存占用                                                        │
│  ✓ 与 ViT 的 patch embedding 兼容                                       │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Patch 设计原则

```
目标：三种模态 patch 后的 scale 大致相同

sMRI:
  ├── VBM voxel-patch: (182,218,182) → 多个 (32,32,32) patches
  └── ROI feature: (N_ROI, 3) → 直接作为 token

fMRI:
  ├── ROI time-series: (T, N_ROI) → time-patch 成 (T_patch, N_ROI)
  ├── Static FC: (N_ROI, N_ROI) → patch 成 (P, P) patches, P=10-20
  └── dFC (W, N_ROI, N_ROI): 先 FC-patch 再 time-patch

dMRI:
  ├── FA voxel-patch: (182,218,182) → 多个 (32,32,32) patches
  └── SC matrix: (N_ROI, N_ROI) → patch 成 (P, P) patches, P=10-20
```

### Scale 平衡策略

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Scale 平衡设计                                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  设计原则：让 patch 后的 token 数量在合理范围                            │
│                                                                         │
│  ────────────────────────────────────────────────────────────────────  │
│                                                                         │
│  sMRI (Metric):                                                        │
│    VBM patch: 每个 patch 是 (32,32,32) = 32K voxels                   │
│    ROI feature: (N_ROI, 3) 作为 N_ROI 个 tokens                       │
│    总计: N_voxel_patches + N_ROI tokens                                │
│                                                                         │
│  ────────────────────────────────────────────────────────────────────  │
│                                                                         │
│  fMRI (Metric):                                                         │
│    Time-patch: 将 (T, N_ROI) 按时间切分                               │
│    每个 time-patch: (T_patch, N_ROI), T_patch 根据数据量决定            │
│    FC-patch: 将 (N_ROI,N_ROI) 切成 (P,P) patches                     │
│    最终: 多个 time-windows × FC-patches                                 │
│                                                                         │
│  ────────────────────────────────────────────────────────────────────  │
│                                                                         │
│  dMRI (Metric):                                                        │
│    FA voxel-patch: 同 sMRI                                              │
│    SC-patch: 将 (N_ROI,N_ROI) 切成 (P,P) patches                     │
│    最终: N_voxel_patches + N_SC_patches                                 │
│                                                                         │
│  ────────────────────────────────────────────────────────────────────  │
│                                                                         │
│  Scale 对齐目标:                                                       │
│    sMRI: ~N_ROI-400 tokens (voxel patches + ROI features)               │
│    fMRI: ~N_ROI-600 tokens (time-windows × FC patches)                  │
│    dMRI: ~N_ROI-400 tokens (voxel patches + SC patches)                  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 具体 Patch 参数

```yaml
patch_config:
  sMRI:
    voxel_patch_size: [32, 32, 32]    # 3D patch 大小
    voxel_patch_stride: [16, 16, 16]  # 3D patch 步长
    roi_feature_dim: 3                # [mean, volume, max]

  fMRI:
    time_patch_length: 50             # 每个 time-patch 的时间步数
    time_patch_stride: 25             # 时间步长 (50% overlap)
    fc_patch_size: 20                 # FC matrix patch 大小
    fc_patch_stride: 10               # FC patch 步长

  dMRI:
    voxel_patch_size: [32, 32, 32]    # 同 sMRI
    voxel_patch_stride: [16, 16, 16]
    sc_patch_size: 20                 # SC matrix patch 大小
    sc_patch_stride: 10
```

---

## 1.5 统一输出规格

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    所有模态统一输出                                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  空间: MNI152 2mm isotropic                                          │
│  Atlas: Schaefer-{N} (7-Networks)                                   │
│  ROI ordering: Schaefer2018_{N}Parcels_7Networks_order.csv           │
│                                                                         │
│  ───────────────────────────────────────────────────────────────────  │
│                                                                         │
│  sMRI:                                                                 │
│    ├─ VBM voxel-patches: List[(32,32,32)]                           │
│    └─ ROI Features: (N_ROI, 3) [mean, volume, max]                  │
│                                                                         │
│  fMRI:                                                                 │
│    ├─ Time-patched ROI TS: List[(T_patch, N_ROI)]                   │
│    └─ FC-patches: List[(P, P)] 或 dFC: List[(T_patch, P, P)]        │
│                                                                         │
│  dMRI:                                                                 │
│    ├─ FA voxel-patches: List[(32,32,32)]                            │
│    └─ SC-patches: List[(P, P)]                                      │
│                                                                         │
│  ───────────────────────────────────────────────────────────────────  │
│                                                                         │
│  数据类型: float16 (节省存储和显存)                                     │
│  存储格式: .pt (PyTorch native format)                                │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 1.6 处理工具链

| 步骤 | 工具 | 用途 |
|------|------|------|
| **sMRI VBM** | SPM12 / ANTs | 组织分割、调制 |
| **fMRI** | FSL / AFNI / SPM | 时间校正、运动校正、滤波 |
| **dMRI** | MRtrix3 / FSL | 扩散处理、张量拟合、追踪 |
| **配准** | ANTs | 非线性配准到 MNI |
| **Atlas** | nilearn /brainspace | Schaefer-{N} parcellation |

---

**下一步**：→ [02_encoder_fusion.md](./02_encoder_fusion.md) - Transformer encoder + tokenization 设计
