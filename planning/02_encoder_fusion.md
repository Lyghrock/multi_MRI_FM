# 2️⃣ Encoder + 多模态融合架构设计

> **文档目标**：定义 Transformer-based encoder + 统一 tokenization + Simplified Latent Hub Fusion  
> **核心思想**：统一 Schaefer-200 atlas 提供 shared geometry，简化 fusion 结构

---

## 2.0 架构变更说明

### 为什么要从 Perceiver 改为 Transformer？

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Backbone 从 Perceiver → Transformer                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Perceiver 的问题：                                                    │
│  ───────────────────────────────────────────────────────────────────  │
│                                                                         │
│  1. Latent bottleneck 限制了表达能力                                    │
│  2. Cross-attention 的 query 是固定的可学习向量                         │
│  3. 难以 scale up 到大规模预训练                                        │
│  4. 不适合后续 patch-token 化                                          │
│                                                                         │
│  Transformer 的优势：                                                  │
│  ───────────────────────────────────────────────────────────────────  │
│                                                                         │
│  1. 更强的表达能力 + 成熟的 scaling 策略                              │
│  2. 更适合处理序列化的 ROI/graph tokens                               │
│  3. 更易兼容 foundation model 预训练范式                              │
│  4. 更适合 modality-specific token sequence                           │
│  5. 生态好：ViT/Swin/DINO 等成熟方案可借鉴                          │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 为什么 Schaefer-200 统一是关键？

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Same-Atlas 是 Simplified Fusion 的前提                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  问题：不同 atlas 下的 cross-attention                                │
│  ───────────────────────────────────────────────────────────────────  │
│                                                                         │
│  不同 atlas → 不同 ROI semantics → 不同 geometry → PE mismatch          │
│                                                                         │
│  Q = W_Q(x + PE_q)  ← ROI_A 的 PE                                  │
│  K = W_K(x + PE_k)  ← ROI_B 的 PE (不同!)                          │
│                                                                         │
│  → Attention score 不稳定 → alignment 失效                           │
│                                                                         │
│  ───────────────────────────────────────────────────────────────────  │
│                                                                         │
│  解决方案：Same Atlas                                                  │
│  ───────────────────────────────────────────────────────────────────  │
│                                                                         │
│  Schaefer-200 统一后：                                                 │
│  ✓ ROI token index 一致                                               │
│  ✓ ROI semantics 一致                                                 │
│  ✓ Positional encoding 一致                                           │
│  ✓ Cross-attention 在 shared geometry 上运作                          │
│  ✓ 无需复杂的 ontology translation                                    │
│                                                                         │
│  结论：same-atlas + simplified fusion = stable + efficient            │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2.1 整体架构

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Complete Architecture                                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐ │
│  │  INPUTS                                                             │ │
│  │  ──────────────────────────────────────────────────────────────── │ │
│  │                                                                    │ │
│  │  sMRI:                                                             │ │
│  │    ├─ VBM (182,218,182) @ 2mm  →  voxel patches → tokens        │ │
│  │    └─ ROI features (200,3)    →  ROI tokens                   │ │
│  │                                                                    │ │
│  │  fMRI:                                                             │ │
│  │    ├─ ROI time-series (T,200)  →  temporal ROI tokens           │ │
│  │    └─ dFC (W,200,200)     →  FC patch tokens                │ │
│  │                                                                    │ │
│  │  dMRI:                                                             │ │
│  │    ├─ FA map (182,218,182) @ 2mm → voxel patches → tokens        │ │
│  │    └─ SC matrix (200,200)   →  SC patch tokens                │ │
│  │                                                                    │ │
│  └──────────────────────────────────────────────────────────────────┘ │
│                                │                                        │
│                                ▼                                        │
│  ┌──────────────────────────────────────────────────────────────────┐ │
│  │  MODALITY-SPECIFIC TRANSFORMER ENCODERS                          │ │
│  │  ──────────────────────────────────────────────────────────────── │ │
│  │                                                                    │ │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐                 │ │
│  │  │ sMRI       │  │ fMRI       │  │ dMRI       │                 │ │
│  │  │ Transformer│  │ Transformer│  │ Transformer│                 │ │
│  │  │ Encoder    │  │ Encoder    │  │ Encoder    │                 │ │
│  │  │            │  │            │  │            │                 │ │
│  │  │ 200→256    │  │ T×200→256 │  │ N_patches→256│                 │ │
│  │  │ tokens     │  │ tokens     │  │ patches    │                 │ │
│  │  └─────┬──────┘  └─────┬──────┘  └─────┬──────┘                 │ │
│  │        │               │               │                        │ │
│  │        └───────────────┴───────────────┘                        │ │
│  │                       │                                        │ │
│  └───────────────────────┼────────────────────────────────────────┘ │
│                          │                                            │
│                          ▼                                            │
│  ┌──────────────────────────────────────────────────────────────────┐ │
│  │  SIMPLIFIED SHARED LATENT HUB                                  │ │
│  │  ──────────────────────────────────────────────────────────────── │ │
│  │                                                                    │ │
│  │  E_s, E_f, E_d ∈ R^(B×N×d) ──► Cross-Attention Fusion        │ │
│  │                                           │                       │ │
│  │                                           ▼                       │ │
│  │                                    H ∈ R^(B×64×d)               │ │
│  │                                                                    │ │
│  └──────────────────────────────────────────────────────────────────┘ │
│                                │                                        │
│                                ▼                                        │
│  ┌──────────────────────────────────────────────────────────────────┐ │
│  │  JEPA TRAINING                                                  │ │
│  │  ──────────────────────────────────────────────────────────────── │ │
│  │                                                                    │ │
│  │  H ──► Context Encoder ──► C ──► Predictor ──► pred           │ │
│  │                                        │                          │ │
│  │       Target (EMA) ◄──────────────────┘                          │ │
│  │                                                                    │ │
│  │  Loss: L_total = L_JEPA + λ·L_VICReg                          │ │
│  │                                                                    │ │
│  └──────────────────────────────────────────────────────────────────┘ │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2.2 Tokenization 策略

### 核心原则：Brain ROI 不是自然序列

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    为什么 Brain ROI 不是自然序列？                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  NLP Sequence:                                                       │
│  ─────────────────────────────────────────────────────────────────── │
│  "The cat sat on the mat"                                            │
│   0    1    2    3    4     5                                      │
│                                                                         │
│  序列有天然顺序：word[i] 紧跟 word[i-1]                              │
│  位置编码 encode 的是"谁在谁旁边"                                     │
│                                                                         │
│  Brain ROI:                                                          │
│  ─────────────────────────────────────────────────────────────────── │
│  Schaefer-200 ROI #1, #2, #3, ..., #200                             │
│                                                                         │
│  ROI[1] 和 ROI[2] 不一定有任何物理/功能邻近关系                       │
│  位置编码 encode 的不是"物理顺序"，而是"拓扑关系"                     │
│                                                                         │
│  ───────────────────────────────────────────────────────────────────  │
│                                                                         │
│  结论：Naive absolute PE 不适用                                       │
│                                                                         │
│  解决方案：                                                           │
│  • 使用 anatomical topology (基于 MNI coordinates) 作为 PE            │
│  • 或使用 learned relative PE                                         │
│  • 或使用 graph structure as PE (Yeo 7-network membership)           │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.2.1 sMRI Tokenization

#### 分支1: Voxel Patch Tokens (VBM Volume)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    sMRI Voxel Patch Tokenization                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Input: VBM Gray Matter Volume ∈ R^(182, 218, 182) @ 2mm MNI         │
│                                                                         │
│  Patchify:                                                           │
│  ───────────────────────────────────────────────────────────────────  │
│                                                                         │
│  3D volume ──► 3D patches (e.g., 16×16×16 voxels)                   │
│                                                                         │
│  patch_size = 16 (在 2mm MNI 空间 = 32mm 物理尺寸)                  │
│  stride = 8 (50% overlap, 可调)                                       │
│                                                                         │
│  计算:                                                                │
│    N_patches = ceil(182/8) × ceil(218/8) × ceil(182/8)             │
│            ≈ 23 × 28 × 23 ≈ 14,800 patches                          │
│                                                                         │
│  每个 patch:                                                          │
│    p = Flatten(16×16×16) = 4096 dims                                 │
│    → Linear(4096, d_patch) = 256 dims (可调)                        │
│                                                                         │
│  输出:                                                                │
│    sMRI_patch_tokens ∈ R^(B × N_patches × d_patch)                   │
│    其中 N_patches ≈ 14,800                                            │
│                                                                         │
│  Positional Encoding:                                                │
│  ───────────────────────────────────────────────────────────────────  │
│  • 使用 3D sin/cos PE (与 Vision Transformer 相同)                   │
│  • 或使用 patch center coordinates as learnable PE                    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

#### 分支2: ROI Tokens (ROI Features)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    sMRI ROI Tokenization                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Input: ROI-level Structural Features ∈ R^(200 × 3)                   │
│         features: [GM_mean, GM_volume, GM_max]                        │
│                                                                         │
│  Tokenization:                                                       │
│  ───────────────────────────────────────────────────────────────────  │
│                                                                         │
│  每个 ROI → 一个 token:                                               │
│                                                                         │
│    token_i = Linear(features_i, d_roi) ∈ R^d_roi                    │
│    d_roi = 256 (可调)                                                │
│                                                                         │
│  输出:                                                                │
│    sMRI_roi_tokens ∈ R^(B × 200 × d_roi)                            │
│                                                                         │
│  Positional Encoding:                                                │
│  ───────────────────────────────────────────────────────────────────  │
│                                                                         │
│  ROI_index PE (learned):                                              │
│    PE ∈ R^(200 × d_roi)                                             │
│    token_i = token_i + PE[i]                                         │
│                                                                         │
│  注意: 每个 ROI 有一个固定的 index PE，与解剖位置无关                  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.2.2 fMRI Tokenization

#### 分支1: Temporal ROI Tokens (Time-Series)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    fMRI Temporal ROI Tokenization                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Input: ROI BOLD Time-Series ∈ R^(T × 200)                            │
│         T: number of time-points                                       │
│                                                                         │
│  方案: Temporal Patching                                               │
│  ───────────────────────────────────────────────────────────────────  │
│                                                                         │
│  将时间序列划分为 windows，每个 window 内对 ROI 取 mean:                │
│                                                                         │
│    window_size = 30 (TRs, 约 30-45秒)                                 │
│    stride = 15 (50% overlap)                                          │
│    N_windows = ceil(T / stride)                                       │
│                                                                         │
│  对每个 window:                                                        │
│    window_t = fmri[t:t+window_size, :] ∈ R^(window_size × 200)       │
│    window_t_mean = Mean(window_t, dim=0) ∈ R^200                     │
│    → Linear(200, d_temporal) = 256 dims                              │
│                                                                         │
│  输出:                                                                │
│    fMRI_temporal_tokens ∈ R^(B × N_windows × d_temporal)             │
│                                                                         │
│  Positional Encoding:                                                │
│  ───────────────────────────────────────────────────────────────────  │
│  • Time position PE: sin/cos based on window index                   │
│  • ROI index PE: 保持与 Schaefer-200 一致的 ROI ordering             │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

#### 分支2: FC Patch Tokens (Dynamic FC Matrix)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    FC Matrix Patch Tokenization                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Input: Dynamic FC ∈ R^(W × N × N)                                    │
│         W: number of sliding windows, N=200 (Schaefer-200)              │
│                                                                         │
│  问题: 直接用 200×200 矩阵 tokens 效率低                               │
│  ───────────────────────────────────────────────────────────────────  │
│                                                                         │
│  解决方案: FC Patchification                                            │
│  ───────────────────────────────────────────────────────────────────  │
│                                                                         │
│  将 FC 矩阵划分为 patches:                                              │
│                                                                         │
│    patch_size = 20  (20×20 patch)                                     │
│    stride = 10  (50% overlap)                                          │
│    N_patches_per_window = ceil(200/10) × ceil(200/10) = 20×20 = 400  │
│                                                                         │
│  每个 FC patch:                                                         │
│    fc_patch ∈ R^(20 × 20) → Flatten → 400 dims                       │
│    → Linear(400, d_patch) → 256 dims                                   │
│                                                                         │
│  输出:                                                                  │
│    fMRI_fc_tokens ∈ R^(B × W × N_patches × d_patch)                        │
│    = R^(B × W × N_patches × d_patch)                                 │
│                                                                         │
│  注意: 时间维度 W 也作为 sequence 处理                                 │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.2.3 dMRI Tokenization

#### 分支1: Voxel Patch Tokens (FA Map)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    dMRI FA Voxel Patch Tokenization                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Input: FA Map ∈ R^(182, 218, 182) @ 2mm MNI                         │
│                                                                         │
│  Patchify: (与 sMRI VBM 相同流程)                                     │
│  ───────────────────────────────────────────────────────────────────  │
│                                                                         │
│  3D volume ──► 3D patches (16×16×16 voxels)                          │
│  N_patches ≈ 14,800 patches                                           │
│  每个 patch → Linear(4096, d_patch)                                   │
│                                                                         │
│  输出:                                                                │
│    dMRI_patch_tokens ∈ R^(B × N_patches × d_patch)                   │
│                                                                         │
│  注意: 与 sMRI VBM 可以共享 patch tokenizer 权重                      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

#### 分支2: SC Patch Tokens (Structural Connectivity Matrix)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    SC Matrix Patch Tokenization                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Input: SC Matrix ∈ R^(N × N)                                        │
│         N=200 (Schaefer-200)                                           │
│         symmetric, diagonal=0, log-normalized streamline count          │
│                                                                         │
│  解决方案: SC Patchification                                            │
│  ───────────────────────────────────────────────────────────────────  │
│                                                                         │
│  将 SC 矩阵划分为 patches:                                              │
│                                                                         │
│    patch_size = 20  (20×20 patch)                                     │
│    stride = 10  (50% overlap)                                          │
│    N_patches = ceil(200/10) × ceil(200/10) = 20×20 = 400            │
│                                                                         │
│  每个 SC patch:                                                        │
│    sc_patch ∈ R^(20 × 20) → Flatten → 400 dims                       │
│    → Linear(400, d_patch) → 256 dims                                   │
│                                                                         │
│  输出:                                                                  │
│    dMRI_sc_tokens ∈ R^(B × N_patches × d_patch)                            │
│    = R^(B × N_patches × d_patch)                                    │
│                                                                         │
│  3D Positional Encoding for FC/SC Patches:                            │
│  ───────────────────────────────────────────────────────────────────  │
│                                                                         │
│  FC/SC patch 的位置也是有物理意义的:                                    │
│                                                                         │
│    patch_row = i * stride + patch_size/2  (ROI index → MNI coord)     │
│    patch_col = j * stride + patch_size/2  (ROI index → MNI coord)     │
│    patch_center = (patch_row, patch_col) ∈ R^2                       │
│                                                                         │
│  使用 2D sinusoidal PE 或 Anatomical PE:                               │
│    PE[patch_id] = AnatomicalCoord(patch_center)                       │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2.3 Transformer Encoder 架构

### 2.3.1 Modality-Specific Encoder

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Modality Transformer Encoder                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Input: modality-specific tokens ∈ R^(B × N × d_in)                   │
│                                                                         │
│  Architecture:                                                         │
│  ───────────────────────────────────────────────────────────────────  │
│                                                                         │
│    tokens ──► Input Projection (Linear) ──► d_model                   │
│         │                                                            │
│         ├──► Transformer Blocks (N_layers)                            │
│         │                                                            │
│         │    ┌──────────────────────────────────────────────┐        │
│         │    │                                              │        │
│         │    │  LayerNorm ──► Multi-Head Self-Attention     │        │
│         │    │      │            │                           │        │
│         │    │      │            ├──► Residual              │        │
│         │    │      ▼                                   │        │
│         │    │  LayerNorm ──► FFN                         │        │
│         │    │      │            │                           │        │
│         │    │      │            ├──► Residual              │        │
│         │    │      ▼                                   │        │
│         │    └──────────────────────────────────────────────┘        │
│         │                                                            │
│         └──► Output Projection (Linear) ──► d_out                   │
│                                                                         │
│  配置 (默认):                                                          │
│  ───────────────────────────────────────────────────────────────────  │
│  • N_layers: 4-6                                                      │
│  • d_model: 256                                                      │
│  • n_heads: 8                                                       │
│  • d_ffn: 1024 (4× d_model)                                         │
│  • dropout: 0.1                                                     │
│  • activation: GELU                                                 │
│                                                                         │
│  输出:                                                                │
│    encoded_tokens ∈ R^(B × N × d_out)                                │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.3.2 ROI Token Transformer (用于所有 ROI-Level 输入)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Shared ROI Token Transformer                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  设计: 所有 ROI-level 输入 (sMRI_roi, fMRI_temporal, fMRI_dFC, dMRI_sc)│
│       使用相同的 Transformer 架构                                       │
│                                                                         │
│  输入:                                                                │
│    tokens ∈ R^(B × 200 × d_in)  (N=200 是固定的 Schaefer-200)        │
│                                                                         │
│  参数:                                                                │
│    • 使用 shared architecture, 可以共享权重或独立                      │
│    • 推荐: 独立权重 (modality-specific adaptation)                    │
│                                                                         │
│  输出:                                                                │
│    encoded_roi ∈ R^(B × 200 × d_model)                               │
│                                                                         │
│  关键点:                                                              │
│  ───────────────────────────────────────────────────────────────────  │
│  • N=200 是固定的，因为 Schaefer-200 是统一的                        │
│  • 每个模态独立编码，但输出 shape 相同                                 │
│  • 便于后续 fusion                                                    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2.4 Positional Encoding 策略

### 为什么 Naive Absolute PE 不适用于 Brain ROI？

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Positional Encoding 设计                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  问题: Brain ROI 不是自然序列                                          │
│  ───────────────────────────────────────────────────────────────────  │
│                                                                         │
│  NLP: "The cat" → position 1,2 有物理顺序意义                        │
│  Brain ROI: "ROI #1, #2" → index 1,2 无物理顺序意义                  │
│                                                                         │
│  ROI #1 和 ROI #200 可能是任意顺序 (由 atlas 决定)                    │
│  因此 index-based PE 没有物理意义                                      │
│                                                                         │
│  ───────────────────────────────────────────────────────────────────  │
│                                                                         │
│  解决方案1: Anatomical Coordinate PE (推荐)                            │
│  ───────────────────────────────────────────────────────────────────  │
│                                                                         │
│  使用 Schaefer-200 每个 ROI 的 MNI centroid coordinates:                │
│                                                                         │
│    coords_i = (x_i, y_i, z_i) ∈ R^3                                 │
│    PE_i = [sin(pos_i/10000^(2j/d)), cos(...)] for j in [0, d/2)     │
│                                                                         │
│  优点: PE encode 了真实解剖位置                                        │
│  缺点: 需要预先存储 coordinates                                        │
│                                                                         │
│  解决方案2: Network Membership PE                                      │
│  ───────────────────────────────────────────────────────────────────  │
│                                                                         │
│  使用 Yeo 7-network membership 作为位置:                               │
│                                                                         │
│    network_i ∈ {1,2,3,4,5,6,7}  (Vis, SomMot, DorsAttn, VentAttn,    │
│                                   Limbic, Cont, Default)              │
│    PE_i = Embedding(network_i, d)                                      │
│                                                                         │
│  优点: encode 功能网络拓扑                                             │
│  缺点: 粗粒度                                                         │
│                                                                         │
│  解决方案3: Learned Relative PE                                       │
│  ───────────────────────────────────────────────────────────────────  │
│                                                                         │
│  使用 relative position bias (如 Swin Transformer):                    │
│                                                                         │
│    Attention = softmax(Q @ K^T / √d + relative_bias)                 │
│    relative_bias 基于 ROI pair 的解剖距离                               │
│                                                                         │
│  优点: 更灵活                                                          │
│  缺点: 需要更多计算                                                    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 推荐 PE 策略

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    推荐的 Positional Encoding 配置                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  对于 Voxel Patches (sMRI VBM, dMRI FA):                              │
│  ───────────────────────────────────────────────────────────────────  │
│  • 3D Sinusoidal PE (与 ViT 相同)                                      │
│  • patch center coordinates                                           │
│                                                                         │
│  对于 ROI Tokens (所有模态):                                           │
│  ───────────────────────────────────────────────────────────────────  │
│  • Anatomical Coordinate PE (MNI centroid)                           │
│  • 或 Network Membership PE + Learnable offset                       │
│  • 两者可 concat                                                     │
│                                                                         │
│  具体实现:                                                             │
│  ───────────────────────────────────────────────────────────────────  │
│                                                                         │
│    # MNI coordinates (200 × 3, 来自 Schaefer atlas)                  │
│    coords = load_schaefer200_mni_coordinates()  # shape: (200, 3)   │
│                                                                         │
│    # Sinusoidal encoding                                              │
│    pe = zeros(200, d_model)                                           │
│    for i in range(200):                                                │
│        for j in range(0, d_model, 2):                                 │
│            pe[i, j] = sin(coords[i, 0] / 10000^(j/d_model))         │
│            pe[i, j+1] = cos(coords[i, 0] / 10000^(j/d_model))       │
│                                                                         │
│    token_i = token_i + pe[i]                                          │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2.5 Simplified Latent Hub Fusion

### 为什么简化 Fusion？

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Simplified Fusion 的理由                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  之前的复杂设计:                                                       │
│  ───────────────────────────────────────────────────────────────────  │
│                                                                         │
│  Iterative latent refinement:                                          │
│  • 需要多层 cross-attention                                           │
│  • 需要 iterative alignment                                          │
│  • 计算量大                                                           │
│  • 优化困难                                                           │
│                                                                         │
│  ───────────────────────────────────────────────────────────────────  │
│                                                                         │
│  新的简化设计:                                                         │
│  ───────────────────────────────────────────────────────────────────  │
│                                                                         │
│  Same-atlas 已经提供:                                                  │
│  ✓ Shared ROI semantics                                               │
│  ✓ Shared token index                                                 │
│  ✓ Shared positional encoding                                          │
│                                                                         │
│  因此只需要 lightweight fusion:                                         │
│  • 单层或双层 cross-attention                                         │
│  • 无需 iterative refinement                                          │
│  • 更容易优化                                                         │
│  • 更高效                                                              │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Fusion 架构

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Simplified Latent Hub Fusion                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  输入:                                                                │
│    E_s ∈ R^(B × 200 × d)  (sMRI encoded)                             │
│    E_f ∈ R^(B × 200 × d)  (fMRI encoded)                             │
│    E_d ∈ R^(B × 200 × d)  (dMRI encoded)                             │
│                                                                         │
│  架构:                                                                │
│  ───────────────────────────────────────────────────────────────────  │
│                                                                         │
│  Step 1: Concatenate                                                  │
│    E_all = concat(E_s, E_f, E_d, dim=1) ∈ R^(B × 1200 × d)          │
│                                                                         │
│  Step 2: Lightweight Cross-Attention Fusion                            │
│    ┌─────────────────────────────────────────────────────────────────┐ │
│    │                                                                  │ │
│    │  # 方法A: Cross-attention with learned queries (Perceiver-style) │ │
│    │  ─────────────────────────────────────────────────────────────── │ │
│    │                                                                  │ │
│    │  Q = learnable_queries ∈ R^(1 × N_hub × d)  (N_hub=64)        │ │
│    │  K = E_all ∈ R^(B × 1200 × d)                                 │ │
│    │  V = E_all ∈ R^(B × 1200 × d)                                 │ │
│    │                                                                  │ │
│    │  H = Attention(Q, K, V) ∈ R^(B × 64 × d)                      │ │
│    │                                                                  │ │
│    └─────────────────────────────────────────────────────────────────┘ │
│                                                                         │
│    ┌─────────────────────────────────────────────────────────────────┐ │
│    │                                                                  │ │
│    │  # 方法B: Simple averaging + projection (更简单)                 │ │
│    │  ─────────────────────────────────────────────────────────────── │ │
│    │                                                                  │ │
│    │  E_mean = (E_s + E_f + E_d) / 3 ∈ R^(B × 200 × d)            │ │
│    │  H = Linear(E_mean) ∈ R^(B × 64 × d)  (project to slots)     │ │
│    │                                                                  │ │
│    └─────────────────────────────────────────────────────────────────┘ │
│                                                                         │
│  输出:                                                                │
│    H ∈ R^(B × 64 × d)  (64 shared latent slots)                     │
│                                                                         │
│  推荐: 方法A (Cross-attention)，表达能力更强                           │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2.6 完整数据流

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Complete Data Flow                                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  sMRI Input:                                                          │
│  ───────────────────────────────────────────────────────────────────  │
│  VBM (182,218,182) ──► Patchify(16³) ──► Patches                     │
│                                          │                             │
│                                          ▼                             │
│                                    3D PE ──► sMRI_patch_tokens         │
│                                          │                             │
│                                          ▼                             │
│                               Transformer Encoder ──► E_s_patch         │
│                                                                         │
│  ROI features (200,3) ──► Linear ──► ROI PE ──► ROI tokens           │
│                                              │                         │
│                                              ▼                         │
│                                   Transformer Encoder ──► E_s_roi      │
│                                                                         │
│  ───────────────────────────────────────────────────────────────────  │
│                                                                         │
│  fMRI Input:                                                          │
│  ───────────────────────────────────────────────────────────────────  │
│  Time-series (T,200) ──► Temporal Patching ──► Windows                │
│                                             │                          │
│                                             ▼                          │
│                                       Time PE ──► fMRI_temp_tokens     │
│                                             │                          │
│                                             ▼                          │
│                                  Transformer Encoder ──► E_f_temp      │
│                                                                         │
│  dFC (W,200,200) ──► FC Patchification ──► FC patches               │
│                                           │                            │
│                                           ▼                            │
│                                    ROI PE ──► fMRI_dFC_tokens          │
│                                           │                            │
│                                           ▼                            │
│                                  Transformer Encoder ──► E_f_dfc        │
│                                                                         │
│  ───────────────────────────────────────────────────────────────────  │
│                                                                         │
│  dMRI Input:                                                          │
│  ───────────────────────────────────────────────────────────────────  │
│  FA (182,218,182) ──► Patchify(16³) ──► Patches                      │
│                                          │                             │
│                                          ▼                             │
│                                    3D PE ──► dMRI_patch_tokens        │
│                                          │                             │
│                                          ▼                             │
│                               Transformer Encoder ──► E_d_patch        │
│                                                                         │
│  SC (200,200) ──► SC Patchification ──► SC patches        │
│                                                     │                   │
│                                                     ▼                   │
│                                        Transformer Encoder ──► E_d_sc │
│                                                                         │
│  ───────────────────────────────────────────────────────────────────  │
│                                                                         │
│  Fusion:                                                              │
│  ───────────────────────────────────────────────────────────────────  │
│  E_s_roi, E_f_temp, E_d_sc                                            │
│          │                                                            │
│          ▼                                                            │
│   Cross-Attention Fusion                                              │
│          │                                                            │
│          ▼                                                            │
│    H ∈ R^(B × 64 × d)                                               │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2.7 参数规格

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    模型参数规格                                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Tokenization:                                                        │
│  ───────────────────────────────────────────────────────────────────  │
│  • sMRI VBM patch: 16³ voxels → 256 dims                            │
│  • sMRI ROI: 3 features → 256 dims                                   │
│  • fMRI temporal: window mean → 256 dims                             │
│  • fMRI dFC node: 4 features → 256 dims                              │
│  • dMRI FA patch: 16³ voxels → 256 dims                             │
│  • dMRI SC node: 4 features → 256 dims                               │
│                                                                         │
│  Transformer Encoders:                                                │
│  ───────────────────────────────────────────────────────────────────  │
│  • d_model: 256                                                      │
│  • n_heads: 8                                                       │
│  • d_ffn: 1024                                                      │
│  • n_layers: 4                                                      │
│  • dropout: 0.1                                                     │
│                                                                         │
│  Fusion:                                                              │
│  ───────────────────────────────────────────────────────────────────  │
│  • N_hub_slots: 64                                                  │
│  • Cross-attention layers: 1-2                                       │
│                                                                         │
│  总参数量估算:                                                        │
│  ───────────────────────────────────────────────────────────────────  │
│  • 每个 Transformer encoder: ~2.5M params                           │
│  • 6 个 encoders: ~15M params                                       │
│  • Fusion: ~1M params                                               │
│  • 总计: ~16M params (可接受范围)                                    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2.8 Future Directions

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Future Extensions                                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  当前版本: Same-Atlas + Simplified Fusion                              │
│                                                                         │
│  Future探索方向:                                                       │
│  ───────────────────────────────────────────────────────────────────  │
│                                                                         │
│  1. Modality-Specific Atlas                                           │
│     • fMRI: Schaefer (功能同质性最优)                                  │
│     • dMRI: Brainnetome (结构连接驱动)                                 │
│     • sMRI: AAL/DK (解剖标准)                                         │
│     • 需要 ontology translation module                                │
│                                                                         │
│  2. Continuous Coordinate PE                                          │
│     • 使用 continuous MNI coordinates 而非 discrete ROI                 │
│     • 适合 voxel-level transformer                                     │
│                                                                         │
│  3. Latent Semantic Alignment                                         │
│     • 不依赖 shared geometry                                         │
│     • 通过 contrastive learning建立 latent alignment                   │
│                                                                         │
│  4. Iterative Latent Refinement                                       │
│     • 多层 cross-attention refinement                                 │
│     • 资源充足时的 advanced version                                   │
│                                                                         │
│  5. Atlas Adapter                                                     │
│     • Modality-specific atlas + shared adapter                        │
│     • Adapter 学习 ontology translation                                │
│                                                                         │
│  注意事项:                                                            │
│  ───────────────────────────────────────────────────────────────────  │
│  • 当前版本不采用以上设计                                              │
│  • 优先保证 stable optimization                                       │
│  • 等基础版本稳定后再探索 advanced features                           │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

**下一步**：
- → [03_training_paradigm.md](./03_training_paradigm.md) - 训练范式
- → [04_loss_function_design.md](./04_loss_function_design.md) - Loss 设计
