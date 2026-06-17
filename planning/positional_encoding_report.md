# Positional Encoding 技术报告

## 1. 概述

Positional Encoding (位置编码) 是 Transformer 架构中不可或缺的组件。由于 Transformer 的 Self-Attention 机制本身是位置无关的（Permutation Invariant），位置编码为模型提供了序列中元素位置的信息。

对于脑 MRI 数据，特别是基于 Schaefer-200 Atlas 的 ROI 数据，我们设计了多种适合脑区结构的 Positional Encoding 策略。

---

## 2. 标准 Sinusoidal Positional Encoding

### 2.1 原理

来自 "Attention Is All You Need" (Vaswani et al., 2017)。

**数学公式：**
```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

其中：
- `pos`: 位置索引 (0, 1, 2, ...)
- `i`: 维度索引
- `d_model`: 模型维度

### 2.2 实现细节

```python
position = torch.arange(max_len).unsqueeze(1)  # (max_len, 1)
div_term = torch.exp(torch.arange(0, d_model, 2) * (-log(10000) / d_model))
pe[:, 0::2] = sin(position * div_term)  # 偶数维度
pe[:, 1::2] = cos(position * div_term)  # 奇数维度
```

### 2.3 特点

| 优点 | 缺点 |
|------|------|
| 可以处理任意长度序列 | 没有利用脑区的先验知识 |
| 不同位置有唯一编码 | 无法区分 ROI 的功能网络归属 |
| 周期性好，支持相对位置 | 与脑区结构无关 |

---

## 3. Brain-Aware Positional Encoding (推荐)

### 3.1 设计动机

Schaefer-200 Atlas 具有明确的结构：

```
Schaefer-200 网络结构：
┌─────────────────────────────────────────────────────────────┐
│ ROI 1-28:   Visual Network (视觉网络)                      │
│ ROI 29-56:  SomatoMotor Network (躯体运动网络)              │
│ ROI 57-78:  Dorsal Attention Network (背侧注意网络)         │
│ ROI 79-92:  Ventral Attention Network (腹侧注意网络)       │
│ ROI 93-100: Limbic System (边缘系统)                      │
│ ROI 101-142: Fronto-Parietal Control (额顶控制网络)         │
│ ROI 143-200: Default Mode Network (默认模式网络)            │
└─────────────────────────────────────────────────────────────┘
         ↑                                          ↑
    左半球                                    右半球 (镜像)
```

### 3.2 编码组成

Brain-Aware PE 将位置信息分解为三个部分：

```python
PE_final = Concat(PE_position, PE_network, PE_hemisphere)
```

| 编码分量 | 维度 | 来源 | 作用 |
|---------|------|------|------|
| **Within-Network Position** | d/3 | Sinusoidal | 网络内 ROI 顺序 |
| **Network ID** | d/3 | Learnable | 7 个功能网络 |
| **Hemisphere** | d/3 | Learnable | 左/右半球 |

### 3.3 具体实现

```python
class BrainAwarePositionalEncoding(nn.Module):
    def __init__(self, d_model, n_rois=200):
        super().__init__()

        # 1. Within-network position (sinusoidal)
        self.pos_encoder = SinusoidalPositionalEncoding(d_model // 3)

        # 2. Network embedding (7 networks)
        self.network_emb = nn.Embedding(7, d_model // 3)

        # 3. Hemisphere embedding (2 hemispheres)
        self.hemisphere_emb = nn.Embedding(2, d_model // 3)

        # 预计算的 ROI 元数据
        self.register_buffer('roi_networks', self._get_roi_networks())
        self.register_buffer('roi_hemispheres', self._get_roi_hemispheres())

    def forward(self, x):
        # 获取每个位置的编码
        pos_enc = self.pos_encoder.pe[:, :N]
        net_enc = self.network_emb(roi_networks)
        hem_enc = self.hemisphere_emb(roi_hemispheres)

        # 拼接
        pe = torch.cat([pos_enc, net_enc, hem_enc], dim=-1)
        return x + pe
```

### 3.4 为什么这样设计？

1. **Within-Network Position**: 网络内的 ROI 是有序的（例如视觉网络的 V1, V2, V3...），保留顺序信息
2. **Network ID**: 不同网络的 ROI 具有完全不同的功能，编码网络身份让模型区分视觉区和运动区
3. **Hemisphere**: 左右半球有功能偏侧化（lateralization），编码半球信息帮助模型学习偏侧化模式

---

## 4. Functional Network Encoding (分层编码)

### 4.1 分层结构

基于脑网络的层次组织：

```
                    ┌─────────────────────────────────────┐
                    │     Association Cortex (联合皮层)     │
                    │  (DorsAttn, VentAttn, FpnCont, DMN) │
                    └──────────────────┬──────────────────┘
                                       │
        ┌──────────────┬───────────────┼───────────────┬──────────────┐
        ▼              ▼               ▼               ▼              ▼
┌──────────┐  ┌──────────────┐  ┌────────────┐  ┌────────────┐  ┌──────────┐
│  Visual  │  │ SomatoMotor  │  │ DorsAttn   │  │  Default   │  │  Limbic  │
└──────────┘  └──────────────┘  └────────────┘  └────────────┘  └──────────┘
      └──────────────────────────────────────────────────────────────────┘
                    Primary Cortex (初级皮层)
```

### 4.2 编码分量

```python
PE_functional = Concat(
    PE_primary_type,    # 初级皮层 vs 联合皮层 (d/4)
    PE_network,         # 具体网络身份 (d/4)
    PE_network_pos,     # 网络内位置 (d/4, sinusoidal)
    PE_hemisphere       # 左/右半球 (d/4)
)
```

---

## 5. Anatomical Positional Encoding (解剖学编码)

### 5.1 原理

使用每个 ROI 在 MNI 标准空间的 3D 坐标来编码位置。

**MNI 坐标系：**
- X 轴：左(-) → 右(+)
- Y 轴：后(-) → 前(+)
- Z 轴：下(-) → 上(+)

### 5.2 实现

```python
def _create_sinusoidal_projection(self, coords):
    # 归一化坐标到 [0, 1]
    coords_norm = (coords - coords_min) / (coords_max - coords_min)

    # 每个维度用不同的频率
    div_term = exp(-log(10000) * (2i / d))
    pe[:, 0::3] = sin(coords_x * div_term)
    pe[:, 1::3] = sin(coords_y * div_term)
    pe[:, 2::3] = sin(coords_z * div_term)
```

### 5.3 空间语义

这种编码保留了空间邻近性：
- 空间上相近的 ROI → 相似的编码
- 前后/左右/上下位置都有明确区分

---

## 6. Rotary Position Embedding (RoPE)

### 6.1 原理

来自 RoFormer (Su et al., 2021)，通过旋转 Query 和 Key 向量来实现位置编码。

**数学形式：**
```python
def rotate_half(x):
    x1 = x[..., :d//2]
    x2 = x[..., d//2:]
    return cat([-x2, x1], dim=-1)

def apply_rotary(q, k, cos, sin):
    q = q * cos + rotate_half(q) * sin
    k = k * cos + rotate_half(k) * sin
    return q, k
```

### 6.2 优势

- 无需在输入中添加 PE
- 支持相对位置编码
- 可以处理任意长度序列

---

## 7. 各编码方式对比

| 编码类型 | 维度利用 | 功能先验 | 空间先验 | 推荐场景 |
|---------|---------|---------|---------|---------|
| Sinusoidal | 完全 | ❌ | ❌ | 通用基准 |
| Learnable | 完全 | ❌ | ❌ | 小数据集 |
| **Brain-Aware** | 完全 | ✅ | 部分 | **最佳选择** |
| Functional | 完全 | ✅✅ | 部分 | 需要层次结构 |
| Anatomical | 完全 | 部分 | ✅✅ | 需要精确空间信息 |
| Rotary | 完全 | ❌ | ❌ | 长序列 |
| Hybrid | 完全 | ✅ | ✅ | 复杂任务 |

---

## 8. 在本项目中的应用

### 8.1 推荐配置

```yaml
model:
  positional_encoding: 'brain_aware'  # 推荐使用 brain_aware

  # 或者使用更复杂的混合编码
  # positional_encoding: 'hybrid'
```

### 8.2 不同模态的使用建议

| 模态 | 推荐 PE 类型 | 原因 |
|------|-------------|------|
| sMRI (ROI) | brain_aware | ROI 正好对应 Schaefer 网络 |
| fMRI (时间序列) | functional | 时间维度和功能网络都重要 |
| dMRI (结构连接) | anatomical | 空间结构连接重要 |

### 8.3 使用示例

```python
from model.backbone.positional_encoding import get_positional_encoding

# 创建 brain-aware PE
pe = get_positional_encoding(
    encoding_type='brain_aware',
    d_model=256,
    n_rois=200,
    dropout=0.1
)

# 在模型中使用
x = embeddings  # (B, 200, 256)
x = pe(x)      # 添加位置编码
```

---

## 9. Schaefer-200 ROI 与编码映射

### 9.1 网络边界 (0-indexed)

```python
NETWORK_BOUNDARIES = {
    'Visual':           (0, 27),    # 28 ROIs
    'SomatoMotor':      (28, 55),   # 28 ROIs
    'DorsalAttention':  (56, 77),   # 22 ROIs
    'VentralAttention': (78, 91),   # 14 ROIs
    'Limbic':           (92, 99),    # 8 ROIs
    'FrontoParietal':   (100, 141), # 42 ROIs
    'DefaultMode':      (142, 199),  # 58 ROIs
}
```

### 9.2 Brain-Aware PE 如何区分网络

```
输入: ROI index = 50

1. 计算网络归属:
   50 ∈ [28, 55] → SomatoMotor Network

2. 计算网络内位置:
   50 - 28 = 22 (网络内第 22 个 ROI)

3. 计算半球:
   50 < 100 → 左半球

4. 生成编码:
   PE[50] = [PE_pos(22), PE_network(1), PE_hemi(0)]
            = [pos_encoding, somatomotor_embedding, left_hemisphere_embedding]
```

---

## 10. 参考论文

1. Vaswani, A., et al. (2017). "Attention Is All You Need." *NeurIPS*.

2. Schaefer, A., et al. (2018). "Local-Global Parcellation of the Human Cerebral Cortex." *Cerebral Cortex*.

3. Su, J., et al. (2021). "RoFormer: Enhanced Transformer with Rotary Position Embedding." *arXiv*.

4. Thomas Yeo, B. T., et al. (2011). "The organization of the human cerebral cortex estimated by intrinsic functional connectivity." *Journal of Neurophysiology*.

---

*文档生成日期: 2026-05-07*
