# 4️⃣ MAE Training Design

> **文档目标**：定义 MAE 训练的完整设计  
> **核心思想**：共享 Encoder + 模态专用 Decoder + High-Ratio Masking + Multi-Task Loss

---

## 4.1 MAE 整体架构

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    MAE 架构：共享 Encoder + 模态 Decoder                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │  FOUNDATION MODEL (共享)                                         │ │
│  │  ──────────────────────────────────────────────────────────── │ │
│  │                                                                  │ │
│  │  Input: sMRI, fMRI, dMRI                                      │ │
│  │       │                                                        │ │
│  │       ▼                                                        │ │
│  │  Modality Encoders (共享权重)                                   │ │
│  │       │                                                        │ │
│  │       ▼                                                        │ │
│  │  Shared Latent Hub                                              │ │
│  │       │                                                        │ │
│  │       ▼                                                        │ │
│  │  H ∈ R^(B×64×d)  (Shared Brain Representation)               │ │
│  └─────────────────────────────────────────────────────────────────┘ │
│                           │                                            │
│                           ▼                                            │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │  MAE DECODERS (模态专用)                                         │ │
│  │  ──────────────────────────────────────────────────────────── │ │
│  │                                                                  │ │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │ │
│  │  │ sMRI Decoder │  │ fMRI Decoder │  │ dMRI Decoder │      │ │
│  │  │  (重建 VBM)  │  │ (重建 ROI)   │  │ (重建 SC)   │      │ │
│  │  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘      │ │
│  │         │                  │                  │                │ │
│  │         └──────────────────┼──────────────────┘                │ │
│  │                            │                                   │ │
│  │                            ▼                                   │ │
│  │               ┌─────────────────────┐                        │ │
│  │               │  Multi-Task Loss   │                        │ │
│  │               │  L_total = Σ L_i   │                        │ │
│  │               └─────────────────────┘                        │ │
│  │                            │                                   │ │
│  │                            ▼                                   │ │
│  │               ┌─────────────────────┐                        │ │
│  │               │  Shared Encoder     │ ← 统一回传梯度           │ │
│  │               │  (Foundation)       │                        │ │
│  │               └─────────────────────┘                        │ │
│  │                                                                  │ │
│  └─────────────────────────────────────────────────────────────────┘ │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 4.2 Component 1: Decoder Structure (模态专用)

### 4.2.1 为什么 decoder 这么简单？

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    MAE Decoder 设计原则                                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  核心洞察:                                                            │
│  ───────────────────────────────────────────────────────────────────  │
│                                                                         │
│  Encoder 已经学会了提取好的表示                                         │
│  Decoder 只需要把这个表示映射回原始空间                               │
│                                                                         │
│  ───────────────────────────────────────────────────────────────────  │
│                                                                         │
│  所以 decoder 应该是:                                                  │
│  • 轻量: 不需要太多参数                                               │
│  • 简单: 主要是线性变换                                               │
│  • 模态专用: 针对每个模态设计                                         │
│                                                                         │
│  不是 Transformer，而是简单的 MLP/Linear                               │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4.2.2 三个模态的 Decoder 设计

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    模态专用 Decoder 设计                                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  sMRI Decoder:                                                        │
│  ───────────────────────────────────────────────────────────────────  │
│  输入: H ∈ R^(B×64×d) → 需要重建: sMRI ROI features (200,3)         │
│                                                                         │
│  结构:                                                                 │
│    H → Linear(d, 256) → GELU → Linear(256, 3) → (B, 200, 3)        │
│                                                                         │
│  重建目标: ROI features [GM_mean, GM_volume, GM_max]                  │
│                                                                         │
│  ───────────────────────────────────────────────────────────────────  │
│                                                                         │
│  fMRI Decoder:                                                        │
│  ───────────────────────────────────────────────────────────────────  │
│  输入: H ∈ R^(B×64×d) → 需要重建: fMRI representation               │
│                                                                         │
│  结构:                                                                 │
│    H → Linear(d, 256) → GELU → Linear(256, 256) → (B, 64, 256)     │
│                                                                         │
│  重建目标: 可以选择重建 ROI time-series 或 dFC matrix                │
│                                                                         │
│  ───────────────────────────────────────────────────────────────────  │
│                                                                         │
│  dMRI Decoder:                                                        │
│  ───────────────────────────────────────────────────────────────────  │
│  输入: H ∈ R^(B×64×d) → 需要重建: dMRI SC matrix (200,200)         │
│                                                                         │
│  结构:                                                                 │
│    H → Linear(d, 512) → GELU → Linear(512, 40000)                  │
│      → reshape → (B, 200, 200)                                      │
│                                                                         │
│  重建目标: SC matrix (上三角 + 对角线置零)                           │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4.2.3 Decoder 实现

```python
class SMRI_Decoder(nn.Module):
    """
    sMRI Decoder: 重建 ROI features
    
    输入: (B, 64, d) shared representation
    输出: (B, 200, 3) ROI features [GM_mean, GM_volume, GM_max]
    """
    def __init__(self, d_model=256, hidden_dim=256, n_roi=200, n_features=3):
        super().__init__()
        self.n_roi = n_roi
        self.n_features = n_features
        
        self.decoder = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_roi * n_features),
        )
    
    def forward(self, H):
        """
        Args:
            H: (B, 64, d) shared representation
        Returns:
            pred: (B, 200, 3) predicted ROI features
        """
        B = H.shape[0]
        x = self.decoder(H)  # (B, 64, 200*3) 或 (B, 64, 600)
        
        # 重新组织形状
        # 假设 H 的维度 64 对应 ROI 位置
        # 或者我们直接预测整个向量然后 reshape
        pred = x.view(B, self.n_roi, self.n_features)
        
        return pred


class FMRI_Decoder(nn.Module):
    """
    fMRI Decoder: 重建 functional representation
    
    输入: (B, 64, d) shared representation
    输出: (B, 64, d) reconstructed functional features
    """
    def __init__(self, d_model=256, hidden_dim=256):
        super().__init__()
        
        self.decoder = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model),
        )
    
    def forward(self, H):
        """
        Args:
            H: (B, 64, d) shared representation
        Returns:
            pred: (B, 64, d) reconstructed functional features
        """
        return self.decoder(H)


class DMRI_Decoder(nn.Module):
    """
    dMRI Decoder: 重建 SC matrix
    
    输入: (B, 64, d) shared representation
    输出: (B, 200, 200) SC matrix
    """
    def __init__(self, d_model=256, hidden_dim=512, n_roi=200):
        super().__init__()
        self.n_roi = n_roi
        
        # 预测上三角 (不包括对角线)
        n_upper = n_roi * (n_roi - 1) // 2
        
        self.decoder = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_upper),
        )
    
    def forward(self, H):
        """
        Args:
            H: (B, 64, d) shared representation
        Returns:
            pred: (B, 200, 200) symmetric SC matrix
        """
        B = H.shape[0]
        x = self.decoder(H)  # (B, 64, n_upper)
        
        # 创建对称矩阵
        SC = torch.zeros(B, self.n_roi, self.n_roi, device=H.device)
        
        # 填充上三角
        i, j = torch.triu_indices(self.n_roi, self.n_roi, offset=1, device=H.device)
        SC[:, i, j] = x
        
        # 对称化
        SC = SC + SC.transpose(-2, -1)
        
        return SC


class MultiModalMAE(nn.Module):
    """
    完整的多模态 MAE 模型
    
    共享 Encoder + 模态专用 Decoder
    """
    def __init__(self, foundation_model, d_model=256, n_roi=200):
        super().__init__()
        
        # 共享的 Foundation Model
        self.foundation = foundation_model
        
        # 模态专用 Decoders
        self.sMRI_decoder = SMRI_Decoder(d_model, n_roi=n_roi)
        self.fMRI_decoder = FMRI_Decoder(d_model)
        self.dMRI_decoder = DMRI_Decoder(d_model, n_roi=n_roi)
        
        # Loss
        self.loss_fn = nn.MSELoss(reduction='mean')
    
    def forward(self, batch, return_reconstructions=False):
        """
        Args:
            batch: dict with 'sMRI', 'fMRI', 'dMRI'
        Returns:
            loss: scalar
            reconstructions: dict (optional)
        """
        # 1. Foundation forward
        H = self.foundation(batch)  # (B, 64, d)
        
        # 2. 模态专用重建
        recon = {}
        
        if 'sMRI' in batch:
            recon['sMRI'] = self.sMRI_decoder(H)
        
        if 'fMRI' in batch:
            recon['fMRI'] = self.fMRI_decoder(H)
        
        if 'dMRI' in batch:
            recon['dMRI'] = self.dMRI_decoder(H)
        
        # 3. 计算损失
        loss = 0
        metrics = {}
        
        if 'sMRI' in batch:
            loss_s = self.loss_fn(recon['sMRI'], batch['sMRI'])
            loss = loss + loss_s
            metrics['loss/sMRI'] = loss_s.item()
        
        if 'fMRI' in batch:
            loss_f = self.loss_fn(recon['fMRI'], batch['fMRI'])
            loss = loss + loss_f
            metrics['loss/fMRI'] = loss_f.item()
        
        if 'dMRI' in batch:
            loss_d = self.loss_fn(recon['dMRI'], batch['dMRI'])
            loss = loss + loss_d
            metrics['loss/dMRI'] = loss_d.item()
        
        if return_reconstructions:
            return loss, metrics, recon
        else:
            return loss, metrics
```

---

## 4.3 Component 2: Masking Strategy

### 4.3.1 MAE Masking 特点

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    MAE Masking 策略                                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  核心: High-Ratio Masking (75%)                                       │
│  ───────────────────────────────────────────────────────────────────  │
│                                                                         │
│  • 随机 mask 掉 75% 的表示                                             │
│  • Decoder 需要从 25% 的 visible 表示重建全部                         │
│  • 强迫 encoder 学习高效的压缩表示                                     │
│                                                                         │
│  ───────────────────────────────────────────────────────────────────  │
│                                                                         │
│  与 JEPA 的区别:                                                        │
│  • JEPA: Curriculum Masking，渐进引入 masking                         │
│  • MAE: High-Ratio Masking，固定高比例                               │
│                                                                         │
│  ───────────────────────────────────────────────────────────────────  │
│                                                                         │
│  为什么 MAE 用高比例?                                                   │
│  • MAE 的目标是重建，需要大量信息去重建                               │
│  • 高比例 masking 使任务更难，迫使 encoder 学习更好的表示              │
│  • 75% 是经验最优值 (来自 ViT-MAE)                                 │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4.3.2 Masking 实现

```python
class MAEMasking:
    """
    MAE Masking: 高比例随机 masking
    """
    def __init__(self, mask_ratio=0.75):
        self.mask_ratio = mask_ratio
    
    def mask(self, H):
        """
        对 shared representation 应用 masking
        
        Args:
            H: (B, N, d) shared representation
        Returns:
            H_visible: (B, N_visible, d) 被保留下来的部分
            mask: (B, N) 1 = masked
            ids_restore: 用于恢复原始顺序
        """
        B, N, d = H.shape
        n_visible = int(N * (1 - self.mask_ratio))
        
        # 随机打乱
        noise = torch.rand(B, N, device=H.device)
        ids_shuffle = noise.argsort(dim=1)
        ids_restore = ids_shuffle.argsort(dim=1)
        
        # 取 visible
        ids_visible = ids_shuffle[:, :n_visible]
        H_visible = torch.gather(H, dim=1, index=ids_visible.unsqueeze(-1).expand(-1, -1, d))
        
        # 创建 mask (1 = masked)
        mask = torch.ones(B, N, device=H.device)
        mask[:, :n_visible] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore).bool()
        
        return H_visible, mask, ids_restore
    
    def unmask(self, x, ids_restore):
        """
        恢复原始顺序 (用于 decoder 输出)
        """
        B = x.shape[0]
        N = ids_restore.shape[1]
        d = x.shape[2] if x.dim() == 3 else 1
        
        if x.dim() == 3:
            x = torch.gather(x, dim=1, index=ids_restore.unsqueeze(-1).expand(-1, -1, d))
        else:
            x = torch.gather(x, dim=1, index=ids_restore)
        
        return x


class MAEModelWithMasking(nn.Module):
    """
    MAE 模型 (带 masking)
    """
    def __init__(self, foundation_model, decoders, mask_ratio=0.75):
        super().__init__()
        self.foundation = foundation_model
        self.decoders = decoders
        self.masking = MAEMasking(mask_ratio)
        
        # 可学习的 mask token
        self.mask_token = nn.Parameter(torch.randn(1, 1, 256))
    
    def forward(self, batch):
        B = batch['sMRI'].shape[0]
        
        # 1. Foundation forward
        H = self.foundation(batch)  # (B, 64, d)
        
        # 2. Apply masking
        H_visible, mask, ids_restore = self.masking.mask(H)
        
        # 3. Decoder forward (只在 visible 上)
        recon = {}
        for name, decoder in self.decoders.items():
            # Decoder 处理 visible tokens
            # 需要将 visible tokens 填回完整序列
            H_full = torch.zeros(B, 64, 256, device=H.device)
            H_visible_expanded = H_visible  # decoder 输出
            recon[name] = decoder(H_visible_expanded)
        
        # 4. 计算损失 (只在 masked 位置)
        loss = 0
        for name in ['sMRI', 'fMRI', 'dMRI']:
            if name in batch:
                pred = recon[name]
                target = batch[name]
                # 只在 masked 位置计算
                loss = loss + F.mse_loss(pred[mask], target[mask])
        
        return loss
```

### 4.3.3 Masking 曲线

```
Mask Ratio
    │
1.0 ┤
    │    ═══════════════════════════════
    │   ╱
0.75 ┤──╱  ← 固定 75% masking
    │
    │
0.0 ┼─────────────────────────────────────→ Epoch
    0         100        300        500
    
    与 JEPA 不同:
    - JEPA: 渐进式引入 masking
    - MAE: 固定高比例 masking
```

### 4.3.4 MAE Masking vs JEPA Masking

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Masking 策略对比                                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  MAE Masking:                                                         │
│  ───────────────────────────────────────────────────────────────────  │
│  • 固定高比例: 75%                                                   │
│  • 在表示层面 masking                                                 │
│  • Decoder 需要从少量 visible 重建全部                                │
│  • 任务: 重建原始输入                                                │
│                                                                         │
│  JEPA Masking:                                                        │
│  ───────────────────────────────────────────────────────────────────  │
│  • 渐进式: 0% → 30%                                                  │
│  • Curriculum: 从简单到复杂                                           │
│  • 在表示层面 masking                                                 │
│  • 任务: 预测 latent target                                           │
│                                                                         │
│  共同点:                                                              │
│  ───────────────────────────────────────────────────────────────────  │
│  • 都在 shared representation H 上操作                                │
│  • 都使用 masking 强迫 encoder 学习好表示                            │
│  • 梯度统一回传到共享 encoder                                        │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 4.4 Component 3: Multi-Task Loss Design

### 4.4.1 Multi-Task Loss 框架

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Multi-Task Loss 设计                                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  L_total = Σ λ_i · L_i                                              │
│                                                                         │
│  其中 L_i 是每个模态的重建损失                                        │
│                                                                         │
│  ───────────────────────────────────────────────────────────────────  │
│                                                                         │
│  L_sMRI = MSE(pred_sMRI, target_sMRI)                                │
│  L_fMRI = MSE(pred_fMRI, target_fMRI)                                │
│  L_dMRI = MSE(pred_dMRI, target_dMRI)                                │
│                                                                         │
│  L_total = λ_sMRI · L_sMRI + λ_fMRI · L_fMRI + λ_dMRI · L_dMRI     │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4.4.2 Loss 实现

```python
class MultiModalMAELoss(nn.Module):
    """
    MAE Multi-Task Loss
    
    每个模态分别计算 loss，统一回传到共享 encoder
    """
    def __init__(self, lambda_sMRI=1.0, lambda_fMRI=1.0, lambda_dMRI=1.0,
                 use_vicreg=True, lambda_vicreg=0.5):
        super().__init__()
        
        # 各模态权重
        self.lambda_sMRI = lambda_sMRI
        self.lambda_fMRI = lambda_fMRI
        self.lambda_dMRI = lambda_dMRI
        
        self.use_vicreg = use_vicreg
        self.lambda_vicreg = lambda_vicreg
        
        if use_vicreg:
            self.vicreg = VICRegLoss()
    
    def forward(self, predictions, targets, H=None):
        """
        Args:
            predictions: dict {
                'sMRI': (B, 200, 3),
                'fMRI': (B, 64, d),
                'dMRI': (B, 200, 200)
            }
            targets: dict {
                'sMRI': (B, 200, 3),
                'fMRI': (B, 64, d),
                'dMRI': (B, 200, 200)
            }
            H: (B, 64, d) shared representation (for VICReg)
        """
        loss = 0
        metrics = {}
        
        # sMRI Loss
        if 'sMRI' in predictions and 'sMRI' in targets:
            L_sMRI = F.mse_loss(predictions['sMRI'], targets['sMRI'])
            loss = loss + self.lambda_sMRI * L_sMRI
            metrics['loss/sMRI'] = L_sMRI.item()
        
        # fMRI Loss
        if 'fMRI' in predictions and 'fMRI' in targets:
            L_fMRI = F.mse_loss(predictions['fMRI'], targets['fMRI'])
            loss = loss + self.lambda_fMRI * L_fMRI
            metrics['loss/fMRI'] = L_fMRI.item()
        
        # dMRI Loss
        if 'dMRI' in predictions and 'dMRI' in targets:
            L_dMRI = F.mse_loss(predictions['dMRI'], targets['dMRI'])
            loss = loss + self.lambda_dMRI * L_dMRI
            metrics['loss/dMRI'] = L_dMRI.item()
        
        # VICReg Regularization
        if self.use_vicreg and H is not None:
            L_vicreg, vicreg_metrics = self.vicreg(H, H)
            loss = loss + self.lambda_vicreg * L_vicreg
            metrics['loss/vicreg'] = L_vicreg.item()
            metrics.update(vicreg_metrics)
        
        metrics['loss/total'] = loss.item()
        
        return loss, metrics
```

### 4.4.3 辅助 Loss 设计

```python
class AuxiliaryLosses:
    """
    辅助 Loss 设计
    
    除了主要的 MSE 重建损失，还可以添加其他辅助损失
    """
    
    @staticmethod
    def contrastive_loss(H, temperature=0.1):
        """
        Contrastive Loss: 同一模态的不同视图应该相似
        
        适用于多模态: sMRI 和 dMRI 都是结构性表示
        """
        # SimCLR-style contrastive loss
        z = F.normalize(H, dim=-1)
        similarity = z @ z.T / temperature
        labels = torch.arange(len(H), device=H.device)
        
        loss = F.cross_entropy(similarity, labels)
        return loss
    
    @staticmethod
    def structure_preserving_loss(H_sMRI, H_dMRI):
        """
        Structure Preserving Loss: 确保不同模态的表示保持拓扑一致性
        
        例如: SC 强的连接对应对应的 ROI 在表示空间也应该接近
        """
        # 计算表示空间的距离
        dist = torch.cdist(H_sMRI, H_dMRI, p=2)
        
        # 与 SC 矩阵的关联
        # 简单版本: 鼓励距离矩阵与 SC 矩阵负相关
        return dist.mean()
    
    @staticmethod
    def orthogonality_loss(H):
        """
        Orthogonality Loss: 鼓励表示的维度之间正交
        
        防止表示维度冗余
        """
        B, N, d = H.shape
        
        # 重新组织为 (B*d, N)
        H_flat = H.transpose(-2, -1).reshape(B * d, N)
        
        # Gram matrix
        G = H_flat @ H_flat.T
        
        # off-diagonal 应该接近 0
        off_diag = G - torch.diag(torch.diag(G))
        
        return (off_diag ** 2).mean()


class FullMAELoss(nn.Module):
    """
    完整的 MAE Loss
    
    包括: MSE + VICReg + Auxiliary
    """
    def __init__(self, lambda_mse=1.0, lambda_vicreg=0.5, 
                 lambda_contrastive=0.1, lambda_orthogonal=0.01):
        super().__init__()
        
        self.lambda_mse = lambda_mse
        self.lambda_vicreg = lambda_vicreg
        self.lambda_contrastive = lambda_contrastive
        self.lambda_orthogonal = lambda_orthogonal
        
        self.vicreg = VICRegLoss()
    
    def forward(self, predictions, targets, H):
        """
        Args:
            predictions: dict of predictions
            targets: dict of targets
            H: (B, N, d) shared representation
        """
        loss = 0
        metrics = {}
        
        # 1. MSE Loss
        L_mse = 0
        for key in predictions:
            if key in targets:
                L_mse = L_mse + F.mse_loss(predictions[key], targets[key])
        loss = loss + self.lambda_mse * L_mse
        metrics['loss/mse'] = L_mse.item()
        
        # 2. VICReg
        if self.lambda_vicreg > 0:
            L_vicreg, vicreg_metrics = self.vicreg(H, H)
            loss = loss + self.lambda_vicreg * L_vicreg
            metrics['loss/vicreg'] = L_vicreg.item()
        
        # 3. Orthogonality
        if self.lambda_orthogonal > 0:
            L_ortho = AuxiliaryLosses.orthogonality_loss(H)
            loss = loss + self.lambda_orthogonal * L_ortho
            metrics['loss/orthogonal'] = L_ortho.item()
        
        metrics['loss/total'] = loss.item()
        
        return loss, metrics
```

### 4.4.4 Loss 权重调度

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Loss 权重调度                                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  策略: 平衡各模态的重要性                                              │
│  ───────────────────────────────────────────────────────────────────  │
│                                                                         │
│  初始:                                                                │
│  • λ_sMRI = 1.0                                                      │
│  • λ_fMRI = 1.0                                                      │
│  • λ_dMRI = 1.0                                                      │
│                                                                         │
│  可选调度:                                                             │
│  • 根据数据质量调整权重                                               │
│  • 根据训练阶段调整权重                                               │
│  • 使用 uncertainty weighting 自动学习权重                              │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

```python
class UncertaintyWeighting(nn.Module):
    """
    Uncertainty Weighting: 自动学习各模态的权重
    
    原理: 每个模态的权重与其不确定性成反比
    """
    def __init__(self, n_tasks=3):
        super().__init__()
        # log(sigma^2), 优化时用 exp(-log_sigma^2) 作为权重
        self.log_vars = nn.Parameter(torch.zeros(n_tasks))
    
    def forward(self, losses):
        """
        Args:
            losses: list of scalar losses
        Returns:
            weighted_loss: scalar
            weights: list of weights
        """
        weights = torch.exp(-self.log_vars)
        weighted_loss = sum(w * l for w, l in zip(weights, losses))
        
        # 添加正则化
        reg = sum(self.log_vars)
        
        return weighted_loss + reg, weights.tolist()
```

---

## 4.5 Component 4: 完整训练循环

### 4.5.1 训练步骤

```python
class MAETrainer:
    """MAE 训练器"""
    def __init__(self, model, optimizer, loss_fn, mask_ratio=0.75):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.mask_ratio = mask_ratio
    
    def train_step(self, batch):
        """单个训练步骤"""
        # 1. Forward
        predictions = self.model(batch)
        
        # 2. 获取 shared representation
        H = self.model.foundation(batch)
        
        # 3. 计算损失
        loss, metrics = self.loss_fn(predictions, batch, H)
        
        # 4. 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return loss.item(), metrics


class JointTrainer:
    """
    JEPA + MAE 联合训练器
    """
    def __init__(self, foundation_model, jepa_predictor, mae_decoders,
                 jepa_loss_fn, mae_loss_fn, optimizer):
        self.foundation = foundation_model
        self.jepa_predictor = jepa_predictor
        self.mae_decoders = mae_decoders
        self.jepa_loss_fn = jepa_loss_fn
        self.mae_loss_fn = mae_loss_fn
        self.optimizer = optimizer
    
    def train_step(self, batch, epoch):
        """联合训练步骤"""
        # 1. Foundation forward
        H = self.foundation(batch)
        
        # 2. MAE loss
        mae_preds = {
            'sMRI': self.mae_decoders['sMRI'](H),
            'dMRI': self.mae_decoders['dMRI'](H),
        }
        mae_loss, mae_metrics = self.mae_loss_fn(mae_preds, batch, H)
        
        # 3. JEPA loss
        # (需要 EMA target encoder...)
        # jepa_loss, jepa_metrics = self.jepa_loss_fn(...)
        
        # 4. 联合损失
        if epoch < 100:
            # Stage 1: 主要 MAE
            total_loss = mae_loss
        elif epoch < 300:
            # Stage 2: MAE + JEPA
            total_loss = 0.5 * mae_loss  # + 0.5 * jepa_loss
        else:
            # Stage 3: 主要 JEPA
            total_loss = 0.2 * mae_loss  # + 0.8 * jepa_loss
        
        # 5. 反向传播
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        return total_loss.item(), {**mae_metrics}
```

### 4.5.2 训练监控

```python
training_metrics = {
    # 总损失
    'loss/total': '总损失',
    
    # MAE 损失
    'loss/mse': 'MSE 重建损失',
    'loss/sMRI': 'sMRI 重建损失',
    'loss/fMRI': 'fMRI 重建损失',
    'loss/dMRI': 'dMRI 重建损失',
    
    # 辅助损失
    'loss/vicreg': 'VICReg 正则化',
    'loss/orthogonal': '正交正则化',
    'loss/contrastive': '对比正则化',
    
    # Masking
    'mask/ratio': 'Masking 比例',
    
    # 表示质量
    'repr/std': '表示标准差',
    'repr/mean': '表示均值',
    'repr/norm': '表示范数',
    
    # 训练稳定性
    'grad/norm': '梯度范数',
}
```

---

## 4.6 JEPA vs MAE 设计对比

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    JEPA vs MAE 设计对比                                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │  Component 1: Predictor/Decoder Structure                       │ │
│  ├─────────────────────────────────────────────────────────────────┤ │
│  │                                                                  │ │
│  │  JEPA:                                                          │ │
│  │  • Predictor: MLP (Linear → GELU → Linear)                   │ │
│  │  • 作用: 映射 context 到 target space                          │ │
│  │  • Target: EMA encoder output                                  │ │
│  │                                                                  │ │
│  │  MAE:                                                           │ │
│  │  • Decoder: 模态专用 (sMRI/fMRI/dMRI)                        │ │
│  │  • 结构: 简单的 MLP                                            │ │
│  │  • 目标: 重建原始输入                                          │ │
│  │                                                                  │ │
│  └─────────────────────────────────────────────────────────────────┘ │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │  Component 2: Masking Strategy                                 │ │
│  ├─────────────────────────────────────────────────────────────────┤ │
│  │                                                                  │ │
│  │  JEPA:                                                          │ │
│  │  • Curriculum Masking                                           │ │
│  │  • Stage 1: 0% (只做 visible)                                  │ │
│  │  • Stage 2: 0% → 30% (渐进引入)                                │ │
│  │  • Stage 3: 30% (稳定)                                         │ │
│  │                                                                  │ │
│  │  MAE:                                                           │ │
│  │  • High-Ratio Masking                                          │ │
│  │  • 固定 75% masking                                            │ │
│  │  • 全程保持高比例                                               │ │
│  │                                                                  │ │
│  └─────────────────────────────────────────────────────────────────┘ │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │  Component 3: Loss Function                                    │ │
│  ├─────────────────────────────────────────────────────────────────┤ │
│  │                                                                  │ │
│  │  JEPA:                                                          │ │
│  │  • L_JEPA = SoftL1(pred, target)                              │ │
│  │  • + λ·L_VICReg                                               │ │
│  │                                                                  │ │
│  │  MAE:                                                           │ │
│  │  • L_MSE = Σ MSE(pred_i, target_i)                           │ │
│  │  • + λ_vicreg·L_VICReg                                        │ │
│  │  • + λ_aux·L_aux (可选)                                      │ │
│  │                                                                  │ │
│  └─────────────────────────────────────────────────────────────────┘ │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │  Component 4: EMA (JEPA only)                                 │ │
│  ├─────────────────────────────────────────────────────────────────┤ │
│  │                                                                  │ │
│  │  JEPA:                                                          │ │
│  │  • Target Encoder: EMA(foundation)                            │ │
│  │  • β = 0.99 → 0.999                                           │ │
│  │                                                                  │ │
│  │  MAE:                                                           │ │
│  │  • 无需 EMA                                                    │ │
│  │  • 直接重建原始输入                                             │ │
│  │                                                                  │ │
│  └─────────────────────────────────────────────────────────────────┘ │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 4.7 MAE 设计总结

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    MAE 设计总结                                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Component 1: Decoder Structure (简单 MLP)                             │
│  ───────────────────────────────────────────────────────────────────  │
│  • sMRI Decoder: H → Linear(d, 256) → Linear(256, 600)              │
│  • fMRI Decoder: H → Linear(d, 256) → Linear(256, d)               │
│  • dMRI Decoder: H → Linear(d, 512) → Linear(512, 200*200)          │
│                                                                         │
│  Component 2: Masking Strategy (High-Ratio)                           │
│  ───────────────────────────────────────────────────────────────────  │
│  • 固定 75% masking                                                 │
│  • 全程保持高比例                                                   │
│  • 强迫 encoder 学习高效表示                                         │
│                                                                         │
│  Component 3: Multi-Task Loss                                          │
│  ───────────────────────────────────────────────────────────────────  │
│  • L_total = Σ λ_i · L_i + λ_vicreg · L_VICReg                     │
│  • 各模态分别计算 loss，统一回传到共享 encoder                       │
│  • 可选辅助 loss: orthogonality, contrastive                        │
│                                                                         │
│  Component 4: 训练循环                                                │
│  ───────────────────────────────────────────────────────────────────  │
│  • 标准反向传播                                                       │
│  • 梯度裁剪                                                         │
│  • 可与 JEPA 联合训练                                               │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

**下一步**：→ [overview.md](./overview.md) - 完整架构总结
