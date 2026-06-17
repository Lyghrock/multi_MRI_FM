# 5️⃣ DINO Training Design

> **文档目标**：定义 DINO 训练的完整设计  
> **核心思想**：Self-Distillation + Momentum Teacher + Multi-Crop + Center + SAM (Sharpness-Aware Minimization)

---

## 5.1 DINO 整体架构

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    DINO 架构：Student-Teacher Self-Distillation                     │
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
│  │  H ∈ R^(B×64×d)  (Shared Brain Representation)                 │ │
│  └─────────────────────────────────────────────────────────────────┘ │
│                           │                                            │
│                           ▼                                            │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │  DINO TRAINING HEAD                                              │ │
│  │  ──────────────────────────────────────────────────────────── │ │
│  │                                                                  │ │
│  │   ┌─────────────────────────────────────────────────────┐      │ │
│  │   │              TEACHER (EMA, 无梯度)                  │      │ │
│  │   │   Output: 概率分布 p_teacher                       │      │ │
│  │   │   更新: θ_teacher ← β·θ_teacher + (1-β)·θ_student │      │ │
│  │   └────────────────────────┬────────────────────────────┘      │ │
│  │                            │                                       │ │
│  │                            ▼                                       │ │
│  │                    ┌──────────────┐                            │ │
│  │                    │   Center C   │  ← 可学习参数，防止collapse │ │
│  │                    └──────────────┘                            │ │
│  │                            │                                       │ │
│  │                            ▼                                       │ │
│  │   ┌─────────────────────────────────────────────────────┐      │ │
│  │   │              STUDENT (有梯度)                       │      │ │
│  │   │   Input: Global View                                │      │ │
│  │   │   Output: 概率分布 p_student                        │      │ │
│  │   └────────────────────────┬────────────────────────────┘      │ │
│  │                            │                                       │ │
│  │                            ▼                                       │ │
│  │                    ┌──────────────┐                            │ │
│  │                    │    Loss      │                            │ │
│  │                    │  Cross-Entropy(q_t, p_t)                   │ │
│  │                    └──────────────┘                            │ │
│  │                                                                  │ │
│  └─────────────────────────────────────────────────────────────────┘ │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 5.2 Component 1: Student-Teacher Architecture

### 5.2.1 为什么需要 Student-Teacher？

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Self-Distillation 的必要性                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  问题: 如何让模型学到好的表示而不需要标签？                              │
│  ───────────────────────────────────────────────────────────────────  │
│                                                                         │
│  传统方法:                                                             │
│  • Contrastive Learning: 需要负样本，容易 collapse                      │
│  • MAE: 需要重建像素，需要 Decoder                                     │
│  • JEPA: 需要 Predictor，间接预测                                      │
│                                                                         │
│  DINO 方法:                                                            │
│  ───────────────────────────────────────────────────────────────────  │
│                                                                         │
│  • Teacher 和 Student 是同一个架构                                      │
│  • Teacher 输出作为 "soft label" 指导 Student                         │
│  • Student 学习让输出分布接近 Teacher                                  │
│  • 通过 EMA 更新 Teacher，避免 Teacher 退化                            │
│                                                                         │
│  优势:                                                                │
│  ✓ 不需要负样本                                                       │
│  ✓ 不需要 Decoder                                                     │
│  ✓ 自然地避免 collapse                                                │
│  ✓ 训练稳定                                                           │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 5.2.2 DINO vs JEPA 对比

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    DINO vs JEPA                                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  共同点:                                                              │
│  ───────────────────────────────────────────────────────────────────  │
│  • 都使用 EMA Teacher                                                  │
│  • 都不需要标签                                                      │
│  • 都使用共享的 Foundation Model                                      │
│                                                                         │
│  差异:                                                                │
│  ───────────────────────────────────────────────────────────────────  │
│                                                                         │
│  JEPA:                                                                │
│  • 有额外的 Predictor 网络                                            │
│  • 预测 latent representation                                        │
│  • 需要设计 target space                                              │
│                                                                         │
│  DINO:                                                                │
│  • 直接在 output 概率分布上蒸馏                                       │
│  • 不需要额外的 Predictor                                             │
│  • 使用 Cross-Entropy Loss                                            │
│  • 有 Center 机制防止 collapse                                        │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 5.2.3 DINO Student-Teacher 实现

```python
class DINOModel(nn.Module):
    """
    DINO Model
    
    Student-Teacher Self-Distillation
    """
    
    def __init__(self, foundation_model, d_model=256, n_crops_global=2, 
                 n_crops_local=8, temperature=0.1, ema_beta=0.996,
                 center_init=0.0):
        super().__init__()
        
        # Foundation Model (Student 和 Teacher 共享)
        self.foundation = foundation_model
        
        # Teacher: EMA 版本的 Foundation (不需要 Predictor)
        self.teacher = copy.deepcopy(foundation_model)
        self.teacher.eval()  # 不更新参数
        
        # EMA 参数
        self.ema_beta = ema_beta
        self._init_shadow()
        
        # 输出头 (MLP)
        # Student 和 Teacher 共享结构，但参数独立
        hidden_dim = d_model * 4
        self.student_head = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model),
        )
        self.teacher_head = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model),
        )
        
        # Center (防止 collapse)
        self.register_buffer('center', torch.tensor(center_init))
        
        # Multi-Crop 设置
        self.n_crops_global = n_crops_global
        self.n_crops_local = n_crops_local
        
        # Temperature
        self.temperature_student = temperature
        self.temperature_teacher = temperature * ema_beta  # 更 sharp
        
        # Teacher 不需要梯度
        for param in self.teacher.parameters():
            param.requires_grad = False
        for param in self.teacher_head.parameters():
            param.requires_grad = False
    
    def _init_shadow(self):
        """初始化 shadow parameters"""
        self.shadow = {}
        for name, param in self.foundation.named_parameters():
            self.shadow[name] = param.data.clone()
    
    @torch.no_grad()
    def _update_teacher(self):
        """更新 Teacher (EMA)"""
        for name, param in self.foundation.named_parameters():
            if param.requires_grad:
                self.shadow[name] = self.ema_beta * self.shadow[name] + \
                                   (1 - self.ema_beta) * param.data
    
    @torch.no_grad()
    def _apply_teacher(self):
        """临时应用 Teacher 参数用于前向传播"""
        self._original_params = {}
        for name, param in self.foundation.named_parameters():
            self._original_params[name] = param.data.clone()
            param.data = self.shadow[name]
    
    def _restore_params(self):
        """恢复 Student 参数"""
        for name, param in self.foundation.named_parameters():
            param.data = self._original_params[name]
    
    def forward_student(self, H, return_embedding=False):
        """
        Student forward
        
        Args:
            H: (B, N, d) shared representation
            return_embedding: 是否返回 embedding 而非 softmax
        
        Returns:
            如果 return_embedding=True: (B, d) embedding
            否则: (B, d) softmax 概率分布
        """
        # Pooling: 对 N 维度取平均
        x = H.mean(dim=1)  # (B, d)
        
        # Head
        x = self.student_head(x)  # (B, d)
        
        if return_embedding:
            return x
        
        # Softmax
        x = F.softmax(x / self.temperature_student, dim=-1)
        return x
    
    @torch.no_grad()
    def forward_teacher(self, H):
        """
        Teacher forward
        
        Args:
            H: (B, N, d) shared representation
        
        Returns:
            (B, d) softmax 概率分布
        """
        # 临时应用 Teacher 参数
        self._apply_teacher()
        
        # Pooling
        x = H.mean(dim=1)  # (B, d)
        
        # Head
        x = self.teacher_head(x)  # (B, d)
        
        # Softmax with center
        x = x - self.center
        x = F.softmax(x / self.temperature_teacher, dim=-1)
        
        # 恢复 Student 参数
        self._restore_params()
        
        return x
```

---

## 5.3 Component 2: Multi-Crop Strategy

### 5.3.1 Multi-Crop 核心思想

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Multi-Crop 数据增强                                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  核心思想:                                                             │
│  ───────────────────────────────────────────────────────────────────  │
│                                                                         │
│  • Global Views (n=2): 完整的 ROI 表示或 volume                         │
│    → Teacher 输入                                                       │
│    → 提供完整的上下文信息                                               │
│                                                                         │
│  • Local Views (n=8): 部分 ROI 或 patches                              │
│    → Student 输入                                                       │
│    → 迫使 Student 学习局部特征                                         │
│                                                                         │
│  ───────────────────────────────────────────────────────────────────  │
│                                                                         │
│  为什么有效:                                                           │
│  • Teacher 只处理全局视图，保持表示的完整性                            │
│  • Student 需要从局部视图中提取足够信息来匹配 Teacher                  │
│  • 这迫使模型学习到可迁移的特征                                        │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 5.3.2 Multi-Crop 实现

```python
class MultiCropTransform:
    """
    Multi-Crop 数据增强
    
    生成 Global 和 Local views
    """
    
    def __init__(self, n_crops_global=2, n_crops_local=8,
                 size_global=200, size_local=64,
                 scales_global=(0.8, 1.0), 
                 scales_local=(0.4, 0.8)):
        """
        Args:
            n_crops_global: Global views 数量
            n_crops_local: Local views 数量
            size_global: Global view 大小 (ROI 数量)
            size_local: Local view 大小
            scales_global: Global view 尺度范围
            scales_local: Local view 尺度范围
        """
        self.n_crops_global = n_crops_global
        self.n_crops_local = n_crops_local
        self.size_global = size_global
        self.size_local = size_local
        self.scales_global = scales_global
        self.scales_local = scales_local
    
    def __call__(self, batch):
        """
        Args:
            batch: {
                'sMRI': (B, 200, 3),
                'fMRI': (B, T, 200),
                'dMRI': (B, 200, 200)
            }
        
        Returns:
            views: list of dict, 每个 dict 包含一个 view 的数据
        """
        views = []
        
        # 1. Global Views (Teacher 输入)
        for _ in range(self.n_crops_global):
            # 对 ROI 进行随机 crop
            view = self._crop_global(batch)
            views.append(view)
        
        # 2. Local Views (Student 输入)
        for _ in range(self.n_crops_local):
            # 对 ROI 的子集进行 crop
            view = self._crop_local(batch)
            views.append(view)
        
        return views
    
    def _crop_global(self, batch):
        """
        生成 Global View
        
        随机选择连续的一段 ROI
        """
        # 对 ROI 特征进行随机裁剪
        n_roi = batch['sMRI'].shape[1]
        crop_ratio = np.random.uniform(*self.scales_global)
        crop_size = int(n_roi * crop_ratio)
        start = np.random.randint(0, n_roi - crop_size + 1)
        
        view = {}
        for key, data in batch.items():
            if key == 'sMRI':
                # (B, 200, 3) → (B, crop_size, 3)
                view[key] = data[:, start:start+crop_size, :]
            elif key == 'fMRI':
                # (B, T, 200) → (B, T, crop_size)
                view[key] = data[:, :, start:start+crop_size]
            elif key == 'dMRI':
                # (B, 200, 200) → 提取子矩阵
                view[key] = data[:, start:start+crop_size, start:start+crop_size]
        
        return view
    
    def _crop_local(self, batch):
        """
        生成 Local View
        
        选择较小的 ROI 子集，可能包含增强
        """
        n_roi = batch['sMRI'].shape[1]
        crop_ratio = np.random.uniform(*self.scales_local)
        crop_size = int(n_roi * crop_ratio)
        start = np.random.randint(0, n_roi - crop_size + 1)
        
        view = {}
        for key, data in batch.items():
            if key == 'sMRI':
                view[key] = data[:, start:start+crop_size, :]
            elif key == 'fMRI':
                view[key] = data[:, :, start:start+crop_size]
            elif key == 'dMRI':
                view[key] = data[:, start:start+crop_size, start:start+crop_size]
        
        # 可能的增强
        # - Random masking
        # - Noise injection
        # - Permutation
        
        return view


class DINOLoss(nn.Module):
    """
    DINO Loss
    
    Teacher 输出 soft label, Student 学习匹配
    """
    
    def __init__(self, temperature=0.1, temp_tch=0.04):
        super().__init__()
        self.temperature = temperature
        self.temp_tch = temp_tch
    
    def forward(self, student_output, teacher_output):
        """
        Args:
            student_output: (B, d) student softmax 输出
            teacher_output: (B, d) teacher softmax 输出
        
        Returns:
            loss: scalar
        """
        # Cross-Entropy
        # 注意: teacher 概率更 sharp (temperature 更低)
        loss = -torch.sum(
            teacher_output * torch.log(student_output + 1e-8),
            dim=-1
        ).mean()
        
        return loss
```

### 5.3.3 Multi-Crop 流程图

```
Input: sMRI/fMRI/dMRI
         │
         ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    Multi-Crop Transform                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────────────────────────────────────────────┐          │
│  │  Global Views (2个)                                  │          │
│  │  ┌─────────┐  ┌─────────┐                           │          │
│  │  │ View G1 │  │ View G2 │  ← Teacher 输入            │          │
│  │  │ (全ROI) │  │ (全ROI) │                           │          │
│  │  └────┬────┘  └────┬────┘                           │          │
│  └───────┼────────────┼────────────────────────────────┘          │
│          │            │                                              │
│          │            ▼                                              │
│          │     ┌─────────────┐                                      │
│          │     │   Teacher   │ → soft label (low temp)             │
│          │     └─────────────┘                                      │
│          │            │                                              │
│          │            ▼                                              │
│          │     ┌─────────────┐                                      │
│          │     │   Loss      │                                      │
│          │     │   CE(q,p)   │                                      │
│          │     └─────────────┘                                      │
│          │            ▲                                              │
│          │            │                                              │
│  ┌───────┴────────────┴────────────────────────────────┐          │
│  │  Local Views (8个)                                   │          │
│  │  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐              │          │
│  │  │View L1│ │View L2│ │View L3│ │View L4│ ...       │          │
│  │  │(局部)│ │(局部)│ │(局部)│ │(局部)│              │          │
│  │  └──┬───┘ └──┬───┘ └──┬───┘ └──┬───┘              │          │
│  └─────┼────────┼────────┼────────┼─────────────────────┘          │
│        └────────┴────────┴────────┘                                │
│                    │                                                │
│                    ▼                                                │
│             ┌─────────────┐                                        │
│             │   Student   │ → 预测 Teacher 的 soft label            │
│             └─────────────┘                                        │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 5.4 Component 3: Center & Sharpness-Aware Minimization

### 5.4.1 Center 机制

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Center 机制 - 防止 Collapse                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  问题: Student 可能学会输出 uniform distribution 来最小化 CE             │
│  ───────────────────────────────────────────────────────────────────  │
│                                                                         │
│  解决方案: Center                                                     │
│  ───────────────────────────────────────────────────────────────────  │
│                                                                         │
│  Teacher output: softmax((h - c) / τ_t)                               │
│                                                                         │
│  其中 c 是可学习的 center，初始化为 0                                  │
│                                                                         │
│  作用:                                                                 │
│  • 减去 center 让分布更 sharp                                          │
│  • 防止 Teacher 输出 uniform distribution                              │
│  • Center 会学习到数据的 "average" 表示                                │
│                                                                         │
│  更新规则:                                                            │
│  ───────────────────────────────────────────────────────────────────  │
│  c ← m·c + (1-m)·mean(q)                                              │
│                                                                         │
│  其中 m 是 momentum (0.9), mean(q) 是 batch mean of teacher output    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 5.4.2 Center 实现

```python
class CenterUpdater:
    """
    Center 更新器
    
    c ← m·c + (1-m)·mean(q)
    """
    
    def __init__(self, momentum=0.9):
        self.momentum = momentum
    
    def update(self, center, teacher_output):
        """
        更新 Center
        
        Args:
            center: (d,) 当前 center
            teacher_output: (B, d) teacher 输出
        
        Returns:
            updated_center: (d,) 更新后的 center
        """
        batch_mean = teacher_output.mean(dim=0)  # (d,)
        new_center = self.momentum * center + (1 - self.momentum) * batch_mean
        return new_center


class DINOWithCenter(nn.Module):
    """
    DINO Model (带 Center)
    """
    
    def __init__(self, foundation_model, d_model=256, 
                 center_momentum=0.9, **kwargs):
        super().__init__()
        
        self.foundation = foundation_model
        self.teacher = copy.deepcopy(foundation_model)
        self.teacher.eval()
        
        # Center (可学习参数)
        self.register_buffer('center', torch.zeros(d_model))
        
        # Head
        self.student_head = self._build_head(d_model)
        self.teacher_head = self._build_head(d_model)
        
        # EMA
        self.ema_beta = 0.996
        self._init_shadow()
        
        # Center updater
        self.center_updater = CenterUpdater(momentum=center_momentum)
        
        # 冻结 Teacher 参数
        for param in self.teacher.parameters():
            param.requires_grad = False
        for param in self.teacher_head.parameters():
            param.requires_grad = False
    
    def _build_head(self, d_model):
        """构建 output head"""
        hidden_dim = d_model * 4
        return nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model),
        )
    
    @torch.no_grad()
    def forward_teacher(self, H):
        """
        Teacher forward with center
        """
        # Pooling
        x = H.mean(dim=1)
        
        # Head
        x = self.teacher_head(x)
        
        # Subtract center
        x = x - self.center
        
        # Softmax
        x = F.softmax(x / 0.04, dim=-1)  # τ_t = 0.04
        
        return x
    
    def forward_student(self, H):
        """
        Student forward
        """
        # Pooling
        x = H.mean(dim=1)
        
        # Head
        x = self.student_head(x)
        
        # Softmax
        x = F.softmax(x / 0.1, dim=-1)  # τ_s = 0.1
        
        return x
    
    def update_center(self, teacher_output):
        """
        更新 Center
        
        Args:
            teacher_output: (B, d) teacher 输出
        """
        new_center = self.center_updater.update(
            self.center, teacher_output
        )
        self.center = new_center
```

### 5.4.3 Sharpness-Aware Minimization (SAM)

```python
class SAMOptimizer:
    """
    Sharpness-Aware Minimization (SAM) Optimizer
    
    SAM 通过在 loss landscape 中寻找 flat minima 来提高泛化能力
    
    核心思想:
    1. 在 w 方向上做梯度上升，找到最大的 loss
    2. 在这个点上计算真实梯度
    3. 用这个梯度更新 w
    """
    
    def __init__(self, base_optimizer, rho=0.05):
        self.base_optimizer = base_optimizer
        self.rho = rho
        self.param_groups = base_optimizer.param_groups
        self.state = {}
    
    @torch.no_grad()
    def first_step(self):
        """
        第一个梯度步骤: 在 w 方向上做梯度上升
        """
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                # 获取 grad norm
                grad_norm = self._get_grad_norm(p)
                
                # 计算 trust region
                trust_radius = self.rho * p.norm()
                
                # 找到上升方向
                if grad_norm > 1e-8:
                    # w ← w + rho * grad / (grad_norm + eps)
                    p.add_(p.grad, alpha=trust_radius / grad_norm)
                
                # 保存当前 w
                if id(p) not in self.state:
                    self.state[id(p)] = {}
                self.state[id(p)]['old_w'] = p.data.clone()
        
        # 重新计算 loss 和梯度
        self.base_optimizer.zero_grad()
    
    @torch.no_grad()
    def second_step(self):
        """
        第二个梯度步骤: 恢复 w 并做真正的更新
        """
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                # 恢复 w
                if id(p) in self.state and 'old_w' in self.state[id(p)]:
                    p.data = self.state[id(p)]['old_w']
        
        # 做真正的梯度下降
        self.base_optimizer.step()
        self.base_optimizer.zero_grad()
    
    def _get_grad_norm(self, p):
        """计算梯度范数"""
        return torch.norm(p.grad)


class DINOWithSAM:
    """
    DINO with SAM
    
    结合 DINO 和 SAM 的优势
    """
    
    def __init__(self, model, dino_lr=0.0005, sam_rho=0.05):
        self.model = model
        
        # 分离需要 SAM 的参数 (Foundation Model)
        foundation_params = list(model.foundation.parameters())
        head_params = list(model.student_head.parameters())
        
        # Foundation 使用 SAM
        self.foundation_optimizer = SAMOptimizer(
            torch.optim.AdamW(foundation_params, lr=dino_lr),
            rho=sam_rho
        )
        
        # Head 使用普通优化器
        self.head_optimizer = torch.optim.AdamW(head_params, lr=dino_lr * 10)
    
    def step(self, loss):
        """
        SAM 优化步骤
        """
        # 1. 计算 loss 并反向传播
        loss.backward()
        
        # 2. Foundation: SAM
        self.foundation_optimizer.first_step()  # w ← w + ρ
        
        # 重新计算 loss
        # (这里需要重新执行 forward)
        # ... (省略)
        
        # 需要重新计算 loss
        # loss_new = ...
        # loss_new.backward()
        
        self.foundation_optimizer.second_step()  # w ← w - lr·grad
        
        # 3. Head: 普通优化
        self.head_optimizer.step()
        self.head_optimizer.zero_grad()
```

---

## 5.5 Component 4: Complete Training Loop

### 5.5.1 DINO 训练步骤

```python
class DINOConfig:
    """DINO 配置"""
    def __init__(self):
        # 模型
        self.d_model = 256
        self.ema_beta = 0.996
        self.center_momentum = 0.9
        
        # Multi-Crop
        self.n_crops_global = 2
        self.n_crops_local = 8
        self.size_global = 200
        self.size_local = 64
        
        # Temperature
        self.temperature_student = 0.1
        self.temperature_teacher = 0.04
        
        # SAM (可选)
        self.use_sam = True
        self.sam_rho = 0.05


class DINOWithSAMTrainer:
    """
    DINO + SAM 训练器
    """
    
    def __init__(self, model, config, train_loader):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        
        # 优化器
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=0.0005,
            weight_decay=0.05
        )
        
        # Multi-Crop
        self.multi_crop = MultiCropTransform(
            n_crops_global=config.n_crops_global,
            n_crops_local=config.n_crops_local,
        )
        
        # Loss
        self.loss_fn = nn.CrossEntropyLoss()
        
        # EMA
        self.ema_beta = config.ema_beta
        self._init_shadow()
        
        # Center
        self.center_momentum = config.center_momentum
        self.center = torch.zeros(config.d_model)
    
    def _init_shadow(self):
        """初始化 EMA shadow"""
        self.shadow = {}
        for name, param in self.model.foundation.named_parameters():
            self.shadow[name] = param.data.clone()
    
    @torch.no_grad()
    def _update_teacher(self):
        """更新 Teacher (EMA)"""
        for name, param in self.model.foundation.named_parameters():
            if param.requires_grad:
                self.shadow[name] = self.ema_beta * self.shadow[name] + \
                                   (1 - self.ema_beta) * param.data
    
    @torch.no_grad()
    def _update_center(self, teacher_output):
        """更新 Center"""
        batch_mean = teacher_output.mean(dim=0)
        self.center = self.center_momentum * self.center + \
                     (1 - self.center_momentum) * batch_mean
    
    def train_step(self, batch):
        """
        单个训练步骤
        
        1. Multi-Crop 生成 views
        2. Teacher 处理 Global views
        3. Student 处理所有 views
        4. 计算 DINO loss
        5. 更新 Teacher 和 Center
        """
        # 1. Multi-Crop
        views = self.multi_crop(batch)
        
        # 2. 分离 Global 和 Local views
        global_views = views[:self.config.n_crops_global]
        all_views = views
        
        # 3. Teacher 处理 Global views
        teacher_outputs = []
        for view in global_views:
            H = self.model.foundation(view)
            with torch.no_grad():
                # 临时应用 Teacher 参数
                self._apply_teacher_params()
                p = self._teacher_forward(H)
                teacher_outputs.append(p)
                self._restore_student_params()
        
        # 合并 Teacher 输出 (平均)
        teacher_output = torch.stack(teacher_outputs).mean(dim=0)
        
        # 4. Student 处理所有 views
        student_outputs = []
        for view in all_views:
            H = self.model.foundation(view)
            p = self._student_forward(H)
            student_outputs.append(p)
        
        # 5. 计算 DINO loss
        # Student 的 Local views 匹配 Teacher 的 Global views
        loss = 0
        for i, student_output in enumerate(student_outputs):
            # Local view i (从 n_crops_global 开始)
            if i >= self.config.n_crops_global:
                # 与所有 Teacher outputs 计算 loss
                for teacher_output_single in teacher_outputs:
                    loss += self.loss_fn(
                        student_output,
                        teacher_output_single.argmax(dim=-1)
                    )
        
        loss = loss / (len(student_outputs) - self.config.n_crops_global) / \
               self.config.n_crops_global
        
        # 6. 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # 7. 更新 Teacher
        self._update_teacher()
        
        # 8. 更新 Center
        self._update_center(teacher_output.detach())
        
        return loss.item()
    
    @torch.no_grad()
    def _apply_teacher_params(self):
        """临时应用 Teacher 参数"""
        self._saved_params = {}
        for name, param in self.model.foundation.named_parameters():
            self._saved_params[name] = param.data.clone()
            param.data = self.shadow[name]
    
    def _restore_student_params(self):
        """恢复 Student 参数"""
        for name, param in self.model.foundation.named_parameters():
            param.data = self._saved_params[name]
    
    def _teacher_forward(self, H):
        """Teacher forward"""
        x = H.mean(dim=1)
        x = self.model.teacher_head(x)
        x = x - self.center
        x = F.softmax(x / self.config.temperature_teacher, dim=-1)
        return x
    
    def _student_forward(self, H):
        """Student forward"""
        x = H.mean(dim=1)
        x = self.model.student_head(x)
        x = F.softmax(x / self.config.temperature_student, dim=-1)
        return x
```

### 5.5.2 训练监控

```python
training_metrics = {
    # 总损失
    'loss/dino': 'DINO Cross-Entropy Loss',
    
    # Teacher
    'teacher/center_mean': 'Center 均值',
    'teacher/center_std': 'Center 标准差',
    'teacher/output_entropy': 'Teacher 输出熵',
    
    # Student
    'student/output_entropy': 'Student 输出熵',
    'student/agreement': 'Student-Teacher 一致性',
    
    # EMA
    'ema/beta': 'EMA beta',
    
    # 表示质量
    'repr/std': '表示标准差',
    'repr/mean': '表示均值',
    'repr/norm': '表示范数',
    
    # 训练稳定性
    'grad/norm': '梯度范数',
}
```

---

## 5.6 DINO 设计总结

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    DINO 设计总结                                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Component 1: Student-Teacher Architecture                               │
│  ───────────────────────────────────────────────────────────────────  │
│  • Foundation Model 同时作为 Student 和 Teacher 的 backbone             │
│  • Teacher 通过 EMA 更新，不回传梯度                                   │
│  • Student 有梯度，回传更新 Foundation Model                          │
│  • EMA: β = 0.996 (比其他方法更慢)                                    │
│                                                                         │
│  Component 2: Multi-Crop Strategy                                       │
│  ───────────────────────────────────────────────────────────────────  │
│  • Global Views (2个): 完整 ROI，Teacher 输入                          │
│  • Local Views (8个): 部分 ROI，Student 输入                           │
│  • 迫使 Student 从局部匹配全局表示                                     │
│                                                                         │
│  Component 3: Center & SAM                                             │
│  ───────────────────────────────────────────────────────────────────  │
│  • Center: 可学习参数，防止 collapse，c ← m·c + (1-m)·mean(q)        │
│  • SAM: 在 flat minima 寻找最优，提高泛化能力                          │
│                                                                         │
│  Component 4: Loss Function                                            │
│  ───────────────────────────────────────────────────────────────────  │
│  • Cross-Entropy: Student 匹配 Teacher 的 soft label                   │
│  • τ_student = 0.1, τ_teacher = 0.04                                 │
│  • Local views 的输出匹配 Global views 的 teacher 输出                 │
│                                                                         │
│  与 JEPA 的关系:                                                        │
│  ───────────────────────────────────────────────────────────────────  │
│  • 都是 self-supervised，不需要标签                                    │
│  • 都使用 EMA Teacher                                                  │
│  • DINO 直接在概率分布上蒸馏，JEPA 在 latent 上预测                    │
│  • DINO 更简单，不需要额外的 Predictor                                │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 5.7 JEPA vs MAE vs DINO 对比

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    三种 Training Guide 对比                                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │  JEPA                                                            │ │
│  ├─────────────────────────────────────────────────────────────────┤ │
│  │  • Predictor: MLP 或 Transformer                               │ │
│  │  • Target: EMA encoder output (latent)                         │ │
│  │  • Loss: Soft-L1 + VICReg                                       │ │
│  │  • Masking: Curriculum (0% → 30%)                             │ │
│  │  • 特点: 预测 latent representation                            │ │
│  └─────────────────────────────────────────────────────────────────┘ │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │  MAE                                                             │ │
│  ├─────────────────────────────────────────────────────────────────┤ │
│  │  • Decoder: 模态专用 MLP                                        │ │
│  │  • Target: 原始输入 (像素/特征)                                 │ │
│  │  • Loss: Multi-Task MSE + VICReg                                │ │
│  │  • Masking: High-Ratio (75%)                                   │ │
│  │  • 特点: 重建原始输入                                            │ │
│  └─────────────────────────────────────────────────────────────────┘ │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │  DINO                                                            │ │
│  ├─────────────────────────────────────────────────────────────────┤ │
│  │  • Student-Teacher: Self-Distillation                          │ │
│  │  • Target: Teacher softmax output (probability)                 │ │
│  │  • Loss: Cross-Entropy                                          │ │
│  │  • Multi-Crop: Global + Local views                             │ │
│  │  • 特点: 不需要 Predictor，直接蒸馏概率分布                      │ │
│  └─────────────────────────────────────────────────────────────────┘ │
│                                                                         │
│  共同点:                                                              │
│  ───────────────────────────────────────────────────────────────────  │
│  • 共享 Foundation Model                                             │
│  • 都不需要标签                                                      │
│  • 都是 self-supervised                                              │
│  • 都可以组合使用                                                    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

**下一步**：→ [overview.md](./overview.md) - 完整架构总结 (已更新)

---

**文档版本**: v1.0
**创建日期**: 2026-05-23
**项目**: Multi-Modal Brain Foundation Model
