# 3️⃣ JEPA Training Design

> **文档目标**：定义 JEPA 训练的完整设计  
> **核心思想**：Predictor + Target Encoder (EMA) + Curriculum Masking + Unified Loss

---

## 3.1 JEPA 整体架构

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    JEPA 架构：Encoder + Predictor + Target                    │
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
│  │  JEPA TRAINING HEAD (JEPA 专用)                                 │ │
│  │  ──────────────────────────────────────────────────────────── │ │
│  │                                                                  │ │
│  │  Context Encoder ──► Predictor ──► pred                        │ │
│  │       │                           │                              │ │
│  │       │                           ▼                              │ │
│  │       │              ┌────────────────┐                        │ │
│  │       │              │ Predictor     │                        │ │
│  │       │              │ (轻量 MLP)    │                        │ │
│  │       │              └────────────────┘                        │ │
│  │       │                                                        │ │
│  │       ▼                                                        │ │
│  │  ┌────────────────────────────────────────────┐              │ │
│  │  │  Target Encoder (EMA, 指数移动平均)         │              │ │
│  │  │  θ_target ← β·θ_target + (1-β)·θ_online  │              │ │
│  │  └────────────────────────────────────────────┘              │ │
│  │                         │                                      │ │
│  │                         ▼                                      │ │
│  │                   target = EMA(H)                              │ │
│  │                                                                  │ │
│  └─────────────────────────────────────────────────────────────────┘ │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 3.2 Component 1: Predictor Structure

### 3.2.1 为什么需要 Predictor？

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Predictor 的必要性                                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  问题: Context Encoder 输出和 Target 不同分布                          │
│  ───────────────────────────────────────────────────────────────────  │
│                                                                         │
│  • Context Encoder: 输入 masked data，学习压缩表示                    │
│  • Target Encoder: EMA，输入 clean data，学习稳定表示                │
│                                                                         │
│  → 两者分布不同，直接预测会很难                                         │
│                                                                         │
│  解决方案: Predictor                                                   │
│  ───────────────────────────────────────────────────────────────────  │
│                                                                         │
│  • Predictor 是一个轻量网络                                            │
│  • 学习从 context 分布到 target 分布的映射                             │
│  • 不需要梯度回传到 Target Encoder                                     │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.2.2 Predictor 结构选择

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Predictor 结构对比                                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Option 1: MLP Predictor (轻量)                                       │
│  ───────────────────────────────────────────────────────────────────  │
│                                                                         │
│  结构:                                                                 │
│    Input → Linear(d, d_hidden) → GELU → Linear(d_hidden, d)        │
│                                                                         │
│  配置:                                                                 │
│    d_hidden = 2×d  或  4×d                                          │
│    n_layers = 2                                                       │
│                                                                         │
│  优点:                                                                │
│    ✓ 轻量，计算效率高                                              │
│    ✓ 足够表达力进行 latent 预测                                      │
│    ✓ 易于训练稳定                                                  │
│                                                                         │
│  缺点:                                                                │
│    ✗ 无法捕获 token 之间的序列关系                                    │
│    ✗ 只做独立变换                                                    │
│                                                                         │
│  ───────────────────────────────────────────────────────────────────  │
│                                                                         │
│  Option 2: Transformer Predictor (轻量)                               │
│  ───────────────────────────────────────────────────────────────────  │
│                                                                         │
│  结构:                                                                 │
│    Input → Transformer Encoder (2 layers) → Output                    │
│                                                                         │
│  配置:                                                                 │
│    n_layers = 2                                                       │
│    n_heads = 4                                                        │
│    d_ffn = 4×d                                                        │
│                                                                         │
│  优点:                                                                │
│    ✓ 可以捕获 token 之间的序列关系                                    │
│    ✓ 更强的表达能力                                                  │
│    ✓ 适合 token-level prediction                                      │
│                                                                         │
│  缺点:                                                                │
│    ✗ 计算量稍大                                                      │
│    ✗ 可能过拟合（层数少则可避免）                                    │
│                                                                         │
│  ───────────────────────────────────────────────────────────────────  │
│                                                                         │
│  选择建议:                                                            │
│  • 如果关注全局表示 → MLP Predictor                                  │
│  • 如果关注 token-level prediction → Transformer Predictor            │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.2.3 MLP Predictor 实现

```python
class MLPPredictor(nn.Module):
    """
    MLP Predictor: 轻量 MLP
    
    将 context representation 映射到 target space
    """
    def __init__(self, d_model=256, d_hidden=None, n_layers=2, dropout=0.1):
        super().__init__()
        
        if d_hidden is None:
            d_hidden = d_model * 4
        
        # MLP layers
        layers = []
        in_dim = d_model
        for i in range(n_layers):
            layers.append(nn.Linear(in_dim, d_hidden))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            in_dim = d_hidden
        
        layers.append(nn.Linear(d_hidden, d_model))
        
        self.mlp = nn.Sequential(*layers)
        
        # 可选: LayerNorm
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, context):
        """
        Args:
            context: (B, N, d) context representation
        Returns:
            pred: (B, N, d) predicted target
        """
        pred = self.mlp(context)
        pred = self.norm(pred)
        return pred
```

### 3.2.4 Transformer Predictor 实现

```python
class TransformerPredictor(nn.Module):
    """
    Transformer Predictor: 轻量 Transformer
    
    使用 Transformer Encoder 来处理 token 之间的关系
    """
    def __init__(self, d_model=256, n_heads=4, n_layers=2, 
                 d_ffn=None, dropout=0.1):
        super().__init__()
        
        if d_ffn is None:
            d_ffn = d_model * 4
        
        # Transformer Encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ffn,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,  # Pre-LN 更稳定
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
        )
        
        # Output projection
        self.proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, context):
        """
        Args:
            context: (B, N, d) context representation
        Returns:
            pred: (B, N, d) predicted target
        """
        # Transformer encoding
        encoded = self.transformer(context)  # (B, N, d)
        
        # Projection
        pred = self.proj(encoded)
        pred = self.norm(pred)
        return pred


class TransformerPredictorBlock(nn.Module):
    """
    Transformer Predictor Block (单独实现)
    
    可选的自定义实现，不依赖 nn.TransformerEncoder
    """
    def __init__(self, d_model=256, n_heads=4, d_ffn=1024, dropout=0.1):
        super().__init__()
        
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        
        # Self-attention
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ffn),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ffn, d_model),
        )
        
        # Norm
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        Args:
            x: (B, N, d)
        """
        # Pre-LN self-attention
        x_norm = self.norm1(x)
        q = self.q_proj(x_norm)
        k = self.k_proj(x_norm)
        v = self.v_proj(x_norm)
        
        # Reshape for multi-head
        B, N, _ = q.shape
        q = q.view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        
        # Attention
        scale = self.d_head ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Output
        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(B, N, -1)
        out = self.out_proj(out)
        x = x + self.dropout(out)
        
        # FFN
        x = x + self.ffn(self.norm2(x))
        
        return x


class LightTransformerPredictor(nn.Module):
    """
    轻量 Transformer Predictor (自定义实现)
    """
    def __init__(self, d_model=256, n_heads=4, n_layers=2, 
                 d_ffn=1024, dropout=0.1):
        super().__init__()
        
        self.layers = nn.ModuleList([
            TransformerPredictorBlock(d_model, n_heads, d_ffn, dropout)
            for _ in range(n_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, context):
        """
        Args:
            context: (B, N, d) context representation
        Returns:
            pred: (B, N, d) predicted target
        """
        for layer in self.layers:
            context = layer(context)
        
        pred = self.norm(context)
        return pred
```

### 3.2.5 Predictor 选择配置

```python
class PredictorFactory:
    """
    Predictor 工厂，根据配置创建合适的 Predictor
    """
    @staticmethod
    def create(predictor_type='mlp', d_model=256, **kwargs):
        """
        Args:
            predictor_type: 'mlp' 或 'transformer'
            d_model: 模型维度
            **kwargs: 传递给 Predictor 的其他参数
        """
        if predictor_type == 'mlp':
            return MLPPredictor(
                d_model=d_model,
                d_hidden=kwargs.get('d_hidden', d_model * 4),
                n_layers=kwargs.get('n_layers', 2),
                dropout=kwargs.get('dropout', 0.1),
            )
        elif predictor_type == 'transformer':
            return TransformerPredictor(
                d_model=d_model,
                n_heads=kwargs.get('n_heads', 4),
                n_layers=kwargs.get('n_layers', 2),
                d_ffn=kwargs.get('d_ffn', d_model * 4),
                dropout=kwargs.get('dropout', 0.1),
            )
        else:
            raise ValueError(f"Unknown predictor type: {predictor_type}")


# 配置示例
PREDICTOR_CONFIGS = {
    'mlp_light': {
        'type': 'mlp',
        'd_hidden': d_model * 2,
        'n_layers': 1,
    },
    'mlp_standard': {
        'type': 'mlp',
        'd_hidden': d_model * 4,
        'n_layers': 2,
    },
    'transformer_light': {
        'type': 'transformer',
        'n_heads': 4,
        'n_layers': 2,
        'd_ffn': d_model * 4,
    },
    'transformer_standard': {
        'type': 'transformer',
        'n_heads': 8,
        'n_layers': 4,
        'd_ffn': d_model * 4,
    },
}
```

---

## 3.3 Component 2: Target Encoder Strategy (EMA)

### 3.3.1 EMA 核心思想

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Exponential Moving Average (EMA)                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  核心思想:                                                            │
│  ───────────────────────────────────────────────────────────────────  │
│                                                                         │
│  θ_online ← 更新 (梯度下降)                                          │
│  θ_target ← β·θ_target + (1-β)·θ_online                           │
│                                                                         │
│  其中 β 通常设为 0.99-0.999                                          │
│                                                                         │
│  ───────────────────────────────────────────────────────────────────  │
│                                                                         │
│  作用:                                                               │
│  • Target Encoder 提供稳定的目标表示                                    │
│  • 不会因为 online encoder 的波动而震荡                               │
│  • 类似于知识蒸馏                                                     │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.3.2 EMA 实现

```python
class EMATargetEncoder:
    """
    EMA Target Encoder Wrapper
    
    定期用 online encoder 的指数移动平均更新 target encoder
    """
    def __init__(self, model, beta=0.99):
        self.model = model
        self.beta = beta
        self.shadow = {}  # 存储 shadow parameters
        
        # 初始化 shadow
        self._init_shadow()
    
    def _init_shadow(self):
        """初始化 shadow parameters"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    @torch.no_grad()
    def update(self):
        """
        更新 target encoder (EMA step)
        
        θ_target ← β·θ_target + (1-β)·θ_online
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.shadow[name] = self.beta * self.shadow[name] + \
                                   (1 - self.beta) * param.data
    
    def get_shadow(self):
        """返回 shadow model 用于计算 target"""
        return self.shadow
    
    @torch.no_grad()
    def forward(self, x):
        """使用 shadow parameters 计算"""
        # 临时替换
        original = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                original[name] = param.data.clone()
                param.data = self.shadow[name]
        
        output = self.model(x)
        
        # 恢复
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = original[name]
        
        return output


class JEPAWithEMA(nn.Module):
    """
    JEPA 模型 (包含 EMA)
    
    支持两种 Predictor: MLP 和 Transformer
    """
    def __init__(self, foundation_model, predictor_type='mlp', 
                 d_model=256, ema_beta=0.99, predictor_kwargs=None):
        super().__init__()
        self.foundation = foundation_model  # Encoder + Latent Hub
        
        # 创建 Predictor (MLP 或 Transformer)
        if isinstance(predictor_type, nn.Module):
            # 直接传入已创建的 predictor
            self.predictor = predictor_type
        else:
            self.predictor = PredictorFactory.create(
                predictor_type=predictor_type,
                d_model=d_model,
                **(predictor_kwargs or {})
            )
        
        # Target encoder: 复制 foundation 模型
        self.target_encoder = copy.deepcopy(foundation_model)
        self.target_encoder.eval()  # 不更新参数
        
        # EMA updater
        self.ema = EMATargetEncoder(self.target_encoder, beta=ema_beta)
        
        # 更新 EMA 的频率
        self.ema_update_every = 1  # 每 step 更新
    
    def update_target_encoder(self):
        """调用这个来更新 EMA"""
        self.ema.update()
    
    def forward(self, batch, predict_masked=True):
        """
        JEPA forward
        
        Args:
            batch: dict of inputs
            predict_masked: 是否只预测 masked 的部分
        """
        # 1. Foundation forward
        H = self.foundation(batch)  # → (B, 64, d)
        
        # 2. Context: 对 H 应用某种 masking
        if predict_masked:
            H_context = self.mask_hidden(H)
        else:
            H_context = H
        
        # 3. Predictor forward
        pred = self.predictor(H_context)  # → (B, 64, d)
        
        # 4. Target forward (no grad)
        with torch.no_grad():
            target = self.target_encoder(batch)  # → (B, 64, d)
        
        return pred, target
    
    def mask_hidden(self, H):
        """
        对 hidden representation 应用 masking
        """
        B, N, d = H.shape
        mask_ratio = 0.3
        
        # 随机 mask
        noise = torch.rand(B, N, device=H.device)
        mask = noise < mask_ratio
        
        H_masked = H.clone()
        H_masked[mask] = 0
        
        return H_masked
```

### 3.3.3 EMA 超参数选择

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    EMA Beta 选择                                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  β = 0.99 (快速 EMA)                                                  │
│  ───────────────────────────────────────────────────────────────────  │
│  • Target 更新快                                                      │
│  • 适合短训练 (< 100 epochs)                                         │
│  • 表示更关注近期更新                                                 │
│                                                                         │
│  β = 0.999 (慢速 EMA)                                                │
│  ───────────────────────────────────────────────────────────────────  │
│  • Target 更新慢                                                      │
│  • 适合长训练 (> 500 epochs)                                         │
│  • 表示更稳定                                                         │
│                                                                         │
│  推荐:                                                                │
│  ───────────────────────────────────────────────────────────────────  │
│  • 训练前期: β = 0.99 (快速适应)                                    │
│  • 训练后期: β = 0.999 (稳定表示)                                   │
│                                                                         │
│  调度策略:                                                            │
│  ───────────────────────────────────────────────────────────────────  │
│  β_schedule = 0.99 → 0.999 (线性或余弦)                              │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 3.4 Component 3: Masking Strategy (Training Paradigm)

### 3.4.1 JEPA Masking vs MAE Masking

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Masking 策略对比                                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  MAE Masking:                                                         │
│  ───────────────────────────────────────────────────────────────────  │
│  • 物理空间 masking: mask 掉 75% 的 tokens                         │
│  • 在像素/特征层面进行                                                │
│  • Encoder 只处理 visible tokens                                     │
│  • 目标是重建被 mask 的像素                                           │
│                                                                         │
│  JEPA Masking:                                                        │
│  ───────────────────────────────────────────────────────────────────  │
│  • 语义空间 masking: mask 掉 latent 表示的某些部分                  │
│  • 在 representation 层面进行                                        │
│  • Encoder 处理全部 tokens，只是部分被 mask                         │
│  • 目标是预测被 mask 的 latent 表示                                    │
│                                                                         │
│  ───────────────────────────────────────────────────────────────────  │
│                                                                         │
│  关键差异:                                                            │
│  • MAE: Encoder 跳过 masked tokens                                  │
│  • JEPA: Encoder 处理全部，Predictor 预测 masked                     │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.4.2 JEPA Masking 类型

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    JEPA Masking 类型                                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  1. Token Masking (ROI/Temporal/Patch)                               │
│  ───────────────────────────────────────────────────────────────────  │
│  • 随机遮挡部分 tokens                                               │
│  • mask_ratio: 30-50%                                              │
│  • 用于 intra-modality 预测                                          │
│                                                                         │
│  2. Modality Masking                                                 │
│  ───────────────────────────────────────────────────────────────────  │
│  • 随机丢弃某个模态的全部表示                                        │
│  • 用于 cross-modality 预测                                          │
│  • 迫使模型学习跨模态对应关系                                         │
│                                                                         │
│  3. Temporal Masking (fMRI 专用)                                     │
│  ───────────────────────────────────────────────────────────────────  │
│  • mask 掉某些时间窗口                                              │
│  • 用于时间序列预测                                                  │
│                                                                         │
│  4. Curriculum Masking                                                │
│  ───────────────────────────────────────────────────────────────────  │
│  • Stage 1: 只做 intra-modality                                     │
│  • Stage 2: 逐步引入 cross-modality                                 │
│  • Stage 3: 稳定训练                                                │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.4.3 Curriculum Masking Schedule

```python
class JEPACurriculumMasking:
    """
    JEPA 课程式 Masking
    
    渐进式引入不同类型的 masking
    """
    def __init__(self, total_epochs=500):
        self.total_epochs = total_epochs
        
        # Stage 配置
        self.stages = {
            'warmup': (0, 100),
            'alignment': (100, 300),
            'refinement': (300, 500),
        }
        
        # 各阶段配置
        self.mask_configs = {
            'warmup': {
                'token_mask': 0.3,       # ROI/token masking
                'temporal_mask': 0.2,    # 时间 masking
                'modality_mask': 0.0,    # 跨模态 masking
                'connectivity_mask': 0.2, # 连接 masking
            },
            'alignment': {
                'token_mask': 0.5,
                'temporal_mask': 0.3,
                'modality_mask': 0.3,
                'connectivity_mask': 0.3,
            },
            'refinement': {
                'token_mask': 0.5,
                'temporal_mask': 0.3,
                'modality_mask': 0.3,
                'connectivity_mask': 0.3,
            },
        }
    
    def get_config(self, epoch):
        """获取当前 epoch 的 masking 配置"""
        if epoch < 100:
            stage = 'warmup'
        elif epoch < 300:
            stage = 'alignment'
            # 线性插值
            t = (epoch - 100) / 200
            return self._interpolate(self.mask_configs['warmup'],
                                    self.mask_configs['alignment'], t)
        else:
            return self.mask_configs['refinement']
    
    def _interpolate(self, cfg1, cfg2, t):
        """线性插值"""
        return {k: cfg1[k] * (1 - t) + cfg2[k] * t for k in cfg1.keys()}
```

### 3.4.4 Masking 曲线

```
Mask Ratio
    │
0.5 ┤                    ═══════════
    │                ════╲
0.3 ┤            ═══════  ╲═══════  ← Modality Masking
    │        ════╱          ╲
0.2 ┤    ════╱              ╲═══════  ← Token/Temporal Masking
    │  ═══╱
0.0 ┼──╱──────────────────────────────→ Epoch
    0    100    200    300    400    500
    
    ├─Warmup─┤├────Alignment────┤├──Refinement─┤
```

---

## 3.5 Component 4: Loss Function Design

### 3.5.1 JEPA Loss 公式

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    JEPA Loss 设计                                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  L_total = L_JEPA + λ_vicreg · L_VICReg                              │
│                                                                         │
│  其中:                                                                 │
│  ───────────────────────────────────────────────────────────────────  │
│                                                                         │
│  L_JEPA = SoftL1(pred, target.detach())                              │
│  • 主要驱动预测任务                                                   │
│  • 预测被 mask 的 latent 表示                                        │
│                                                                         │
│  L_VICReg = L_inv + L_var + L_cov                                   │
│  • 防止表示 collapse                                                │
│  • 鼓励表示多样性                                                    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.5.2 Soft-L1 Loss 实现

```python
def soft_l1_loss(pred, target, mask=None):
    """
    Soft-L1 (Huber) Loss for JEPA
    
    Args:
        pred: (B, N, d) 预测
        target: (B, N, d) 目标 (EMA)
        mask: (B, N) 可选，只计算 masked 位置
    """
    if mask is not None:
        # 只在被 mask 的位置计算
        loss = F.smooth_l1_loss(pred, target, reduction='none')
        # mask: 1 = masked, 0 = visible
        loss = (loss * mask.unsqueeze(-1)).sum()
        loss = loss / (mask.sum() + 1e-8)
    else:
        loss = F.smooth_l1_loss(pred, target, reduction='mean')
    
    return loss
```

### 3.5.3 VICReg Loss 实现

```python
class VICRegLoss(nn.Module):
    """
    VICReg: Variance, Invariance, Covariance
    
    防止表示 collapse，鼓励维度解耦
    """
    def __init__(self, lambda_inv=1.0, lambda_var=25.0, lambda_cov=1.0, gamma=1.0):
        super().__init__()
        self.lambda_inv = lambda_inv
        self.lambda_var = lambda_var
        self.lambda_cov = lambda_cov
        self.gamma = gamma
    
    def forward(self, z1, z2):
        """
        Args:
            z1, z2: (B, N, d) 两个表示 (可以是相同的)
        """
        B, N, d = z1.shape
        
        # 1. Invariance: 确保表示稳定
        L_inv = ((z1 - z2) ** 2).mean()
        
        # 2. Variance: 每个维度有足够方差
        std_z1 = torch.sqrt(z1.var(dim=0) + 1e-4)
        std_z2 = torch.sqrt(z2.var(dim=0) + 1e-4)
        L_var = torch.relu(self.gamma - std_z1).mean() + \
                torch.relu(self.gamma - std_z2).mean()
        
        # 3. Covariance: 减少维度间相关性
        z1_norm = (z1 - z1.mean(0)) / (z1.std(0) + 1e-4)
        C1 = (z1_norm.transpose(0,1) @ z1_norm) / B
        off_diag1 = C1 - torch.diag(torch.diag(C1))
        L_cov = (off_diag1 ** 2).sum() / d
        
        # 总损失
        L_total = self.lambda_inv * L_inv + \
                  self.lambda_var * L_var + \
                  self.lambda_cov * L_cov
        
        return L_total, {
            'L_inv': L_inv.item(),
            'L_var': L_var.item(),
            'L_cov': L_cov.item(),
        }
```

### 3.5.4 完整 JEPA Loss 类

```python
class JEPALoss(nn.Module):
    """
    完整的 JEPA Loss
    """
    def __init__(self, lambda_vicreg=0.5, lambda_mae=1.0):
        super().__init__()
        self.lambda_vicreg = lambda_vicreg
        self.lambda_mae = lambda_mae
        self.vicreg = VICRegLoss()
    
    def forward(self, pred, target, mask, context=None):
        """
        Args:
            pred: (B, N, d) Predictor 输出
            target: (B, N, d) EMA Target
            mask: (B, N) 1=masked
            context: (B, N, d) Context Encoder 输出 (用于 VICReg)
        """
        # 1. Soft-L1 Loss (只在 masked 位置)
        L_jepa = soft_l1_loss(pred, target, mask)
        
        # 2. VICReg (可选)
        L_vicreg = 0
        if context is not None and self.lambda_vicreg > 0:
            L_vicreg, vicreg_metrics = self.vicreg(context, target.detach())
        else:
            vicreg_metrics = {}
        
        # 3. 总损失
        L_total = L_jepa + self.lambda_vicreg * L_vicreg
        
        return L_total, {
            'L_jepa': L_jepa.item(),
            'L_vicreg': L_vicreg.item() if isinstance(L_vicreg, torch.Tensor) else L_vicreg,
            **vicreg_metrics
        }
```

### 3.5.5 Loss 系数调度

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Loss 系数调度                                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  λ_vicreg: 0.25 → 0.5                                              │
│  ───────────────────────────────────────────────────────────────────  │
│                                                                         │
│  • 前期: 弱正则化，让预测损失主导                                     │
│  • 后期: 强正则化，防止 collapse                                     │
│                                                                         │
│  调度:                                                                │
│  epoch 0-100:   λ_vicreg = 0.25                                      │
│  epoch 100-300: λ_vicreg = 0.25 → 0.5 (线性)                        │
│  epoch 300+:   λ_vicreg = 0.5                                       │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 3.6 Component 5: 完整训练循环

### 3.6.1 训练步骤

```python
class JEPATrainer:
    """JEPA 训练器"""
    def __init__(self, model, optimizer, mask_scheduler, loss_fn):
        self.model = model
        self.optimizer = optimizer
        self.mask_scheduler = mask_scheduler
        self.loss_fn = loss_fn
        self.epoch = 0
    
    def train_step(self, batch):
        """单个训练步骤"""
        # 1. 获取 masking 配置
        mask_cfg = self.mask_scheduler.get_config(self.epoch)
        
        # 2. JEPA forward
        pred, target = self.model(batch, predict_masked=True)
        
        # 3. 应用 masking
        mask = self._create_mask(pred.shape, mask_cfg)
        
        # 4. 计算损失
        context = self.model.foundation(batch)  # 需要修改 forward 返回 context
        loss, metrics = self.loss_fn(pred, target, mask, context)
        
        # 5. 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # 6. 更新 EMA
        self.model.update_target_encoder()
        
        return loss.item(), metrics
    
    def _create_mask(self, shape, mask_cfg):
        """创建 mask tensor"""
        B, N, d = shape
        mask_ratio = mask_cfg['token_mask']
        
        noise = torch.rand(B, N, device=shape[0].device if isinstance(shape[0], torch.Tensor) else 'cpu')
        mask = (noise < mask_ratio).float()
        
        return mask
```

### 3.6.2 训练监控

```python
training_metrics = {
    # 主要损失
    'loss/total': '总损失',
    'loss/L_jepa': 'JEPA 预测损失 (Soft-L1)',
    
    # VICReg
    'loss/L_vicreg': 'VICReg 正则化',
    'loss/L_inv': 'Invariance 损失',
    'loss/L_var': 'Variance 损失',
    'loss/L_cov': 'Covariance 损失',
    
    # Masking
    'mask/token_ratio': 'Token masking 比例',
    'mask/modality_ratio': 'Modality masking 比例',
    
    # EMA
    'ema/beta': '当前 EMA beta',
    
    # 表示质量
    'repr/std': '表示标准差',
    'repr/mean': '表示均值',
    
    # 训练稳定性
    'grad/norm': '梯度范数',
}
```

---

## 3.7 JEPA 设计总结

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    JEPA 设计总结                                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Component 1: Predictor Structure                                    │
│  ───────────────────────────────────────────────────────────────────  │
│                                                                         │
│  Option A: MLP Predictor (轻量)                                      │
│  • Linear(d, 4d) → GELU → Linear(4d, d)                            │
│  • 轻量，计算效率高                                                  │
│                                                                         │
│  Option B: Transformer Predictor (轻量)                               │
│  • Transformer Encoder (2 layers, 4 heads)                           │
│  • 可捕获 token 之间的序列关系                                        │
│                                                                         │
│  Component 2: Target Encoder (EMA)                                    │
│  ───────────────────────────────────────────────────────────────────  │
│  • 复制 foundation encoder                                            │
│  • EMA 更新: θ_target = β·θ_target + (1-β)·θ_online                │
│  • β = 0.99 → 0.999 (可调度)                                        │
│                                                                         │
│  Component 3: Masking Strategy                                        │
│  ───────────────────────────────────────────────────────────────────  │
│  • Curriculum Masking: 渐进引入 masking                               │
│  • Stage 1: 只 intra-modality                                        │
│  • Stage 2: 引入 cross-modality                                       │
│  • Stage 3: 稳定训练                                                 │
│                                                                         │
│  Component 4: Loss Function                                          │
│  ───────────────────────────────────────────────────────────────────  │
│  • L_JEPA = SoftL1(pred, target)                                     │
│  • L_VICReg = L_inv + L_var + L_cov                                 │
│  • L_total = L_JEPA + λ·L_VICReg                                    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

**下一步**：→ [04_MAE_training_design.md](./04_MAE_training_design.md) - MAE Training Design
