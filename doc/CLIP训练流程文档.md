# CLIP雷达数据训练流程文档

## 概述

本文档详细介绍了当前代码中基于雷达数据的CLIP（Contrastive Language-Image Pre-training）模型训练流程。该系统实现了多视图雷达数据（距离-时间、多普勒-时间、方位-时间）与文本描述之间的对比学习。

## 1. 系统架构

### 1.1 核心组件

- **RadarEncoder**: 三视图雷达数据并行编码器
- **TextEncoder**: 文本编码器
- **Transformer**: 用于处理融合雷达特征的转换器
- **CLIP**: 主模型类，整合所有组件

### 1.2 数据流程

```
雷达数据(.npz) → 数据预处理 → RadarEncoder → Transformer → 特征归一化 → 对比学习损失
文本描述 → Token化 → TextEncoder → 特征归一化 → 对比学习损失
```

## 2. 数据处理流程

### 2.1 雷达数据格式

雷达数据存储在NPZ文件中，包含三个视图：
- **range_time**: 距离-时间频谱 (256 × T)
- **doppler_time**: 多普勒-时间频谱 (128 × T)
- **azimuth_time**: 方位-时间频谱 (128 × T)

### 2.2 数据预处理

#### 2.2.1 裁剪和填充
```python
def _crop_radar_view(radar_view, opt):
    """将雷达视图裁剪到单位长度边界"""
    T = radar_view.shape[1]
    T = (T // opt.unit_length) * opt.unit_length  # 确保长度是unit_length的倍数
    idx = random.randint(0, radar_view.shape[1] - T)
    radar_view = radar_view[:, idx: idx + T]
    return radar_view, T
```

#### 2.2.2 归一化处理
```python
def _apply_radar_normalization(radar_view, opt):
    """根据策略对单个雷达视图应用归一化"""
    normalize_type = opt.get('normalize', 'none')

    if normalize_type == 'per_frame':
        return _normalize_per_frame(radar_view)
    elif normalize_type == 'global':
        return _normalize_global(radar_view)
    elif normalize_type == 'log':
        return _normalize_log(radar_view)
    else:  # 'none'
        return radar_view
```

#### 2.2.3 自适应分块处理
系统支持自适应分块大小，确保三个视图具有相同的序列长度：
- 距离视图: (32, 16) → 8×31 = 248 tokens
- 多普勒视图: (16, 16) → 8×31 = 248 tokens
- 方位视图: (16, 16) → 8×31 = 248 tokens

## 3. 模型架构详解

### 3.1 RadarEncoder类

RadarEncoder采用三视图并行编码架构：

```python
class RadarEncoder(nn.Module):
    def __init__(self, model_name, embed_dim, fusion_method='add',
                 adaptive_patch_size=False, **kwargs):
        # 创建三个独立的编码器
        self.range_encoder = self._create_encoder(...)
        self.doppler_encoder = self._create_encoder(...)
        self.azimuth_encoder = self._create_encoder(...)

        # 融合层
        if fusion_method == 'concat':
            self.fusion_proj = nn.Linear(embed_dim * 3, embed_dim)
        elif fusion_method == 'add':
            # 简单相加融合
            pass
        elif fusion_method == 'attention':
            self.attention = nn.MultiheadAttention(embed_dim, num_heads=8)
```

#### 3.1.1 视图编码器创建
每个视图使用ViT（Vision Transformer）作为基础编码器：

```python
def _create_encoder(self, model_name, embed_dim, resolution, pretrained, view_type, **kwargs):
    if 'vit' in model_name:
        encoder = _create_vision_transformer(
            model_name,
            pretrained=pretrained,
            in_chans=1,  # 单通道输入
            img_size=resolution,
            num_classes=0,  # 不使用分类头
            patch_size=patch_size  # 自适应分块大小
        )
        # 添加投影层到目标嵌入维度
        encoder.proj = nn.Linear(encoder.embed_dim, embed_dim)
```

#### 3.1.2 特征融合策略

**拼接融合 (concat)**:
```python
if self.fusion_method == 'concat':
    if self.adaptive_patch_size:
        # 堆叠特征: [b, n, 3, embed_dim] → [b, n, embed_dim*3]
        stacked = torch.stack([range_features, doppler_features, azimuth_features], dim=2)
        fused = stacked.view(b, n, n_views * d)
        fused = self.fusion_proj(fused)
```

**加法融合 (add)**:
```python
elif self.fusion_method == 'add':
    # 填充到相同长度后相加
    max_len = max(...)
    fused = (range_padded + doppler_padded + azimuth_padded) / 3.0
```

**注意力融合 (attention)**:
```python
elif self.fusion_method == 'attention':
    # 使用多头注意力融合
    attended, _ = self.attention(reshaped, reshaped, reshaped)
    fused = attended.mean(dim=2)
```

### 3.2 TextEncoder类

文本编码器使用预训练的Transformer模型：

```python
class TextEncoder(nn.Module):
    def __init__(self, model_name, text_pooling='pooler', unfreeze_last_layer_num=0):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.text_encoder = AutoModel.from_pretrained(model_name)

        # 参数冻结策略
        for name, param in self.text_encoder.named_parameters():
            unfreeze_param = False
            # 根据层级决定是否解冻参数
            if unfreeze_param:
                param.requires_grad = True
            else:
                param.requires_grad = False
```

#### 3.2.1 文本编码过程
```python
def encode(self, text, device='cuda'):
    inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = self.text_encoder(**inputs)

    if self.text_pooling == 'mean':
        out = outputs.last_hidden_state.mean(dim=1)
    elif self.text_pooling == 'pooler':
        out = outputs.pooler_output
    elif self.text_pooling == 'max':
        out = outputs.last_hidden_state.max(dim=1)[0]
    return out
```

### 3.3 Transformer处理器

用于处理融合后的雷达特征：

```python
class Transformer(nn.Module):
    def __init__(self, width, layers, heads, mlp_ratio=4, attn_mask=None):
        self.resblocks = nn.Sequential(*[
            ResidualAttentionBlock(width, heads, mlp_ratio, attn_mask)
            for _ in range(layers)
        ])
```

### 3.4 CLIP主模型

```python
class CLIP(pl.LightningModule):
    def __init__(self, encoder_cfg, text_cfg, context_length,
                 transformer_width, transformer_layers, transformer_heads,
                 temperature, learning_rate, max_epochs):

        # 雷达编码器
        self.radar_encoder = RadarEncoder(**encoder_cfg)

        # 文本编码器
        self.text_encoder = TextEncoder(**text_cfg)
        self.text_projection = nn.Linear(text_embed_dim, encoder_embed_dim)

        # Transformer设置
        self.context_length = context_length
        self.transformer = Transformer(...)
        self.positional_embedding = nn.Parameter(torch.empty(context_length, transformer_width))
        self.eot_token = nn.Parameter(scale * torch.randn(transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        # 损失函数
        self.loss_fn = create_loss(loss_args)
```

## 4. 训练流程

### 4.1 数据加载

```python
class HumanDInterface(DInterface):
    def setup(self, stage=None):
        if stage == 'fit':
            self.train_dataset = Text2DopplerDatasetV2(opt, train_split, train_ratio)
            self.val_dataset = Text2DopplerDatasetV2(opt, val_split, val_ratio)
            self.test_dataset = Text2DopplerDatasetV2(opt, test_split, test_ratio)
```

### 4.2 前向传播

```python
def forward(self, radar_data, text):
    # 编码雷达数据
    radar_features = self.encode_radar(
        radar_data['range_time'],
        radar_data['doppler_time'],
        radar_data['azimuth_time']
    )

    # 编码文本
    text_features = self.encode_text(text, device=radar_data['range_time'].device)

    # 特征归一化
    radar_features = radar_features / radar_features.norm(dim=1, keepdim=True)
    text_features = text_features / text_features.norm(dim=1, keepdim=True)

    return radar_features, text_features
```

#### 4.2.1 雷达编码详细过程
```python
def encode_radar(self, range_data, doppler_data, azimuth_data):
    # 1. 三视图并行编码
    x = self.radar_encoder(range_data, doppler_data, azimuth_data)  # [b, n, embed_dim]

    # 2. 添加EOT标记
    x = torch.cat([x, self.eot_token.to(x.dtype) + torch.zeros_like(x[:, :1, :])], dim=1)

    # 3. 添加位置编码
    x = x + self.positional_embedding.to(x.dtype)

    # 4. Transformer处理
    x = self.transformer(x)

    # 5. 获取EOT标记表示
    x = self.ln_final(x[:, -1, :])

    return x
```

### 4.3 损失计算

使用对比学习损失函数：

```python
def compute_loss(self, batch):
    radar_features = batch['radar_features']
    text_features = batch['text_features']

    if not self.use_siglip:
        loss_clip = self.loss_fn(radar_features, text_features, logit_scale=self.logit_scale)
    else:
        loss_clip = self.loss_fn(radar_features, text_features,
                                logit_scale=self.logit_scale, logit_bias=self.logit_bias)
    return {'loss_clip': loss_clip}
```

### 4.4 训练步骤

```python
def training_step(self, batch, batch_idx):
    # 1. 共享步骤处理数据
    batch = self.shared_step(batch, batch_idx, phase='train')

    # 2. 计算损失
    loss_dict = self.compute_loss(batch)

    # 3. 记录损失
    self.log_loss(loss_dict, phase='train')
    self.log('train_loss', loss_dict['loss_clip'], prog_bar=True)

    return loss_dict['loss_clip']
```

#### 4.4.1 共享步骤详解
```python
def shared_step(self, batch, batch_idx, phase='train'):
    self.batch_size = batch['input_wave_range'].size(0)

    # 构建雷达数据字典
    radar_data = {
        'range_time': batch['input_wave_range'],    # [b, 256, T]
        'doppler_time': batch['input_wave_doppler'],  # [b, 128, T]
        'azimuth_time': batch['input_wave_azimuth']   # [b, 128, T]
    }
    text = batch['caption']

    # 前向传播
    radar_features, text_features = self.forward(radar_data, text)

    # 更新批次数据
    batch['radar_features'] = radar_features
    batch['image_features'] = radar_features  # 兼容性
    batch['text_features'] = text_features

    return batch
```

### 4.5 优化器配置

```python
def configure_optimizers(self):
    lr = self.learning_rate

    opt = torch.optim.AdamW([
        {'params': self.text_encoder.parameters(), 'lr': lr / 2},      # 文本编码器学习率减半
        {'params': self.text_projection.parameters(), 'lr': lr},
        {'params': self.radar_encoder.parameters(), 'lr': lr},
        {'params': self.transformer.parameters(), 'lr': lr},
    ], betas=(0.5, 0.9), weight_decay=0.01)

    # 余弦退火学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.max_epochs, eta_min=0)

    return [opt], [scheduler]
```

## 5. 配置文件详解

### 5.1 训练参数配置

```yaml
model_cfg:
  target: src.model.clip.CLIP
  params:
    max_epochs: 50
    learning_rate: 1.0e-04
    temperature: 0.07              # CLIP温度参数
```

### 5.2 雷达编码器配置

```yaml
encoder_cfg:
  model_name: 'vit_base_patch16_clip_224.openai'  # ViT模型
  embed_dim: 256                               # 特征嵌入维度
  fusion_method: 'add'                         # 融合方法

  # 雷达信号分辨率
  range_resolution: [256, 496]    # 距离-时间频谱维度
  doppler_resolution: [128, 496]  # 多普勒-时间频谱维度
  azimuth_resolution: [128, 496]  # 方位-时间频谱维度

  pretrained: false               # 不使用预训练权重
  adaptive_patch_size: true       # 启用自适应分块
```

### 5.3 文本编码器配置

```yaml
text_cfg:
  model_name: 'sentence-transformers/paraphrase-MiniLM-L6-v2'
  embed_dim: 384
  text_pooling: 'pooler'
  unfreeze_last_layer_num: 0
```

### 5.4 Transformer配置

```yaml
context_length: 249  # 雷达编码器序列长度(248) + 1个EOT标记
transformer_width: 256
transformer_heads: 8
transformer_layers: 1
```

## 6. 关键技术点

### 6.1 自适应分块技术

- **问题**: 三个雷达视图具有不同的空间分辨率
- **解决方案**: 为每个视图计算最优分块大小，确保统一的序列长度
- **实现**:
  - 距离视图 (256×496) → 32×16分块 → 8×31=248 tokens
  - 多普勒视图 (128×496) → 16×16分块 → 8×31=248 tokens
  - 方位视图 (128×496) → 16×16分块 → 8×31=248 tokens

### 6.2 多视图融合策略

支持三种融合方式：
1. **拼接融合**: 沿特征维度拼接，通过线性层投影
2. **加法融合**: 元素级相加平均
3. **注意力融合**: 使用多头注意力机制学习最优融合权重

### 6.3 对比学习

使用InfoNCE损失函数进行对比学习：
```python
# 计算相似度矩阵
logits = torch.matmul(radar_features, text_features.t()) * logit_scale
# 创建标签
labels = torch.arange(batch_size, device=device)
# 计算交叉熵损失
loss_i2t = F.cross_entropy(logits, labels)
loss_t2i = F.cross_entropy(logits.t(), labels)
loss = (loss_i2t + loss_t2i) / 2
```

## 7. 训练监控

### 7.1 损失记录

系统使用SwanLab进行训练监控：
- 训练损失: `train_loss`
- 验证损失: `val_loss`
- 学习率: `lr`

### 7.2 模型检查点

支持从检查点恢复训练：
```bash
python run_model.py --config config/clip.yaml --resume-checkpoint path/to/checkpoint.ckpt
```

## 8. 运行命令

### 8.1 启动训练

```bash
python run_model.py --config config/clip.yaml --resume-checkpoint checkpoints/model.ckpt
```

### 8.2 分布式训练

系统支持DDP（Distributed Data Parallel）训练：
```bash
torchrun --nproc_per_node=2 run_model.py --config config/clip.yaml
```

## 9. 性能优化建议

### 9.1 数据加载优化

- 增加`num_workers`数量以加速数据加载
- 使用`pin_memory=True`加速GPU数据传输
- 调整`batch_size`以平衡内存使用和训练效率

### 9.2 模型优化

- 根据GPU内存调整模型大小
- 使用混合精度训练（FP16）减少内存使用
- 考虑梯度累积以支持更大的有效批次大小

### 9.3 超参数调优

- 学习率: 建议从1e-4开始，根据收敛情况调整
- 温度参数: 影响对比学习的难度，通常0.07-0.1
- 融合方法: 根据任务需求选择合适的融合策略

## 10. 故障排除

### 10.1 常见错误

1. **维度不匹配错误**
   - 检查`embed_dim`和`transformer_width`是否一致
   - 确保`context_length`包含EOT标记

2. **数据类型错误**
   - 确保所有数据转换为float32
   - 检查模型参数和数据类型匹配

3. **内存不足错误**
   - 减小`batch_size`
   - 降低图像分辨率或模型复杂度

### 10.2 调试技巧

- 使用小数据集测试代码正确性
- 打印中间张量形状验证数据流
- 检查梯度是否正常计算

---

本文档详细介绍了CLIP雷达数据训练的完整流程，为后续的模型优化和功能扩展提供了参考基础。