# CLIP雷达数据训练流程文档

## 概述

本文档详细介绍了mmExpert框架中基于雷达数据的CLIP（Contrastive Language-Image Pre-training）模型训练流程。该系统基于现代化的抽象层架构设计，实现了多视图雷达数据（距离-时间、多普勒-时间、方位-时间）与文本描述之间的对比学习，支持序列级相似度计算和分布式训练。

## 1. 系统架构

### 1.1 核心组件

基于新的抽象层架构，系统包含以下核心组件：

- **CLIPModel**: 主模型类（位于 `src/model/clip_model.py`），整合所有组件
- **RadarEncoder**: 雷达数据编码器（位于 `src/encoders/radar_encoder.py`）
- **TextEncoder**: 文本编码器（位于 `src/encoders/text_encoder.py`）
- **ClipLoss/SigLipLoss**: 损失函数实现（位于 `src/model/clip_loss.py`）
- **SequenceSimilarity**: 序列相似度计算（位于 `src/model/sequence_similarity.py`）
- **抽象层**: 核心接口定义（位于 `src/core/base.py`）

### 1.2 抽象层设计

系统采用现代化的抽象层设计：

```python
# 核心抽象类
class BaseEncoder:           # 编码器基类
class BaseModel:             # 模型基类
class ModalityData:          # 多模态数据容器
class EncodingResult:        # 编码结果容器
class ModalityType:          # 模态类型枚举
```

### 1.3 数据流程

```
雷达数据 → ModalityData → RadarEncoder → EncodingResult → 特征归一化 → 对比学习损失
文本描述 → ModalityData → TextEncoder → EncodingResult → 特征归一化 → 对比学习损失
                                   ↘ 序列相似度计算（可选）
```

### 1.4 注册系统

系统使用装饰器进行组件注册：

```python
@register_model(name="clip_model", ...)
class CLIPModel(BaseModel): ...

@register_encoder(name="radar_encoder", ...)
class RadarEncoder(BaseEncoder): ...
```

## 2. 数据处理流程

### 2.1 雷达数据格式

雷达数据以字典格式传递给编码器，包含三个视图：
- **range_time**: 距离-时间频谱 (256 × T)
- **doppler_time**: 多普勒-时间频谱 (128 × T)
- **azimuth_time**: 方位-时间频谱 (128 × T)

### 2.2 数据加载器

系统使用 `HumanDInterface` 数据加载器：

```python
class HumanDInterface(DInterface):
    def setup(self, stage=None):
        if stage == 'fit':
            self.train_dataset = Text2DopplerDatasetV2(opt, train_split, train_ratio)
            self.val_dataset = Text2DopplerDatasetV2(opt, val_split, val_ratio)
            self.test_dataset = Text2DopplerDatasetV2(opt, test_split, test_ratio)
```

### 2.3 ModalityData封装

所有数据通过 `ModalityData` 类进行封装：

```python
# 雷达数据封装
radar_data = ModalityData(
    data=radar_dict,
    modality=ModalityType.RADAR,
    metadata={"format": "multi_view"}
)

# 文本数据封装
text_data = ModalityData(
    data=text_list,
    modality=ModalityType.TEXT,
    metadata={"format": "string_list"}
)
```

### 2.4 雷达数据预处理

#### 2.4.1 多视图处理
RadarEncoder支持多视图雷达数据的并行处理：

```python
def _encode_multi_view(self, radar_data: Dict[str, torch.Tensor]) -> torch.Tensor:
    """编码多视图雷达数据"""
    view_features = []

    # 处理每个可用视图
    for view_name, encoder in self.view_encoders.items():
        if view_name in radar_data:
            view_data = radar_data[view_name]
            # 确保数据格式为 [batch, channels, time]
            encoded_view = encoder(view_data)
            view_features.append(encoded_view)
```

#### 2.4.2 卷积编码
每个视图使用1D卷积网络进行编码：

```python
def _create_view_encoder(self, input_dim: int) -> nn.Module:
    """为单个雷达视图创建编码器"""
    return nn.Sequential(
        nn.Conv1d(input_dim, self.embed_dim // 2, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv1d(self.embed_dim // 2, self.embed_dim, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool1d(self.max_sequence_length)
    )
```

### 2.5 文本数据预处理

文本编码器使用预训练的Transformer模型：

```python
def encode(self, data: ModalityData, **kwargs) -> EncodingResult:
    """编码文本数据"""
    text_inputs = self.tokenizer(
        data.data,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=self.max_length
    )

    outputs = self.text_encoder(**text_inputs)

    # 根据池化策略获取特征
    if self.pooling_strategy == 'cls':
        features = outputs.last_hidden_state[:, 0]  # CLS token
    elif self.pooling_strategy == 'mean':
        features = outputs.last_hidden_state.mean(dim=1)
    # ...
```

## 3. 模型架构详解

### 3.1 CLIPModel主模型

CLIPModel是基于PyTorch Lightning的主模型类，继承自BaseModel：

```python
@register_model(name="clip_model")
class CLIPModel(pl.LightningModule):
    def __init__(self,
                 name: str = "clip_model",
                 modality_types: List[str] = None,
                 embed_dim: int = 512,
                 temperature: float = 0.07,
                 use_siglip: bool = False,
                 learning_rate: float = 1e-4,
                 max_epochs: int = 50,
                 encoder_configs: Dict[str, Any] = None,
                 use_sequence_similarity: bool = False,
                 **kwargs):
```

#### 3.1.1 初始化参数
- `embed_dim`: 特征嵌入维度
- `temperature`: CLIP温度参数
- `use_siglip`: 是否使用SigLIP损失
- `use_sequence_similarity`: 是否启用序列相似度计算
- `encoder_configs`: 编码器配置字典

#### 3.1.2 统一前向传播
```python
def forward(self,
            radar_data: Dict[str, torch.Tensor],
            text: List[str],
            return_sequences: bool = False,
            compute_loss: bool = False,
            **kwargs):
    """统一的前向传播入口点"""
    # 统一编码
    encoding_results = self._encode_data(radar_data, text, return_sequences)

    # 提取特征
    radar_features = encoding_results[ModalityType.RADAR].features
    text_features = encoding_results[ModalityType.TEXT].features

    # 特征归一化
    radar_features = F.normalize(radar_features, p=2, dim=-1)
    text_features = F.normalize(text_features, p=2, dim=-1)

    if compute_loss:
        return self._compute_losses_from_features(radar_features, text_features, encoding_results)
    # ...
```

### 3.2 RadarEncoder类

RadarEncoder继承自BaseEncoder，处理多视图雷达数据：

```python
@register_encoder(name="radar_encoder")
class RadarEncoder(BaseEncoder):
    def __init__(self,
                 embed_dim: int = 512,
                 input_dims: Dict[str, int] = None,
                 dropout: float = 0.1,
                 max_sequence_length: int = 496,
                 use_layer_norm: bool = True,
                 use_positional_encoding: bool = False,
                 **kwargs):
```

#### 3.2.1 视图编码器创建
每个视图使用1D卷积网络：

```python
def _create_view_encoder(self, input_dim: int) -> nn.Module:
    """为单个雷达视图创建编码器"""
    return nn.Sequential(
        nn.Conv1d(input_dim, self.embed_dim // 2, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv1d(self.embed_dim // 2, self.embed_dim, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool1d(self.max_sequence_length)
    )
```

#### 3.2.2 多视图融合
```python
def _encode_multi_view(self, radar_data: Dict[str, torch.Tensor]) -> torch.Tensor:
    """编码多视图雷达数据"""
    view_features = []

    # 处理每个可用视图
    for view_name, encoder in self.view_encoders.items():
        if view_name in radar_data and radar_data[view_name] is not None:
            encoded_view = encoder(view_data)  # [batch, embed_dim, seq_len]
            encoded_view = encoded_view.transpose(1, 2)  # [batch, seq_len, embed_dim]
            view_features.append(encoded_view)

    # 拼接视图特征并融合
    concatenated = torch.cat(view_features, dim=-1)  # [batch, seq_len, embed_dim * num_views]
    features = self.fusion_layer(concatenated)  # [batch, seq_len, embed_dim]

    return features
```

### 3.3 TextEncoder类

TextEncoder处理文本编码，支持多种预训练模型：

```python
class TextEncoder(BaseEncoder):
    def __init__(self,
                 model_name: str = "sentence-transformers/paraphrase-MiniLM-L6-v2",
                 embed_dim: int = 384,
                 max_length: int = 77,
                 pooling_strategy: str = "pooler",
                 freeze_backbone: bool = False):
```

#### 3.3.1 编码过程
```python
def encode(self, data: ModalityData, return_sequence: bool = False, **kwargs) -> EncodingResult:
    """编码文本数据"""
    # Token化
    text_inputs = self.tokenizer(
        data.data,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=self.max_length
    )

    # 获取模型输出
    outputs = self.text_encoder(**text_inputs)

    # 根据池化策略获取特征
    if self.pooling_strategy == 'cls':
        features = outputs.last_hidden_state[:, 0]  # CLS token
    elif self.pooling_strategy == 'mean':
        features = outputs.last_hidden_state.mean(dim=1)
    elif self.pooling_strategy == 'pooler':
        features = outputs.pooler_output

    # 投影到目标维度
    features = self.projection_layer(features)

    return EncodingResult(features=features, metadata={...})
```

### 3.4 损失函数

系统支持两种损失函数：

#### 3.4.1 ClipLoss（标准CLIP损失）
```python
class ClipLoss(nn.Module):
    def __init__(self, local_loss=False, gather_with_grad=False,
                 cache_labels=True, rank=0, world_size=1, use_horovod=False):

    def forward(self, image_features, text_features, logit_scale):
        """计算CLIP对比损失"""
        # 计算相似度矩阵
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logit_scale * text_features @ image_features.t()

        # 计算交叉熵损失
        labels = self.get_ground_truth(image_features.device, logits_per_image.shape[0])
        loss_i2t = F.cross_entropy(logits_per_image, labels)
        loss_t2i = F.cross_entropy(logits_per_text, labels)

        return (loss_i2t + loss_t2i) / 2
```

#### 3.4.2 SigLipLoss（Sigmoid损失）
```python
class SigLipLoss(nn.Module):
    def forward(self, image_features, text_features, logit_scale, logit_bias):
        """计算SigLIP损失"""
        logits = logit_scale * image_features @ text_features.t() + logit_bias
        # 使用sigmoid交叉熵而不是softmax
        loss = -torch.mean(torch.log1p(torch.exp(-logits)))
        return loss
```

### 3.5 序列相似度计算

当启用序列相似度时，系统提供多种计算策略：

```python
class SequenceSimilarity(nn.Module):
    def __init__(self, embed_dim: int, similarity_type: str = "combined",
                 window_size: int = 16, temperature: float = 0.07):
        """
        Args:
            similarity_type: "global", "local", "attention", "temporal", "combined"
            window_size: 局部相似度的窗口大小
        """
```

#### 3.5.1 相似度计算策略
- **global**: 全局平均池化后计算相似度
- **local**: 滑动窗口局部相似度
- **attention**: 注意力机制相似度
- **temporal**: 时序对齐相似度
- **combined**: 多种策略的组合

## 4. 训练流程

### 4.1 数据加载器配置

系统使用 `HumanDInterface` 作为数据接口：

```python
data_cfg:
  target: src.data_interface.HumanDInterface
  params:
    cfg:
      # 数据集分割
      train_split: ['dataset/HumanML3D/_split/train.json']
      val_split: ['dataset/HumanML3D/_split/val.json']
      test_split: ['dataset/HumanML3D/_split/test.json']

      # 雷达数据配置
      opt:
        max_motion_length: 496      # 最大填充长度
        min_motion_len: 96          # 最小动作长度
        max_text_len: 20            # 最大文本标题长度
        unit_length: 16             # 处理单元长度
        log_norm: true              # 启用对数归一化
        radar_views: 'all'          # 使用的雷达视图

      batch_size: 64                # 批次大小
      num_workers: 1                # 数据加载工作进程数
```

### 4.2 Lightning训练步骤

CLIPModel基于PyTorch Lightning实现训练逻辑：

```python
def training_step(self, batch, batch_idx):
    """Lightning训练步骤"""
    # 从批次中提取数据
    radar_data = batch.get('radar', batch.get('radar_data'))
    text_data = batch.get('text', batch.get('caption'))

    # 获取批次大小用于日志记录
    batch_size = self._get_batch_size_from_batch(batch)

    # 数据验证
    if radar_data is None:
        self.print("警告：批次中未找到雷达数据，使用虚拟数据")
        radar_data = {"range_time": torch.zeros(batch_size, 256, 100),
                     "doppler_time": torch.zeros(batch_size, 128, 100),
                     "azimuth_time": torch.zeros(batch_size, 128, 100)}
    if text_data is None:
        self.print("警告：批次中未找到文本数据，使用虚拟数据")
        text_data = ["虚拟文本"] * batch_size

    # 使用统一的前向方法计算损失
    losses = self.forward(radar_data, text_data, compute_loss=True)

    # 获取主要损失
    main_loss = losses.get('loss_total', losses['loss_clip'])

    # 记录损失
    self._log_training_losses(losses, batch_size, 'train')

    return main_loss
```

### 4.3 验证步骤

```python
def validation_step(self, batch, batch_idx):
    """Lightning验证步骤"""
    # 与训练步骤相同的逻辑，但使用验证数据
    radar_data = batch.get('radar', batch.get('radar_data'))
    text_data = batch.get('text', batch.get('caption'))
    batch_size = self._get_batch_size_from_batch(batch)

    # 计算损失
    losses = self.forward(radar_data, text_data, compute_loss=True)
    main_loss = losses.get('loss_total', losses['loss_clip'])

    # 记录验证损失
    self._log_training_losses(losses, batch_size, 'valid')

    return main_loss
```

### 4.4 损失计算

系统支持多层损失计算：

```python
def _compute_losses_from_features(self,
                                 radar_features: torch.Tensor,
                                 text_features: torch.Tensor,
                                 encoding_results: Dict[ModalityType, EncodingResult]) -> Dict[str, torch.Tensor]:
    """从特征计算所有损失"""
    losses = self._compute_clip_loss_only(radar_features, text_features)

    # 如果启用序列相似度，计算序列损失
    if self.use_sequence_similarity:
        radar_seq = encoding_results[ModalityType.RADAR].sequence_features
        text_seq = encoding_results[ModalityType.TEXT].sequence_features

        if radar_seq is not None and text_seq is not None:
            # 计算序列相似度矩阵
            seq_similarities = self.sequence_similarity(radar_seq, text_seq)

            # 转换为损失
            batch_size = radar_features.size(0)
            seq_labels = torch.arange(batch_size, device=radar_features.device)
            seq_loss = F.cross_entropy(seq_similarities, seq_labels)

            losses["loss_seq"] = seq_loss * self.sequence_similarity_weight
            losses["loss_total"] = losses["loss_clip"] + losses["loss_seq"]

    return losses
```

### 4.5 优化器配置

使用AdamW优化器和余弦退火调度器：

```python
def configure_optimizers(self):
    """配置训练优化器"""
    # 创建不同学习率的参数组
    param_groups = []

    # 文本编码器使用较低学习率
    if hasattr(self, 'text_encoder'):
        param_groups.append({
            'params': self.text_encoder.parameters(),
            'lr': self.learning_rate / 2
        })

    # 雷达编码器使用标准学习率
    if hasattr(self, 'radar_encoder'):
        param_groups.append({
            'params': self.radar_encoder.parameters(),
            'lr': self.learning_rate
        })

    # 损失函数参数
    if hasattr(self, 'criterion'):
        param_groups.append({
            'params': self.criterion.parameters(),
            'lr': self.learning_rate
        })

    # SigLIP特定参数
    if self.use_siglip and hasattr(self, 'logit_scale') and hasattr(self, 'logit_bias'):
        param_groups.append({
            'params': [self.logit_scale, self.logit_bias],
            'lr': self.learning_rate
        })

    optimizer = torch.optim.AdamW(param_groups, betas=(0.5, 0.9), weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.max_epochs, eta_min=0)

    return {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": scheduler,
            "interval": "epoch",
            "frequency": 1,
        },
    }
```

### 4.6 训练脚本

主训练脚本 `run_clip.py` 的关键部分：

```python
if __name__ == '__main__':
    args = parse_args()

    # 加载配置
    cfg = load_yaml(args.config)

    # 数据加载
    data_cfg = cfg.data_cfg
    data_cfg.params.cfg.batch_size = data_cfg.params.cfg.batch_size // args.world_size
    data = instantiate_from_config(data_cfg)
    data.setup('fit')

    # 模型创建
    model_cfg = cfg.model_cfg
    model = instantiate_from_config(model_cfg)

    # SwanLab日志记录
    logger = SwanLabLogger(name=args.version, project='mmExpert')

    # 检查点回调
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(cfg.log_dir, args.version),
        monitor='valid/loss_clip',
        filename='epoch_{epoch:02d}_val_{valid/loss_clip:.4f}',
        save_top_k=10,
        mode='min',
        save_last=True
    )

    # 训练器
    trainer = Trainer(
        accelerator='gpu',
        devices=args.world_size,
        strategy=cfg.strategy,  # 支持'ddp'分布式训练
        logger=logger,
        max_epochs=cfg.model_cfg.params.max_epochs,
        callbacks=[checkpoint_callback]
    )

    # 开始训练
    trainer.fit(model, train_dataloaders=data.train_dataloader(),
                val_dataloaders=[data.val_dataloader()],
                ckpt_path=args.resume_checkpoint)
```

## 5. 配置文件详解

### 5.1 主配置文件结构 (config/clip.yaml)

```yaml
# 日志和分布式训练配置
log_dir: 'log/'
strategy: 'ddp'                    # 分布式训练策略

# 数据配置
data_cfg:
  target: src.data_interface.HumanDInterface
  params:
    cfg:
      # 数据集分割
      train_split: ['dataset/HumanML3D/_split/train.json']
      train_ratio: [1.0]
      val_split: ['dataset/HumanML3D/_split/val.json']
      val_ratio: [1.0]
      test_split: ['dataset/HumanML3D/_split/test.json']
      test_ratio: [1.0]

      # 雷达数据处理配置
      opt:
        max_motion_length: 496      # 最大动作长度
        min_motion_len: 96          # 最小动作长度
        max_text_len: 20            # 最大文本长度
        unit_length: 16             # 单元长度
        log_norm: true              # 启用对数归一化
        radar_views: 'all'          # 雷达视图选择

      batch_size: 64                # 批次大小
      num_workers: 1                # 数据加载进程数

# 模型配置
model_cfg:
  target: src.model.clip_model.CLIPModel
  params:
    # 训练参数
    max_epochs: 200
    learning_rate: 1.0e-04
    temperature: 0.07              # CLIP温度参数
    use_siglip: false              # 是否使用SigLIP损失

    # 序列相似度配置
    use_sequence_similarity: false
    sequence_similarity_type: "combined"
    sequence_similarity_weight: 0.5
    sequence_similarity_window_size: 16

    # 编码器配置
    encoder_configs:
      radar:
        embed_dim: 256             # 雷达特征维度
        dropout: 0.1               # Dropout率
        max_sequence_length: 496   # 最大序列长度
        use_layer_norm: true       # 启用层归一化
        use_positional_encoding: false

      text:
        model_name: 'sentence-transformers/paraphrase-MiniLM-L6-v2'
        embed_dim: 256
        max_length: 77
        pooling_strategy: 'pooler'   # 'cls', 'mean', 'pooler', 'max'
        freeze_backbone: false       # 是否冻结骨干网络
```

### 5.2 SigLIP配置 (config/siglip.yaml)

```yaml
# 基础配置与clip.yaml相同，但启用SigLIP损失
model_cfg:
  target: src.model.clip_model.CLIPModel
  params:
    max_epochs: 200
    learning_rate: 1.0e-04
    temperature: 0.07
    use_siglip: true              # 启用SigLIP损失

    # 编码器配置保持相同
    encoder_configs:
      radar:
        embed_dim: 256
        dropout: 0.1
      text:
        model_name: 'sentence-transformers/paraphrase-MiniLM-L6-v2'
        embed_dim: 256
```

### 5.3 序列相似度配置

启用序列相似度的配置示例：

```yaml
model_cfg:
  params:
    # 启用序列相似度计算
    use_sequence_similarity: true
    sequence_similarity_type: "combined"  # "global", "local", "attention", "temporal", "combined"
    sequence_similarity_weight: 0.5       # 序列损失权重
    sequence_similarity_window_size: 16   # 局部窗口大小

    encoder_configs:
      radar:
        embed_dim: 256
        # 启用位置编码（序列相似度需要）
        use_positional_encoding: true
        max_sequence_length: 496
```

### 5.4 分布式训练配置

多GPU训练配置：

```yaml
# 训练脚本参数
strategy: 'ddp'                   # 分布式数据并行
log_dir: 'log/'

# 批次大小会自动调整
# 在run_clip.py中：
# data_cfg.params.cfg.batch_size = data_cfg.params.cfg.batch_size // args.world_size
```

### 5.5 数据处理参数详解

```yaml
opt:
  max_motion_length: 496          # 数据填充到的最大长度
  min_motion_len: 96              # 最小有效动作长度
  max_text_len: 20                # 文本描述的最大token数量
  unit_length: 16                 # 处理的基本单元长度
  log_norm: true                  # 对数归一化：log(1 + |x|)
  radar_views: 'all'              # 可选：'all', 'doppler', 'range', 'azimuth'
```

### 5.6 编码器参数详解

#### 雷达编码器参数
- `embed_dim`: 特征嵌入维度
- `dropout`: Dropout率，防止过拟合
- `max_sequence_length`: 序列的最大长度
- `use_layer_norm`: 是否使用层归一化
- `use_positional_encoding`: 是否添加位置编码（序列特征需要）
- `input_dims`: 各视图的输入维度（可选，默认为256, 128, 128）

#### 文本编码器参数
- `model_name`: 预训练模型名称
- `embed_dim`: 目标嵌入维度
- `max_length`: 最大文本长度
- `pooling_strategy`: 特征池化策略
  - `cls`: 使用[CLS] token
  - `mean`: 平均池化
  - `pooler`: 模型自带的池化输出
  - `max`: 最大池化
- `freeze_backbone`: 是否冻结预训练模型参数

## 6. 关键技术点

### 6.1 抽象层设计模式

系统采用现代化的抽象层设计，提供清晰的接口和可扩展性：

```python
# 核心抽象接口
class BaseEncoder(ABC):
    @abstractmethod
    def encode(self, data: ModalityData, **kwargs) -> EncodingResult: ...

class BaseModel(ABC):
    @abstractmethod
    def forward(self, *args, **kwargs): ...

# 数据容器
@dataclass
class ModalityData:
    data: Any
    modality: ModalityType
    metadata: Dict[str, Any]

@dataclass
class EncodingResult:
    features: torch.Tensor
    sequence_features: Optional[torch.Tensor] = None
    metadata: Dict[str, Any] = None
```

**优势**:
- 统一的接口设计
- 易于扩展新模态
- 类型安全
- 清晰的数据流

### 6.2 多视图雷达编码

RadarEncoder支持多视图雷达数据的并行处理：

#### 6.2.1 卷积编码架构
```python
def _create_view_encoder(self, input_dim: int) -> nn.Module:
    return nn.Sequential(
        nn.Conv1d(input_dim, self.embed_dim // 2, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv1d(self.embed_dim // 2, self.embed_dim, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool1d(self.max_sequence_length)  # 统一序列长度
    )
```

#### 6.2.2 特征融合策略
- **拼接融合**: 各视图特征拼接后通过线性层投影
- **自适应池化**: 确保所有视图具有相同的序列长度
- **层归一化**: 提高训练稳定性

### 6.3 序列相似度计算

系统支持高级的序列级相似度计算：

#### 6.3.1 计算策略
```python
class SequenceSimilarity(nn.Module):
    def __init__(self, similarity_type: str = "combined"):
        # similarity_type: "global", "local", "attention", "temporal", "combined"
```

**策略说明**:
- **global**: 全局平均池化后的相似度
- **local**: 滑动窗口局部相似度计算
- **attention**: 基于注意力机制的相似度
- **temporal**: 考虑时序对齐的相似度
- **combined**: 多种策略的加权组合

#### 6.3.2 损失计算
```python
def _compute_losses_from_features(self, radar_features, text_features, encoding_results):
    losses = self._compute_clip_loss_only(radar_features, text_features)

    if self.use_sequence_similarity:
        # 计算序列相似度矩阵
        seq_similarities = self.sequence_similarity(radar_seq, text_seq)
        seq_labels = torch.arange(batch_size, device=radar_features.device)
        seq_loss = F.cross_entropy(seq_similarities, seq_labels)

        losses["loss_seq"] = seq_loss * self.sequence_similarity_weight
        losses["loss_total"] = losses["loss_clip"] + losses["loss_seq"]
```

### 6.4 分布式训练支持

系统完全支持分布式训练：

#### 6.4.1 DDP配置
```python
# 训练脚本中
trainer = Trainer(
    accelerator='gpu',
    devices=args.world_size,
    strategy='ddp',  # 分布式数据并行
    logger=logger,
    max_epochs=max_epochs
)
```

#### 6.4.2 损失函数分布式优化
```python
class ClipLoss(nn.Module):
    def __init__(self, local_loss=False, gather_with_grad=False,
                 cache_labels=True, rank=0, world_size=1):
        # 支持分布式训练的损失计算
        # local_loss: 本地损失计算
        # gather_with_grad: 梯度收集
        # cache_labels: 标签缓存优化
```

### 6.5 对比学习策略

系统支持两种对比学习损失：

#### 6.5.1 标准CLIP损失
```python
# InfoNCE损失
logits_per_image = logit_scale * image_features @ text_features.t()
logits_per_text = logit_scale * text_features @ image_features.t()
loss = (F.cross_entropy(logits_per_image, labels) +
        F.cross_entropy(logits_per_text, labels)) / 2
```

#### 6.5.2 SigLIP损失
```python
# Sigmoid损失，无需负样本对
logits = logit_scale * image_features @ text_features.t() + logit_bias
loss = -torch.mean(torch.log1p(torch.exp(-logits)))
```

**SigLIP优势**:
- 训练更稳定
- 不需要复杂的负采样
- 内存效率更高

### 6.6 实验跟踪和监控

集成SwanLab进行实验管理：

```python
# 自动日志记录
logger = SwanLabLogger(name=args.version, project='mmExpert')

# 训练过程中的指标记录
self.log(f'{prefix}/loss_clip', losses['loss_clip'], on_step=True, on_epoch=True)
self.log(f'{prefix}/loss_seq', losses['loss_seq'], on_step=True, on_epoch=True)
```

**监控指标**:
- 训练/验证损失
- 学习率变化
- 梯度统计
- 模型检查点自动保存

## 7. 训练监控和实验管理

### 7.1 SwanLab集成

系统完全集成SwanLab进行实验跟踪：

```python
# 自动实验日志记录
logger = SwanLabLogger(name=args.version, project='mmExpert')
```

### 7.2 监控指标

系统自动记录以下训练指标：

#### 7.2.1 损失指标
- `train/loss_clip`: 训练CLIP损失
- `valid/loss_clip`: 验证CLIP损失
- `train/loss_radar_to_text`: 雷达到文本损失
- `train/loss_text_to_radar`: 文本到雷达损失
- `train/loss_seq`: 序列相似度损失（如果启用）
- `train/loss_total`: 总损失（包含序列损失）

#### 7.2.2 系统指标
- 学习率变化
- 梯度范数
- 批次处理时间
- GPU内存使用情况

### 7.3 模型检查点

自动保存和管理模型检查点：

```python
checkpoint_callback = ModelCheckpoint(
    dirpath=os.path.join(cfg.log_dir, args.version),
    monitor='valid/loss_clip',
    filename='epoch_{epoch:02d}_val_{valid/loss_clip:.4f}',
    save_top_k=10,
    mode='min',
    save_last=True,
    save_weights_only=False
)
```

## 8. 运行命令

### 8.1 基本训练命令

```bash
# 标准训练
python run_clip.py --config config/clip.yaml

# 从检查点恢复训练
python run_clip.py --config config/clip.yaml --resume-checkpoint path/to/checkpoint.ckpt

# 指定实验版本
python run_clip.py --config config/clip.yaml --version my_experiment_v1
```

### 8.2 分布式训练

#### 8.2.1 多GPU训练
```bash
# 使用torchrun进行分布式训练
torchrun --nproc_per_node=4 run_clip.py --config config/clip.yaml

# 或使用slurm
srun --nodes=1 --ntasks-per-node=4 python run_clip.py --config config/clip.yaml
```

#### 8.2.2 分布式配置
配置文件中设置：
```yaml
strategy: 'ddp'  # 分布式数据并行
```

### 8.3 测试和评估

```bash
# 模型测试
python run_clip.py --config config/clip.yaml --test --resume-checkpoint model.ckpt

# 独立评估
python evaluate_clip.py --model_path model.ckpt --config_path config.yaml
```

### 8.4 环境变量设置

```bash
# 设置并行处理
export TOKENIZERS_PARALLELISM=true

# 设置CUDA设备
export CUDA_VISIBLE_DEVICES=0,1,2,3

# 设置分布式训练环境变量
export RANK=0
export WORLD_SIZE=4
export MASTER_ADDR=localhost
export MASTER_PORT=12355
```

## 9. 性能优化建议

### 9.1 训练优化

#### 9.1.1 批次大小调整
```yaml
# 根据GPU内存调整
batch_size: 64  # RTX 3090建议值
batch_size: 32  # RTX 2080 Ti建议值
batch_size: 16  # 较小GPU建议值
```

#### 9.1.2 数据加载优化
```yaml
# 增加工作进程数
num_workers: 4  # 根据CPU核心数调整
pin_memory: true  # 加速GPU数据传输
prefetch_factor: 2  # 预取因子
```

#### 9.1.3 模型配置优化
```yaml
encoder_configs:
  radar:
    embed_dim: 256    # 平衡性能和效率
    dropout: 0.1      # 防止过拟合
  text:
    max_length: 77    # 标准长度
    freeze_backbone: false  # 根据数据量决定
```

### 9.2 内存优化

#### 9.2.1 梯度检查点
```python
# 在模型中启用梯度检查点
model = torch.utils.checkpoint.checkpoint_sequential(model, segments=2)
```

#### 9.2.2 混合精度训练
```python
# 在Trainer中启用
trainer = Trainer(
    precision='16-mixed',  # 混合精度
    # ...
)
```

### 9.3 超参数调优建议

#### 9.3.1 学习率策略
- **雷达编码器**: 1e-4 (标准学习率)
- **文本编码器**: 5e-5 (较低学习率，防止破坏预训练权重)
- **调度器**: CosineAnnealingLR，余弦退火

#### 9.3.2 温度参数
```yaml
temperature: 0.07  # 标准CLIP温度
# 范围: 0.01 - 0.1，根据任务难度调整
```

#### 9.3.3 序列相似度参数
```yaml
use_sequence_similarity: true
sequence_similarity_weight: 0.5  # 权重平衡
sequence_similarity_window_size: 16  # 局部窗口大小
```

## 10. 故障排除

### 10.1 常见错误和解决方案

#### 10.1.1 模型创建错误
**错误**: `ValueError: No encoder found for modality: radar`

**解决方案**:
```yaml
# 检查配置文件中的encoder_configs
encoder_configs:
  radar:  # 确保存在radar配置
    embed_dim: 256
```

#### 10.1.2 维度不匹配错误
**错误**: `RuntimeError: mat1 and mat2 shapes cannot be multiplied`

**解决方案**:
```yaml
# 确保所有编码器的embed_dim一致
encoder_configs:
  radar:
    embed_dim: 256
  text:
    embed_dim: 256  # 必须相同
```

#### 10.1.3 数据加载错误
**错误**: `FileNotFoundError: dataset/HumanML3D/_split/train.json`

**解决方案**:
```yaml
# 检查数据路径配置
train_split: ['path/to/your/train.json']
val_split: ['path/to/your/val.json']
test_split: ['path/to/your/test.json']
```

#### 10.1.4 内存不足错误
**错误**: `CUDA out of memory`

**解决方案**:
```yaml
# 减小批次大小
batch_size: 16  # 从64减小到16

# 或启用梯度检查点
# 在代码中添加model.enable_gradient_checkpointing()
```

#### 10.1.5 分布式训练错误
**错误**: `RuntimeError: Default process group has not been initialized`

**解决方案**:
```bash
# 确保使用正确的启动命令
torchrun --nproc_per_node=2 run_clip.py --config config/clip.yaml
```

### 10.2 调试技巧

#### 10.2.1 数据流验证
```python
# 在training_step中添加调试代码
def training_step(self, batch, batch_idx):
    print(f"Batch keys: {batch.keys()}")
    print(f"Radar data shape: {batch['radar']['range_time'].shape}")
    print(f"Text data: {batch['text'][:2]}")  # 打印前两个文本
```

#### 10.2.2 模型输出检查
```python
# 在forward方法中添加检查
def forward(self, radar_data, text, **kwargs):
    encoding_results = self._encode_data(radar_data, text, return_sequences=True)

    radar_features = encoding_results[ModalityType.RADAR].features
    print(f"Radar features shape: {radar_features.shape}")
    print(f"Radar features norm: {radar_features.norm(dim=1).mean()}")
```

#### 10.2.3 梯度检查
```python
# 在training_step后添加
def on_before_optimizer_step(self, optimizer, optimizer_idx):
    # 检查梯度范数
    total_norm = 0
    for p in self.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    self.log('grad_norm', total_norm)
```

### 10.3 性能问题诊断

#### 10.3.1 训练速度慢
- 增加`num_workers`到4-8
- 使用SSD存储数据
- 检查网络带宽（如果数据在远程）

#### 10.3.2 内存使用过高
- 启用梯度检查点
- 使用混合精度训练
- 减小`max_sequence_length`

#### 10.3.3 收敛问题
- 调整学习率（尝试1e-5到1e-3）
- 检查数据质量和预处理
- 调整温度参数

---

本文档详细介绍了mmExpert框架中CLIP雷达数据训练的完整流程，包括基于抽象层的现代化架构设计、序列相似度计算、分布式训练支持等高级功能，为后续的模型优化和功能扩展提供了全面的参考基础。