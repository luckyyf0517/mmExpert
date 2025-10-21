# Text Encoder 冻结策略指南

## 🎯 功能概述

TextEncoder 现在支持三种灵活的参数冻结策略，帮助你在微调时防止过拟合：

1. **freeze_pattern** - 基于正则表达式的灵活冻结（最强大）
2. **freeze_layers** - 基于层数的冻结（最常用）
3. **freeze_backbone** - 全部冻结或全部不冻结（最简单）

**优先级**: `freeze_pattern` > `freeze_layers` > `freeze_backbone`

---

## 📝 使用示例

### 方案 1: Pattern-based Freezing (推荐用于精细控制)

```yaml
# config/clip.yaml
encoder_configs:
  text:
    model_name: 'sentence-transformers/paraphrase-MiniLM-L6-v2'
    embed_dim: 256
    max_length: 77
    pooling_strategy: 'pooler'
    
    # 使用正则表达式模式冻结参数
    freeze_pattern: 'encoder.layer.[0-3]'  # 冻结前 4 层
```

#### 常用 Pattern 示例:

```yaml
# 1. 只冻结 embeddings
freeze_pattern: 'embeddings'

# 2. 冻结前 4 层 (layer 0-3)
freeze_pattern: 'encoder.layer.[0-3]'

# 3. 冻结前 3 层 (更精确的语法)
freeze_pattern: 'encoder.layer.(0|1|2)'

# 4. 冻结 embeddings + 前 2 层
freeze_pattern: 'embeddings|encoder.layer.[0-1]'

# 5. 冻结所有 attention 层
freeze_pattern: 'attention'

# 6. 冻结所有 LayerNorm
freeze_pattern: 'LayerNorm'

# 7. 冻结所有参数
freeze_pattern: 'all'
```

---

### 方案 2: Layer-count Freezing (推荐用于简单场景)

```yaml
encoder_configs:
  text:
    model_name: 'sentence-transformers/paraphrase-MiniLM-L6-v2'
    embed_dim: 256
    
    # 冻结前 4 层 (共 6 层)
    freeze_layers: 4
    # 这会自动冻结: embeddings + layer 0-3
    # 可训练的: layer 4-5 + projection
```

#### freeze_layers 选项:

```yaml
freeze_layers: 0   # 不冻结任何层
freeze_layers: 4   # 冻结 embeddings + 前 4 层
freeze_layers: -1  # 冻结所有层 (等同于 freeze_backbone: true)
```

---

### 方案 3: Simple All-or-Nothing (最简单)

```yaml
encoder_configs:
  text:
    model_name: 'sentence-transformers/paraphrase-MiniLM-L6-v2'
    embed_dim: 256
    
    # 简单的全部冻结或不冻结
    freeze_backbone: true   # 冻结所有 backbone 参数
    # 或
    freeze_backbone: false  # 不冻结任何参数
```

---

## 🎓 最佳实践

### 针对不同数据集规模的推荐

| 数据集大小 | 推荐策略 | 配置示例 |
|-----------|---------|---------|
| **小型** (<5K) | 冻结大部分层 | `freeze_layers: 5` (仅微调最后1层) |
| **中型** (5K-50K) | 冻结前半部分 | `freeze_layers: 3` (微调后半部分) |
| **大型** (>50K) | 少量冻结或不冻结 | `freeze_layers: 1` 或 `freeze_backbone: false` |

### MiniLM-L6-v2 的层结构 (6层)

```
embeddings          ← Layer 0, 通用特征
encoder.layer.0     ← Layer 1, 语法特征
encoder.layer.1     ← Layer 2, 语法特征
encoder.layer.2     ← Layer 3, 语法特征
encoder.layer.3     ← Layer 4, 语义特征
encoder.layer.4     ← Layer 5, 语义特征
encoder.layer.5     ← Layer 6, 任务相关特征
projection          ← 永不冻结
```

**推荐配置**:
```yaml
# 防止过拟合的标准配置
freeze_layers: 4   # 冻结 embeddings + layer 0-3
                   # 微调 layer 4-5 + projection
                   # 可训练参数: ~30% (约 7M)
```

---

## 🔍 查看冻结效果

训练时会打印冻结信息：

```bash
# Pattern-based
TextEncoder: Froze 89 parameters matching pattern 'encoder.layer.[0-3]'

# Layer-count based
TextEncoder: Froze embeddings + first 4/6 layers (245 parameters)

# All-or-nothing
TextEncoder: Froze all backbone parameters (356 total)

# No freezing
TextEncoder: All 356/356 backbone parameters are trainable
```

---

## 📊 参数统计 (MiniLM-L6-v2)

| 策略 | 可训练参数 | 占比 | 过拟合风险 | 适应能力 |
|------|-----------|------|-----------|---------|
| `freeze_backbone: true` | ~0.5M | 2% | 很低 | 低 |
| `freeze_layers: 4` | ~7M | 30% | 低 | 中 |
| `freeze_layers: 2` | ~14M | 60% | 中 | 中高 |
| `freeze_backbone: false` | ~23M | 100% | 高⚠️ | 很高 |

---

## 🚀 快速解决过拟合问题

如果你的模型在测试集上过拟合，按以下顺序尝试：

### Step 1: 冻结更多层
```yaml
# 从当前的 freeze_backbone: false
# 改为
freeze_layers: 4  # 先试 4 层
```

### Step 2: 如果还过拟合，增加冻结层数
```yaml
freeze_layers: 5  # 只微调最后 1 层
```

### Step 3: 如果仍然过拟合，完全冻结
```yaml
freeze_backbone: true  # 只训练 projection
```

### Step 4: 配合其他策略
```yaml
text:
  freeze_layers: 4
  dropout: 0.2        # 增加 dropout
  
# 同时调整训练参数
max_epochs: 50        # 减少 epoch (从 200)
learning_rate: 5e-5   # 降低学习率 (从 1e-4)
```

---

## 🧪 调试技巧

### 查看具体哪些参数被冻结

添加调试代码:
```python
# 在训练脚本中
for name, param in model.text_encoder.backbone.named_parameters():
    if not param.requires_grad:
        print(f"Frozen: {name}")
```

### 验证参数数量

```python
trainable = sum(p.numel() for p in model.text_encoder.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.text_encoder.parameters())
print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")
```

---

## ⚠️ 注意事项

1. **优先级规则**: 如果同时设置多个策略，只有最高优先级的生效
   ```yaml
   freeze_pattern: 'encoder.layer.[0-3]'  # ✅ 生效
   freeze_layers: 2                        # ❌ 被忽略
   freeze_backbone: true                   # ❌ 被忽略
   ```

2. **Pattern 语法**: 使用 Python 正则表达式
   - `.` 匹配任意字符
   - `[0-3]` 匹配 0,1,2,3 中的任意一个
   - `(0|1|2)` 精确匹配 0, 1, 或 2
   - `.*` 匹配任意长度的字符串

3. **梯度设置**: 冻结在初始化时完成，训练过程中不会改变

---

## 💡 高级技巧

### 渐进式解冻 (Progressive Unfreezing)

虽然当前不直接支持，但可以通过多阶段训练实现：

```yaml
# Stage 1: config/clip_stage1.yaml (50 epochs)
freeze_layers: 5   # 只微调最后1层

# Stage 2: config/clip_stage2.yaml (30 epochs)
freeze_layers: 3   # 解冻到微调后3层

# Stage 3: config/clip_stage3.yaml (20 epochs)
freeze_layers: 0   # 全部微调
```

### 差异化学习率

在 `clip_model.py` 的 `configure_optimizers` 中已经实现：
```python
# Text encoder 使用一半的学习率
param_groups.append({
    'params': self.text_encoder.parameters(),
    'lr': self.learning_rate / 2
})
```

配合冻结策略效果更佳！

---

## 📚 参考资料

- [CLIP 论文](https://arxiv.org/abs/2103.00020)
- [Fine-tuning Best Practices](https://huggingface.co/docs/transformers/training)
- [Progressive Unfreezing](https://arxiv.org/abs/1801.06146)

