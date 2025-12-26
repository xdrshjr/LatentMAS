# Latent PRM Fine-tuning Guide

本文档介绍如何使用收集的训练数据对 Qwen-0.6B 模型进行全参数微调，训练一个 Process Reward Model (PRM) 来预测推理路径的质量分数。

## 📋 目录

1. [概述](#概述)
2. [前置条件](#前置条件)
3. [快速开始](#快速开始)
4. [详细配置](#详细配置)
5. [训练监控](#训练监控)
6. [使用微调后的模型](#使用微调后的模型)
7. [故障排除](#故障排除)

---

## 概述

### 训练流程

```
收集数据 (collect_training_data.sh)
    ↓
训练数据 (.pt files in prm_data/)
    ↓
微调模型 (train_latent_prm.sh)
    ↓
微调后的模型 (checkpoints/qwen_prm/)
```

### 核心特性

- ✅ **全参数微调** - 不使用 LoRA，直接微调所有参数
- ✅ **序列处理** - 使用完整的 latent sequence（不仅仅是平均值）
- ✅ **PRM Score** - 基于最终答案正确性的分数作为训练目标
- ✅ **进度显示** - 实时显示训练进度和 loss 变化
- ✅ **灵活配置** - 可配置保存频率、batch size 等参数

---

## 前置条件

### 1. 收集训练数据

首先需要运行数据收集脚本：

```bash
bash collect_training_data.sh
```

这将生成训练数据文件到 `prm_data/` 目录。

### 2. 验证数据

检查数据是否正确收集：

```bash
ls -lh prm_data/
```

应该看到：
- `*.pt` 文件（包含 latent sequences 和 scores）
- `*_metadata.json` 文件（包含数据集统计信息）

### 3. 环境要求

- Python 3.10+
- PyTorch 2.0+
- transformers
- tqdm
- CUDA GPU（推荐至少 8GB 显存）

---

## 快速开始

### 基本用法

```bash
bash train_latent_prm.sh
```

这将使用默认配置开始训练。

### 自定义配置

编辑 `train_latent_prm.sh` 中的配置参数：

```bash
# 训练超参数
NUM_EPOCHS=5                       # 训练轮数
BATCH_SIZE=4                       # Batch size
LEARNING_RATE=2e-5                 # 学习率
GRADIENT_ACCUMULATION_STEPS=4      # 梯度累积步数

# 保存配置
SAVE_STEPS=100                     # 每 N 步保存一次 checkpoint
LOGGING_STEPS=10                   # 每 N 步记录一次日志
```

---

## 详细配置

### 训练超参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `NUM_EPOCHS` | 5 | 训练轮数 |
| `BATCH_SIZE` | 4 | 每个 batch 的样本数 |
| `LEARNING_RATE` | 2e-5 | 学习率（全参数微调推荐 1e-5 到 5e-5） |
| `WEIGHT_DECAY` | 0.01 | 权重衰减（正则化） |
| `WARMUP_RATIO` | 0.1 | Warmup 步数占比 |
| `GRADIENT_ACCUMULATION_STEPS` | 4 | 梯度累积步数 |
| `MAX_GRAD_NORM` | 1.0 | 梯度裁剪阈值 |

**有效 Batch Size** = `BATCH_SIZE × GRADIENT_ACCUMULATION_STEPS`

例如：`4 × 4 = 16`（相当于 batch size 16）

### 模型配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `POOLING_STRATEGY` | "mean" | 序列池化方式（mean/last/max） |
| `DROPOUT_PROB` | 0.1 | Dropout 概率 |
| `MAX_SEQ_LENGTH` | 512 | 最大序列长度 |
| `USE_PRM_SCORE` | true | 使用 prm_score（基于最终答案正确性） |

### 日志和保存

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `SAVE_STEPS` | 100 | 每 N 步保存 checkpoint |
| `LOGGING_STEPS` | 10 | 每 N 步记录日志 |
| `LOG_LEVEL` | "INFO" | 日志级别 |

### 设备配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `DEVICE` | "cuda" | 训练设备 |
| `MIXED_PRECISION` | true | 混合精度训练（fp16） |
| `SEED` | 42 | 随机种子 |

---

## 训练监控

### 控制台输出

训练时会显示实时进度条和 loss：

```
Epoch 1/5
100%|████████████| 250/250 [10:23<00:00, loss=0.0234, lr=1.98e-05]

[Trainer] Step 100: loss=0.0234, avg_loss=0.0245, lr=1.98e-05
[Trainer] Checkpoint saved to: checkpoints/qwen_prm_20251226_120000/checkpoint-100
```

### 日志文件

详细日志保存在：

```
checkpoints/qwen_prm_TIMESTAMP/training.log
```

### 训练统计

训练完成后，统计信息保存在：

```
checkpoints/qwen_prm_TIMESTAMP/training_stats.json
```

包含：
- 总步数
- 每个 epoch 的 loss
- 最佳 loss
- 训练时间等

---

## Checkpoint 说明

训练过程中会保存多个 checkpoint：

### 1. 定期 Checkpoint

每 `SAVE_STEPS` 步保存一次：

```
checkpoints/qwen_prm_TIMESTAMP/checkpoint-100/
checkpoints/qwen_prm_TIMESTAMP/checkpoint-200/
...
```

### 2. Epoch Checkpoint

每个 epoch 结束时保存：

```
checkpoints/qwen_prm_TIMESTAMP/epoch_1/
checkpoints/qwen_prm_TIMESTAMP/epoch_2/
...
```

### 3. 最佳 Checkpoint

训练过程中 loss 最低的 checkpoint：

```
checkpoints/qwen_prm_TIMESTAMP/best/
```

**推荐使用这个进行推理！**

### 4. 最终 Checkpoint

训练结束时的 checkpoint：

```
checkpoints/qwen_prm_TIMESTAMP/final/
```

### Checkpoint 内容

每个 checkpoint 包含：

```
checkpoint-XXX/
├── pytorch_model.bin      # 模型权重
├── optimizer.pt           # 优化器状态
├── scheduler.pt           # 学习率调度器状态
├── training_state.json    # 训练状态
└── config.json            # 模型配置
```

---

## 使用微调后的模型

### 加载模型

```python
from methods.latent_prm import QwenLatentPRM
import torch

# 加载最佳 checkpoint
checkpoint_path = "checkpoints/qwen_prm_20251226_120000/best"
model = QwenLatentPRM(model_path="Qwen/Qwen3-0.6B")
model.load_state_dict(torch.load(f"{checkpoint_path}/pytorch_model.bin"))
model.eval()
model.to("cuda")
```

### 预测路径分数

```python
# 准备 latent sequence
latent_sequence = torch.randn(1, 10, 896)  # [batch, seq_len, hidden_dim]
attention_mask = torch.ones(1, 10)

# 预测分数
with torch.no_grad():
    scores = model.predict_scores(latent_sequence, attention_mask)
    print(f"Predicted score: {scores[0]:.4f}")
```

---

## 故障排除

### 问题 1: CUDA Out of Memory

**症状**：训练时出现 `CUDA out of memory` 错误

**解决方案**：

1. 减小 `BATCH_SIZE`（例如从 4 改为 2）
2. 增加 `GRADIENT_ACCUMULATION_STEPS`（保持有效 batch size）
3. 减小 `MAX_SEQ_LENGTH`（例如从 512 改为 256）

```bash
# 修改 train_latent_prm.sh
BATCH_SIZE=2                       # 减小 batch size
GRADIENT_ACCUMULATION_STEPS=8      # 增加梯度累积
MAX_SEQ_LENGTH=256                 # 减小序列长度
```

### 问题 2: 找不到训练数据

**症状**：

```
✗ ERROR: No .pt files found in prm_data
```

**解决方案**：

1. 先运行数据收集脚本：

```bash
bash collect_training_data.sh
```

2. 确认数据目录正确：

```bash
ls -lh prm_data/
```

### 问题 3: Loss 不下降

**症状**：训练多个 epoch 后 loss 仍然很高或不变

**可能原因和解决方案**：

1. **学习率太高或太低**

```bash
# 尝试调整学习率
LEARNING_RATE=1e-5    # 降低学习率
# 或
LEARNING_RATE=5e-5    # 提高学习率
```

2. **数据质量问题**

检查收集的数据：

```bash
# 查看数据统计
cat prm_data/*_metadata.json
```

确认：
- 样本数量足够（至少 100+）
- Score 分布合理（不是全部相同）

3. **模型配置问题**

尝试不同的池化策略：

```bash
POOLING_STRATEGY="last"    # 使用最后一个 token
# 或
POOLING_STRATEGY="max"     # 使用 max pooling
```

### 问题 4: 训练速度慢

**解决方案**：

1. 启用混合精度训练（默认已启用）：

```bash
MIXED_PRECISION=true
```

2. 增加 batch size（如果显存允许）：

```bash
BATCH_SIZE=8
```

3. 减少保存频率：

```bash
SAVE_STEPS=500    # 从 100 改为 500
```

---

## 高级用法

### 命令行直接调用

除了使用 shell 脚本，也可以直接调用 Python 模块：

```bash
python -m methods.latent_prm.trainer \
  --model_path /path/to/Qwen-0.6B \
  --data_dir prm_data \
  --output_dir checkpoints/my_experiment \
  --num_epochs 10 \
  --batch_size 8 \
  --learning_rate 3e-5 \
  --save_steps 50 \
  --logging_steps 5
```

### 查看所有参数

```bash
python -m methods.latent_prm.trainer --help
```

### 从 Checkpoint 继续训练

目前不支持从 checkpoint 继续训练，但可以通过修改 `trainer.py` 添加此功能。

---

## 性能建议

### GPU 显存使用

| Batch Size | Gradient Acc | Effective BS | 显存使用 | 推荐 GPU |
|-----------|--------------|--------------|---------|----------|
| 2 | 8 | 16 | ~6GB | RTX 3060 |
| 4 | 4 | 16 | ~8GB | RTX 3070 |
| 8 | 2 | 16 | ~12GB | RTX 3080 |
| 16 | 1 | 16 | ~20GB | A100 |

### 训练时间估算

以 1000 个样本为例：

- Batch size 4, 5 epochs: ~30-45 分钟（RTX 3080）
- Batch size 2, 5 epochs: ~60-90 分钟（RTX 3060）

---

## 最佳实践

### 1. 数据收集

- 收集至少 1000+ 样本
- 确保数据多样性（不同问题类型）
- 验证 prm_score 分布合理

### 2. 训练配置

- 从较小的学习率开始（1e-5 到 2e-5）
- 使用 warmup（10% 步数）
- 启用梯度裁剪（max_grad_norm=1.0）
- 使用混合精度训练节省显存

### 3. 监控训练

- 观察 loss 曲线是否平滑下降
- 检查学习率调度是否合理
- 定期保存 checkpoint（每 100 步）

### 4. 模型选择

- 使用 best checkpoint 进行推理
- 如果 best 过拟合，尝试 epoch_3 或 epoch_4

---

## 相关文档

- [数据收集指南](prm-data-collection-guide.md)
- [多路径推理指南](multipath-guide.md)
- [实验指南](experiment-guide.md)

---

## 常见问题 (FAQ)

**Q: 训练需要多少数据？**

A: 推荐至少 1000 个样本。更多数据通常会带来更好的性能。

**Q: 训练需要多长时间？**

A: 取决于数据量和硬件。1000 样本，5 epochs，RTX 3080 约需 30-45 分钟。

**Q: 可以使用 LoRA 而不是全参数微调吗？**

A: 当前实现是全参数微调。如需 LoRA，需要修改 `model.py` 添加 PEFT 支持。

**Q: 如何评估微调后的模型？**

A: 可以在验证集上计算 MSE/MAE，或者在下游任务（如 GSM8K）上评估准确率。

**Q: 微调后的模型如何用于推理？**

A: 加载 checkpoint 后，可以用于预测路径质量分数，从而改进路径选择策略。

---

## 更新日志

- **2024-12-26**: 初始版本
  - 支持全参数微调
  - 支持序列处理
  - 支持 prm_score 作为训练目标
  - 添加进度条和实时 loss 显示

---

如有问题或建议，请查看项目文档或提交 issue。

