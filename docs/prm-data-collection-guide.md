# Latent PRM Training Data Collection Guide

## 概述 (Overview)

本指南介绍如何使用LatentMAS系统收集多路径潜在推理数据，用于训练Qwen-0.6B模型的过程奖励模型(Process Reward Model, PRM)。

## 功能特性 (Features)

### 1. 多路径数据收集
- 收集每个agent生成的所有推理路径
- 保存完整的latent向量历史
- 记录路径之间的父子关系（树状结构）
- 基于最终答案正确性计算PRM评分

### 2. 路径树结构
- 构建完整的推理路径树
- 支持多层级agent架构
- 反向传播评分（从叶子节点到根节点）
- 保存树的拓扑结构和统计信息

### 3. 数据存储
- 使用PyTorch .pt格式保存latent向量
- JSON格式保存元数据和树结构
- 支持批量保存和单个问题保存
- 自动创建数据索引文件

### 4. 评分机制
- 叶子节点：基于最终答案正确性（1.0或0.0）
- 内部节点：子节点PRM分数的平均值
- 支持路径质量分析和关键路径识别

## 快速开始 (Quick Start)

### 1. 使用Shell脚本收集数据

最简单的方式是使用提供的shell脚本：

```bash
bash collect_training_data.sh
```

### 2. 自定义参数

编辑 `collect_training_data.sh` 中的配置参数：

```bash
# 数据集配置
TASK="gsm8k"                    # 数据集
MAX_SAMPLES=100                 # 收集的问题数量

# 多路径配置
NUM_PATHS=10                    # 每个agent生成的路径数
LATENT_STEPS=5                  # 每条路径的latent步骤数
DIVERSITY_STRATEGY="hybrid"     # 多样性策略

# 输出配置
OUTPUT_DIR="output/prm_data"    # 数据保存目录
```

### 3. 使用Python命令行

也可以直接使用Python命令：

```bash
python run.py \
  --method latent_mas_multipath \
  --model_name Qwen/Qwen3-0.6B \
  --task gsm8k \
  --max_samples 100 \
  --num_paths 10 \
  --latent_steps 5 \
  --collect_prm_data \
  --prm_output_dir output/prm_data \
  --prm_disable_pruning \
  --prm_disable_merging
```

## 命令行参数 (Command Line Arguments)

### PRM数据收集专用参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--collect_prm_data` | flag | False | 启用PRM数据收集模式 |
| `--prm_output_dir` | str | "output/prm_data" | 数据保存目录 |
| `--prm_disable_pruning` | flag | False | 禁用路径剪枝（收集所有路径） |
| `--prm_disable_merging` | flag | False | 禁用路径合并（收集所有路径） |

### 多路径配置参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--num_paths` | int | 5 | 每个agent生成的路径数 |
| `--latent_steps` | int | 10 | 每条路径的latent步骤数 |
| `--diversity_strategy` | str | "hybrid" | 多样性策略 (temperature/noise/hybrid) |
| `--temperature` | float | 0.7 | 基准温度 |

## 数据格式 (Data Format)

### 保存的文件结构

```
output/prm_data/
├── batch_20241226_120000.pt          # 批量数据文件
├── batch_20241226_120000_metadata.json  # 元数据文件
├── question_000001_20241226_120001.pt   # 单个问题数据
├── question_000002_20241226_120002.pt
└── dataset_index.json                # 数据集索引
```

### 数据文件内容 (.pt文件)

```python
{
    "question_id": "q_0",
    "question": "问题文本",
    "gold_answer": "正确答案",
    "final_answer": "模型预测答案",
    "is_correct": True/False,
    
    "paths": [
        {
            "path_id": 0,
            "agent_name": "planner",
            "agent_idx": 0,
            "parent_path_id": None,
            "child_path_ids": [1, 2, 3],
            "score": 0.85,                    # 原始路径评分
            "prm_score": 1.0,                 # PRM评分
            "latent_history": torch.Tensor,   # [num_steps, hidden_dim]
            "hidden_states": torch.Tensor,    # [hidden_dim]
            "num_latent_steps": 5,
            "metadata": {...}
        },
        # ... 更多路径
    ],
    
    "tree_structure": {
        "nodes": [...],
        "edges": [...],
        "root_ids": [0, 1],
        "is_correct": True,
        "num_nodes": 30,
        "num_edges": 27,
        "max_depth": 3
    }
}
```

### 树结构说明

- **nodes**: 所有路径节点的列表
- **edges**: 路径之间的连接关系 [(parent_id, child_id), ...]
- **root_ids**: 根节点ID列表（第一个agent的路径）
- **max_depth**: 树的最大深度（等于agent数量-1）

## PRM评分机制 (PRM Scoring)

### 评分策略

1. **叶子节点**（最后一个agent的路径）
   - 如果最终答案正确：`prm_score = 1.0`
   - 如果最终答案错误：`prm_score = 0.0`

2. **内部节点**（中间agent的路径）
   - `prm_score = mean(children_prm_scores)`
   - 反向传播：从叶子节点向根节点计算

3. **评分传播示例**

```
                Root (score=0.75)
               /    |    \
              /     |     \
         0.5      1.0     0.75  (内部节点)
         /|\      /|\      /|\
        / | \    / | \    / | \
       0  1  1  1  1  1  0  1  1  (叶子节点，基于最终答案)
```

## 数据加载示例 (Loading Data)

### Python代码示例

```python
import torch
from pathlib import Path

# 加载单个问题的数据
data = torch.load("output/prm_data/question_000001_20241226_120001.pt")

print(f"Question: {data['question']}")
print(f"Correct: {data['is_correct']}")
print(f"Num paths: {len(data['paths'])}")

# 访问路径数据
for path in data['paths']:
    print(f"Path {path['path_id']}: agent={path['agent_name']}, "
          f"prm_score={path['prm_score']:.4f}")
    
    # 访问latent向量
    latent_history = path['latent_history']  # torch.Tensor [num_steps, hidden_dim]
    print(f"  Latent shape: {latent_history.shape}")

# 访问树结构
tree = data['tree_structure']
print(f"Tree: {tree['num_nodes']} nodes, {tree['num_edges']} edges")
```

### 批量加载

```python
# 加载批量数据
batch_data = torch.load("output/prm_data/batch_20241226_120000.pt")

print(f"Batch: {batch_data['batch_name']}")
print(f"Questions: {batch_data['num_questions']}")

for question_data in batch_data['questions']:
    print(f"Q: {question_data['question_id']}, "
          f"Correct: {question_data['is_correct']}, "
          f"Paths: {len(question_data['paths'])}")
```

## 数据统计 (Statistics)

### 查看收集的数据统计

```python
from methods.latent_prm import PRMDataStorage

storage = PRMDataStorage(output_dir="output/prm_data")

# 获取统计信息
stats = storage.get_statistics()
print(f"Total files: {stats['num_files']}")
print(f"Total size: {stats['total_size_mb']:.2f} MB")

# 创建数据集索引
index_path = storage.create_dataset_index()
print(f"Index created: {index_path}")
```

## 训练PRM模型 (Training PRM)

收集的数据可以用于训练过程奖励模型：

### 训练目标

- **输入**: 路径的latent向量序列
- **输出**: 路径的质量评分（0-1之间）
- **标签**: PRM评分（基于最终答案正确性反向传播）

### 训练策略

1. **监督学习**: 使用PRM评分作为标签
2. **对比学习**: 区分高质量路径和低质量路径
3. **序列建模**: 学习latent向量序列的质量模式

### 模型架构建议

```python
class LatentPRM(nn.Module):
    def __init__(self, hidden_dim=896):
        super().__init__()
        self.encoder = nn.TransformerEncoder(...)
        self.scorer = nn.Linear(hidden_dim, 1)
    
    def forward(self, latent_sequence):
        # latent_sequence: [batch, seq_len, hidden_dim]
        encoded = self.encoder(latent_sequence)
        score = torch.sigmoid(self.scorer(encoded[:, -1, :]))
        return score
```

## 最佳实践 (Best Practices)

### 1. 数据收集配置

- **路径数量**: 建议10-20条路径，平衡多样性和计算成本
- **Latent步骤**: 建议3-5步，捕获足够的推理过程
- **禁用剪枝/合并**: 确保收集所有路径，最大化数据多样性

### 2. 数据质量

- 收集多个数据集的数据（GSM8K, AIME, 等）
- 确保正确和错误答案的平衡
- 定期检查数据完整性

### 3. 存储管理

- 定期备份收集的数据
- 使用数据索引文件快速检索
- 监控磁盘空间使用

## 故障排查 (Troubleshooting)

### 问题1: 内存不足

**症状**: CUDA out of memory

**解决方案**:
- 减少 `--num_paths` 参数
- 减少 `--latent_steps` 参数
- 使用 `--generate_bs 1` 单个处理

### 问题2: 数据文件过大

**症状**: .pt文件太大

**解决方案**:
- 减少latent步骤数
- 减少路径数量
- 分批保存数据

### 问题3: 路径树不完整

**症状**: 某些路径缺少父子关系

**解决方案**:
- 确保使用 `--prm_disable_pruning`
- 确保使用 `--prm_disable_merging`
- 检查日志中的警告信息

## 日志示例 (Log Examples)

### 正常运行日志

```
[LatentMASMultiPathMethod] Enabling PRM data collection mode
[LatentMASMultiPathMethod] Output directory: output/prm_data
[PRM DataCollection] Starting data collection for batch
[DataCollector] Starting question q_0
[DataCollector] Recording 10 paths for agent planner
[PathTreeBuilder] Building tree from 30 paths
[PathTreeBuilder] Found 10 leaf nodes
[PRMScorer] Scoring 30 paths
[PRMScorer] Final answer correct: True
[PRM DataCollection] Saved batch data to: output/prm_data/batch_20241226_120000.pt
```

## 参考资料 (References)

- [LatentMAS论文](https://arxiv.org/abs/...)
- [Process Reward Models](https://arxiv.org/abs/...)
- [Qwen模型文档](https://github.com/QwenLM/Qwen)

## 联系与支持 (Contact)

如有问题或建议，请提交Issue或联系开发团队。

