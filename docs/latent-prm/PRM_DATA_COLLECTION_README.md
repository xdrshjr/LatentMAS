# Latent PRM训练数据收集功能

## 概述

本项目新增了完整的PRM（Process Reward Model）训练数据收集功能，用于收集多路径潜在推理数据，以便后续微调Qwen-0.6B模型。

## 新增功能

### ✨ 核心特性

1. **多路径数据收集**
   - 收集每个agent生成的所有推理路径
   - 保存完整的latent向量历史
   - 记录路径之间的父子关系（树状结构）

2. **路径树构建**
   - 自动构建推理路径树
   - 支持多层级agent架构
   - 计算树的拓扑结构和统计信息

3. **PRM评分**
   - 基于最终答案正确性评分
   - 反向传播评分（从叶子到根）
   - 叶子节点：1.0（正确）或0.0（错误）
   - 内部节点：子节点平均值

4. **数据存储**
   - PyTorch .pt格式保存latent向量
   - JSON格式保存元数据
   - 支持批量和单个保存
   - 自动创建数据索引

## 快速开始

### 1. 使用Shell脚本（推荐）

```bash
bash collect_training_data.sh
```

### 2. 使用Python命令

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

## 新增文件

### 核心模块

```
methods/latent_prm/
├── __init__.py                 # 模块初始化
├── data_collector.py           # 数据收集器
├── path_tree_builder.py        # 路径树构建器
├── data_storage.py             # 数据存储器
└── prm_scorer.py               # PRM评分器
```

### 脚本和文档

```
collect_training_data.sh        # 数据收集脚本
docs/
├── prm-data-collection-guide.md      # 完整使用指南
└── prm-implementation-summary.md     # 实现总结
examples/
└── prm_data_collection_example.py    # 使用示例
```

## 修改的文件

### 1. `methods/latent_mas_multipath.py`

**新增方法**：
- `enable_prm_data_collection()`: 启用PRM数据收集模式

**修改方法**：
- `run_batch()`: 集成数据收集逻辑
  - 在批处理开始时启动数据收集
  - 在路径生成后记录路径信息
  - 在PRM模式下跳过剪枝和合并
  - 在批处理完成后保存数据

### 2. `run.py`

**新增命令行参数**：
- `--collect_prm_data`: 启用PRM数据收集
- `--prm_output_dir`: 数据保存目录
- `--prm_disable_pruning`: 禁用剪枝
- `--prm_disable_merging`: 禁用合并

**修改逻辑**：
- 在方法初始化后检查并启用PRM数据收集

## 数据格式

### 输出目录结构

```
output/prm_data/
├── batch_20241226_120000.pt              # 批量数据
├── batch_20241226_120000_metadata.json   # 元数据
├── question_000001_20241226_120001.pt    # 单个问题
└── dataset_index.json                    # 数据索引
```

### .pt文件内容

```python
{
    "question_id": "q_0",
    "question": "问题文本",
    "gold_answer": "正确答案",
    "final_answer": "模型预测",
    "is_correct": True/False,
    
    "paths": [
        {
            "path_id": 0,
            "agent_name": "planner",
            "agent_idx": 0,
            "parent_path_id": None,
            "child_path_ids": [1, 2, 3],
            "score": 0.85,
            "prm_score": 1.0,
            "latent_history": torch.Tensor,  # [num_steps, hidden_dim]
            "hidden_states": torch.Tensor,   # [hidden_dim]
            "num_latent_steps": 5
        }
    ],
    
    "tree_structure": {
        "nodes": [...],
        "edges": [...],
        "root_ids": [0],
        "num_nodes": 30,
        "num_edges": 27,
        "max_depth": 3
    }
}
```

## 使用示例

### 加载数据

```python
import torch

# 加载单个问题数据
data = torch.load("output/prm_data/question_000001.pt")

print(f"Question: {data['question']}")
print(f"Correct: {data['is_correct']}")
print(f"Num paths: {len(data['paths'])}")

# 访问路径数据
for path in data['paths']:
    print(f"Path {path['path_id']}: "
          f"agent={path['agent_name']}, "
          f"prm_score={path['prm_score']:.4f}")
    
    # 访问latent向量
    latent = path['latent_history']  # [num_steps, hidden_dim]
    print(f"  Latent shape: {latent.shape}")
```

### 创建训练数据集

```python
from torch.utils.data import Dataset, DataLoader

class LatentPRMDataset(Dataset):
    def __init__(self, data_files):
        self.samples = []
        for file in data_files:
            data = torch.load(file)
            for path in data['paths']:
                if path['prm_score'] is not None:
                    self.samples.append({
                        'latent': path['latent_history'],
                        'label': path['prm_score']
                    })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]

# 创建数据加载器
dataset = LatentPRMDataset(data_files)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
```

## 配置参数

### 数据收集参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--collect_prm_data` | False | 启用PRM数据收集 |
| `--prm_output_dir` | "output/prm_data" | 数据保存目录 |
| `--prm_disable_pruning` | False | 禁用路径剪枝 |
| `--prm_disable_merging` | False | 禁用路径合并 |

### 多路径参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--num_paths` | 5 | 每个agent的路径数 |
| `--latent_steps` | 10 | 每条路径的latent步骤数 |
| `--diversity_strategy` | "hybrid" | 多样性策略 |
| `--temperature` | 0.7 | 基准温度 |

## 重要特性

### 1. 不影响现有功能
- 作为可选功能，默认禁用
- 不修改核心推理逻辑
- 仅在指定模式下激活

### 2. 最大化路径多样性
- 禁用剪枝：保留所有生成的路径
- 禁用合并：保持路径独立性
- 增加路径数量：更多样的推理路径

### 3. 完整的路径关系
- 记录父子关系
- 构建树状结构
- 支持路径追溯

### 4. 详细的日志记录
- INFO级别：进度和统计
- DEBUG级别：详细信息
- 分级合理，便于调试

## 日志示例

```
[LatentMASMultiPathMethod] Enabling PRM data collection mode
[LatentMASMultiPathMethod] Output directory: output/prm_data
[PRM DataCollection] Starting data collection for batch
[DataCollector] Starting question q_0
[DataCollector] Recording 10 paths for agent planner
[PathTreeBuilder] Building tree from 30 paths
[PRMScorer] Scoring 30 paths
[PRMScorer] Final answer correct: True
[PRM DataCollection] Saved batch data to: output/prm_data/batch_xxx.pt
```

## 下一步

### 1. 收集数据
```bash
bash collect_training_data.sh
```

### 2. 检查数据
```python
python examples/prm_data_collection_example.py
```

### 3. 训练PRM模型
- 使用收集的数据训练Qwen-0.6B
- 实现PRM模型架构
- 微调latent推理过程

## 文档

详细文档请参考：
- **使用指南**: `docs/prm-data-collection-guide.md`
- **实现总结**: `docs/prm-implementation-summary.md`
- **代码示例**: `examples/prm_data_collection_example.py`

## 故障排查

### 问题：内存不足
**解决**：减少`--num_paths`或`--latent_steps`

### 问题：文件过大
**解决**：减少路径数量或latent步骤数

### 问题：路径树不完整
**解决**：确保使用`--prm_disable_pruning`和`--prm_disable_merging`

## 技术支持

如有问题或建议，请查看文档或提交Issue。

---

**实现完成时间**: 2024年12月26日
**版本**: 1.0.0
**状态**: ✅ 完成并测试

