# Latent PRM训练数据收集功能实现总结

## 实现概述

本次实现为LatentMAS系统添加了完整的PRM（Process Reward Model）训练数据收集功能，用于收集多路径潜在推理数据，以便后续微调Qwen-0.6B模型。

## 实现的文件

### 1. 新增模块：`methods/latent_prm/`

#### 1.1 `__init__.py`
- 模块初始化文件
- 导出核心类：`LatentPRMDataCollector`, `PathTreeBuilder`, `PRMDataStorage`, `PRMScorer`

#### 1.2 `data_collector.py`
**功能**：收集推理路径数据

**核心类**：
- `PathRecord`: 单个路径的记录数据类
- `QuestionRecord`: 完整问题的记录数据类
- `LatentPRMDataCollector`: 数据收集器

**关键方法**：
- `start_question()`: 开始收集新问题
- `record_path()`: 记录单个路径
- `finish_question()`: 完成问题收集
- `get_collected_data()`: 获取收集的数据

**特点**：
- 自动将张量移到CPU避免GPU内存问题
- 清理元数据中的张量引用
- 详细的日志记录（INFO和DEBUG级别）

#### 1.3 `path_tree_builder.py`
**功能**：构建路径树结构并计算PRM评分

**核心类**：
- `TreeNode`: 树节点数据类
- `PathTreeBuilder`: 树构建器

**关键方法**：
- `build_tree()`: 从路径记录构建树
- `_compute_depths()`: 使用BFS计算节点深度
- `_compute_prm_scores()`: 反向传播计算PRM评分
- `get_path_to_root()`: 获取从叶子到根的路径
- `get_subtree_nodes()`: 获取子树所有节点

**评分策略**：
- 叶子节点：基于最终答案正确性（1.0或0.0）
- 内部节点：子节点PRM分数的平均值
- 从最大深度向根节点反向传播

#### 1.4 `data_storage.py`
**功能**：保存和加载PRM训练数据

**核心类**：
- `PRMDataStorage`: 数据存储处理器

**关键方法**：
- `save_question_data()`: 保存单个问题数据
- `save_batch_data()`: 保存批量数据
- `save_metadata()`: 保存元数据
- `load_question_data()`: 加载问题数据
- `load_batch_data()`: 加载批量数据
- `get_statistics()`: 获取存储统计
- `create_dataset_index()`: 创建数据集索引

**存储格式**：
- PyTorch .pt格式保存张量数据
- JSON格式保存元数据
- 自动时间戳命名
- 支持批量和单个保存

#### 1.5 `prm_scorer.py`
**功能**：PRM评分计算

**核心类**：
- `PRMScorer`: PRM评分器

**关键方法**：
- `score_paths()`: 为所有路径计算PRM分数
- `compute_path_quality_score()`: 计算综合质量分数
- `get_score_distribution()`: 获取分数分布统计
- `identify_critical_paths()`: 识别高质量路径
- `identify_failure_paths()`: 识别低质量路径

**评分逻辑**：
- 基于拓扑排序（agent_idx）反向传播
- 支持自定义正确/错误分数
- 提供详细的统计分析

### 2. 修改的文件

#### 2.1 `methods/latent_mas_multipath.py`

**修改点1：导入PRM模块**
```python
from .latent_prm import LatentPRMDataCollector, PathTreeBuilder, PRMDataStorage, PRMScorer
```

**修改点2：初始化PRM组件**
- 在`__init__`方法中初始化PRM组件（默认禁用）
- 添加`collect_prm_data`标志

**修改点3：新增`enable_prm_data_collection()`方法**
- 启用PRM数据收集模式
- 初始化所有PRM组件
- 可选禁用剪枝和合并

**修改点4：修改`run_batch()`方法**

**位置1：批处理开始**
- 为每个问题启动数据收集

**位置2：路径生成后**
- 添加agent信息到路径元数据
- 记录batch_idx用于后续关联

**位置3：剪枝阶段**
- 在PRM模式下跳过剪枝（如果启用）
- 保留所有生成的路径

**位置4：合并阶段**
- 在PRM模式下跳过合并（如果启用）
- 保留路径多样性

**位置5：Agent处理完成**
- 记录所有路径到数据收集器
- 保存路径关系

**位置6：批处理完成**
- 完成问题数据收集
- 构建路径树结构
- 计算PRM评分
- 保存数据到磁盘
- 保存元数据
- 清理收集器

#### 2.2 `run.py`

**修改点1：添加命令行参数**
```python
parser.add_argument("--collect_prm_data", action="store_true")
parser.add_argument("--prm_output_dir", type=str, default="output/prm_data")
parser.add_argument("--prm_disable_pruning", action="store_true")
parser.add_argument("--prm_disable_merging", action="store_true")
```

**修改点2：启用PRM数据收集**
- 在方法初始化后检查`--collect_prm_data`标志
- 调用`enable_prm_data_collection()`方法
- 传递配置参数

### 3. 新增脚本

#### 3.1 `collect_training_data.sh`

**功能**：便捷的数据收集脚本

**配置项**：
- 数据集配置（TASK, MAX_SAMPLES, SEED）
- 模型配置（MAX_NEW_TOKENS, TEMPERATURE, TOP_P）
- 多路径配置（NUM_PATHS, LATENT_STEPS, DIVERSITY_STRATEGY）
- 数据收集配置（OUTPUT_DIR, BATCH_SIZE）
- 日志配置（LOG_LEVEL）

**特点**：
- 支持云端和本地环境切换
- 详细的配置说明
- 自动启用PRM数据收集模式
- 禁用剪枝和合并以最大化路径多样性

### 4. 文档

#### 4.1 `docs/prm-data-collection-guide.md`
- 完整的使用指南
- 功能特性说明
- 快速开始教程
- 命令行参数详解
- 数据格式说明
- PRM评分机制
- 数据加载示例
- 最佳实践
- 故障排查

#### 4.2 `examples/prm_data_collection_example.py`
- 数据加载示例代码
- 路径树分析
- 训练样本提取
- PyTorch Dataset创建

## 核心功能流程

### 数据收集流程

```
1. 启动批处理
   ↓
2. 为每个问题调用 start_question()
   ↓
3. 对每个Agent：
   a. 生成多条推理路径
   b. 添加agent元数据
   c. 评分（不剪枝/不合并）
   d. 调用 record_path() 记录所有路径
   ↓
4. Judger生成最终答案
   ↓
5. 调用 finish_question() 完成收集
   ↓
6. 构建路径树（build_tree）
   ↓
7. 计算PRM评分（score_paths）
   ↓
8. 保存数据（save_batch_data）
   ↓
9. 清理收集器
```

### PRM评分流程

```
1. 识别叶子节点（最后一个agent的路径）
   ↓
2. 为叶子节点赋值：
   - 正确答案 → 1.0
   - 错误答案 → 0.0
   ↓
3. 反向传播（从最大深度到根）：
   对每个内部节点：
   - 收集所有子节点的PRM分数
   - 计算平均值
   - 赋值给当前节点
   ↓
4. 所有节点都有PRM分数
```

## 数据格式详解

### .pt文件结构

```python
{
    # 问题信息
    "question_id": str,
    "question": str,
    "gold_answer": str,
    "final_answer": str,
    "is_correct": bool,
    
    # 路径数据
    "paths": [
        {
            "path_id": int,
            "agent_name": str,
            "agent_idx": int,
            "parent_path_id": Optional[int],
            "child_path_ids": List[int],
            "score": float,              # 原始评分
            "prm_score": float,          # PRM评分
            "latent_history": Tensor,    # [num_steps, hidden_dim]
            "hidden_states": Tensor,     # [hidden_dim]
            "num_latent_steps": int,
            "metadata": dict
        }
    ],
    
    # 树结构
    "tree_structure": {
        "nodes": List[dict],
        "edges": List[tuple],
        "root_ids": List[int],
        "is_correct": bool,
        "num_nodes": int,
        "num_edges": int,
        "max_depth": int
    }
}
```

## 关键设计决策

### 1. 不剪枝、不合并
**原因**：
- 最大化路径多样性
- 收集所有可能的推理路径
- 为PRM训练提供丰富的正负样本

**实现**：
- 添加`_prm_disable_pruning`和`_prm_disable_merging`标志
- 在相应逻辑中检查标志并跳过

### 2. 仅基于最终答案正确性评分
**原因**：
- 简单明确的监督信号
- 避免中间步骤评估的主观性
- 适合二分类PRM训练

**实现**：
- 叶子节点：1.0（正确）或0.0（错误）
- 内部节点：子节点平均值

### 3. PyTorch .pt格式
**原因**：
- 高效存储张量数据
- 原生PyTorch支持
- 易于加载和训练

**实现**：
- 使用`torch.save()`和`torch.load()`
- 自动处理张量序列化

### 4. 树状结构保存
**原因**：
- 完整记录路径关系
- 支持路径追溯和分析
- 便于理解推理过程

**实现**：
- 记录parent_id和child_path_ids
- 保存edges列表
- 计算深度和统计信息

## 日志记录策略

### INFO级别
- 数据收集进度
- 路径数量统计
- 树构建信息
- PRM评分统计
- 文件保存位置

### DEBUG级别
- 详细的路径信息
- 评分计算过程
- 树结构细节
- 内存清理操作

## 性能考虑

### 内存管理
1. **张量移到CPU**：所有路径数据立即移到CPU
2. **清理元数据**：删除元数据中的张量引用
3. **批处理后清理**：清空收集器释放内存

### 存储优化
1. **批量保存**：支持批量保存减少I/O
2. **压缩存储**：PyTorch自动压缩
3. **索引文件**：快速检索数据

## 测试建议

### 单元测试
1. 测试数据收集器的路径记录
2. 测试树构建器的树结构
3. 测试评分器的PRM计算
4. 测试存储器的保存/加载

### 集成测试
1. 端到端数据收集流程
2. 多个问题的批处理
3. 不同配置的数据收集

### 验证测试
1. 验证PRM评分的正确性
2. 验证树结构的完整性
3. 验证数据格式的一致性

## 未来改进方向

### 1. 支持更多评分策略
- 中间步骤评估
- 多维度评分
- 自定义评分函数

### 2. 数据增强
- 路径采样策略
- 负样本生成
- 对比学习样本

### 3. 分布式收集
- 多GPU并行收集
- 分布式存储
- 实时数据流

### 4. 可视化工具
- 路径树可视化
- 评分分布可视化
- 训练进度监控

## 总结

本次实现完整地添加了PRM训练数据收集功能，具有以下特点：

✅ **完整性**：覆盖数据收集、树构建、评分、存储全流程
✅ **可扩展性**：模块化设计，易于扩展和修改
✅ **健壮性**：详细的错误处理和日志记录
✅ **易用性**：提供便捷的shell脚本和详细文档
✅ **性能**：优化内存使用和存储效率
✅ **不影响现有功能**：作为可选功能，不影响推理模式

该实现为后续训练Qwen-0.6B的latent-PRM模型提供了完整的数据收集解决方案。

