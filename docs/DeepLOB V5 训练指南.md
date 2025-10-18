# DeepLOB V5 训练指南

## 概述

V5 训练系统专为 **Triple-Barrier 标签** 和 **样本权重** 设计，是 DeepLOB 项目的核心训练管线。

### 核心特性

- ✅ **V5 标签**：{0:下跌, 1:持平, 2:上涨}（Triple-Barrier 方法）
- ✅ **样本权重**：训练/验证 loss 加权，测试输出加权/不加权指标
- ✅ **数据验证**：启动时完整的安全检查
- ✅ **温度校准**：降低加权 NLL/ECE
- ✅ **完整报表**：混淆矩阵、per-class PR/RC、加权/不加权指标
- ✅ **混合精度**：支持 AMP (FP16)，RTX 5090 优化
- ✅ **早停机制**：基于加权 macro-F1
- ✅ **TensorBoard**：实时监控训练进度

## 快速开始

### 1. 验证数据

```bash
# 检查 V5 数据是否存在
ls data/processed_v5/npz/

# 应包含：
# - stock_embedding_train.npz
# - stock_embedding_val.npz
# - stock_embedding_test.npz
# - normalization_meta.json
```

### 2. 检查配置

```bash
cat configs/train_v5.yaml

# 关键配置：
# - data.use_sample_weights: true
# - loss.class_weights: "auto"
# - calibration.enabled: true
# - train.early_stop.metric: "val.f1_macro_weighted"
```

### 3. 开始训练

```bash
# 基础训练（使用默认配置）
python scripts/train_deeplob_v5.py --config configs/train_v5.yaml

# 覆盖配置项
python scripts/train_deeplob_v5.py \
    --config configs/train_v5.yaml \
    --override optim.lr=0.0005 train.epochs=100

# 监控训练（另开终端）
tensorboard --logdir logs/deeplob_v5/
```

### 4. 查看结果

```bash
# 训练完成后，检查输出目录
ls checkpoints/v5/

# 输出文件：
# - deeplob_v5_best.pth          # 最佳模型（checkpoint）
# - confusion_matrix_test.png    # 测试集混淆矩阵
# - test_metrics.json            # 测试指标（加权/不加权）
# - calibration.pt               # 温度校准参数
# - label_mapping.json           # 标签映射
# - normalization_meta.json      # 正规化参数（副本）
# - config.yaml                  # 完整配置（副本）
```

## 数据格式

### V5 NPZ 文件结构

```python
{
    'X': (N, 100, 20),      # LOB 序列
    'y': (N,),              # 标签 {0, 1, 2}
    'weights': (N,),        # 样本权重
    'stock_ids': (N,)       # 股票 ID（可选）
}
```

### 标签映射

```json
{
  "0": "下跌 (down)",
  "1": "持平 (stationary/flat)",
  "2": "上涨 (up)"
}
```

## 评估指标

### 训练/验证阶段

每个 epoch 输出：

**加权指标**（用于早停和模型选择）：
- Loss（加权）
- Macro-F1（加权）

**不加权指标**（用于性能评估）：
- Accuracy
- Macro-F1
- Per-class Precision/Recall

### 测试阶段

最终报告包含：

1. **加权指标**：
   - Weighted Loss
   - Weighted Macro-F1

2. **不加权指标**：
   - Accuracy
   - Unweighted Macro-F1
   - Per-class Precision
   - Per-class Recall

3. **混淆矩阵**：
   - 归一化版本（PNG 图片）
   - 原始版本（JSON 数组）

4. **温度校准**（如果启用）：
   - 最优温度值
   - 校准前 NLL
   - 校准后 NLL

## 配置说明

### 核心配置项

#### 数据配置

```yaml
data:
  use_sample_weights: true    # 必须为 true（V5 核心特性）
  weights_normalize: "mean_to_1"  # 权重归一化到均值=1
  fail_on_missing_keys: true  # 缺必备键即报错
```

#### 损失函数

```yaml
loss:
  type: "ce"                  # CrossEntropy（或 "focal"）
  class_weights: "auto"       # 自动计算逆频率权重
  label_smoothing:
    global: 0.05              # 标签平滑
```

#### 优化器

```yaml
optim:
  name: "adamw"
  lr: 0.001
  weight_decay: 0.0001
  grad_clip: 1.0              # 梯度裁剪
  amp: true                   # 混合精度
```

#### 早停

```yaml
train:
  early_stop:
    metric: "val.f1_macro_weighted"  # 以加权 F1 为准
    patience: 10
    mode: "max"
```

#### 校准

```yaml
calibration:
  enabled: true               # 启用温度校准
  type: "temperature"
  opt_metric: "weighted_nll"  # 优化加权 NLL
```

## 验收标准

根据需求文档，V5 训练系统应满足：

- [x] 1. 仅用 `--config configs/train_v5.yaml` 可训练；V5 为预设模式
- [x] 2. 训练/验证 loss 以样本权重计算并以 `sum(w)` 正规化；测试输出加权/不加权指标
- [x] 3. 报表类别固定 `[下跌(0), 持平(1), 上涨(2)]`；保存 `label_mapping.json`
- [x] 4. 报表包含：不加权 acc/macro-F1/per-class PR/RC + 混淆矩阵、加权 macro-F1/loss
- [x] 5. 温度校准能降低加权 NLL/ECE，报告前后对照
- [x] 6. 输出工件齐备（模型、配置、混淆矩阵、指标、校准参数）
- [x] 7. 缺鍵/不一致/非法值会明确报错，不会默默忽略
- [x] 8. 启动时打印安全检查与数据摘要

## 常见问题

### Q1: 权重为什么要归一化？

**A**: 归一化到均值=1 可以：
- 保持 loss 的数值范围稳定
- 避免梯度过大/过小
- 使不同数据集的 loss 可比较

### Q2: 加权 vs 不加权指标有什么区别？

**A**:
- **加权指标**：考虑样本重要性，用于模型选择和早停
- **不加权指标**：平等对待所有样本，用于公平评估

### Q3: 为什么测试集同时输出两种指标？

**A**:
- **加权指标**：反映模型在"重要样本"上的表现（如高收益交易）
- **不加权指标**：反映模型在"所有样本"上的表现（如整体准确率）

### Q4: 温度校准有什么用？

**A**: 温度校准可以：
- 降低模型的过度自信（overly confident predictions）
- 使输出概率更接近真实概率
- 提升概率预测的可靠性（calibration）

### Q5: 如何调整学习率？

**A**: 使用命令行覆盖：

```bash
python scripts/train_deeplob_v5.py \
    --config configs/train_v5.yaml \
    --override optim.lr=0.0005
```

### Q6: 训练多久会停止？

**A**:
- 最多训练 50 epochs（默认配置）
- 如果连续 10 epochs 验证集加权 F1 无提升，则早停
- 可通过 `--override train.epochs=100` 调整

## 与 V4 的差异

| 特性 | V4 | V5 |
|------|----|----|
| **标签方法** | 固定 k 步 Δ% | Triple-Barrier |
| **标签含义** | {0:上涨, 1:持平, 2:下跌} | {0:下跌, 1:持平, 2:上涨} |
| **样本权重** | ❌ 无 | ✅ 必备 |
| **加权指标** | ❌ 无 | ✅ 完整支持 |
| **温度校准** | ❌ 无 | ✅ 内建 |
| **数据验证** | ⚠️ 部分 | ✅ 完整 |
| **早停指标** | val_acc | val.f1_macro_weighted |

## 下一步

完成训练后，可以：

1. **评估模型**：查看 `test_metrics.json` 和混淆矩阵
2. **分析错误**：检查哪些类别预测不准确
3. **超参数调优**：尝试不同的 lr、dropout、class_weights
4. **部署模型**：使用 `deeplob_v5_best.pth` 进行推理

## 支持

如有问题，请参考：
- 主配置文件：`configs/train_v5.yaml`
- 训练脚本：`scripts/train_deeplob_v5.py`
- 项目文档：`CLAUDE.md`

---

**最后更新**: 2025-10-18
**版本**: v1.0
