# V5 Pro 技术规格文档

## 📋 目录

1. [概述](#概述)
2. [核心组件](#核心组件)
3. [数据流程](#数据流程)
4. [技术实现细节](#技术实现细节)
5. [API 参考](#api-参考)
6. [性能优化](#性能优化)
7. [错误处理](#错误处理)

---

## 1. 概述

### 1.1 版本信息

- **版本号**: 5.0.0
- **发布日期**: 2025-10-18
- **基于文档**: `V5_Pro_NoMLFinLab_Guide.md`

### 1.2 核心升级

| 模块 | V4 | V5 Pro |
|------|-----|--------|
| 波动率估计 | 简单相对波动率 | EWMA / GARCH (arch 库) |
| 标签生成 | 固定 k 步价格变动 | Triple-Barrier 自适应标签 |
| 样本权重 | 无权重 | 收益加权 + 时间衰减 + 类别平衡 |
| 输出格式 | X, y | X, y, weights + 详细 metadata |

### 1.3 依赖套件

```python
# 必需依赖
numpy >= 1.26
pandas >= 2.0
ruamel.yaml >= 0.17  # YAML 配置管理（项目使用 yaml_manager）
scikit-learn >= 1.3
triple-barrier  # PyPI 版本
arch >= 6.0     # GARCH 波动率

# DeepLOB 训练依赖（继承自 V4）
torch >= 2.0
```

---

## 2. 核心组件

### 2.1 波动率估计模块

#### 2.1.1 EWMA 波动率

**函数签名**:
```python
def ewma_vol(close: pd.Series, halflife: int = 60) -> pd.Series
```

**原理**:
- 指数加权移动平均（Exponential Weighted Moving Average）
- 对数收益率的指数加权方差的平方根
- 半衰期参数控制历史数据的权重衰减速度

**公式**:
```
返回率: r_t = log(P_t / P_{t-1})
方差: σ²_t = EWMA(r²_t, α=2/(halflife+1))
波动率: σ_t = sqrt(σ²_t)
```

**适用场景**:
- 通用场景，计算速度快
- 适合高频数据（分钟级）
- 平滑噪音，响应速度可调

#### 2.1.2 GARCH(1,1) 波动率

**函数签名**:
```python
def garch11_vol(close: pd.Series) -> pd.Series
```

**原理**:
- 广义自回归条件异方差模型（GARCH）
- 捕捉波动率的聚集效应（Volatility Clustering）
- 使用 `arch` 库的专业实现

**模型方程**:
```
r_t = μ + ε_t
ε_t = σ_t × z_t,  z_t ~ N(0,1)
σ²_t = ω + α×ε²_{t-1} + β×σ²_{t-1}
```

**适用场景**:
- 金融时间序列波动率建模
- 捕捉波动率的时间依赖性
- 需要至少 50 个数据点

**回退机制**:
- 数据点不足（< 50）时，自动回退到 EWMA
- GARCH 拟合失败时，自动回退到 EWMA

---

### 2.2 Triple-Barrier 标签生成

#### 2.2.1 原理

Triple-Barrier 方法同时设置三个退出条件：
1. **止盈屏障（Take Profit）**: 价格上涨达到 `pt_mult × volatility`
2. **止损屏障（Stop Loss）**: 价格下跌达到 `sl_mult × volatility`
3. **时间屏障（Time Barrier）**: 达到最大持有期 `max_holding`

**退出逻辑**:
- 三个屏障中**最先触发**的决定标签
- 触发止盈 → 标签 = 2（上涨）
- 触发止损 → 标签 = 0（下跌）
- 触发时间 → 标签 = 1（持平，或根据收益判断）

#### 2.2.2 函数签名

```python
def tb_labels(
    close: pd.Series,
    vol: pd.Series,
    pt_mult: float = 2.0,
    sl_mult: float = 2.0,
    max_holding: int = 200,
    min_return: float = 0.0001
) -> pd.DataFrame
```

**返回值**:
```python
{
    "y": {-1, 0, 1},        # 标签（-1=下跌, 0=持平, 1=上涨）
    "ret": float,           # 实际收益
    "tt": int,              # 触发时间步数
    "why": str,             # 触发原因 {'up', 'down', 'time'}
    "up_p": float,          # 止盈价格
    "dn_p": float           # 止损价格
}
```

#### 2.2.3 参数调优建议

| 参数 | 保守配置 | 标准配置 | 激进配置 |
|------|---------|---------|---------|
| `pt_mult` | 3.0σ | 2.0σ | 1.5σ |
| `sl_mult` | 3.0σ | 2.0σ | 1.5σ |
| `max_holding` | 300 bars | 200 bars | 100 bars |
| `min_return` | 0.0002 (0.02%) | 0.0001 (0.01%) | 0.00005 (0.005%) |

**影响分析**:
- `pt_mult / sl_mult` **越小** → 标签变化**越频繁** → 交易信号**越多**
- `max_holding` **越短** → 持平标签**越多** → 避免长期持仓
- `min_return` **越大** → 持平标签**越多** → 忽略微小价格变动

---

### 2.3 样本权重计算

#### 2.3.1 权重公式

**三阶段加权**:
```python
# 1. 收益权重（基础）
w_return = |收益| × return_scaling × exp(-触发时间 / tau)

# 2. 类别平衡权重
w_class = sklearn.compute_class_weight('balanced', ...)

# 3. 最终权重
w_final = w_return × w_class
w_final = w_final / mean(w_final)  # 归一化到均值为 1
```

#### 2.3.2 参数说明

| 参数 | 含义 | 默认值 | 建议范围 |
|------|------|--------|---------|
| `tau` | 时间衰减参数 | 100.0 | 50-200 |
| `return_scaling` | 收益缩放系数 | 10.0 | 5-20 |
| `balance_classes` | 启用类别平衡 | True | True/False |

**设计理念**:
- **收益权重**: 高收益样本权重高（更重要）
- **时间衰减**: 快速退出的样本权重高（决策准确）
- **类别平衡**: 避免样本不平衡导致的偏差

#### 2.3.3 使用示例

```python
from scripts.extract_tw_stock_data_v5 import make_sample_weight

# 计算样本权重
weights = make_sample_weight(
    ret=tb_df["ret"],         # 实际收益序列
    tt=tb_df["tt"],           # 触发时间步数
    y=y_tb,                   # 标签序列 {0,1,2}
    tau=100.0,                # 时间衰减参数
    scale=10.0,               # 收益缩放系数
    balance=True              # 启用类别平衡
)
```

---

## 3. 数据流程

### 3.1 完整流水线

```
原始 LOB 数据 (.txt)
  ↓
【步骤 1】解析与品质检查 (parse_line)
  ├─ 移除试撮（IsTrialMatch='1'）
  ├─ 时间窗口过滤（09:00-13:30）
  ├─ 价差检查（bid-ask spread）
  └─ 价格限制检查（涨跌停）
  ↓
【步骤 2】去重 (dedup_by_timestamp_keep_last)
  ├─ 按 (时间戳, TotalVolume) 去重
  └─ 同一组合保留最后一笔
  ↓
【步骤 3】10 事件聚合 (aggregate_chunks_of_10)
  ├─ 每 10 笔快照 → 1 时间点
  ├─ 价格/数量取视窗末端
  └─ 输出: (N, 20) 特征 + mid 价格
  ↓
【步骤 4】按股票串接（跨日合并）
  ├─ 以股票为单位合并所有天数据
  └─ 按日期排序
  ↓
【步骤 5】波动率估计 (ewma_vol / garch11_vol)
  ├─ 基于 mid 价格序列
  └─ 输出: (N,) 波动率序列
  ↓
【步骤 6】Triple-Barrier 标签 (tb_labels)
  ├─ 输入: close, vol, 参数
  └─ 输出: y {-1,0,1}, ret, tt, why
  ↓
【步骤 7】样本权重计算 (make_sample_weight)
  ├─ 输入: ret, tt, y
  └─ 输出: weights (N,)
  ↓
【步骤 8】滑窗抽样 (100 timesteps)
  ├─ 输入: (N, 20) + y + weights
  └─ 输出: X (M, 100, 20), y (M,), w (M,)
  ↓
【步骤 9】Z-Score 正规化
  ├─ 基于训练集计算 μ, σ
  └─ 应用到 train/val/test
  ↓
【步骤 10】70/15/15 切分
  ├─ 按股票数量切分（不是样本数）
  └─ 确保股票不跨 split
  ↓
【输出】NPZ 文件 + Metadata
  ├─ stock_embedding_train.npz
  ├─ stock_embedding_val.npz
  ├─ stock_embedding_test.npz
  └─ normalization_meta.json
```

### 3.2 数据形状变化

```
原始数据:  (raw_events,)          例: 10,000,000 事件
  ↓ 清洗与品质检查
清洗后:    (cleaned_events,)      例: 8,500,000 事件
  ↓ 10 事件聚合
聚合后:    (aggregated_points, 20) 例: 850,000 × 20
  ↓ 跨日合并（按股票）
合并后:    (total_points, 20)     例: 850,000 × 20 (195 档股票)
  ↓ 滑窗抽样
滑窗后:    (samples, 100, 20)     例: 750,000 × 100 × 20
  ↓ 70/15/15 切分
训练集:    (train_samples, 100, 20)  例: 525,000 × 100 × 20
验证集:    (val_samples, 100, 20)    例: 112,500 × 100 × 20
测试集:    (test_samples, 100, 20)   例: 112,500 × 100 × 20
```

---

## 4. 技术实现细节

### 4.1 时间序列窗口设计

**窗口配置**:
- **窗口大小**: 100 timesteps（固定）
- **步长**: 1 timestep（滑窗）
- **标签偏移**: horizon = 0（当前时间点）

**示例**:
```
时间点:  0   1   2 ... 98  99  100 101 ...
窗口 1:  [----------------]  ← 标签来自 t=99
窗口 2:      [----------------]  ← 标签来自 t=100
窗口 3:          [----------------]  ← 标签来自 t=101
```

**边界处理**:
- 前端：跳过前 99 个时间点（不足 100）
- 后端：移除末端 `max_holding` 个时间点（标签无效）

### 4.2 Z-Score 正规化

**计算公式**:
```python
# 训练集拟合
μ = mean(X_train, axis=0)  # (20,)
σ = std(X_train, axis=0)   # (20,)

# 应用到所有集合
X_norm = (X - μ) / σ
```

**注意事项**:
- μ 和 σ **仅使用训练集**计算
- 验证集和测试集使用**相同的** μ 和 σ（避免数据泄漏）
- σ < 1e-8 时，设为 1.0（避免除零）

### 4.3 股票级别切分

**切分策略**:
```python
# 按股票数量切分（不是样本数）
n_stocks = 195
n_train = int(0.7 * n_stocks)  # 136 档
n_val = int(0.15 * n_stocks)   # 29 档
n_test = 195 - 136 - 29        # 30 档

# 按流动性排序（时间点数量）
stocks.sort(key=lambda x: x.n_points, reverse=True)

# 分配股票到各集合
train_stocks = stocks[:n_train]
val_stocks = stocks[n_train:n_train+n_val]
test_stocks = stocks[n_train+n_val:]
```

**优点**:
- 避免股票跨 split（防止数据泄漏）
- 流动性高的股票优先进入训练集
- 测试集包含完整的股票交易历史

---

## 5. API 参考

### 5.1 主函数

#### `extract_tw_stock_data_v5.py::main()`

**功能**: 完整的 V5 数据流水线

**命令行参数**:
```bash
python scripts/extract_tw_stock_data_v5.py \
    --input-dir ./data/temp \              # 输入目录
    --output-dir ./data/processed_v5 \     # 输出目录
    --config ./configs/config_pro_v5.yaml \  # 配置文件
    --make-npz                              # 生成 NPZ 文件
```

**配置文件格式** (`config_pro_v5.yaml`):
```yaml
version: "5.0.0"

volatility:
  method: 'ewma'     # 'ewma' 或 'garch'
  halflife: 60

triple_barrier:
  pt_multiplier: 2.0
  sl_multiplier: 2.0
  max_holding: 200
  min_return: 0.0001

sample_weights:
  enabled: true
  tau: 100.0
  return_scaling: 10.0
  balance_classes: true

data:
  aggregation_factor: 10
  seq_len: 100
  alpha: 0.002
  horizons: [1, 2, 3, 5, 10]
  trading_start: 90000
  trading_end: 133000

split:
  train_ratio: 0.7
  val_ratio: 0.15
  test_ratio: 0.15
```

### 5.2 输出格式

#### NPZ 文件结构

```python
# 加载数据
data = np.load("stock_embedding_train.npz")

# 访问字段
X = data['X']           # (N, 100, 20) - LOB 特征窗口
y = data['y']           # (N,) - 标签 {0, 1, 2}
weights = data['weights']  # (N,) - 样本权重
stock_ids = data['stock_ids']  # (N,) - 股票代码
```

#### Metadata 结构

```json
{
  "format": "deeplob_v5_pro",
  "version": "5.0.0",
  "creation_date": "2025-10-18T12:00:00",
  "seq_len": 100,
  "feature_dim": 20,

  "volatility": {
    "method": "ewma",
    "halflife": 60
  },

  "triple_barrier": {
    "pt_multiplier": 2.0,
    "sl_multiplier": 2.0,
    "max_holding": 200,
    "min_return": 0.0001
  },

  "sample_weights": {
    "enabled": true,
    "tau": 100.0,
    "return_scaling": 10.0,
    "balance_classes": true
  },

  "normalization": {
    "method": "zscore",
    "computed_on": "train_set",
    "feature_means": [...],  # (20,)
    "feature_stds": [...]    # (20,)
  },

  "data_quality": {
    "total_raw_events": 10000000,
    "cleaned_events": 8500000,
    "aggregated_points": 850000,
    "valid_windows": 750000,
    "tb_success": 195
  },

  "data_split": {
    "method": "by_stock_count",
    "train_stocks": 136,
    "val_stocks": 29,
    "test_stocks": 30,
    "total_stocks": 195,
    "results": {
      "train": {
        "samples": 525000,
        "label_dist": [180000, 165000, 180000],
        "weight_stats": {"mean": 1.0, "std": 0.8, "max": 5.2}
      },
      ...
    }
  }
}
```

---

## 6. 性能优化

### 6.1 计算性能

**瓶颈分析**:
| 步骤 | 时间占比 | 优化建议 |
|------|---------|---------|
| 数据读取 | 10% | 使用 SSD，并行读取 |
| Triple-Barrier | 40% | NumPy 向量化，避免 Python 循环 |
| GARCH 拟合 | 30% | 使用 EWMA（速度快 10 倍） |
| 滑窗抽样 | 15% | NumPy stride_tricks（内存视图） |
| Z-Score | 5% | 一次性批量计算 |

**优化后性能**:
- 195 档股票，1000 万事件
- RTX 5090（单GPU）: ~15 分钟
- CPU（32 核）: ~45 分钟

### 6.2 内存优化

**内存使用**:
```
原始数据（文本）: ~2 GB
清洗后（内存）: ~4 GB
NPZ 输出（压缩）: ~1.5 GB
峰值内存占用: ~8 GB
```

**降低内存策略**:
1. 分批处理股票（每次 50 档）
2. 使用 `np.float32` 代替 `np.float64`
3. 启用 NPZ 压缩（`np.savez_compressed`）

---

## 7. 错误处理

### 7.1 异常类型

| 异常 | 原因 | 处理方式 |
|------|------|---------|
| `ValueError: 不支援的波动率方法` | 配置错误 | 检查 `config.volatility.method` |
| `Triple-Barrier 失败` | 数据点不足或参数错误 | 检查股票时间序列长度 |
| `数据点不足` | 股票数据太短 | 自动跳过，记录警告 |
| `GARCH 拟合失败` | 数据质量或数量问题 | 自动回退到 EWMA |

### 7.2 日志级别

```python
# 配置日志
logging.basicConfig(
    level=logging.INFO,  # INFO / DEBUG / WARNING / ERROR
    format='%(asctime)s - %(levelname)s - %(message)s'
)
```

**日志示例**:
```
2025-10-18 12:00:00 - INFO - V5 Pro 资料流水线启动
2025-10-18 12:00:01 - INFO - 找到 100 个档案待处理
2025-10-18 12:01:30 - WARNING - 2330.TW: 跳过（只有 80 个点）
2025-10-18 12:15:00 - INFO - ✅ 已保存: ./data/processed_v5/npz/stock_embedding_train.npz
2025-10-18 12:15:01 - INFO - [完成] V5 转换成功
```

---

## 附录 A: 与 DeepLOB 训练集成

### A.1 数据载入

```python
import numpy as np

# 加载 V5 数据
data = np.load("data/processed_v5/npz/stock_embedding_train.npz")
X_train = data['X']       # (N, 100, 20)
y_train = data['y']       # (N,)
w_train = data['weights'] # (N,)

# 加载 metadata
import json
with open("data/processed_v5/npz/normalization_meta.json") as f:
    meta = json.load(f)
```

### A.2 权重训练

**PyTorch 示例**:
```python
import torch
from torch.utils.data import Dataset, DataLoader

class WeightedLOBDataset(Dataset):
    def __init__(self, X, y, weights):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)
        self.weights = torch.from_numpy(weights)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.weights[idx]

# 创建 DataLoader
dataset = WeightedLOBDataset(X_train, y_train, w_train)
loader = DataLoader(dataset, batch_size=64, shuffle=True)

# 训练循环
criterion = torch.nn.CrossEntropyLoss(reduction='none')

for X_batch, y_batch, w_batch in loader:
    logits = model(X_batch)
    loss_vec = criterion(logits, y_batch)
    loss = (loss_vec * w_batch).mean()  # 加权平均

    loss.backward()
    optimizer.step()
```

---

## 附录 B: 参数调优指南

### B.1 波动率参数

| 场景 | 推荐方法 | 参数 |
|------|---------|------|
| 高频交易（分钟级） | EWMA | halflife=30-60 |
| 日内交易 | EWMA | halflife=60-120 |
| 波动率聚集明显 | GARCH | 默认参数 |
| 计算资源有限 | EWMA | halflife=60 |

### B.2 Triple-Barrier 参数

**调优目标**:
- 提高标签质量（Sharpe Ratio）
- 平衡标签分布（避免过度不平衡）
- 适应市场波动性

**调优流程**:
1. 初始配置：`pt_mult=2.0, sl_mult=2.0, max_holding=200`
2. 观察标签分布（目标：上涨 30-40%，持平 20-40%，下跌 30-40%）
3. 若持平过多 → 降低 `max_holding` 或增加 `min_return`
4. 若上涨/下跌过少 → 降低 `pt_mult / sl_mult`

### B.3 样本权重参数

| 参数 | 影响 | 调优建议 |
|------|------|---------|
| `tau` | 时间衰减速度 | 50（快） - 200（慢） |
| `return_scaling` | 收益权重强度 | 5（弱） - 20（强） |
| `balance_classes` | 类别平衡 | True（推荐） |

---

**文档版本**: 1.0
**最后更新**: 2025-10-18
**维护者**: DeepLOB-Pro Team
