# V5 Pro 使用指南

## 📋 目录

1. [快速开始](#快速开始)
2. [安装与环境配置](#安装与环境配置)
3. [基本使用](#基本使用)
4. [进阶配置](#进阶配置)
5. [常见问题](#常见问题)
6. [最佳实践](#最佳实践)
7. [故障排查](#故障排查)

---

## 1. 快速开始

### 1.1 三步骤快速上手

```bash
# 步骤 1: 安装依赖
pip install triple-barrier arch ruamel.yaml pandas numpy scikit-learn

# 步骤 2: 运行脚本（使用默认配置）
python scripts/extract_tw_stock_data_v5.py \
    --input-dir ./data/temp \
    --output-dir ./data/processed_v5

# 步骤 3: 查看输出
ls -lh ./data/processed_v5/npz/
```

### 1.2 预期输出

```
data/processed_v5/npz/
├── stock_embedding_train.npz      # 训练集（70%）
├── stock_embedding_val.npz        # 验证集（15%）
├── stock_embedding_test.npz       # 测试集（15%）
└── normalization_meta.json        # 元数据（包含所有配置和统计）
```

### 1.3 输出说明

**NPZ 文件内容**:
```python
import numpy as np

# 加载数据
data = np.load("data/processed_v5/npz/stock_embedding_train.npz")

# 查看形状
print("X shape:", data['X'].shape)          # (N, 100, 20) - LOB 特征窗口
print("y shape:", data['y'].shape)          # (N,) - 标签 {0, 1, 2}
print("weights shape:", data['weights'].shape)  # (N,) - 样本权重 (V5 新增)
print("stock_ids:", data['stock_ids'][:5])  # ['2330.TW', '2317.TW', ...]
```

**标签含义**:
- `0`: 下跌（Stop Loss 触发）
- `1`: 持平（Time Barrier 触发或收益 < min_return）
- `2`: 上涨（Take Profit 触发）

---

## 2. 安装与环境配置

### 2.1 创建虚拟环境（推荐）

```bash
# 使用 conda
conda create -n deeplob-v5 python=3.11 -y
conda activate deeplob-v5

# 或使用 venv
python -m venv deeplob-v5-env
source deeplob-v5-env/bin/activate  # Linux/Mac
# deeplob-v5-env\Scripts\activate  # Windows
```

### 2.2 安装依赖

#### 必需依赖

```bash
# 核心数据处理
pip install numpy>=1.26 pandas>=2.0 scipy>=1.11

# V5 专业套件
pip install triple-barrier arch scikit-learn>=1.3

# 配置文件支持（使用项目的 yaml_manager）
pip install ruamel.yaml
```

#### 可选依赖（DeepLOB 训练）

```bash
# PyTorch（根据 CUDA 版本选择）
pip install torch==2.5.0 torchvision==0.20.0 --index-url https://download.pytorch.org/whl/cu124  # CUDA 12.4
# pip install torch==2.5.0 torchvision==0.20.0  # CPU 版本

# 其他训练工具
pip install tensorboard matplotlib seaborn
```

### 2.3 验证安装

```bash
python -c "import triple_barrier; print('✅ triple-barrier installed')"
python -c "import arch; print('✅ arch installed')"
python -c "from ruamel.yaml import YAML; print('✅ ruamel.yaml installed')"
python -c "import pandas; print(f'✅ pandas {pandas.__version__}')"
```

---

## 3. 基本使用

### 3.1 命令行参数

#### 完整参数列表

```bash
python scripts/extract_tw_stock_data_v5.py \
    --input-dir ./data/temp \                  # 输入目录（必需）
    --output-dir ./data/processed_v5 \         # 输出目录（必需）
    --config ./configs/config_pro_v5.yaml \    # 配置文件（可选，默认值存在）
    --make-npz                                  # 生成 NPZ 文件（默认启用）
```

#### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--input-dir` | str | `./data/temp` | 包含原始 .txt 文件的目录 |
| `--output-dir` | str | `./data/processed_v5` | 输出目录 |
| `--config` | str | `./configs/config_pro_v5.yaml` | V5 配置文件路径 |
| `--make-npz` | flag | True | 是否生成 NPZ 文件 |

### 3.2 输入数据格式

**文件组织**:
```
data/temp/
├── 20240101_TW_LOB.txt
├── 20240102_TW_LOB.txt
├── 20240103_TW_LOB.txt
└── ...
```

**文件内容**（每行一笔报价，`||` 分隔，无表头）:
```
QType||Symbol||Name||RefPrice||UpperPrice||LowerPrice||...||MatchTime||IsTrialMatch
S||2330.TW||台积电||590.0||620.0||560.0||...||093000||0
S||2330.TW||台积电||590.0||620.0||560.0||...||093001||0
...
```

**关键字段**（34 个字段）:
- `[0]`: QType（报价类型）
- `[1]`: Symbol（股票代码）
- `[2]`: Name（股票名称）
- `[3-5]`: RefPrice, UpperPrice, LowerPrice（参考价、涨停价、跌停价）
- `[12-21]`: Bid1~Bid5 (P,Q)×5（买方五档）
- `[22-31]`: Ask1~Ask5 (P,Q)×5（卖方五档）
- `[32]`: MatchTime（成交时间，HHMMSS）
- `[33]`: IsTrialMatch（是否试撮，1=是）

### 3.3 运行示例

#### 示例 1: 基本运行（使用默认配置）

```bash
python scripts/extract_tw_stock_data_v5.py \
    --input-dir ./data/temp \
    --output-dir ./data/processed_v5
```

**预期日志**:
```
2025-10-18 12:00:00 - INFO - ============================================================
2025-10-18 12:00:00 - INFO - V5 Pro 资料流水线启动
2025-10-18 12:00:00 - INFO - ============================================================
2025-10-18 12:00:00 - INFO - 输入目录: ./data/temp
2025-10-18 12:00:00 - INFO - 输出目录: ./data/processed_v5
2025-10-18 12:00:00 - INFO - 配置版本: 5.0.0
2025-10-18 12:00:00 - INFO - 波动率方法: ewma
2025-10-18 12:00:00 - INFO - Triple-Barrier: PT=2.0σ, SL=2.0σ, MaxHold=200
2025-10-18 12:00:00 - INFO - ============================================================

2025-10-18 12:00:01 - INFO - 找到 100 个档案待处理
2025-10-18 12:01:30 - INFO - 处理日期 20240101，共 5 个档案
...
2025-10-18 12:15:00 - INFO - ✅ 已保存: ./data/processed_v5/npz/stock_embedding_train.npz
2025-10-18 12:15:01 - INFO - ✅ 已保存: ./data/processed_v5/npz/stock_embedding_val.npz
2025-10-18 12:15:02 - INFO - ✅ 已保存: ./data/processed_v5/npz/stock_embedding_test.npz
2025-10-18 12:15:03 - INFO - ✅ Metadata 已保存: ./data/processed_v5/npz/normalization_meta.json

2025-10-18 12:15:04 - INFO - ============================================================
2025-10-18 12:15:04 - INFO - [完成] V5 转换成功，输出资料夹: ./data/processed_v5
2025-10-18 12:15:04 - INFO - ============================================================
2025-10-18 12:15:04 - INFO - 统计资料:
2025-10-18 12:15:04 - INFO -   原始事件数: 10,000,000
2025-10-18 12:15:04 - INFO -   清洗后: 8,500,000
2025-10-18 12:15:04 - INFO -   聚合后时间点: 850,000
2025-10-18 12:15:04 - INFO -   有效窗口: 750,000
2025-10-18 12:15:04 - INFO -   Triple-Barrier 成功: 195
2025-10-18 12:15:04 - INFO - ============================================================
```

#### 示例 2: 使用自定义配置

```bash
# 创建自定义配置文件
cat > my_config.yaml << EOF
version: "5.0.0"
volatility:
  method: 'garch'    # 使用 GARCH 波动率
  halflife: 60
triple_barrier:
  pt_multiplier: 1.5  # 更激进的止盈
  sl_multiplier: 1.5  # 更激进的止损
  max_holding: 100    # 更短的持有期
  min_return: 0.0002  # 更高的最小收益阈值
sample_weights:
  enabled: true
  tau: 50.0           # 更快的时间衰减
  return_scaling: 15.0  # 更强的收益权重
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
EOF

# 使用自定义配置运行
python scripts/extract_tw_stock_data_v5.py \
    --input-dir ./data/temp \
    --output-dir ./data/processed_v5_custom \
    --config ./my_config.yaml
```

---

## 4. 进阶配置

### 4.1 波动率方法选择

#### EWMA（推荐，默认）

**优点**:
- 计算速度快（~10倍于 GARCH）
- 稳定性好，适合高频数据
- 参数少，易于调优

**缺点**:
- 无法捕捉波动率聚集效应

**配置示例**:
```yaml
volatility:
  method: 'ewma'
  halflife: 60  # 半衰期（bars）
```

**参数调优**:
- `halflife=30`: 快速响应（适合高频交易）
- `halflife=60`: 标准配置（日内交易）
- `halflife=120`: 平滑响应（中长期）

#### GARCH（适合波动率聚集明显的市场）

**优点**:
- 捕捉波动率的时间依赖性
- 专业金融模型

**缺点**:
- 计算慢（需要迭代优化）
- 需要足够数据点（≥ 50）
- 可能拟合失败（自动回退到 EWMA）

**配置示例**:
```yaml
volatility:
  method: 'garch'
  halflife: 60  # 回退到 EWMA 时使用
```

### 4.2 Triple-Barrier 参数调优

#### 保守配置（适合低频交易）

```yaml
triple_barrier:
  pt_multiplier: 3.0  # 更宽的止盈屏障
  sl_multiplier: 3.0  # 更宽的止损屏障
  max_holding: 300    # 更长的持有期
  min_return: 0.0002  # 更高的最小收益阈值
```

**特点**:
- 交易信号少，质量高
- 持平标签多（~50-60%）
- 避免过度交易

#### 激进配置（适合高频交易）

```yaml
triple_barrier:
  pt_multiplier: 1.5  # 更窄的止盈屏障
  sl_multiplier: 1.5  # 更窄的止损屏障
  max_holding: 100    # 更短的持有期
  min_return: 0.00005  # 更低的最小收益阈值
```

**特点**:
- 交易信号多
- 持平标签少（~20-30%）
- 捕捉小幅价格波动

#### 平衡配置（推荐，默认）

```yaml
triple_barrier:
  pt_multiplier: 2.0
  sl_multiplier: 2.0
  max_holding: 200
  min_return: 0.0001
```

**特点**:
- 交易频率适中
- 标签分布较均衡（上涨 ~35%，持平 ~30%，下跌 ~35%）

### 4.3 样本权重配置

#### 禁用样本权重（简化训练）

```yaml
sample_weights:
  enabled: false
  tau: 100.0
  return_scaling: 10.0
  balance_classes: true
```

**使用场景**:
- 快速测试
- 标签分布已经平衡
- 不关心收益大小，只关心方向

#### 启用样本权重（推荐）

```yaml
sample_weights:
  enabled: true
  tau: 100.0           # 时间衰减参数（50-200）
  return_scaling: 10.0  # 收益缩放系数（5-20）
  balance_classes: true # 类别平衡
```

**参数说明**:
- `tau` **越小** → 快速退出的样本权重**越高**
- `return_scaling` **越大** → 高收益样本权重**越高**
- `balance_classes=true` → 自动平衡类别权重

---

## 5. 常见问题

### Q1: 运行时报错 "不支援的波动率方法"

**错误信息**:
```
ValueError: 不支援的波动率方法: xyz，请使用 'ewma' 或 'garch'
```

**解决方法**:
检查配置文件 `config_pro_v5.yaml`，确保 `volatility.method` 只能是 `'ewma'` 或 `'garch'`：

```yaml
volatility:
  method: 'ewma'  # 只能是 'ewma' 或 'garch'
```

---

### Q2: Triple-Barrier 标签生成失败

**错误信息**:
```
ERROR - Triple-Barrier 失败: ...
```

**可能原因**:
1. 股票数据点太少（< 150 个点）
2. `triple-barrier` 库未正确安装
3. 波动率序列包含 NaN 或 Inf

**解决方法**:

**方法 1: 检查数据点数量**
```python
# 查看每档股票的数据点数
import glob
import pandas as pd

for file in glob.glob("data/temp/*.txt"):
    df = pd.read_csv(file, sep="||", header=None, engine='python')
    print(f"{file}: {len(df)} 行")
```

**方法 2: 重新安装 triple-barrier**
```bash
pip uninstall triple-barrier
pip install triple-barrier
```

**方法 3: 检查波动率配置**
```yaml
# 确保波动率方法正确
volatility:
  method: 'ewma'  # 推荐使用 EWMA（稳定性高）
  halflife: 60
```

---

### Q3: 输出的标签分布极度不平衡

**现象**:
```
标签分布: 上涨=5,000, 持平=90,000, 下跌=5,000
```

**原因**:
- `max_holding` 太长，导致大量样本触发时间屏障
- `pt_mult / sl_mult` 太大，导致很少触发止盈/止损

**解决方法**:

**调整 Triple-Barrier 参数**:
```yaml
triple_barrier:
  pt_multiplier: 1.5  # 从 2.0 降低到 1.5
  sl_multiplier: 1.5
  max_holding: 100    # 从 200 降低到 100
  min_return: 0.0002  # 从 0.0001 提高到 0.0002
```

**预期效果**:
- 上涨/下跌标签增加（触发止盈/止损更容易）
- 持平标签减少

---

### Q4: 内存不足（OOM）

**错误信息**:
```
MemoryError: Unable to allocate ... for an array with shape ...
```

**解决方法**:

**方法 1: 降低数据精度**
修改脚本，将 `np.float64` 改为 `np.float32`（内存减半）

**方法 2: 分批处理股票**
修改脚本，每次只处理部分股票（例如 50 档）

**方法 3: 增加系统内存**
- 关闭其他程序
- 使用服务器（64GB+ 内存）

---

### Q5: 处理速度太慢

**现象**:
- 195 档股票处理超过 2 小时

**可能原因**:
1. 使用 GARCH 波动率（计算慢）
2. 数据量过大
3. 硬盘速度慢（HDD）

**解决方法**:

**方法 1: 使用 EWMA 波动率**
```yaml
volatility:
  method: 'ewma'  # 速度快 10 倍
```

**方法 2: 使用 SSD**
- 将数据存储在 SSD 上
- 读写速度提升 5-10 倍

**方法 3: 并行处理（需修改脚本）**
```python
# 示例：使用 multiprocessing 并行处理股票
from multiprocessing import Pool

with Pool(processes=8) as pool:
    results = pool.map(process_stock, stock_list)
```

---

## 6. 最佳实践

### 6.1 数据组织

**推荐目录结构**:
```
project/
├── data/
│   ├── raw/                    # 原始数据（只读）
│   │   └── temp/
│   │       ├── 20240101_TW_LOB.txt
│   │       └── ...
│   └── processed/              # 处理后数据
│       ├── v4/                 # V4 数据（保留）
│       └── v5/                 # V5 数据（新）
├── configs/
│   ├── config_pro_v5.yaml      # 默认配置
│   ├── config_conservative.yaml # 保守配置
│   └── config_aggressive.yaml   # 激进配置
├── scripts/
│   └── extract_tw_stock_data_v5.py
└── docs/
    ├── V5_Technical_Specification.md
    └── V5_User_Guide.md
```

### 6.2 配置管理

**为不同实验创建配置文件**:
```bash
# 实验 1: 保守配置（低频交易）
cp configs/config_pro_v5.yaml configs/exp1_conservative.yaml
# 编辑 exp1_conservative.yaml...

# 实验 2: 激进配置（高频交易）
cp configs/config_pro_v5.yaml configs/exp2_aggressive.yaml
# 编辑 exp2_aggressive.yaml...

# 运行实验
python scripts/extract_tw_stock_data_v5.py \
    --config configs/exp1_conservative.yaml \
    --output-dir data/processed/exp1

python scripts/extract_tw_stock_data_v5.py \
    --config configs/exp2_aggressive.yaml \
    --output-dir data/processed/exp2
```

### 6.3 版本控制

**记录关键信息**:
```bash
# 创建实验日志
cat > data/processed/v5/EXPERIMENT_LOG.md << EOF
# 实验记录

## 实验 1: V5 默认配置
- 日期: 2025-10-18
- 配置: config_pro_v5.yaml
- 数据: 2024-01-01 ~ 2024-12-31 (195 档股票)
- 波动率: EWMA (halflife=60)
- Triple-Barrier: PT=2.0σ, SL=2.0σ, MaxHold=200
- 样本权重: 启用 (tau=100, scaling=10)
- 输出目录: data/processed/v5/

## 结果
- 总样本: 750,000
- 标签分布: 上涨=260,000 (34.7%), 持平=230,000 (30.7%), 下跌=260,000 (34.7%)
- 权重统计: mean=1.0, std=0.8, max=5.2
EOF
```

### 6.4 数据验证

**验证输出数据**:
```python
import numpy as np
import json

# 加载数据
data = np.load("data/processed/v5/npz/stock_embedding_train.npz")
X = data['X']
y = data['y']
w = data['weights']

# 基本检查
assert X.shape == (len(y), 100, 20), "形状错误"
assert np.all((y >= 0) & (y <= 2)), "标签超出范围"
assert np.all(w > 0), "权重包含非正值"
assert np.allclose(w.mean(), 1.0, atol=0.01), "权重未正确归一化"

# 加载 metadata
with open("data/processed/v5/npz/normalization_meta.json") as f:
    meta = json.load(f)

# 检查配置一致性
assert meta['version'] == "5.0.0"
assert meta['seq_len'] == 100
assert meta['feature_dim'] == 20

print("✅ 数据验证通过")
```

---

## 7. 故障排查

### 7.1 日志分析

**开启调试日志**:
```python
# 修改脚本开头的日志配置
logging.basicConfig(
    level=logging.DEBUG,  # 改为 DEBUG
    format='%(asctime)s - %(levelname)s - %(message)s'
)
```

**关键日志位置**:
```
INFO - 处理日期 20240101，共 5 个档案      # 每日数据处理进度
INFO - 2330.TW: 5,000 个样本 (共 8,000 个点)  # 每档股票样本数
WARNING - 1234.TW: 跳过（只有 80 个点）    # 数据点不足的股票
ERROR - Triple-Barrier 失败: ...          # 标签生成错误
```

### 7.2 常见错误代码

| 错误代码 | 含义 | 解决方法 |
|---------|------|---------|
| `return 1` | 程式执行失败 | 查看日志，定位具体错误 |
| `ValueError` | 参数错误 | 检查配置文件 |
| `FileNotFoundError` | 文件不存在 | 检查输入目录路径 |
| `MemoryError` | 内存不足 | 降低数据精度或分批处理 |

### 7.3 性能诊断

**检查瓶颈**:
```python
import time

# 在关键函数前后添加计时
start = time.time()
tb_df = tb_labels(close, vol, ...)
print(f"Triple-Barrier 耗时: {time.time() - start:.2f} 秒")
```

**优化建议**:
- Triple-Barrier 太慢 → 使用 EWMA 替代 GARCH
- 滑窗抽样太慢 → 检查是否有 Python 循环（应使用 NumPy）
- 数据读取太慢 → 使用 SSD 或压缩格式

---

## 附录 A: 配置文件完整示例

### config_pro_v5.yaml

```yaml
# V5 Pro 專業流水線配置
version: "5.0.0"
description: "使用專業套件的 DeepLOB 資料流水線"

# 波動率設定
volatility:
  method: 'ewma'  # 'ewma' 或 'garch'
  halflife: 60    # EWMA 半衰期（bars）

# Triple-Barrier 參數
triple_barrier:
  pt_multiplier: 2.0   # 止盈倍數（2.0 = 2σ）
  sl_multiplier: 2.0   # 止損倍數
  max_holding: 200     # 最大持有期（bars）
  min_return: 0.0001   # 最小報酬閾值（0.01%）

# 樣本權重
sample_weights:
  enabled: true        # 是否啟用樣本權重
  tau: 100.0           # 時間衰減參數
  return_scaling: 10.0 # 報酬縮放係數
  balance_classes: true # 類別平衡

# 輸出設定
output:
  save_meta: true         # 保存詳細 metadata
  save_weights: true      # 保存樣本權重
  compression: 'compressed'  # NPZ 壓縮

# 資料處理參數（與 V4 相同）
data:
  aggregation_factor: 10
  seq_len: 100
  alpha: 0.002
  horizons: [1, 2, 3, 5, 10]
  trading_start: 90000
  trading_end: 133000

# 資料切分
split:
  train_ratio: 0.7
  val_ratio: 0.15
  test_ratio: 0.15
```

---

## 附录 B: Python API 使用示例

### 加载 V5 数据

```python
import numpy as np
import json

# 加载训练集
train_data = np.load("data/processed/v5/npz/stock_embedding_train.npz")
X_train = train_data['X']       # (N, 100, 20)
y_train = train_data['y']       # (N,)
w_train = train_data['weights'] # (N,)
stock_ids = train_data['stock_ids']  # (N,)

# 加载 metadata
with open("data/processed/v5/npz/normalization_meta.json") as f:
    meta = json.load(f)

# 提取配置
volatility_method = meta['volatility']['method']
pt_mult = meta['triple_barrier']['pt_multiplier']
feature_means = np.array(meta['normalization']['feature_means'])
feature_stds = np.array(meta['normalization']['feature_stds'])

print(f"数据集: {X_train.shape[0]:,} 个样本")
print(f"标签分布: {np.bincount(y_train)}")
print(f"权重范围: [{w_train.min():.3f}, {w_train.max():.3f}]")
```

### 整合到 PyTorch DataLoader

```python
import torch
from torch.utils.data import Dataset, DataLoader

class WeightedLOBDataset(Dataset):
    def __init__(self, npz_path):
        data = np.load(npz_path)
        self.X = torch.from_numpy(data['X'])
        self.y = torch.from_numpy(data['y']).long()
        self.weights = torch.from_numpy(data['weights']).float()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.weights[idx]

# 创建 DataLoader
train_dataset = WeightedLOBDataset("data/processed/v5/npz/stock_embedding_train.npz")
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)

# 训练循环
criterion = torch.nn.CrossEntropyLoss(reduction='none')

for X_batch, y_batch, w_batch in train_loader:
    X_batch = X_batch.to(device)
    y_batch = y_batch.to(device)
    w_batch = w_batch.to(device)

    # 前向传播
    logits = model(X_batch)

    # 加权损失
    loss_vec = criterion(logits, y_batch)
    loss = (loss_vec * w_batch).mean()

    # 反向传播
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

---

## 附录 C: 版本变更记录

### V5.0.0 (2025-10-18)

**新增功能**:
- ✅ 专业波动率估计（EWMA / GARCH）
- ✅ Triple-Barrier 标签生成
- ✅ 样本权重计算
- ✅ 详细 metadata 输出

**移除功能**:
- ❌ V4 回退机制（失败时直接停止）
- ❌ 简单波动率计算

**改进**:
- 🔧 更准确的标签（基于波动率倍数）
- 🔧 更平衡的样本权重（收益 + 时间 + 类别）
- 🔧 更详细的日志输出

---

**文档版本**: 1.0
**最后更新**: 2025-10-18
**维护者**: DeepLOB-Pro Team

**反馈与支持**:
- GitHub Issues: [链接]
- 技术文档: `docs/V5_Technical_Specification.md`
