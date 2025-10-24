# extract_tw_stock_data_v7.py 功能與流程說明

**版本**: v7.0.0-simplified
**更新日期**: 2025-10-24
**腳本位置**: `scripts/extract_tw_stock_data_v7.py`

---

## 一、核心理念

> **"預處理已完成，V7 只做數據組織"**

V7 是一個**輕量級數據組織工具**，專注於將預處理完成的數據轉換為訓練格式，不重複計算任何特徵或標籤。

---

## 二、主要功能

### 2.1 功能概述

`extract_tw_stock_data_v7.py` 負責將預處理好的 NPZ 檔案轉換為 DeepLOB 訓練格式：

1. ✅ **讀取預處理數據** - 從 `preprocessed_v5/daily/` 載入 NPZ 檔案
2. ✅ **數據選擇與過濾** - 支持 JSON 檔案或配置過濾
3. ✅ **滑動窗口生成** - 創建 100 timesteps 的時間序列樣本
4. ✅ **數據集劃分** - 按股票分割 train/val/test (70/15/15)
5. ✅ **權重分配** - 從預處理 metadata 提取樣本權重
6. ✅ **標籤轉換** - 將 {-1, 0, 1} 轉換為 {0, 1, 2}
7. ✅ **輸出 NPZ** - 保存訓練就緒的數據集

### 2.2 V7 簡化設計

**不做的事情**（已在預處理完成）：
- ❌ 不重新計算標籤（直接使用 NPZ 的 `labels` 字段）
- ❌ 不重新計算波動率（不需要）
- ❌ 不重新計算權重（從 `metadata` 讀取）
- ❌ 不重新標準化（features 已是原始價格）

**只做的事情**：
- ✅ 數據選擇與過濾
- ✅ 滑動窗口生成
- ✅ 數據集劃分
- ✅ 格式轉換

---

## 三、完整流程

### 階段 0: 環境準備

```bash
# 確保已完成預處理
scripts\batch_preprocess.bat

# 檢查預處理結果
dir data\preprocessed_v5\daily\20250901\*.npz
```

### 階段 1: 載入預處理數據

#### 1.1 掃描 NPZ 檔案

```python
# 掃描 data/preprocessed_v5/daily/YYYYMMDD/*.npz
npz_files = glob.glob("data/preprocessed_v5/daily/*/*.npz")
```

#### 1.2 載入並驗證

每個 NPZ 檔案包含：
- `features` - (T, 20) 原始 LOB 特徵（未標準化）
- `labels` - (T,) 標籤 {-1, 0, 1}
- `mids` - (T,) 中間價（用於驗證）
- `metadata` - JSON 格式的元數據

**數據質量驗證**：
- ✅ `mids` 不能為 0（預處理應已移除）
- ✅ `mids` 不能為 NaN
- ✅ `mids` 不能為負數
- ✅ `features` 不能含 NaN
- ✅ `features` 和 `labels` 長度必須匹配

#### 1.3 版本檢查

V7 要求 NPZ v2.0+（必須包含 `labels` 字段）：

```python
if 'labels' not in data:
    logging.error("❌ NPZ 版本過舊（v1.0）")
    return None
```

### 階段 2: 數據選擇與過濾

支持兩種過濾模式（按優先級）：

#### 2.1 優先級 1: 使用 `dataset_selection.json`

**生成 JSON**：
```bash
python scripts/analyze_label_distribution.py \
    --preprocessed-dir data/preprocessed_v5 \
    --mode smart_recommend \
    --target-dist "0.30,0.40,0.30" \
    --output results/dataset_selection_auto.json
```

**JSON 格式**：
```json
{
  "metadata": {
    "total_samples": 500000,
    "total_files": 50
  },
  "file_list": [
    {"date": "20250901", "symbol": "2330", "samples": 10000},
    {"date": "20250901", "symbol": "2317", "samples": 8500}
  ]
}
```

**使用 JSON**：
```bash
# 方式 1: 命令行參數（最高優先級）
python scripts/extract_tw_stock_data_v7.py \
    --json results/dataset_selection_auto.json

# 方式 2: 配置文件
# config.yaml:
# data_selection:
#   json_file: "results/dataset_selection_auto.json"
```

#### 2.2 優先級 2: 使用配置過濾

```yaml
# configs/config_pro_v7_optimal.yaml
data_selection:
  start_date: "20250901"      # 起始日期
  end_date: "20251015"        # 結束日期（可選）
  num_days: 30                # 限制天數（可選）
  symbols: ["2330", "2317"]   # 指定股票（可選）
  sample_ratio: 0.8           # 隨機採樣 80%（可選）
  random_seed: 42
```

### 階段 3: 按股票分組

```python
# 按股票代碼分組
stock_data = defaultdict(list)
for date, symbol, features, labels, meta in preprocessed_data:
    stock_data[symbol].append((date, features, labels))

# 同時保存 metadata（用於權重提取）
stock_metadata[symbol] = meta
```

### 階段 4: 數據集劃分

按股票代碼劃分 train/val/test（70/15/15）：

```python
symbols = sorted(stock_data.keys())
n_symbols = len(symbols)

n_train = int(n_symbols * 0.70)
n_val = int(n_symbols * 0.15)

train_symbols = set(symbols[:n_train])
val_symbols = set(symbols[n_train:n_train + n_val])
test_symbols = set(symbols[n_train + n_val:])
```

**劃分原則**：
- ✅ 按股票劃分（避免數據洩漏）
- ✅ 同一股票的所有天數在同一集合
- ✅ 確保測試集評估的泛化能力

### 階段 5: 提取權重策略

從預處理 metadata 提取權重策略：

```python
# 從配置獲取權重策略名稱
weight_strategy_name = config['sample_weights']['strategy']  # 例如: "balanced"

# 從 metadata 提取權重
meta = stock_metadata[symbol]
weight_strategies = meta['weight_strategies']

if weight_strategy_name in weight_strategies:
    class_weights = weight_strategies[weight_strategy_name]['class_weights']
    weight_down = class_weights['-1']      # 例如: 1.5
    weight_neutral = class_weights['0']    # 例如: 0.8
    weight_up = class_weights['1']         # 例如: 1.2
else:
    # 默認無權重
    weight_down = weight_neutral = weight_up = 1.0
```

**預處理提供的權重策略**：
1. `uniform` - 均勻權重 (1.0, 1.0, 1.0)
2. `balanced` - sklearn 平衡權重
3. `balanced_sqrt` - 平衡權重的平方根
4. `inverse_freq` - 逆頻率權重
5. `focal_alpha` - Focal Loss 風格權重

### 階段 6: 滑動窗口生成

#### 6.1 合併時間序列

```python
# 合併該股票所有天的數據
all_features = []
all_labels = []

for date, features, labels in sorted(stock_data[symbol], key=lambda x: x[0]):
    all_features.append(features)
    all_labels.append(labels)

concat_features = np.vstack(all_features)  # (T_total, 20)
concat_labels = np.hstack(all_labels)      # (T_total,)
```

#### 6.2 生成窗口

```python
SEQ_LEN = 100

for i in range(T - SEQ_LEN):
    # 提取 100 timesteps 窗口
    X_window = concat_features[i:i+SEQ_LEN]  # (100, 20)

    # 標籤取最後一個時間步
    y_label = concat_labels[i+SEQ_LEN-1]
```

#### 6.3 標籤轉換與權重分配

```python
# 標籤轉換 {-1, 0, 1} → {0, 1, 2}
if y_label == -1:
    y_label = 0                    # 下跌 → 0
    sample_weight = weight_down    # 例如: 1.5
elif y_label == 0:
    y_label = 1                    # 持平 → 1
    sample_weight = weight_neutral # 例如: 0.8
elif y_label == 1:
    y_label = 2                    # 上漲 → 2
    sample_weight = weight_up      # 例如: 1.2
```

**為什麼要轉換？**
- DeepLOB 模型要求類別標籤從 0 開始
- PyTorch 的 `CrossEntropyLoss` 要求標籤為 `[0, num_classes-1]`

#### 6.4 分配到對應集合

```python
if symbol in train_symbols:
    train_X.append(X_window)
    train_y.append(y_label)
    train_weights.append(sample_weight)
    train_stock_ids.append(symbol)
elif symbol in val_symbols:
    val_X.append(X_window)
    # ...
else:  # test_symbols
    test_X.append(X_window)
    # ...
```

### 階段 7: 轉換為 NumPy 陣列

```python
train_X = np.array(train_X, dtype=np.float32)     # (N_train, 100, 20)
train_y = np.array(train_y, dtype=np.int32)       # (N_train,)
train_weights = np.array(train_weights, dtype=np.float32)  # (N_train,)
train_stock_ids = np.array(train_stock_ids, dtype='<U10')  # (N_train,)
```

### 階段 8: 保存 NPZ 檔案

#### 8.1 保存三個數據集

```python
# 輸出路徑
out_dir = "data/processed_v7/npz"

# 保存 train/val/test
np.savez_compressed(
    'stock_embedding_train.npz',
    X=train_X,           # (N, 100, 20) - 特徵窗口
    y=train_y,           # (N,) - 標籤 {0, 1, 2}
    weights=train_weights,  # (N,) - 樣本權重
    stock_ids=train_stock_ids  # (N,) - 股票代碼
)
```

**NPZ 檔案結構**：
```
stock_embedding_train.npz:
  - X: (10293665, 100, 20) float32
  - y: (10293665,) int32
  - weights: (10293665,) float32
  - stock_ids: (10293665,) <U10
```

#### 8.2 聚合權重策略

```python
# 收集所有股票的權重策略
all_strategy_names = set()
for meta in stock_metadata.values():
    all_strategy_names.update(meta['weight_strategies'].keys())

# 對每個策略計算平均權重
for strategy_name in all_strategy_names:
    weights_by_class = {'-1': [], '0': [], '1': []}

    for meta in stock_metadata.values():
        if strategy_name in meta['weight_strategies']:
            cw = meta['weight_strategies'][strategy_name]['class_weights']
            weights_by_class['-1'].append(cw['-1'])
            weights_by_class['0'].append(cw['0'])
            weights_by_class['1'].append(cw['1'])

    # 平均權重
    avg_weights = {
        '-1': np.mean(weights_by_class['-1']),
        '0': np.mean(weights_by_class['0']),
        '1': np.mean(weights_by_class['1'])
    }
```

#### 8.3 保存 Metadata

```json
{
  "format": "deeplob_v7_simplified",
  "version": "7.0.0-simplified",
  "creation_date": "2025-10-24T10:30:00",
  "normalization": {
    "method": "none",
    "note": "V7 數據來自預處理 NPZ，features 為原始價格（未標準化）",
    "feature_means": [0.0, 0.0, ..., 0.0],  // 20 個 0
    "feature_stds": [1.0, 1.0, ..., 1.0]    // 20 個 1
  },
  "data_source": {
    "preprocessed_files": 195,
    "symbols_count": 195
  },
  "data_split": {
    "method": "by_symbol",
    "train": {
      "symbols": 136,
      "samples": 10293665,
      "label_dist": [...]
    },
    "val": {...},
    "test": {...}
  },
  "label_source": {
    "method": "preprocessed",
    "note": "直接使用預處理 NPZ 的 labels 字段，未重新計算"
  },
  "sample_weights": {
    "enabled": true,
    "strategy_used": "balanced",
    "available_strategies": ["uniform", "balanced", "balanced_sqrt", "inverse_freq", "focal_alpha"],
    "note": "權重策略從預處理數據聚合而來（平均值）"
  },
  "weight_strategies": {
    "uniform": {
      "class_weights": {"-1": 1.0, "0": 1.0, "1": 1.0},
      "type": "class_weight",
      "description": "Uniform weights (no rebalancing)",
      "aggregation_method": "mean",
      "n_stocks": 195
    },
    "balanced": {
      "class_weights": {"-1": 1.15, "0": 0.82, "1": 1.23},
      "type": "class_weight",
      "description": "sklearn balanced class weights",
      "aggregation_method": "mean",
      "n_stocks": 195
    },
    // ... 其他策略
  }
}
```

**Metadata 關鍵欄位**：
- `normalization` - 標準化信息（V7 為 identity，因為 features 未標準化）
- `data_split` - 數據集劃分統計
- `label_source` - 標籤來源說明
- `sample_weights` - 權重配置
- `weight_strategies` - 所有可用的權重策略（聚合後）

---

## 四、使用方式

### 4.1 基本用法

```bash
# 確保環境
conda activate deeplob-pro

# 執行 V7
python scripts/extract_tw_stock_data_v7.py \
    --preprocessed-dir ./data/preprocessed_v5 \
    --output-dir ./data/processed_v7 \
    --config ./configs/config_pro_v7_optimal.yaml
```

### 4.2 使用 JSON 選擇數據

```bash
# 步驟 1: 生成 JSON
python scripts/analyze_label_distribution.py \
    --preprocessed-dir data/preprocessed_v5 \
    --mode smart_recommend \
    --start-date 20250901 \
    --target-dist "0.30,0.40,0.30" \
    --min-samples 50000 \
    --output results/dataset_selection_auto.json

# 步驟 2: 使用 JSON 生成訓練數據
python scripts/extract_tw_stock_data_v7.py \
    --preprocessed-dir ./data/preprocessed_v5 \
    --output-dir ./data/processed_v7 \
    --config ./configs/config_pro_v7_optimal.yaml \
    --json results/dataset_selection_auto.json
```

### 4.3 使用配置過濾

```yaml
# configs/config_pro_v7_optimal.yaml
data_selection:
  start_date: "20250901"
  num_days: 30
  symbols: ["2330", "2317", "2454"]
  sample_ratio: 1.0
```

```bash
python scripts/extract_tw_stock_data_v7.py \
    --config ./configs/config_pro_v7_optimal.yaml
```

### 4.4 配置權重策略

```yaml
# configs/config_pro_v7_optimal.yaml
sample_weights:
  enabled: true
  strategy: "balanced"  # 可選: uniform, balanced, balanced_sqrt, inverse_freq, focal_alpha
```

---

## 五、輸出檔案

### 5.1 NPZ 檔案

```
data/processed_v7/npz/
├── stock_embedding_train.npz  # 訓練集
├── stock_embedding_val.npz    # 驗證集
├── stock_embedding_test.npz   # 測試集
└── normalization_meta.json    # 元數據
```

### 5.2 載入訓練數據

```python
# 載入訓練集
data = np.load('data/processed_v7/npz/stock_embedding_train.npz')

X = data['X']           # (N, 100, 20) - 特徵窗口
y = data['y']           # (N,) - 標籤 {0, 1, 2}
weights = data['weights']  # (N,) - 樣本權重
stock_ids = data['stock_ids']  # (N,) - 股票代碼

# 標籤含義
# 0: 下跌 (Down)
# 1: 持平 (Neutral)
# 2: 上漲 (Up)
```

### 5.3 使用樣本權重訓練

```python
import torch
from torch.utils.data import TensorDataset, DataLoader

# 創建 Dataset
X_tensor = torch.FloatTensor(X)
y_tensor = torch.LongTensor(y)
weights_tensor = torch.FloatTensor(weights)

dataset = TensorDataset(X_tensor, y_tensor, weights_tensor)
dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

# 訓練循環
for X_batch, y_batch, w_batch in dataloader:
    # 計算損失
    outputs = model(X_batch)
    loss = criterion(outputs, y_batch)

    # 應用樣本權重
    weighted_loss = (loss * w_batch).mean()

    # 反向傳播
    optimizer.zero_grad()
    weighted_loss.backward()
    optimizer.step()
```

---

## 六、配置文件範例

### 6.1 完整配置範例

```yaml
# configs/config_pro_v7_optimal.yaml
version: "7.0.0"

# 數據選擇
data_selection:
  # 選項 1: 使用 JSON 檔案（優先級最高）
  json_file: "results/dataset_selection_auto.json"

  # 選項 2: 使用配置過濾（JSON 不存在時使用）
  start_date: "20250901"
  end_date: "20251015"
  num_days: null          # null 表示不限制
  symbols: null           # null 表示所有股票
  sample_ratio: 1.0       # 1.0 表示使用全部數據
  random_seed: 42

# 樣本權重
sample_weights:
  enabled: true
  strategy: "balanced"    # 可選: uniform, balanced, balanced_sqrt, inverse_freq, focal_alpha

# 滑動窗口
sliding_window:
  seq_len: 100

# 數據劃分
data_split:
  method: "by_symbol"
  train_ratio: 0.70
  val_ratio: 0.15
  test_ratio: 0.15
```

### 6.2 不同場景配置

#### 場景 1: 精確控制數據（推薦）

```yaml
data_selection:
  json_file: "results/dataset_selection_auto.json"

sample_weights:
  enabled: true
  strategy: "balanced"
```

#### 場景 2: 快速測試

```yaml
data_selection:
  start_date: "20250901"
  num_days: 5
  sample_ratio: 0.1  # 僅使用 10% 數據

sample_weights:
  enabled: false  # 不使用權重
```

#### 場景 3: 特定股票訓練

```yaml
data_selection:
  symbols: ["2330", "2317", "2454"]
  start_date: "20250801"

sample_weights:
  enabled: true
  strategy: "focal_alpha"
```

---

## 七、常見問題

### Q1: V7 和預處理的關係？

**A**: V7 依賴預處理階段 (`preprocess_single_day.py`)：
- 預處理：清洗、標準化、計算標籤、計算權重
- V7：讀取預處理結果，生成滑動窗口，劃分數據集

### Q2: 為什麼標籤要轉換 {-1,0,1} → {0,1,2}？

**A**: DeepLOB 模型要求：
- PyTorch 的 `CrossEntropyLoss` 要求標籤為 `[0, num_classes-1]`
- 負數標籤會導致索引錯誤

### Q3: weights 和 weight_strategies 的區別？

**A**:
- `weights` (NPZ): 每個樣本的實際權重值（float32 陣列）
- `weight_strategies` (metadata): 權重計算方法的說明（JSON）

### Q4: 如何切換權重策略？

**A**: 修改配置檔案：

```yaml
sample_weights:
  strategy: "balanced_sqrt"  # 從 balanced 改為 balanced_sqrt
```

重新執行 V7 即可。

### Q5: 為什麼 normalization 是 identity？

**A**: V7 的 features 來自預處理 NPZ，是**原始價格**（未標準化）。

如需標準化，應在**訓練腳本**中處理：

```python
# 訓練時標準化
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_norm = scaler.fit_transform(X_train.reshape(-1, 20)).reshape(-1, 100, 20)
X_val_norm = scaler.transform(X_val.reshape(-1, 20)).reshape(-1, 100, 20)
```

### Q6: 如何驗證輸出數據？

```bash
# 檢查 NPZ 結構
python -c "import numpy as np; data = np.load('data/processed_v7/npz/stock_embedding_train.npz'); print('Keys:', list(data.keys())); print('Shapes:', {k: data[k].shape for k in data.keys()})"

# 檢查標籤分布
python -c "import numpy as np; data = np.load('data/processed_v7/npz/stock_embedding_train.npz'); y = data['y']; print('Label distribution:', np.bincount(y)); print('Percentages:', np.bincount(y) / len(y) * 100)"

# 檢查權重策略
type data\processed_v7\npz\normalization_meta.json | findstr "weight_strategies"
```

---

## 八、性能統計

### 8.1 執行時間

| 數據量 | 預處理時間 | V7 時間 | 總時間 |
|--------|-----------|---------|--------|
| 50 天 | 15 分鐘 | 1 分鐘 | 16 分鐘 |
| 100 天 | 30 分鐘 | 2 分鐘 | 32 分鐘 |
| 200 天 | 60 分鐘 | 4 分鐘 | 64 分鐘 |

**優勢**：
- ✅ V7 僅佔總時間的 6-7%
- ✅ 調整參數只需重跑 V7（快速迭代）

### 8.2 代碼複雜度

| 指標 | 數值 |
|------|------|
| 總行數 | 978 行 |
| 核心函數 | 5 個 |
| 註釋覆蓋率 | 40% |
| 代碼複雜度 | 低 |

---

## 九、最佳實踐

### 9.1 推薦工作流

```bash
# 1. 預處理（首次執行）
scripts\batch_preprocess.bat

# 2. 生成最佳數據集選擇
python scripts/analyze_label_distribution.py \
    --mode smart_recommend \
    --target-dist "0.30,0.40,0.30" \
    --output results/best_dataset.json

# 3. 生成訓練數據
python scripts/extract_tw_stock_data_v7.py \
    --json results/best_dataset.json \
    --config configs/config_pro_v7_optimal.yaml

# 4. 驗證輸出
python scripts/verify_npz_labels.py --npz-dir data/processed_v7/npz
```

### 9.2 權重策略選擇建議

| 場景 | 推薦策略 | 理由 |
|------|---------|------|
| 標籤平衡 (30/40/30) | `uniform` | 無需重新平衡 |
| 標籤不平衡 (10/70/20) | `balanced` | 強力平衡 |
| 輕微不平衡 (25/50/25) | `balanced_sqrt` | 溫和平衡 |
| 極端不平衡 (5/90/5) | `focal_alpha` | Focal Loss 風格 |
| 自定義需求 | `inverse_freq` | 完全逆頻率 |

### 9.3 數據選擇建議

| 目標 | 方法 | 配置 |
|------|------|------|
| 最佳標籤分布 | JSON | `smart_recommend` |
| 最新數據 | 配置 | `start_date: "20250901"` |
| 快速測試 | 配置 | `num_days: 5, sample_ratio: 0.1` |
| 特定股票 | 配置 | `symbols: ["2330", "2317"]` |

---

## 十、總結

### 10.1 V7 核心價值

1. ✅ **簡單** - 專注數據組織，不重複計算
2. ✅ **快速** - 僅需 2-4 分鐘（相比 V6 的 10 分鐘）
3. ✅ **靈活** - 支持 JSON 精確控制
4. ✅ **可靠** - 不重複計算，避免不一致
5. ✅ **可維護** - 代碼簡潔，邏輯清晰

### 10.2 輸入與輸出

**輸入**：
- 預處理 NPZ：`data/preprocessed_v5/daily/YYYYMMDD/*.npz`
- 配置文件：`configs/config_pro_v7_optimal.yaml`
- 選擇 JSON（可選）：`results/dataset_selection_auto.json`

**輸出**：
- 訓練集：`stock_embedding_train.npz` (X, y, weights, stock_ids)
- 驗證集：`stock_embedding_val.npz`
- 測試集：`stock_embedding_test.npz`
- 元數據：`normalization_meta.json`

### 10.3 關鍵特性

| 特性 | 說明 |
|------|------|
| 標籤轉換 | {-1, 0, 1} → {0, 1, 2} |
| 權重支持 | 5 種預定義策略 |
| 數據選擇 | JSON + 配置雙模式 |
| 股票追蹤 | stock_ids 字段 |
| 質量驗證 | 多重數據檢查 |

---

**文檔版本**: v1.0
**最後更新**: 2025-10-24
**維護者**: DeepLOB-Pro Team
