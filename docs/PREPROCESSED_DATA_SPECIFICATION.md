# 預處理數據規格說明書

**版本**: v2.0
**最後更新**: 2025-10-23
**適用腳本**: `scripts/preprocess_single_day.py`
**目標**: 提供完整的預處理數據格式、內容、用途說明

---

## 📋 目錄

- [概述](#概述)
- [目錄結構](#目錄結構)
- [NPZ 文件格式](#npz-文件格式)
- [預處理流程](#預處理流程)
- [數據讀取範例](#數據讀取範例)
- [常見問題](#常見問題)

---

## 概述

### 什麼是預處理數據？

預處理數據是從原始 LOB (Limit Order Book) TXT 文件處理後的**中間數據**，用於：
1. **快速檢查**：查看每日、每檔股票的標籤分布
2. **知情選擇**：在生成訓練數據前，依標籤分布選擇合適的股票
3. **避免重複處理**：預處理一次，可多次用於不同訓練配置

### 處理階段

```
原始 TXT → preprocess_single_day.py → NPZ (預處理數據) → extract_tw_stock_data_v6.py → 訓練數據
         (清洗、聚合、標籤預覽)                      (時間序列窗口、標準化)
```

---

## 目錄結構

```
data/
└── preprocessed_v5_1hz/              # 預處理數據根目錄
    └── daily/                        # 按日期組織
        ├── 20250901/                 # 單一交易日
        │   ├── 0050.npz              # 元大台灣50
        │   ├── 2330.npz              # 台積電
        │   ├── 2317.npz              # 鴻海
        │   ├── ...                   # 其他股票
        │   └── summary.json          # 當日處理摘要
        ├── 20250902/
        │   └── ...
        └── ...
```

### 目錄說明

| 路徑 | 說明 |
|------|------|
| `preprocessed_v5_1hz` | 預處理根目錄（v5 配置 + 1Hz 聚合） |
| `daily/` | 按日期組織的子目錄 |
| `daily/20250901/` | 單一交易日目錄 |
| `*.npz` | 個股預處理數據（NumPy 壓縮格式） |
| `summary.json` | 當日處理摘要（股票數、標籤統計等） |

---

## NPZ 文件格式

### 文件內容

每個 `.npz` 文件包含以下陣列和元數據：

```python
import numpy as np

data = np.load('data/preprocessed_v5_1hz/daily/20250901/2330.npz', allow_pickle=True)

# 可用的 keys:
data.keys()
# dict_keys(['features', 'mids', 'bucket_event_count', 'bucket_mask', 'metadata', 'labels'])
```

### 1. features (必要)

**類型**: `np.ndarray`, shape `(T, 20)`, dtype `float32`
**說明**: LOB 特徵矩陣（5 檔買賣價量）

| 列索引 | 特徵名稱 | 說明 |
|-------|---------|------|
| 0-4   | ask_price_1 ~ ask_price_5 | 賣方 5 檔價格 |
| 5-9   | ask_volume_1 ~ ask_volume_5 | 賣方 5 檔量 |
| 10-14 | bid_price_1 ~ bid_price_5 | 買方 5 檔價格 |
| 15-19 | bid_volume_1 ~ bid_volume_5 | 買方 5 檔量 |

**特點**:
- ✅ 已清洗異常值（價格/量為 0 或異常跳動）
- ✅ 按時間聚合（1Hz 模式：每秒 1 個快照）
- ❌ 未標準化（原始價格和量）

**範例**:
```python
features = data['features']
print(f"Shape: {features.shape}")  # (15957, 20)
print(f"第 1 筆賣 1 價: {features[0, 0]}")  # 587.0
print(f"第 1 筆買 1 價: {features[0, 10]}")  # 586.0
```

---

### 2. mids (必要)

**類型**: `np.ndarray`, shape `(T,)`, dtype `float64`
**說明**: 中間價時序（買 1 價 + 賣 1 價）/ 2

**公式**:
```python
mid_price = (bid_price_1 + ask_price_1) / 2
```

**用途**:
- 計算標籤（Triple-Barrier）
- 視覺化價格走勢
- 計算報酬率

**範例**:
```python
mids = data['mids']
print(f"Shape: {mids.shape}")  # (15957,)
print(f"價格範圍: {mids.min():.2f} ~ {mids.max():.2f}")
# 價格範圍: 585.50 ~ 592.00
```

---

### 3. bucket_event_count (選用)

**類型**: `np.ndarray`, shape `(T,)`, dtype `int32`
**說明**: 每個時間桶（bucket）內的事件數量

**用途**:
- 檢查數據質量（事件數太少可能不可靠）
- 識別高頻交易時段

**範例**:
```python
bucket_event_count = data['bucket_event_count']
print(f"平均每秒事件數: {bucket_event_count.mean():.2f}")  # 5.3
print(f"最大事件數: {bucket_event_count.max()}")  # 120
```

---

### 4. bucket_mask (選用)

**類型**: `np.ndarray`, shape `(T,)`, dtype `int32`
**說明**: 時間桶遮罩（0=缺失數據, 1=有效數據）

**用途**:
- 標記缺失數據位置
- 過濾無效樣本

**範例**:
```python
bucket_mask = data['bucket_mask']
valid_ratio = bucket_mask.mean()
print(f"有效數據比例: {valid_ratio:.2%}")  # 98.5%
```

---

### 5. metadata (必要)

**類型**: JSON 字串（需解析）
**說明**: 元數據，包含股票資訊、過濾狀態、標籤預覽等

**解析方式**:
```python
import json
metadata_str = str(data['metadata'])
metadata = json.loads(metadata_str)
```

**完整欄位列表**:

| 欄位 | 類型 | 說明 | 範例 |
|------|------|------|------|
| **基本資訊** |
| `symbol` | str | 股票代碼 | `"2330"` |
| `date` | str | 交易日期 | `"20250901"` |
| `n_points` | int | 數據點數 | `15957` |
| **價格統計** |
| `high` | float | 最高價 | `592.0` |
| `low` | float | 最低價 | `585.5` |
| `open` | float | 開盤價 | `587.0` |
| `close` | float | 收盤價 | `590.0` |
| `range_pct` | float | 價格範圍（%） | `1.11` |
| `return_pct` | float | 日報酬率（%） | `0.51` |
| **過濾資訊** |
| `pass_filter` | bool | 是否通過過濾 | `true` |
| `filter_threshold` | float | 過濾閾值 | `0.005` |
| `filter_method` | str | 過濾方法 | `"range_pct"` |
| `filter_reason` | str | 過濾原因 | `"pass"` 或 `"range_pct too low"` |
| **處理參數** |
| `sampling_mode` | str | 聚合模式 | `"time_agg"` |
| `bucket_seconds` | int | 時間桶大小（秒） | `1` |
| `agg_reducer` | str | 聚合方式 | `"last"` |
| `ffill_limit` | int | 前向填充上限 | `10` |
| **數據質量** |
| `raw_events` | int | 原始事件數 | `85432` |
| `aggregated_points` | int | 聚合後點數 | `15957` |
| `ffill_ratio` | float | 填充比例 | `0.015` |
| `missing_ratio` | float | 缺失比例 | `0.012` |
| `multi_event_ratio` | float | 多事件比例 | `0.65` |
| `max_gap_sec` | int | 最大間隔（秒） | `5` |
| `n_seconds` | int | 總秒數 | `16200` |
| **時間戳記** |
| `processed_at` | str | 處理時間 | `"2025-10-23 10:30:45"` |
| **標籤預覽** |
| `label_preview` | dict | 標籤統計 | 見下方 |

**label_preview 結構**:
```json
{
    "total_labels": 15956,
    "down_count": 2273,
    "neutral_count": 11395,
    "up_count": 2288,
    "down_pct": 0.1425,
    "neutral_pct": 0.7142,
    "up_pct": 0.1434
}
```

**weight_strategies 結構** 🆕 (v2.0):
```json
{
    "balanced": {
        "class_weights": {"-1": 1.11, "0": 2.38, "1": 1.11},
        "description": "Standard balanced weights (total / (n_classes * count))",
        "type": "class_weight"
    },
    "sqrt_balanced": {
        "class_weights": {"-1": 1.05, "0": 1.54, "1": 1.05},
        "description": "Square root of balanced weights (gentler, more stable)",
        "type": "class_weight"
    },
    "effective_num_0999": {
        "class_weights": {"-1": 1.02, "0": 1.92, "1": 1.04},
        "description": "Effective Number of Samples (beta=0.999, CVPR 2019)",
        "type": "class_weight"
    },
    "uniform": {
        "class_weights": {"-1": 1.0, "0": 1.0, "1": 1.0},
        "description": "No weighting (all classes equal)",
        "type": "class_weight"
    },
    "focal_loss": {
        "type": "focal",
        "gamma": 2.0,
        "class_weights": {"-1": 1.0, "0": 1.0, "1": 1.0},
        "description": "Use Focal Loss (gamma=2.0) during training"
    }
}
```

**weight_strategies 說明**:
- **用途**: 預先計算多種權重策略，訓練時靈活選擇
- **類型**: 包含 11 種策略（balanced, sqrt_balanced, log_balanced, effective_num_09/099/0999/09999, cb_focal_099/0999, uniform, focal_loss）
- **優點**: 避免訓練時重複計算，支援快速實驗不同權重策略
- **文件增加**: 每個策略約 100 bytes，11 個策略約 1.1 KB/股票（幾乎可忽略）

**使用範例**:
```python
# 載入權重策略
metadata = json.loads(str(data['metadata']))
weight_strategies = metadata.get('weight_strategies', {})

# 選擇策略
strategy_name = 'effective_num_0999'  # 或從配置文件讀取
strategy = weight_strategies[strategy_name]

# 使用權重
class_weights = strategy['class_weights']
print(f"Down weight: {class_weights['-1']}")
print(f"Neutral weight: {class_weights['0']}")
print(f"Up weight: {class_weights['1']}")

# 用於訓練
import torch
weight_tensor = torch.FloatTensor([
    class_weights['-1'],
    class_weights['0'],
    class_weights['1']
])
criterion = nn.CrossEntropyLoss(weight=weight_tensor)
```

**讀取範例**:
```python
# 基本資訊
print(f"股票: {metadata['symbol']}")
print(f"日期: {metadata['date']}")
print(f"數據點數: {metadata['n_points']}")

# 價格統計
print(f"開盤: {metadata['open']}, 收盤: {metadata['close']}")
print(f"日報酬率: {metadata['return_pct']:.2%}")

# 過濾狀態
if metadata['pass_filter']:
    print("✅ 通過過濾")
else:
    print(f"❌ 未通過過濾: {metadata['filter_reason']}")

# 標籤預覽
lp = metadata['label_preview']
print(f"\n標籤分布:")
print(f"  Down: {lp['down_pct']:.2%} ({lp['down_count']})")
print(f"  Neutral: {lp['neutral_pct']:.2%} ({lp['neutral_count']})")
print(f"  Up: {lp['up_pct']:.2%} ({lp['up_count']})")
```

---

### 6. labels (選用，v2.0 新增) 🆕

**類型**: `np.ndarray`, shape `(T,)`, dtype `float32`
**說明**: Triple-Barrier 標籤陣列

**標籤值**:
- `-1`: Down（價格下跌）
- `0`: Neutral（價格持平）
- `1`: Up（價格上漲）
- `NaN`: 未計算（邊界點）

**計算方式**:
- 使用 `compute_label_preview()` 函數
- 基於 Triple-Barrier 方法
- 參數來自 `config_pro_v5_ml_optimal.yaml`

**範例**:
```python
labels = data.get('labels')  # 可能不存在（舊版 NPZ）

if labels is not None:
    # 統計標籤分布
    unique, counts = np.unique(labels[~np.isnan(labels)], return_counts=True)
    for label, count in zip(unique, counts):
        print(f"Label {int(label)}: {count}")

    # Label -1: 2273
    # Label 0: 11395
    # Label 1: 2288
else:
    print("此 NPZ 文件不包含標籤數據（舊版格式）")
```

**注意事項**:
- ⚠️ 只有**通過過濾**的股票才有標籤
- ⚠️ 舊版 NPZ（v1.0）不包含此欄位
- ⚠️ 標籤數量可能比 `T` 少（邊界點被設為 NaN）

---

## 預處理流程

### 完整流程圖

```
原始 TXT 文件 (Ticker_20250901.txt)
    │
    ├─ Step 1: 清洗異常值
    │   ├─ 移除價格/量為 0 的記錄
    │   ├─ 移除異常價格跳動
    │   └─ 移除無效股票代碼
    │
    ├─ Step 2: 時間聚合（1Hz）
    │   ├─ 按股票代碼分組
    │   ├─ 按秒聚合（每秒取最後一筆）
    │   ├─ 前向填充缺失值（最多 10 秒）
    │   └─ 計算中間價
    │
    ├─ Step 3: 動態過濾閾值決策
    │   ├─ 計算各股票波動率 (range_pct)
    │   ├─ 嘗試多個閾值（0.003 ~ 0.010）
    │   ├─ 模擬標籤分布
    │   └─ 選擇最接近目標分布的閾值（30/40/30）
    │
    ├─ Step 4: 計算標籤預覽 🆕
    │   ├─ 對通過過濾的股票計算 Triple-Barrier 標籤
    │   ├─ 統計 down/neutral/up 數量和比例
    │   └─ 保存標籤陣列到 NPZ
    │
    ├─ Step 5: 保存 NPZ
    │   ├─ features, mids, bucket_event_count, bucket_mask
    │   ├─ metadata (JSON)
    │   └─ labels (標籤陣列) 🆕
    │
    └─ Step 6: 生成 summary.json
        ├─ 處理統計（股票數、過濾數）
        ├─ 整體標籤分布
        └─ 個股列表
```

### 關鍵參數（來自 config_pro_v5_ml_optimal.yaml）

#### Triple-Barrier 配置
```yaml
triple_barrier:
  pt_mult: 2.0              # 上邊界倍數
  sl_mult: 2.0              # 下邊界倍數
  max_holding: 200          # 最大持有時間（秒）
  min_return: 0.0001        # 最小報酬率閾值
  ewma_halflife: 60         # EWMA 半衰期（用於波動率）
```

#### 過濾配置
```yaml
filter:
  target_down_pct: 0.30     # 目標 Down 標籤比例
  target_neutral_pct: 0.40  # 目標 Neutral 標籤比例
  target_up_pct: 0.30       # 目標 Up 標籤比例
  threshold_candidates:     # 候選閾值
    - 0.003
    - 0.005
    - 0.007
    - 0.010
```

#### 聚合配置
```yaml
preprocessing:
  sampling_mode: "time_agg" # 時間聚合模式
  bucket_seconds: 1         # 1Hz (每秒 1 個快照)
  agg_reducer: "last"       # 聚合方式（取最後一筆）
  ffill_limit: 10           # 前向填充上限（秒）
```

---

## 數據讀取範例

### 完整讀取範例

```python
import numpy as np
import json
from pathlib import Path

def load_preprocessed_stock(npz_path: str):
    """
    載入預處理股票數據

    Args:
        npz_path: NPZ 文件路徑

    Returns:
        dict: 包含所有數據的字典
    """
    # 載入 NPZ
    data = np.load(npz_path, allow_pickle=True)

    # 解析 metadata
    metadata_str = str(data['metadata'])
    metadata = json.loads(metadata_str)

    return {
        'features': data['features'],
        'mids': data['mids'],
        'bucket_event_count': data.get('bucket_event_count'),
        'bucket_mask': data.get('bucket_mask'),
        'labels': data.get('labels'),  # 可能為 None
        'metadata': metadata
    }

# 使用範例
npz_path = 'data/preprocessed_v5_1hz/daily/20250901/2330.npz'
stock_data = load_preprocessed_stock(npz_path)

# 檢查數據
print(f"股票: {stock_data['metadata']['symbol']}")
print(f"數據點數: {len(stock_data['mids'])}")
print(f"是否有標籤: {'是' if stock_data['labels'] is not None else '否'}")
```

### 批次讀取範例

```python
from pathlib import Path

def scan_daily_directory(daily_dir: str):
    """掃描日期目錄，返回所有股票"""
    daily_path = Path(daily_dir)
    npz_files = list(daily_path.glob("*.npz"))

    stocks = {}
    for npz_file in npz_files:
        symbol = npz_file.stem
        stocks[symbol] = str(npz_file)

    return stocks

# 使用範例
daily_dir = 'data/preprocessed_v5_1hz/daily/20250901'
stocks = scan_daily_directory(daily_dir)

print(f"找到 {len(stocks)} 檔股票:")
for symbol in sorted(stocks.keys())[:10]:
    print(f"  - {symbol}")
```

### 標籤分布分析範例

```python
import numpy as np

def analyze_label_distribution(daily_dir: str):
    """分析整個日期目錄的標籤分布"""
    stocks = scan_daily_directory(daily_dir)

    overall_counts = {-1: 0, 0: 0, 1: 0}
    stocks_with_labels = 0

    for symbol, npz_path in stocks.items():
        data = load_preprocessed_stock(npz_path)
        labels = data.get('labels')

        if labels is not None:
            stocks_with_labels += 1
            # 統計標籤（排除 NaN）
            valid_labels = labels[~np.isnan(labels)]
            unique, counts = np.unique(valid_labels, return_counts=True)

            for label, count in zip(unique, counts):
                overall_counts[int(label)] += count

    # 計算比例
    total = sum(overall_counts.values())
    print(f"總股票數: {len(stocks)}")
    print(f"有標籤股票: {stocks_with_labels}")
    print(f"\n整體標籤分布:")
    print(f"  Down (-1): {overall_counts[-1]:,} ({overall_counts[-1]/total:.2%})")
    print(f"  Neutral (0): {overall_counts[0]:,} ({overall_counts[0]/total:.2%})")
    print(f"  Up (1): {overall_counts[1]:,} ({overall_counts[1]/total:.2%})")

# 使用範例
analyze_label_distribution('data/preprocessed_v5_1hz/daily/20250901')
```

輸出:
```
總股票數: 195
有標籤股票: 187

整體標籤分布:
  Down (-1): 58,341 (30.12%)
  Neutral (0): 77,485 (40.01%)
  Up (1): 57,882 (29.87%)
```

---

## 常見問題

### Q1: 為什麼有些股票沒有標籤？

**A**: 只有**通過過濾**的股票才會計算標籤。未通過過濾的原因包括：
- 波動率太低（`range_pct < filter_threshold`）
- 數據點數太少
- 缺失值過多

檢查方式：
```python
metadata = stock_data['metadata']
if not metadata['pass_filter']:
    print(f"未通過過濾: {metadata['filter_reason']}")
```

---

### Q2: labels 陣列長度為何與 mids 不同？

**A**: `labels` 陣列可能包含 `NaN`（未計算的邊界點）。

處理方式：
```python
labels = stock_data['labels']
mids = stock_data['mids']

# 只使用有效標籤
valid_mask = ~np.isnan(labels)
valid_labels = labels[valid_mask]
valid_mids = mids[valid_mask]

print(f"總數據點: {len(mids)}")
print(f"有效標籤: {len(valid_labels)}")
```

---

### Q3: 如何知道 NPZ 是新版（有標籤）還是舊版？

**A**: 檢查 `labels` 欄位是否存在：

```python
data = np.load(npz_path, allow_pickle=True)
if 'labels' in data.keys():
    print("✅ 新版 NPZ（v2.0，含標籤）")
else:
    print("⚠️ 舊版 NPZ（v1.0，僅標籤預覽）")
```

---

### Q4: label_preview 和 labels 有什麼差別？

| 項目 | label_preview | labels |
|------|--------------|--------|
| **類型** | metadata 中的統計資訊 | NPZ 中的實際陣列 |
| **內容** | 標籤計數和比例 | 逐點標籤值 (-1, 0, 1, NaN) |
| **用途** | 快速檢查分布 | 訓練、視覺化 |
| **大小** | 固定（幾十 bytes） | 與數據等長（幾 KB ~ MB） |
| **版本** | v1.0+ | v2.0+ |

**使用建議**:
- 快速統計 → 使用 `label_preview`
- 詳細分析 → 使用 `labels` 陣列

---

### Q5: 如何選擇合適的股票用於訓練？

**A**: 使用 `analyze_label_distribution.py` 腳本：

```bash
python scripts/analyze_label_distribution.py \
    --preprocessed-dir data/preprocessed_v5_1hz \
    --mode recommend \
    --target-dist 0.30,0.40,0.30
```

會根據標籤分布推薦合適的股票組合。

---

### Q6: 預處理數據可以直接用於訓練嗎？

**A**: ❌ **不行！** 預處理數據還需要：

1. **時間序列窗口化**（例如：100 timesteps）
2. **標準化** (Z-Score)
3. **按股票劃分** train/val/test

這些步驟由 `extract_tw_stock_data_v6.py` 完成。

**完整流程**:
```bash
# Step 1: 預處理（本文檔）
python scripts/preprocess_single_day.py ...

# Step 2: 生成訓練數據
python scripts/extract_tw_stock_data_v6.py \
    --preprocessed-dir data/preprocessed_v5_1hz \
    --output-dir data/processed_v6

# Step 3: 訓練
python scripts/train_deeplob_generic.py ...
```

---

### Q7: summary.json 包含什麼資訊？

**A**: 當日處理摘要，範例：

```json
{
  "date": "20250901",
  "total_symbols": 195,
  "symbols_passed": 187,
  "symbols_filtered": 8,
  "filter_threshold": 0.005,
  "filter_method": "range_pct",
  "overall_label_dist": {
    "down_pct": 0.3012,
    "neutral_pct": 0.4001,
    "up_pct": 0.2987
  },
  "stocks": ["0050", "2330", "2317", ...]
}
```

---

### Q8: 如何視覺化預處理數據？

**A**: 使用 `label_viewer` 工具：

```bash
cd label_viewer
python app_preprocessed.py
```

在瀏覽器打開 http://localhost:8051，可以：
- 查看中間價時序圖
- 查看標籤分布（圓餅圖/柱狀圖）
- 查看標籤疊加在價格圖上
- 查看元數據表格

---

### Q9: 數據佔用多少空間？

**A**: 單檔股票單日數據：

- **無標籤版 (v1.0)**: ~200-500 KB
- **有標籤版 (v2.0)**: ~250-600 KB
- **全部 195 檔單日**: ~50-100 MB
- **全年數據（約 250 交易日）**: ~12-25 GB

---

### Q10: 如何更新舊版 NPZ 到新版（添加標籤）？

**A**: 重新運行預處理腳本：

```bash
# 單一天
python scripts/preprocess_single_day.py \
    --input data/raw/FI2010/Ticker_20250901.txt \
    --output data/preprocessed_v5_1hz \
    --config configs/config_pro_v5_ml_optimal.yaml

# 批次處理
scripts\batch_preprocess.bat
```

預處理腳本會覆蓋舊文件並生成包含標籤的新版 NPZ。

---

## 版本歷史

### v2.0 (2025-10-23) 🆕
- ✅ 新增 `labels` 陣列到 NPZ
- ✅ 修改 `compute_label_preview()` 支持返回完整標籤
- ✅ 更新 `save_preprocessed_npz()` 保存標籤
- ✅ Label Viewer 支持標籤視覺化

### v1.0 (2025-10-15)
- ✅ 基礎預處理流程
- ✅ 動態過濾閾值決策
- ✅ label_preview 統計
- ✅ summary.json 生成

---

## 相關文檔

- **[LABEL_PREVIEW_GUIDE.md](LABEL_PREVIEW_GUIDE.md)** - 標籤預覽完整指南
- **[V6_TWO_STAGE_PIPELINE_GUIDE.md](V6_TWO_STAGE_PIPELINE_GUIDE.md)** - V6 雙階段處理流程
- **[CLAUDE.md](../CLAUDE.md)** - 專案總體說明

---

## 技術支援

如有問題，請參考：
1. 本文檔的[常見問題](#常見問題)章節
2. 運行 `python scripts/preprocess_single_day.py --help`
3. 查看日誌文件 `logs/preprocess_*.log`

**最後更新**: 2025-10-23
**文檔版本**: v2.0
