# Dataset Selection JSON 格式說明

## 概述

`analyze_label_distribution.py` 的輸出 JSON 包含完整的「日期+股票」配對列表，每個配對對應一個 NPZ 檔案，可直接用於訓練數據載入。

---

## JSON 結構

### 完整範例

```json
{
  "description": "平衡方案（高精度，偏差 < 2%）",
  "date_range": "20250901-20250905",
  "dates": [
    "20250901",
    "20250902",
    "20250903",
    "20250904",
    "20250905"
  ],
  "num_dates": 5,
  "symbols": [
    "1101",
    "1102",
    "1216",
    "1301",
    "1303",
    "2002",
    "2301",
    "2317",
    "2330",
    "2454"
  ],
  "num_stocks": 10,
  "total_records": 50,
  "file_list": [
    {
      "date": "20250901",
      "symbol": "1101",
      "file_path": "data/preprocessed_v5/daily/20250901/1101.npz",
      "n_points": 4567,
      "total_labels": 4467,
      "down_count": 1340,
      "neutral_count": 1787,
      "up_count": 1340
    },
    {
      "date": "20250901",
      "symbol": "1102",
      "file_path": "data/preprocessed_v5/daily/20250901/1102.npz",
      "n_points": 3891,
      "total_labels": 3791,
      "down_count": 1137,
      "neutral_count": 1517,
      "up_count": 1137
    },
    {
      "date": "20250902",
      "symbol": "1101",
      "file_path": "data/preprocessed_v5/daily/20250902/1101.npz",
      "n_points": 4321,
      "total_labels": 4221,
      "down_count": 1266,
      "neutral_count": 1689,
      "up_count": 1266
    }
  ],
  "distribution": {
    "total_stocks": 10,
    "total_samples": 234567,
    "down_count": 71453,
    "neutral_count": 92012,
    "up_count": 71102,
    "down_pct": 0.3045,
    "neutral_pct": 0.3923,
    "up_pct": 0.3032
  },
  "deviation": 0.0089,
  "level": 0.01
}
```

---

## 欄位說明

### 1. 方案描述

| 欄位 | 類型 | 說明 | 範例 |
|------|------|------|------|
| `description` | string | 方案描述 | "平衡方案（高精度，偏差 < 2%）" |
| `deviation` | float | 偏差度（L2距離） | 0.0089 |
| `level` | float | 偏差閾值 | 0.01 |

### 2. 日期資訊

| 欄位 | 類型 | 說明 | 範例 |
|------|------|------|------|
| `date_range` | string | 日期範圍 | "20250901-20250905" |
| `dates` | array[string] | 日期列表（不重複） | ["20250901", "20250902", ...] |
| `num_dates` | int | 天數 | 5 |

### 3. 個股資訊

| 欄位 | 類型 | 說明 | 範例 |
|------|------|------|------|
| `symbols` | array[string] | 個股代碼列表（不重複） | ["1101", "1102", ...] |
| `num_stocks` | int | 個股數量 | 10 |

### 4. 檔案配對列表（核心）⭐

| 欄位 | 類型 | 說明 | 範例 |
|------|------|------|------|
| `file_list` | array[object] | **完整的日期+股票配對列表** | 見下表 |
| `total_records` | int | 總配對數（日期×股票） | 50 |

#### `file_list` 中的每個物件：

| 欄位 | 類型 | 說明 | 範例 |
|------|------|------|------|
| `date` | string | 日期 | "20250901" |
| `symbol` | string | 股票代碼 | "1101" |
| `file_path` | string | NPZ 檔案路徑 | "data/preprocessed_v5/daily/20250901/1101.npz" |
| `n_points` | int | 數據點數 | 4567 |
| `total_labels` | int | 標籤總數 | 4467 |
| `down_count` | int | Down 標籤數 | 1340 |
| `neutral_count` | int | Neutral 標籤數 | 1787 |
| `up_count` | int | Up 標籤數 | 1340 |

### 5. 標籤分布統計

| 欄位 | 類型 | 說明 | 範例 |
|------|------|------|------|
| `distribution` | object | 整體標籤分布統計 | 見下表 |

#### `distribution` 物件：

| 欄位 | 類型 | 說明 | 範例 |
|------|------|------|------|
| `total_stocks` | int | 去重後的個股數 | 10 |
| `total_samples` | int | 總樣本數（所有配對加總） | 234567 |
| `down_count` | int | Down 標籤總數 | 71453 |
| `neutral_count` | int | Neutral 標籤總數 | 92012 |
| `up_count` | int | Up 標籤總數 | 71102 |
| `down_pct` | float | Down 標籤比例 | 0.3045 (30.45%) |
| `neutral_pct` | float | Neutral 標籤比例 | 0.3923 (39.23%) |
| `up_pct` | float | Up 標籤比例 | 0.3032 (30.32%) |

---

## 使用範例

### 1. 載入 JSON 並提取檔案列表

```python
import json
import numpy as np
from pathlib import Path

# 載入選取結果
with open('dataset_selection.json', 'r', encoding='utf-8') as f:
    selection = json.load(f)

# 提取檔案配對列表
file_list = selection['file_list']

print(f"共 {len(file_list)} 個檔案配對")
print(f"日期範圍: {selection['date_range']}")
print(f"個股數量: {selection['num_stocks']} 檔")
print(f"總樣本數: {selection['distribution']['total_samples']:,}")

# 顯示前 5 個配對
for i, item in enumerate(file_list[:5], 1):
    print(f"{i}. {item['date']}-{item['symbol']} → {item['total_labels']} 樣本")
```

**輸出**:
```
共 50 個檔案配對
日期範圍: 20250901-20250905
個股數量: 10 檔
總樣本數: 234,567

1. 20250901-1101 → 4467 樣本
2. 20250901-1102 → 3791 樣本
3. 20250902-1101 → 4221 樣本
4. 20250902-1102 → 3654 樣本
5. 20250903-1101 → 4089 樣本
```

---

### 2. 載入所有 NPZ 檔案

```python
import json
import numpy as np
from pathlib import Path
from typing import List, Dict

def load_dataset_from_json(json_path: str) -> List[np.lib.npyio.NpzFile]:
    """
    從 JSON 載入選取的數據集

    Args:
        json_path: dataset_selection.json 路徑

    Returns:
        載入的 NPZ 檔案列表
    """
    # 載入 JSON
    with open(json_path, 'r', encoding='utf-8') as f:
        selection = json.load(f)

    # 載入所有 NPZ 檔案
    data_list = []
    file_list = selection['file_list']

    for item in file_list:
        file_path = item['file_path']

        # 檢查檔案存在
        if not Path(file_path).exists():
            print(f"⚠️  檔案不存在: {file_path}")
            continue

        # 載入 NPZ
        try:
            data = np.load(file_path, allow_pickle=True)
            data_list.append(data)
            print(f"✅ {item['date']}-{item['symbol']}: {item['total_labels']} 樣本")
        except Exception as e:
            print(f"❌ 載入失敗 {file_path}: {e}")

    print(f"\n✅ 成功載入 {len(data_list)}/{len(file_list)} 個檔案")
    return data_list


# 使用範例
dataset = load_dataset_from_json('dataset_selection.json')
```

---

### 3. 按日期分組載入

```python
import json
from collections import defaultdict
from pathlib import Path
import numpy as np

def load_dataset_by_date(json_path: str) -> Dict[str, List[np.ndarray]]:
    """
    按日期分組載入數據集

    Returns:
        {date: [npz_data1, npz_data2, ...]}
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        selection = json.load(f)

    # 按日期分組
    grouped_data = defaultdict(list)

    for item in selection['file_list']:
        date = item['date']
        file_path = item['file_path']

        if Path(file_path).exists():
            data = np.load(file_path, allow_pickle=True)
            grouped_data[date].append(data)

    # 顯示統計
    for date, data_list in sorted(grouped_data.items()):
        print(f"{date}: {len(data_list)} 檔")

    return dict(grouped_data)


# 使用範例
dataset_by_date = load_dataset_by_date('dataset_selection.json')

# 訓練時可按日期切分 train/val/test
dates = sorted(dataset_by_date.keys())
train_dates = dates[:int(len(dates) * 0.7)]
val_dates = dates[int(len(dates) * 0.7):int(len(dates) * 0.85)]
test_dates = dates[int(len(dates) * 0.85):]

print(f"\nTrain dates: {train_dates}")
print(f"Val dates: {val_dates}")
print(f"Test dates: {test_dates}")
```

---

### 4. 按個股分組載入

```python
import json
from collections import defaultdict
from pathlib import Path
import numpy as np

def load_dataset_by_symbol(json_path: str) -> Dict[str, List[np.ndarray]]:
    """
    按個股分組載入數據集

    Returns:
        {symbol: [npz_data1, npz_data2, ...]}
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        selection = json.load(f)

    # 按個股分組
    grouped_data = defaultdict(list)

    for item in selection['file_list']:
        symbol = item['symbol']
        file_path = item['file_path']

        if Path(file_path).exists():
            data = np.load(file_path, allow_pickle=True)
            grouped_data[symbol].append(data)

    # 顯示統計
    for symbol, data_list in sorted(grouped_data.items()):
        total_samples = sum(d['metadata'].item()['label_preview']['total_labels']
                          for d in data_list if 'metadata' in d)
        print(f"{symbol}: {len(data_list)} 天, {total_samples:,} 樣本")

    return dict(grouped_data)


# 使用範例
dataset_by_symbol = load_dataset_by_symbol('dataset_selection.json')
```

---

### 5. 統計標籤分布（驗證）

```python
import json
import numpy as np
from pathlib import Path

def verify_label_distribution(json_path: str):
    """驗證 JSON 中的標籤統計是否正確"""
    with open(json_path, 'r', encoding='utf-8') as f:
        selection = json.load(f)

    # 從 file_list 重新計算
    total_down = 0
    total_neutral = 0
    total_up = 0

    for item in selection['file_list']:
        total_down += item['down_count']
        total_neutral += item['neutral_count']
        total_up += item['up_count']

    total_all = total_down + total_neutral + total_up

    # 與 distribution 欄位比對
    dist = selection['distribution']

    print("【驗證標籤統計】")
    print(f"Down:    {total_down:>10,} (JSON: {dist['down_count']:>10,}) {'✅' if total_down == dist['down_count'] else '❌'}")
    print(f"Neutral: {total_neutral:>10,} (JSON: {dist['neutral_count']:>10,}) {'✅' if total_neutral == dist['neutral_count'] else '❌'}")
    print(f"Up:      {total_up:>10,} (JSON: {dist['up_count']:>10,}) {'✅' if total_up == dist['up_count'] else '❌'}")
    print(f"Total:   {total_all:>10,} (JSON: {dist['total_samples']:>10,}) {'✅' if total_all == dist['total_samples'] else '❌'}")

    # 計算比例
    print(f"\n【標籤比例】")
    print(f"Down:    {total_down / total_all:.2%}")
    print(f"Neutral: {total_neutral / total_all:.2%}")
    print(f"Up:      {total_up / total_all:.2%}")


# 使用範例
verify_label_distribution('dataset_selection.json')
```

---

## 與 NPZ 檔案的對應關係

### NPZ 檔案結構

每個 NPZ 檔案（`file_path` 欄位指向的檔案）包含：

```python
data = np.load('data/preprocessed_v5/daily/20250901/1101.npz', allow_pickle=True)

# 欄位：
# - 'lob': LOB 數據 (shape: [n_points, 20])
# - 'mid': 中間價 (shape: [n_points])
# - 'volatility': 波動率 (shape: [n_points])
# - 'labels': 標籤 (shape: [n_labels], 通常 n_labels ≈ n_points - lookforward)
# - 'metadata': 元數據（包含 label_preview）
```

### 對應關係

| JSON 欄位 | NPZ 欄位 | 說明 |
|-----------|----------|------|
| `file_path` | - | NPZ 檔案路徑 |
| `n_points` | `len(data['lob'])` | 原始數據點數 |
| `total_labels` | `len(data['labels'])` | 標籤數量 |
| `down_count` | `(data['labels'] == -1).sum()` | Down 標籤數 |
| `neutral_count` | `(data['labels'] == 0).sum()` | Neutral 標籤數 |
| `up_count` | `(data['labels'] == 1).sum()` | Up 標籤數 |

---

## 常見問題

### Q1: 為什麼 `total_records` ≠ `num_dates` × `num_stocks`？

**原因**: 不是每個日期都有所有股票的數據（可能某些股票當天停牌或未通過過濾）。

**範例**:
```
dates = [20250901, 20250902, 20250903]  # 3 天
symbols = [1101, 1102, 2330, 2454]      # 4 檔

理論配對數 = 3 × 4 = 12
實際配對數 = 10（可能某些配對不存在）
```

---

### Q2: 如何快速載入大量 NPZ 檔案？

**建議**: 使用多進程並行載入

```python
from multiprocessing import Pool
import numpy as np

def load_npz(file_path: str):
    try:
        return np.load(file_path, allow_pickle=True)
    except:
        return None

# 並行載入
with Pool(processes=8) as pool:
    file_paths = [item['file_path'] for item in selection['file_list']]
    data_list = pool.map(load_npz, file_paths)
    data_list = [d for d in data_list if d is not None]
```

---

### Q3: JSON 太大怎麼辦？

如果 `file_list` 太大（如 > 1000 個配對），可以：

1. **壓縮 JSON**
   ```bash
   gzip dataset_selection.json
   ```

2. **只保存檔案路徑**（不包含統計資訊）
   ```python
   # 簡化版 JSON（只有路徑）
   simplified = {
       'file_paths': [item['file_path'] for item in selection['file_list']]
   }
   ```

3. **分批載入**
   ```python
   # 分批處理 file_list
   batch_size = 100
   for i in range(0, len(file_list), batch_size):
       batch = file_list[i:i+batch_size]
       # 處理 batch...
   ```

---

**版本**: v2.0
**更新**: 2025-10-23
**相關文檔**:
- [ANALYZE_LABEL_DISTRIBUTION_V2_GUIDE.md](ANALYZE_LABEL_DISTRIBUTION_V2_GUIDE.md)
- [PREPROCESSED_DATA_SPECIFICATION.md](PREPROCESSED_DATA_SPECIFICATION.md)
