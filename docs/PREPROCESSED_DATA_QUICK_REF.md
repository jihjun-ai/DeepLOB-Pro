# 預處理數據快速參考卡

**📖 完整文檔**: [PREPROCESSED_DATA_SPECIFICATION.md](PREPROCESSED_DATA_SPECIFICATION.md)

---

## 🗂️ 文件位置

```
data/preprocessed_v5_1hz/daily/20250901/2330.npz
```

---

## 📦 NPZ 內容一覽

| Key | Shape | 類型 | 說明 |
|-----|-------|------|------|
| `features` | (T, 20) | float32 | LOB 5檔買賣價量 |
| `mids` | (T,) | float64 | 中間價時序 |
| `labels` | (T,) | float32 | 標籤 (-1/0/1/NaN) 🆕 |
| `bucket_event_count` | (T,) | int32 | 每秒事件數 |
| `bucket_mask` | (T,) | int32 | 有效性遮罩 |
| `metadata` | - | JSON | 元數據（見下） |

---

## 📊 Features 結構 (20維)

```
[0-4]   ask_price_1~5   賣方 5檔價格
[5-9]   ask_volume_1~5  賣方 5檔量
[10-14] bid_price_1~5   買方 5檔價格
[15-19] bid_volume_1~5  買方 5檔量
```

---

## 📋 Metadata 關鍵欄位

### 基本資訊
- `symbol`: 股票代碼
- `date`: 交易日期
- `n_points`: 數據點數

### 價格統計
- `high`, `low`, `open`, `close`
- `range_pct`: 價格範圍（%）
- `return_pct`: 日報酬率（%）

### 過濾資訊
- `pass_filter`: 是否通過過濾
- `filter_threshold`: 過濾閾值
- `filter_reason`: 過濾原因

### 標籤預覽
```json
"label_preview": {
    "total_labels": 15956,
    "down_count": 2273,
    "neutral_count": 11395,
    "up_count": 2288,
    "down_pct": 0.1425,
    "neutral_pct": 0.7142,
    "up_pct": 0.1434
}
```

---

## 💻 快速讀取代碼

```python
import numpy as np
import json

# 載入
data = np.load('path/to/stock.npz', allow_pickle=True)

# 基本數據
features = data['features']      # (T, 20)
mids = data['mids']              # (T,)
labels = data.get('labels')      # (T,) or None

# 元數據
metadata = json.loads(str(data['metadata']))
symbol = metadata['symbol']
date = metadata['date']
pass_filter = metadata['pass_filter']

# 標籤預覽
if metadata['label_preview']:
    lp = metadata['label_preview']
    print(f"Down: {lp['down_pct']:.2%}")
    print(f"Neutral: {lp['neutral_pct']:.2%}")
    print(f"Up: {lp['up_pct']:.2%}")
```

---

## 🎯 常用操作

### 檢查是否有標籤
```python
if 'labels' in data.keys() and data['labels'] is not None:
    print("✅ 有標籤")
else:
    print("❌ 無標籤（舊版或未通過過濾）")
```

### 統計有效標籤
```python
labels = data['labels']
valid_labels = labels[~np.isnan(labels)]
unique, counts = np.unique(valid_labels, return_counts=True)
```

### 計算中間價報酬率
```python
mids = data['mids']
returns = np.diff(mids) / mids[:-1]
volatility = returns.std()
```

### 檢查數據質量
```python
metadata = json.loads(str(data['metadata']))
print(f"填充比例: {metadata['ffill_ratio']:.2%}")
print(f"缺失比例: {metadata['missing_ratio']:.2%}")
print(f"最大間隔: {metadata['max_gap_sec']} 秒")
```

---

## 🔍 視覺化工具

```bash
cd label_viewer
python app_preprocessed.py
```

開啟瀏覽器：http://localhost:8051

---

## ⚠️ 重要提醒

1. **labels 可能為 None**（舊版 NPZ 或未通過過濾）
2. **labels 包含 NaN**（邊界點未計算）
3. **不可直接用於訓練**（需經過 `extract_tw_stock_data_v6.py`）
4. **只有通過過濾的股票有標籤**

---

## 📞 更多資訊

詳見：[PREPROCESSED_DATA_SPECIFICATION.md](PREPROCESSED_DATA_SPECIFICATION.md)
