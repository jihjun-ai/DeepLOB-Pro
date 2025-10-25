# 數據增強快速修改總結

**日期**: 2025-10-24
**目標**: 在 NPZ 文件中添加價格和成交量（parts[9], parts[10], parts[11]）

---

## 🎯 需要修改的文件

### 1. `scripts/preprocess_single_day.py` (3 處修改)

#### 修改 A: `parse_line()` 函數 (第 252 行後添加)

```python
# 當前代碼（第 251-252 行）:
last_px = to_float(parts[IDX_LASTPRICE], 0.0)
tv = max(0, int(to_float(parts[IDX_TV], 0.0)))

# ⭐ 添加這一行:
last_vol = max(0, int(to_float(parts[IDX_LASTVOL], 0.0)))  # parts[10]

# 修改 rec 字典（第 263-272 行）:
rec = {
    "feat": feat,
    "mid": mid,
    "ref": ref,
    "upper": upper,
    "lower": lower,
    "last_px": last_px,
    "last_vol": last_vol,    # ⭐ 添加這一行
    "tv": tv,
    "raw": raw.strip()
}
```

---

#### 修改 B: `aggregate_to_1hz()` 函數簽名和返回值

**位置**: 第 295-450 行

**步驟 1**: 修改函數簽名（第 295-298 行）
```python
# 修改返回類型
def aggregate_to_1hz(
    seq: List[Tuple[int, Dict[str,Any]]],
    reducer: str = 'last',
    ffill_limit: int = 120
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
#                                                            ^^^^^^^^  ^^^^^^^^  ^^^^^^^^
#                                                            last_prices last_volumes total_volumes
```

**步驟 2**: 添加初始化列表（第 ~330 行，在 `mids_list = []` 後）
```python
features_list = []
mids_list = []
event_counts = []
masks = []
last_prices_list = []      # ⭐ 添加
last_volumes_list = []     # ⭐ 添加
total_volumes_list = []    # ⭐ 添加
```

**步驟 3**: 單事件處理（第 ~375 行，在 `mid = rec['mid']` 後）
```python
if len(bucket) == 1:
    rec = bucket[0]
    feat = rec['feat']
    mid = rec['mid']
    last_price = rec.get('last_px', 0.0)      # ⭐ 添加
    last_volume = rec.get('last_vol', 0)      # ⭐ 添加
    total_volume = rec.get('tv', 0)           # ⭐ 添加

    features_list.append(feat)
    mids_list.append(mid)
    last_prices_list.append(last_price)       # ⭐ 添加
    last_volumes_list.append(last_volume)     # ⭐ 添加
    total_volumes_list.append(total_volume)   # ⭐ 添加
```

**步驟 4**: 多事件處理（第 ~400 行，在 reducer 邏輯後）
```python
else:  # 多事件
    # ... 現有 reducer 邏輯 (feat, mid) ...

    # ⭐ 添加這三行:
    last_price = bucket[-1].get('last_px', 0.0)
    last_volume = sum(r.get('last_vol', 0) for r in bucket)
    total_volume = max(r.get('tv', 0) for r in bucket)

    features_list.append(feat)
    mids_list.append(mid)
    last_prices_list.append(last_price)       # ⭐ 添加
    last_volumes_list.append(last_volume)     # ⭐ 添加
    total_volumes_list.append(total_volume)   # ⭐ 添加
```

**步驟 5**: 轉換為 numpy 數組（第 ~440 行，在最後添加）
```python
# 現有代碼
features = np.array(features_list)
mids = np.array(mids_list, dtype=np.float64)
bucket_event_count = np.array(event_counts, dtype=np.int32)
bucket_mask = np.array(masks, dtype=np.int32)

# ⭐ 添加這三行:
last_prices = np.array(last_prices_list, dtype=np.float64)
last_volumes = np.array(last_volumes_list, dtype=np.int64)
total_volumes = np.array(total_volumes_list, dtype=np.int64)

# ⭐ 前值填補（在現有 ffill 邏輯後添加）
for i in range(1, len(last_prices)):
    if bucket_mask[i] == 1:  # ffill
        if last_prices[i] == 0:
            last_prices[i] = last_prices[i-1]
        if total_volumes[i] == 0:
            total_volumes[i] = total_volumes[i-1]

# ⭐ 修改 return 語句（最後一行）
return features, mids, bucket_event_count, bucket_mask, last_prices, last_volumes, total_volumes
```

---

#### 修改 C: 調用 `aggregate_to_1hz()` 的地方（第 ~1050 行）

```python
# 當前代碼:
features, mids, bucket_event_count, bucket_mask = aggregate_to_1hz(
    sorted_seq,
    reducer=reducer,
    ffill_limit=config.get('ffill_limit', 120)
)

# ⭐ 修改為:
features, mids, bucket_event_count, bucket_mask, last_prices, last_volumes, total_volumes = aggregate_to_1hz(
    sorted_seq,
    reducer=reducer,
    ffill_limit=config.get('ffill_limit', 120)
)
```

---

#### 修改 D: 保存數據（第 1078-1090 行）

```python
# 準備保存的數據字典
save_data = {
    'features': features.astype(np.float32),
    'mids': mids.astype(np.float64),
    'bucket_event_count': bucket_event_count.astype(np.int32),
    'bucket_mask': bucket_mask.astype(np.int32),
    'last_prices': last_prices.astype(np.float64),       # ⭐ 添加
    'last_volumes': last_volumes.astype(np.int64),       # ⭐ 添加
    'total_volumes': total_volumes.astype(np.int64),     # ⭐ 添加
    'volume_deltas': np.diff(total_volumes, prepend=total_volumes[0]).astype(np.int64),  # ⭐ 添加
    'metadata': json.dumps(metadata, ensure_ascii=False)
}
```

---

### 2. `scripts/extract_tw_stock_data_v7.py` (2 處修改)

#### 修改 A: 讀取 NPZ 數據（搜尋 `data = np.load`）

```python
# 讀取預處理 NPZ
data = np.load(npz_path, allow_pickle=True)

features = data['features']  # (T, 20)
mids = data['mids']          # (T,)
labels = data.get('labels', None)

# ⭐ 添加這四行:
last_prices = data.get('last_prices', mids)      # 回退到 mids
last_volumes = data.get('last_volumes', np.zeros_like(mids, dtype=np.int64))
total_volumes = data.get('total_volumes', np.zeros_like(mids, dtype=np.int64))
volume_deltas = data.get('volume_deltas', np.zeros_like(mids, dtype=np.int64))
```

---

#### 修改 B: 滑動窗口生成（搜尋 `for i in range(len(features) - window_size`）

```python
# 初始化列表
X_list = []
y_list = []
prices_list = []          # ⭐ 添加
volumes_list = []         # ⭐ 添加
stock_ids_list = []

# 滑動窗口循環
for i in range(len(features) - window_size + 1):
    X_window = features[i:i+window_size]
    y_label = labels[i+window_size-1]

    # ⭐ 添加這兩行:
    price_window = last_prices[i:i+window_size]
    volume_window = last_volumes[i:i+window_size]

    X_list.append(X_window)
    y_list.append(y_label)
    prices_list.append(price_window)          # ⭐ 添加
    volumes_list.append(volume_window)        # ⭐ 添加
    stock_ids_list.append(stock_id)

# 轉換為 numpy 數組
X = np.array(X_list)
y = np.array(y_list)
prices = np.array(prices_list)          # ⭐ 添加
volumes = np.array(volumes_list)        # ⭐ 添加
stock_ids = np.array(stock_ids_list)
```

---

#### 修改 C: 最終保存（搜尋 `np.savez_compressed(output_npz_path`）

```python
# 保存最終數據
np.savez_compressed(
    output_npz_path,
    X=X,
    y=y,
    weights=weights,
    stock_ids=stock_ids,
    prices=prices,          # ⭐ 添加
    volumes=volumes         # ⭐ 添加
)

logger.info(f"✅ 保存數據: X={X.shape}, y={y.shape}, prices={prices.shape}, volumes={volumes.shape}")
```

---

## 🚀 執行步驟

### 步驟 1: 修改腳本
```bash
# 打開文件進行修改
code scripts/preprocess_single_day.py
code scripts/extract_tw_stock_data_v7.py
```

### 步驟 2: 重新預處理（30-45 分鐘）
```bash
conda activate deeplob-pro
scripts\batch_preprocess.bat
```

### 步驟 3: 重新生成訓練數據（2-3 分鐘）
```bash
python scripts/extract_tw_stock_data_v7.py \
    --preprocessed-dir ./data/preprocessed_v5 \
    --output-dir ./data/processed_v7_enhanced \
    --config ./configs/config_pro_v7_optimal.yaml
```

### 步驟 4: 驗證數據完整性
```bash
python -c "
import numpy as np

# 檢查預處理數據
data = np.load('data/preprocessed_v5/daily/20250901/0050.npz', allow_pickle=True)
print('預處理 NPZ 鍵:', list(data.keys()))
assert 'last_prices' in data.keys()
assert 'last_volumes' in data.keys()
assert 'total_volumes' in data.keys()
print('✅ 預處理數據包含所有字段')

# 檢查訓練數據
data = np.load('data/processed_v7_enhanced/npz/stock_embedding_train.npz')
print('\\n訓練 NPZ 鍵:', list(data.keys()))
assert 'prices' in data.keys()
assert 'volumes' in data.keys()
print('✅ 訓練數據包含所有字段')
"
```

---

## 📊 新增字段說明

| 字段 | 來源 | 類型 | 用途 |
|------|------|------|------|
| `last_prices` | parts[9] | float64 | 真實成交價，用於 PnL 計算 |
| `last_volumes` | parts[10] | int64 | **當次成交量，用於市場活躍度分析** ⭐⭐ |
| `total_volumes` | parts[11] | int64 | 累計成交量，用於 VWAP |
| `volume_deltas` | derived | int64 | 每秒成交量增量（= last_volumes 的聚合）|

**關鍵**: `last_volumes` (parts[10]) 是每筆交易的成交量，聚合後可反映每秒的成交強度！

---

## ⚠️ 重要提醒

1. **備份當前數據**（可選）:
   ```bash
   mv data/preprocessed_v5 data/preprocessed_v5_backup
   mv data/processed_v7 data/processed_v7_backup
   ```

2. **修改後測試單檔**:
   ```bash
   python scripts/preprocess_single_day.py --date 20250901 --test
   ```

3. **向後兼容**: V7 腳本已添加回退邏輯，舊 NPZ 文件仍可讀取

---

**文檔版本**: v1.0
**創建日期**: 2025-10-24
**預計修改時間**: 15-20 分鐘（代碼修改） + 30-45 分鐘（重新生成數據）
