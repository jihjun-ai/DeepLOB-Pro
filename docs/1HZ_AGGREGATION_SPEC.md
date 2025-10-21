# 1Hz 時間聚合規格說明

**版本**: 1.0.0
**日期**: 2025-10-21
**狀態**: ✅ 已實作

---

## 📋 概述

將原本的 **10 事件聚合** 改為 **1Hz (秒級) 時間聚合**，以提供更精確的時間對齊和更細粒度的數據。

### 核心改變

| 項目 | 舊版 (10 事件聚合) | 新版 (1Hz 聚合) |
|------|-------------------|----------------|
| 聚合基準 | 每 10 筆事件 | 每 1 秒 |
| 時間對齊 | 不固定 | 精確到秒 |
| 無事件處理 | 跳過 | ffill 或標記缺失 |
| 多事件處理 | N/A | 支援 3 種 reducer |
| 輸出標記 | 無 | bucket_mask (4 種狀態) |

---

## 🎯 設計規格

### 時間框架

- **交易時段**: 09:00:00 – 13:30:00
- **秒軸**: 逐秒建立時間桶 `[t, t+1s)`
- **Latency Guard**: 禁止使用 `t+1s` 之外的資料（無偷看未來）

### Bucket 事件計數

每個秒桶 `bucket_event_count[t]` 表示該秒內的事件數：

- `0` → 無事件
- `1` → 單事件
- `≥2` → 多事件

---

## 🔧 聚合策略（Reducer）

當 `bucket_event_count ≥ 2` 時，根據 `reducer` 參數聚合：

### 1. `last` (預設) ⭐

- **策略**: 取該秒內**最後一筆快照**
- **優點**: 簡單快速，最接近秒末狀態
- **用途**: 通用場景

```python
rec = bucket[-1]
feat = rec['feat']
mid = rec['mid']
```

### 2. `median`

- **策略**: 對價量做逐欄**中位數**
- **優點**: 穩健，抗異常值
- **缺點**: 計算稍重
- **用途**: 噪音較大的市場

```python
feats_array = np.array([r['feat'] for r in bucket])
feat = np.median(feats_array, axis=0)
mid = np.median([r['mid'] for r in bucket])
```

### 3. `vwap-mid`

- **策略**: 以成交量加權平均 mid
- **優點**: 反映真實交易活動
- **回退**: 若無成交量資訊則回退到 `last`
- **用途**: 有成交量數據的場景

```python
if has_volume:
    total_vol = sum(r.get('tv', 0) for r in bucket)
    mid = sum(r['mid'] * r['tv'] for r in bucket) / total_vol
    feat = bucket[-1]['feat']  # 特徵仍用 last
else:
    # 回退到 last
```

---

## 📊 無事件處理（Bucket Count == 0）

### ffill 規則

若距上次實際更新時間的間隔 `gap_sec ≤ ffill_limit`（預設 120 秒）：

- **動作**: 前值填補 (forward fill)
- **標記**: `bucket_mask = 1`

```python
if gap <= ffill_limit and last_valid_feat is not None:
    features_list.append(last_valid_feat)
    mids_list.append(last_valid_mid)
    masks.append(1)  # ffill
```

### 缺失標記

若間隔 `gap_sec > ffill_limit` 或無前值：

- **動作**: 標記為缺失
- **填充**: 零向量（下游需處理）
- **標記**: `bucket_mask = 2`

```python
else:
    features_list.append(np.zeros(20))
    mids_list.append(0.0)
    masks.append(2)  # missing
```

### 首尾缺失剔除

- **邏輯**: 只剔除序列**首尾**連續缺失
- **保留**: 中間缺失保留給下游決策

```python
# 找到第一個和最後一個有效點
first_valid = 找到第一個 mask != 2 的索引
last_valid = 找到最後一個 mask != 2 的索引

# 截取有效範圍
output = output[first_valid:last_valid+1]
```

---

## 🏷️ Bucket Mask 標記系統

**輸出**: `bucket_mask[T]` (int32 陣列)

| 值 | 意義 | 來源 | 說明 |
|----|------|------|------|
| `0` | 原生-單事件 | 該秒 1 筆事件 | 直接取用，最可靠 |
| `1` | ffill | 該秒 0 筆，在 ffill_limit 內 | 前值填補 |
| `2` | 缺失 | 該秒 0 筆，超過 ffill_limit | 需下游處理 |
| `3` | 原生-多事件聚合 | 該秒 ≥2 筆 | 經 reducer 聚合 |

### 使用建議

**訓練階段**：
- 含 `mask=2` (缺失) 的窗口：**丟棄**或切片重算
- `mask=0` (單事件)：最高權重
- `mask=3` (多事件)：正常權重
- `mask=1` (ffill)：可保留但可降權

**回測階段**：
- `mask=2` 窗口：不交易或加風控
- `mask=1` 窗口：可用但需注意時效性

---

## 📦 輸出格式

### NPZ 文件內容

```python
{
    "features": np.ndarray,          # (T, 20) - LOB 特徵
    "mids": np.ndarray,              # (T,) - 中間價
    "bucket_event_count": np.ndarray, # (T,) - 每秒事件數
    "bucket_mask": np.ndarray,       # (T,) - 標記 {0,1,2,3}
    "metadata": json.dumps({
        # ... 原有欄位 ...

        # 新增：1Hz 聚合資訊
        "sampling_mode": "time",
        "bucket_seconds": 1,
        "ffill_limit": 120,
        "agg_reducer": "last",
        "n_seconds": 16200,          # 實際秒數
        "ffill_ratio": 0.05,         # ffill 比例
        "missing_ratio": 0.02,       # 缺失比例
        "multi_event_ratio": 0.15,   # 多事件比例
        "max_gap_sec": 45            # 最大間隔
    })
}
```

### Metadata 統計

```python
# 基於 bucket_mask 計算
ffill_ratio = (bucket_mask == 1).sum() / len(bucket_mask)
missing_ratio = (bucket_mask == 2).sum() / len(bucket_mask)
multi_event_ratio = (bucket_mask == 3).sum() / len(bucket_mask)

# 最大間隔（秒）
valid_indices = np.where(bucket_mask != 2)[0]
max_gap_sec = np.max(np.diff(valid_indices)) if len(valid_indices) > 1 else 0
```

---

## ✅ 驗收標準

### 必過測試

1. **對齊性**: 每秒都有一個 `bucket_mask` 標記且和 `features/mids` 對齊
   ```python
   assert len(features) == len(mids) == len(bucket_mask) == len(bucket_event_count)
   ```

2. **無事件標記**: `bucket_event_count==0` 的秒，`bucket_mask` 只能是 1 或 2
   ```python
   zero_event_mask = bucket_mask[bucket_event_count == 0]
   assert all((zero_event_mask == 1) | (zero_event_mask == 2))
   ```

3. **多事件標記**: `bucket_event_count≥2` 的秒，`bucket_mask==3`
   ```python
   multi_event_mask = bucket_mask[bucket_event_count >= 2]
   assert all(multi_event_mask == 3)
   ```

4. **無偷看未來**: 任何點不會使用 `t+1s` 之外的事件
   ```python
   # 在實作中保證：桶分配時只用 [t, t+1s) 的事件
   ```

5. **統計一致性**: `ffill_ratio + missing_ratio ≤ 1` 且統計正確
   ```python
   assert ffill_ratio + missing_ratio <= 1.0
   assert ffill_ratio == (bucket_mask == 1).sum() / len(bucket_mask)
   ```

---

## 🧪 測試腳本

**位置**: `scripts/test_1hz_aggregation.py`

### 使用方式

```bash
conda activate deeplob-pro
python scripts/test_1hz_aggregation.py
```

### 測試案例

1. **基本功能**:
   - 正常事件序列
   - 無事件間隔 (ffill)
   - 同一秒多筆 (聚合)
   - 長間隔 (missing)

2. **邊界案例**:
   - 空序列
   - 單一事件
   - 同一秒 10+ 筆
   - 超過 ffill_limit 間隔

3. **Reducer 測試**:
   - `last`
   - `median`
   - `vwap-mid`

---

## 📈 性能影響

### 時間複雜度

- **舊版**: O(N / 10) - N 為事件數
- **新版**: O(T) - T 為秒數 (固定 ~16200)

### 空間需求

- **舊版**: 約 N / 10 個點
- **新版**: 固定約 16200 個點（09:00:00 - 13:30:00）

### 數據量變化

| 場景 | 舊版樣本數 | 新版樣本數 | 變化 |
|------|-----------|-----------|------|
| 低頻股票 | ~500 | ~16200 | +32x |
| 高頻股票 | ~5000 | ~16200 | +3x |
| 平均 | ~2000 | ~16200 | +8x |

**注意**: 新版樣本數固定，但包含 ffill 和 missing 標記，實際有效樣本數取決於 `bucket_mask`。

---

## 🔄 下游適配

### extract_tw_stock_data_v6.py 需修改

**讀取預處理數據時**:

```python
def load_preprocessed_npz(npz_path):
    data = np.load(npz_path, allow_pickle=True)

    features = data['features']
    mids = data['mids']
    bucket_event_count = data['bucket_event_count']  # 新增
    bucket_mask = data['bucket_mask']                # 新增
    meta = json.loads(str(data['metadata']))

    return features, mids, bucket_event_count, bucket_mask, meta
```

**滑窗生成時**:

```python
# 排除含缺失的窗口
for t in range(SEQ_LEN - 1, len(mids)):
    window_mask = bucket_mask[t-SEQ_LEN+1:t+1]

    # 若窗口含缺失 (mask=2)，跳過
    if (window_mask == 2).any():
        continue

    # 可選：計算窗口質量分數
    quality_score = (window_mask == 0).sum() / SEQ_LEN  # 單事件比例

    # 生成樣本...
```

---

## 📚 參考

- **實作**: [scripts/preprocess_single_day.py](../scripts/preprocess_single_day.py) - `aggregate_to_1hz()`
- **測試**: [scripts/test_1hz_aggregation.py](../scripts/test_1hz_aggregation.py)
- **V6 指南**: [V6_TWO_STAGE_PIPELINE_GUIDE.md](V6_TWO_STAGE_PIPELINE_GUIDE.md)

---

## 🎯 總結

### 核心優勢

✅ **精確時間對齊**: 秒級精度，便於標籤計算
✅ **完整數據保留**: 無事件秒也有記錄（ffill/missing）
✅ **靈活聚合策略**: 支援 3 種 reducer
✅ **透明標記系統**: 4 種 bucket_mask 狀態
✅ **下游友好**: metadata 提供完整統計

### 使用建議

1. **預設配置**: `reducer='last'`, `ffill_limit=120`
2. **訓練數據**: 排除含 `mask=2` 的窗口
3. **回測驗證**: 檢查 `ffill_ratio` 和 `missing_ratio`
4. **質量監控**: 定期查看 metadata 統計

---

**版本**: 1.0.0
**更新**: 2025-10-21
**狀態**: ✅ 已實作並測試
