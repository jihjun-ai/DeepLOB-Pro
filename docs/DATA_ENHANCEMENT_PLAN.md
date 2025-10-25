# 數據增強計劃：添加價格和成交量到 NPZ 文件

**日期**: 2025-10-24
**目標**: 在預處理階段將價格（price）和成交量（volume）保存到 NPZ 文件中
**原因**: 長期來看，在數據源頭保存完整信息更可靠，避免在環境層重複計算

---

## 📊 當前數據結構分析

### 原始 TXT 數據字段（台股 LOB）

**正確欄位對應**（感謝用戶提供）:
```
0   QType           市場代碼
1   Symbol          股票代號
2   Name            股票名稱
3   ReferencePrice  參考價
4   UpperPrice      漲停價
5   LowerPrice      跌停價
6   OpenPrice       開盤價
7   HighPrice       最高價
8   LowPrice        最低價
9   LastPrice       最新成交價 ⭐
10  LastVolume      當次成交量 ⭐⭐ (重要！)
11  TotalVolume     累計成交量 ⭐
12-21 Bid1~Bid5    買1~買5 (Price, Volume) 交錯格式
22-31 Ask1~Ask5    賣1~賣5 (Price, Volume) 交錯格式
32  MatchTime      時間戳 (HHMMSS)
33  IsTrialMatch   試撮標記
```

**關鍵字段**:
- `parts[9]`: LastPrice - 最新成交價
- `parts[10]`: LastVolume - **當次成交量**（每筆交易的量）⭐⭐
- `parts[11]`: TotalVolume - 累計成交量（累加）

### 當前 preprocess_single_day.py 輸出

**NPZ 文件結構** (`data/preprocessed_v5/daily/YYYYMMDD/{symbol}.npz`):
```python
{
    'features': (T, 20),      # LOB 五檔價量
    'mids': (T,),             # 中價 (bid1 + ask1) / 2
    'bucket_event_count': (T,),  # 每秒事件數
    'bucket_mask': (T,),      # 聚合狀態標記
    'labels': (T,),           # Triple-Barrier 標籤（可選）
    'metadata': str           # JSON 格式元數據
}
```

**當前代碼已讀取但未保存的字段**:
- ✅ `rec['last_px']` - 從 parts[9] 讀取（已在代碼中，未保存到 NPZ）
- ✅ `rec['tv']` - 從 parts[11] 讀取（已在代碼中，未保存到 NPZ）
- ❌ `rec['last_vol']` - parts[10] **未讀取**（需添加）⭐

**缺失字段**:
- ❌ 最新成交價 (last_price) - 已讀取但未保存
- ❌ 當次成交量 (last_volume) - 未讀取
- ❌ 累計成交量 (total_volume) - 已讀取但未保存

### 當前 extract_tw_stock_data_v7.py 輸出

**NPZ 文件結構** (`data/processed_v7/npz/stock_embedding_{train/val/test}.npz`):
```python
{
    'X': (N, 100, 20),        # 時間序列 LOB 特徵
    'y': (N,),                # 標籤
    'weights': (N,),          # 樣本權重
    'stock_ids': (N,)         # 股票 ID
}
```

**缺失字段**:
- ❌ 價格序列 (prices)
- ❌ 成交量序列 (volumes)

---

## 🎯 目標數據結構

### 階段一：preprocess_single_day.py 輸出增強

**新增字段**:

```python
{
    # === 現有字段 ===
    'features': (T, 20),      # LOB 五檔價量
    'mids': (T,),             # 中價
    'bucket_event_count': (T,),
    'bucket_mask': (T,),
    'labels': (T,),           # 可選
    'metadata': str,

    # === 新增字段 ===
    'last_prices': (T,),      # 最新成交價序列 ⭐ NEW
    'last_volumes': (T,),     # 當次成交量序列 ⭐⭐ NEW (parts[10])
    'total_volumes': (T,),    # 累計成交量序列 ⭐ NEW (parts[11])
    'volume_deltas': (T,),    # 每秒成交量增量 ⭐ NEW (derived)
}
```

**字段說明**:

| 字段 | 類型 | 說明 | 來源 | 用途 |
|------|------|------|------|------|
| `last_prices` | float64 | 最新成交價（每秒） | `parts[9]` | PnL 計算、回報率 |
| `last_volumes` | int64 | 當次成交量（每筆交易）| `parts[10]` | **成交強度、市場活躍度** ⭐ |
| `total_volumes` | int64 | 累計成交量（每秒） | `parts[11]` | 總量統計、VWAP |
| `volume_deltas` | int64 | 每秒成交量增量 | `diff(total_volumes)` | 流量分析 |

**聚合策略**（多事件同秒）:
- `last_prices`: 使用 **last**（最後一筆成交價）
- `last_volumes`: 使用 **sum**（同秒內所有成交量加總）⭐
- `total_volumes`: 使用 **max**（累計量取最大值）
- `volume_deltas`: 計算後再聚合（= last_volumes 的聚合）

---

### 階段二：extract_tw_stock_data_v7.py 輸出增強

**新增字段**:

```python
{
    # === 現有字段 ===
    'X': (N, 100, 20),        # LOB 特徵
    'y': (N,),                # 標籤
    'weights': (N,),          # 權重
    'stock_ids': (N,),        # 股票 ID

    # === 新增字段 ===
    'prices': (N, 100),       # 價格序列（100 時間步） ⭐ NEW
    'volumes': (N, 100),      # 成交量序列（100 時間步） ⭐ NEW
    'mid_prices': (N, 100),   # 中價序列（100 時間步） ⭐ NEW (optional)
}
```

**用途**:
- `prices`: 用於計算 PnL、回報率
- `volumes`: 用於 VWAP、成交量指標、流動性分析
- `mid_prices`: 用於訂單簿中價分析

---

## 🔧 實作修改方案

### 修改 1: `scripts/preprocess_single_day.py`

#### 1.1 修改 `parse_line()` 函數

**位置**: 第 211-275 行

**需要添加**: 讀取 `parts[10]` (LastVolume)

**修改**:
```python
def parse_line(raw: str) -> Tuple[str, int, Optional[Dict[str, Any]]]:
    """解析單行數據（增強版：添加 last_price, last_volume 和 total_volume）"""
    # ... 現有代碼 ...

    # 取價格和成交量
    last_px = to_float(parts[IDX_LASTPRICE], 0.0)      # parts[9]  ✅ 已存在
    last_vol = max(0, int(to_float(parts[IDX_LASTVOL], 0.0)))  # parts[10] ⭐ NEW
    tv = max(0, int(to_float(parts[IDX_TV], 0.0)))     # parts[11] ✅ 已存在

    # ... 現有代碼 ...

    rec = {
        "feat": feat,
        "mid": mid,
        "ref": ref,
        "upper": upper,
        "lower": lower,
        "last_px": last_px,      # ✅ 已存在
        "last_vol": last_vol,    # ⭐ NEW
        "tv": tv,                # ✅ 已存在
        "raw": raw.strip()
    }
```

**修改點**:
- ✅ `last_px` (parts[9]) - 已存在，無需修改
- ⭐ `last_vol` (parts[10]) - **需要添加**
- ✅ `tv` (parts[11]) - 已存在，無需修改

---

#### 1.2 修改 `aggregate_to_1hz()` 函數

**位置**: 第 295-450 行

**修改**: 添加 `last_prices`, `last_volumes`, `total_volumes` 的聚合

```python
def aggregate_to_1hz(
    seq: List[Tuple[int, Dict[str,Any]]],
    reducer: str = 'last',
    ffill_limit: int = 120
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    1Hz 時間聚合（增強版：添加價格和成交量）

    Returns:
        features: (T, 20)
        mids: (T,)
        bucket_event_count: (T,)
        bucket_mask: (T,)
        last_prices: (T,)      # ⭐ NEW
        last_volumes: (T,)     # ⭐⭐ NEW
        total_volumes: (T,)    # ⭐ NEW
    """
    # ... 現有初始化代碼 ...

    features_list = []
    mids_list = []
    event_counts = []
    masks = []
    last_prices_list = []      # ⭐ NEW
    last_volumes_list = []     # ⭐⭐ NEW
    total_volumes_list = []    # ⭐ NEW

    # ... 現有聚合邏輯 ...

    for bucket in buckets:
        if len(bucket) == 1:
            # 單事件
            rec = bucket[0]
            feat = rec['feat']
            mid = rec['mid']
            last_price = rec.get('last_px', 0.0)      # ⭐ NEW
            last_volume = rec.get('last_vol', 0)      # ⭐⭐ NEW
            total_volume = rec.get('tv', 0)           # ⭐ NEW

            features_list.append(feat)
            mids_list.append(mid)
            last_prices_list.append(last_price)       # ⭐ NEW
            last_volumes_list.append(last_volume)     # ⭐⭐ NEW
            total_volumes_list.append(total_volume)   # ⭐ NEW
            # ...

        else:  # 多事件同秒
            # ... 現有 reducer 邏輯 ...

            # 新增：聚合價格和成交量
            last_price = bucket[-1].get('last_px', 0.0)           # 最後一筆成交價
            last_volume = sum(r.get('last_vol', 0) for r in bucket)  # 同秒內所有成交量加總 ⭐⭐
            total_volume = max(r.get('tv', 0) for r in bucket)   # 最大累計量

            last_prices_list.append(last_price)       # ⭐ NEW
            last_volumes_list.append(last_volume)     # ⭐⭐ NEW
            total_volumes_list.append(total_volume)   # ⭐ NEW

    # 轉換為 numpy 數組
    last_prices = np.array(last_prices_list, dtype=np.float64)
    last_volumes = np.array(last_volumes_list, dtype=np.int64)
    total_volumes = np.array(total_volumes_list, dtype=np.int64)

    # ⭐ 處理缺失值（前值填補）
    for i in range(1, len(last_prices)):
        if masks[i] == 1:  # ffill
            last_prices[i] = last_prices[i-1] if last_prices[i] == 0 else last_prices[i]
            # last_volumes 不填補（0 表示該秒無成交）
            total_volumes[i] = total_volumes[i-1] if total_volumes[i] == 0 else total_volumes[i]

    return features, mids, bucket_event_count, bucket_mask, last_prices, last_volumes, total_volumes
```

---

#### 1.3 修改保存部分

**位置**: 第 1074-1092 行

**修改**:
```python
# 準備保存的數據字典
save_data = {
    'features': features.astype(np.float32),
    'mids': mids.astype(np.float64),
    'bucket_event_count': bucket_event_count.astype(np.int32),
    'bucket_mask': bucket_mask.astype(np.int32),
    'last_prices': last_prices.astype(np.float64),       # ⭐ NEW (parts[9])
    'last_volumes': last_volumes.astype(np.int64),       # ⭐⭐ NEW (parts[10])
    'total_volumes': total_volumes.astype(np.int64),     # ⭐ NEW (parts[11])
    'volume_deltas': np.diff(total_volumes, prepend=total_volumes[0]).astype(np.int64),  # ⭐ NEW (派生)
    'metadata': json.dumps(metadata, ensure_ascii=False)
}

# 如果有標籤數據，添加到保存字典中
if labels is not None:
    save_data['labels'] = labels.astype(np.float32)

np.savez_compressed(npz_path, **save_data)
```

---

### 修改 2: `scripts/extract_tw_stock_data_v7.py`

#### 2.1 修改數據讀取部分

**位置**: 搜尋 `np.load(npz_path)`

**修改**: 讀取新增字段

```python
# 讀取預處理 NPZ
data = np.load(npz_path, allow_pickle=True)

features = data['features']  # (T, 20)
mids = data['mids']          # (T,)
labels = data.get('labels', None)  # (T,) 可選

# ⭐ NEW: 讀取價格和成交量
last_prices = data.get('last_prices', None)      # (T,)
total_volumes = data.get('total_volumes', None)  # (T,)
volume_deltas = data.get('volume_deltas', None)  # (T,)

# 回退方案：如果沒有 last_prices，從 mids 計算
if last_prices is None:
    logger.warning(f"⚠️  未找到 last_prices，使用 mids 作為替代")
    last_prices = mids

# 回退方案：如果沒有 total_volumes，填零
if total_volumes is None:
    logger.warning(f"⚠️  未找到 total_volumes，填充為 0")
    total_volumes = np.zeros_like(mids, dtype=np.int64)
```

---

#### 2.2 修改滑動窗口生成部分

**位置**: 滑動窗口循環

**修改**: 同時提取價格和成交量序列

```python
# 滑動窗口生成樣本
for i in range(len(features) - window_size + 1):
    # LOB 特徵窗口
    X_window = features[i:i+window_size]  # (100, 20)

    # 標籤（當前時間步）
    y_label = labels[i+window_size-1]

    # ⭐ NEW: 價格和成交量窗口
    price_window = last_prices[i:i+window_size]      # (100,)
    volume_window = volume_deltas[i:i+window_size]   # (100,)
    mid_price_window = mids[i:i+window_size]         # (100,)

    # 保存
    X_list.append(X_window)
    y_list.append(y_label)
    prices_list.append(price_window)       # ⭐ NEW
    volumes_list.append(volume_window)     # ⭐ NEW
    mid_prices_list.append(mid_price_window)  # ⭐ NEW (optional)
    stock_ids_list.append(stock_id)
```

---

#### 2.3 修改最終保存部分

**位置**: `np.savez_compressed(output_npz_path, ...)`

**修改**:
```python
# 保存最終數據
np.savez_compressed(
    output_npz_path,
    X=X,                    # (N, 100, 20)
    y=y,                    # (N,)
    weights=weights,        # (N,)
    stock_ids=stock_ids,    # (N,)
    prices=prices,          # (N, 100) ⭐ NEW
    volumes=volumes,        # (N, 100) ⭐ NEW
    mid_prices=mid_prices   # (N, 100) ⭐ NEW (optional)
)

logger.info(f"✅ 保存數據: X={X.shape}, y={y.shape}, prices={prices.shape}, volumes={volumes.shape}")
```

---

## 🔄 重新生成數據流程

### 步驟 1: 重新預處理（生成增強版 NPZ）

```bash
# 激活環境
conda activate deeplob-pro

# 批次重新預處理所有日期
scripts\batch_preprocess.bat

# 或單日測試
python scripts/preprocess_single_day.py \
    --date 20250901 \
    --input-dir data/temp \
    --output-dir data/preprocessed_v5_enhanced \
    --config configs/config_pro_v5_ml_optimal.yaml
```

**預計時間**: 30-45 分鐘（195 檔股票 × N 天）

---

### 步驟 2: 重新生成訓練數據（V7）

```bash
# 從增強版 NPZ 生成訓練數據
python scripts/extract_tw_stock_data_v7.py \
    --preprocessed-dir ./data/preprocessed_v5_enhanced \
    --output-dir ./data/processed_v7_enhanced \
    --config ./configs/config_pro_v7_optimal.yaml
```

**預計時間**: 2-3 分鐘

---

### 步驟 3: 驗證數據完整性

```bash
# 驗證 NPZ 文件包含所有字段
python -c "
import numpy as np

# 檢查預處理數據
data = np.load('data/preprocessed_v5_enhanced/daily/20250901/0050.npz', allow_pickle=True)
print('預處理 NPZ 鍵:', list(data.keys()))
print('last_prices 形狀:', data.get('last_prices', 'NOT FOUND'))
print('total_volumes 形狀:', data.get('total_volumes', 'NOT FOUND'))

# 檢查最終訓練數據
data = np.load('data/processed_v7_enhanced/npz/stock_embedding_train.npz', allow_pickle=True)
print('\n訓練 NPZ 鍵:', list(data.keys()))
print('prices 形狀:', data.get('prices', 'NOT FOUND'))
print('volumes 形狀:', data.get('volumes', 'NOT FOUND'))
"
```

---

### 步驟 4: 更新環境使用新數據

修改 [src/envs/tw_lob_trading_env.py](../src/envs/tw_lob_trading_env.py):

```python
def _load_data(self):
    """載入台股數據（增強版：包含價格和成交量）"""
    # ... 現有載入邏輯 ...

    # ⭐ 讀取價格（優先使用 prices，回退到 mids）
    if 'prices' in data.files:
        self.prices = data['prices'][sampled_indices]  # (N, 100)
        # 使用最後一個時間步的價格
        self.prices = self.prices[:, -1]
        logger.info(f"✅ 使用 NPZ 中的真實價格")
    else:
        logger.warning(f"⚠️  未找到 prices 字段，從 LOB 計算中價")
        # ... 現有回退邏輯 ...

    # ⭐ 讀取成交量（可選）
    if 'volumes' in data.files:
        self.volumes = data['volumes'][sampled_indices]  # (N, 100)
        logger.info(f"✅ 載入成交量數據: {self.volumes.shape}")
    else:
        logger.warning(f"⚠️  未找到 volumes 字段")
        self.volumes = None
```

---

## 📊 數據完整性檢查

### 檢查點 1: 預處理輸出

```bash
# 檢查單個 NPZ 文件
python -c "
import numpy as np
data = np.load('data/preprocessed_v5_enhanced/daily/20250901/0050.npz', allow_pickle=True)

assert 'last_prices' in data.keys(), '缺少 last_prices'
assert 'total_volumes' in data.keys(), '缺少 total_volumes'
assert 'volume_deltas' in data.keys(), '缺少 volume_deltas'

print('✅ 預處理數據完整性檢查通過')
print(f'  - last_prices: {data[\"last_prices\"].shape}')
print(f'  - total_volumes: {data[\"total_volumes\"].shape}')
print(f'  - volume_deltas: {data[\"volume_deltas\"].shape}')
"
```

### 檢查點 2: 訓練數據輸出

```bash
# 檢查最終訓練 NPZ
python -c "
import numpy as np
data = np.load('data/processed_v7_enhanced/npz/stock_embedding_train.npz', allow_pickle=True)

assert 'prices' in data.keys(), '缺少 prices'
assert 'volumes' in data.keys(), '缺少 volumes'

X_shape = data['X'].shape
prices_shape = data['prices'].shape
volumes_shape = data['volumes'].shape

assert X_shape[0] == prices_shape[0], '樣本數量不匹配'
assert X_shape[1] == prices_shape[1] == 100, '時間步數量不匹配'

print('✅ 訓練數據完整性檢查通過')
print(f'  - X: {X_shape}')
print(f'  - prices: {prices_shape}')
print(f'  - volumes: {volumes_shape}')
"
```

---

## 🎯 預期效益

### 數據質量提升

| 改進點 | 當前 | 增強後 |
|--------|------|--------|
| **價格來源** | 從 LOB 計算中價（可能不準） | 真實成交價 ✅ |
| **成交量** | 無 | 完整成交量序列 ✅ |
| **可靠性** | 依賴環境層計算 | 數據源頭保存 ✅ |
| **靈活性** | 固定中價計算 | 多種價格指標可選 ✅ |

### 新增功能可能性

**強化學習環境增強**:
- ✅ 使用真實成交價計算 PnL
- ✅ 基於成交量的流動性指標
- ✅ VWAP (Volume-Weighted Average Price)
- ✅ 成交量加權獎勵函數
- ✅ 高/低流動性環境區分

**新增特徵工程**:
- ✅ 價格波動率（從真實成交價計算）
- ✅ 成交量異常檢測
- ✅ 價格-成交量相關性
- ✅ 訂單不平衡指標

---

## ⚠️ 注意事項

### 1. 向後兼容性

**問題**: 舊代碼可能無法讀取新格式 NPZ

**解決方案**: 在讀取時添加回退邏輯
```python
# 回退方案
if 'prices' in data.files:
    prices = data['prices']
else:
    logger.warning("未找到 prices，使用 mids 替代")
    prices = data['mids']
```

### 2. 磁碟空間

**新增數據量估算**:
- `last_prices`: (T,) × 8 bytes (float64)
- `total_volumes`: (T,) × 8 bytes (int64)
- `volume_deltas`: (T,) × 8 bytes (int64)

**每檔股票每天**: ~24 bytes × 20,000 秒 ≈ **480 KB**
**195 檔 × 30 天**: 195 × 30 × 480 KB ≈ **2.7 GB**

**結論**: 增加的磁碟空間可忽略（< 10%）

### 3. 處理時間

**預處理階段**: 幾乎無影響（僅多保存 3 個數組）
**V7 生成階段**: 增加 ~10-20% 時間（滑動窗口需處理更多數組）

---

## 🚀 執行計劃

### 短期（立即執行）

1. ✅ **完成階段二訓練**（使用當前數據）
   - 先用現有數據完成 SB3 訓練
   - 驗證訓練管線正確性
   - 獲得基線性能

### 中期（階段三前執行）

2. ⏳ **實施數據增強**
   - 修改 `preprocess_single_day.py`
   - 修改 `extract_tw_stock_data_v7.py`
   - 重新生成所有數據
   - 更新環境使用新數據

### 長期（持續優化）

3. ⏳ **利用新數據改進模型**
   - 設計基於成交量的獎勵函數
   - 添加流動性感知特徵
   - 實現 VWAP 策略對比

---

## 📝 修改文件清單

### 必須修改

- [ ] `scripts/preprocess_single_day.py` - 添加價格和成交量聚合
- [ ] `scripts/extract_tw_stock_data_v7.py` - 添加價格和成交量窗口
- [ ] `src/envs/tw_lob_trading_env.py` - 使用 NPZ 中的價格

### 可選修改

- [ ] `configs/config_pro_v7_optimal.yaml` - 更新配置註釋
- [ ] `docs/V6_TWO_STAGE_PIPELINE_GUIDE.md` - 更新文檔

---

## 🎯 總結

**建議策略**: **分階段實施**

1. **現階段（立即）**: 使用當前數據完成階段二訓練
   - 目標：驗證 SB3 管線正確性
   - 時間：1-2 天

2. **下一階段（階段三前）**: 實施數據增強
   - 目標：獲得更可靠的價格和成交量數據
   - 時間：半天修改 + 半天重新生成數據

3. **未來階段（持續）**: 利用新數據優化模型
   - 目標：設計更精確的獎勵函數和特徵
   - 時間：持續迭代

**核心優勢**:
- ✅ 數據源頭保存，避免重複計算
- ✅ 向後兼容，不影響現有流程
- ✅ 為未來優化打下基礎
- ✅ 磁碟和時間成本可忽略

---

**文檔版本**: v1.0
**創建日期**: 2025-10-24
**最後更新**: 2025-10-24
