# ChatGPT 建議分析報告

**日期**: 2025-10-21
**分析者**: Claude (Sonnet 4.5)
**目的**: 評估 7 項建議的必要性、可行性和優先級

---

## 總覽

| 建議 | ChatGPT 優先級 | 我的評估 | 採納建議 | 理由 |
|------|---------------|----------|---------|------|
| A. ffill 上限管控 | 中度 | **高優先** | ✅ **採納** | 已有部分實現，需補強監控 |
| B. 動態過濾退化保護 | 中度 | **中優先** | ⚠️ **部分採納** | 極端情況罕見，可觀察後決定 |
| C. 切分穩定性 | 高優先 | **高優先** | ✅ **立即採納** | 關鍵問題，已有 seed 但需驗證 |
| D. 零價統計一致性 | 中優先 | **中優先** | ✅ **採納** | 審計需求，易實現 |
| E. 標籤邊界檢查 | 中優先 | **高優先** | ✅ **立即採納** | 數據完整性關鍵 |
| F. 計算資源優化 | 低優先 | **低優先** | ⏳ **延後** | 當前規模無瓶頸 |
| G. 批次器健壯性 | 低優先 | **中優先** | ✅ **採納** | Windows 環境重要 |

---

## A. ffill 上限與長缺口管控

### ChatGPT 建議
- **問題**: 盤中長時間無報價，ffill 把舊狀態延伸到現在 → 假訊號
- **建議**:
  1. `ffill_max_gap` 設定上限（30-60 秒）
  2. 超過上限標記為缺失
  3. metadata 記錄 `max_gap_sec`, `ffill_ratio`
  4. 滑窗時排除「缺失占比過高」的窗口

### 現狀檢查 ✅ 已有基礎實現

**preprocess_single_day.py:233**:
```python
def aggregate_to_1hz(seq, reducer='last', ffill_limit=120):
    """
    ffill_limit: 前值填補最大間隔（秒）
    """
```

**已實現**:
- ✅ `ffill_limit=120` 秒（2 分鐘）
- ✅ `bucket_mask` 標記 {0: 單事件, 1: ffill, 2: 缺失, 3: 多事件}
- ✅ metadata 記錄 `ffill_ratio`, `missing_ratio`, `max_gap_sec`

**當前值**（來自 load_npz.py 輸出）:
```
mask=1 (ffill): 27.4%
mask=2 (missing): 1.1%
max_gap_sec: 181 秒
```

### 評估

#### ✅ 優點（已實現部分）
1. **ffill_limit 已設定**: 120 秒是合理值（台股盤中冷門股可能數分鐘無報價）
2. **mask 機制健全**: 清楚區分原始/ffill/缺失/聚合
3. **統計已記錄**: metadata 包含所需資訊

#### ⚠️ 不足之處
1. **缺少滑窗階段的品質過濾**:
   - 當前：生成所有滑窗，不管缺失占比
   - 建議：若窗口內 `mask=1` (ffill) 占比 > 50%，應標記低質量

2. **ffill_limit 偏大**:
   - 120 秒 = 2 分鐘，對高頻交易偏長
   - 建議調整為 **60 秒**（1 分鐘）

3. **缺少異常監控**:
   - max_gap_sec=181 秒（3 分鐘）超過 ffill_limit=120 秒
   - 應在 summary.json 中標註「長缺口股票」

### 建議行動 ✅ **採納**

#### 優先級：**高**

#### 實現方案

**階段 1: 預處理階段**（已完成 ✅）
```python
# preprocess_single_day.py
# 當前已記錄：
vol_stats['ffill_ratio'] = float((bucket_mask == 1).sum() / len(bucket_mask))
vol_stats['max_gap_sec'] = int(np.max(np.diff(...)))
```

**階段 2: 訓練數據生成階段**（需補充）
```python
# extract_tw_stock_data_v6.py - 滑窗生成時
for t in range(SEQ_LEN - 1, max_t):
    window_mask = bucket_mask[window_start:t+1]

    # 新增：品質檢查
    ffill_ratio = (window_mask == 1).sum() / len(window_mask)
    if ffill_ratio > 0.5:  # 超過 50% 是 ffill
        # 選項 1: 跳過（推薦）
        continue
        # 選項 2: 降權
        # weight *= 0.5
```

**階段 3: 摘要報告**（需補充）
```python
# preprocess_single_day.py - summary.json
"data_quality": {
    "long_gap_symbols": [  # 新增
        {"symbol": "3048", "max_gap_sec": 181},
        ...
    ],
    "high_ffill_symbols": [  # 新增
        {"symbol": "2330", "ffill_ratio": 0.45},
        ...
    ]
}
```

#### 時程
- **立即**: 降低 ffill_limit 從 120 → 60 秒
- **本週**: 添加滑窗品質過濾
- **下週**: 增強摘要報告

---

## B. 動態過濾的「退化保護」

### ChatGPT 建議
- **問題**: 極端行情日（大多數標的停滯），分位數集中 → 過濾失效
- **建議**:
  1. 若 `IQR < 1e-6`，回退到寬鬆過濾（P10 或 none）
  2. summary.json 記錄 `filter_confidence`, `range_iqr`

### 現狀檢查

**preprocess_single_day.py:515-564**:
```python
def determine_adaptive_threshold(daily_stats, config):
    range_values = [s['range_pct'] for s in daily_stats]

    candidates = {
        'P10': np.percentile(range_values, 10),
        'P25': np.percentile(range_values, 25),
        'P50': np.percentile(range_values, 50),
        'none': 0.0
    }

    # 選擇最接近目標分布的閾值
```

### 評估

#### ✅ 優點（現有設計）
1. **多候選閾值**: P10/P25/P50/none，有退路
2. **目標分布驅動**: 選擇最接近 30/40/30 的閾值
3. **已有 'none' 選項**: 極端情況可以不過濾

#### ⚠️ 潛在問題
1. **極端行情測試不足**:
   - 例如：漲跌停日、熔斷日
   - 所有股票 range_pct ≈ 0.1（漲停）或 ≈ 0（停牌）

2. **IQR 檢查缺失**:
   - 當前沒有檢測分位數重疊的邏輯

#### 🤔 實際發生概率
- **台股實況**:
  - 全面漲跌停：極罕見（1997 亞洲金融風暴、2020 疫情暴跌）
  - 當前有熔斷機制，難以出現極端情況
  - 即使極端日，仍有部分股票波動（權值股、概念股）

- **數據期間**:
  - 當前數據：2025-09-01 ~ 2025-09-10（10 天）
  - 極端情況出現概率：< 0.1%

### 建議行動 ⚠️ **部分採納**

#### 優先級：**中低**（觀察後決定）

#### 實現方案（輕量級）

**方案 1: 保守監控（推薦）**
```python
# preprocess_single_day.py - determine_adaptive_threshold()
range_values = [s['range_pct'] for s in daily_stats]

# 計算 IQR
q1, q3 = np.percentile(range_values, [25, 75])
iqr = q3 - q1

# 記錄但不干預（先觀察）
if iqr < 0.001:  # 0.1% 波動
    logging.warning(
        f"⚠️ 當日波動極低: IQR={iqr:.6f}\n"
        f"   這可能是極端行情日（停牌/漲跌停）\n"
        f"   建議手動檢查過濾結果"
    )

# 寫入 summary.json
summary['filter_confidence'] = {
    'iqr': float(iqr),
    'warning': 'low_volatility' if iqr < 0.001 else 'normal'
}
```

**方案 2: 主動回退（若確認需要）**
```python
if iqr < 0.0005:  # 極端閾值
    logging.warning(f"⚠️ IQR={iqr:.6f} 過低，強制使用 'none' 過濾")
    return 0.0, 'none_forced', {'down': 0.33, 'neutral': 0.34, 'up': 0.33}
```

#### 時程
- **本週**: 添加 IQR 監控和日誌
- **觀察 1-2 個月**: 是否出現極端情況
- **若頻繁出現**: 實現主動回退邏輯

#### 結論
**暫不實現主動回退**，原因：
1. 極端情況罕見（< 0.1%）
2. 現有 'none' 選項已提供退路
3. 過早優化可能引入複雜度
4. 建議先監控，累積數據後決定

---

## C. 切分穩定性與隨機種子

### ChatGPT 建議
- **問題**: 按股票數切分 70/15/15 沒問題，但若沒固定 seed，重跑會抖動
- **建議**:
  1. 明確設定 `random.seed(seed)` 或 `np.random.default_rng(seed)`
  2. 將 seed 寫入 `normalization_meta.json`
  3. 測試：兩次重跑 stock_ids 的 split 一致

### 現狀檢查 ✅ 已有實現但需驗證

**extract_tw_stock_data_v6.py:413-420**:
```python
# 步驟 3: 按股票切分 70/15/15
import random
SPLIT_SEED = config.get('split', {}).get('seed', 42)
random.Random(SPLIT_SEED).shuffle(valid_stocks)

n_train = max(1, int(n_stocks * config['split']['train_ratio']))
n_val = max(1, int(n_stocks * config['split']['val_ratio']))
```

**config_pro_v5_ml_optimal.yaml:94**:
```yaml
split:
  train_ratio: 0.7
  val_ratio: 0.15
  test_ratio: 0.15
  seed: 42  # ✅ 已設定
```

**normalization_meta.json** (需檢查是否已寫入):
```json
{
  "data_split": {
    "method": "by_stock_count",
    "total_stocks": 250,
    // ❓ 缺少 seed 記錄
  }
}
```

### 評估

#### ✅ 優點
1. **seed 已設定**: `seed: 42`
2. **使用 random.Random(seed)**: 正確的隔離用法
3. **確定性切分**: 理論上可復現

#### ⚠️ 潛在問題
1. **seed 未寫入 metadata**:
   - 無法從輸出反推使用的 seed
   - 若配置文件丟失，無法復現

2. **未測試復現性**:
   - 缺少哈希校驗或單元測試
   - 無法確認兩次運行 split 完全一致

3. **多處隨機性**:
   - `random.shuffle()` - 股票切分 ✅（已固定）
   - NumPy 隨機操作（若有）- ❓（需檢查）

### 建議行動 ✅ **立即採納**

#### 優先級：**高**（關鍵可復現性）

#### 實現方案

**Step 1: 記錄 seed 到 metadata**（簡單）
```python
# extract_tw_stock_data_v6.py - meta 字典
meta = {
    ...
    "data_split": {
        "method": "by_stock_count",
        "seed": SPLIT_SEED,  # 新增
        "train_stocks": len(train_stocks),
        ...
    }
}
```

**Step 2: 添加復現性測試**（推薦）
```python
# tests/test_reproducibility.py
def test_split_reproducibility():
    """測試兩次運行 split 一致"""
    # 運行兩次
    split1 = run_pipeline(seed=42)
    split2 = run_pipeline(seed=42)

    # 驗證 stock_ids 一致
    assert split1['train_symbols'] == split2['train_symbols']
    assert split1['val_symbols'] == split2['val_symbols']

    # 哈希校驗
    hash1 = hashlib.md5(str(split1['train_symbols']).encode()).hexdigest()
    hash2 = hashlib.md5(str(split2['train_symbols']).encode()).hexdigest()
    assert hash1 == hash2
```

**Step 3: 文檔化 seed 使用**
```markdown
# 可復現性保證

本專案使用固定隨機種子確保結果可復現：

- **預處理階段**: 無隨機性
- **訓練數據生成階段**: seed=42（股票切分）
- **模型訓練階段**: seed=42（PyTorch, NumPy）

重新運行時，確保：
1. 使用相同配置文件（seed=42）
2. 使用相同預處理數據
3. Python/NumPy/PyTorch 版本一致
```

#### 時程
- **立即**: 記錄 seed 到 metadata（5 分鐘）
- **本週**: 添加復現性測試
- **本週**: 驗證兩次運行 split 一致

---

## D. 零價/異常價處置一致性

### ChatGPT 建議
- **問題**: 異常價處理規則存在，但統計不完整
- **建議**:
  1. 增加異常統計字段：`zero_price_rows`, `outlier_price_rows`
  2. 寫入每日 summary.json
  3. 提供「被剔除/修正的異常比例」
  4. 若發現異常集中，加白名單/黑名單（可選）

### 現狀檢查

**preprocess_single_day.py:179-181**（異常檢測）:
```python
# 零值處理
for p, q in zip(bids_p + asks_p, bids_q + asks_q):
    if p == 0.0 and q != 0.0:
        return (sym, t, None)  # 直接丟棄
```

**preprocess_single_day.py:189-192**（價格限制）:
```python
# 價格限制檢查
prices_to_check = [p for p in bids_p + asks_p if p > 0]
if not all(within_limits(p, lower, upper) for p in prices_to_check):
    return (sym, t, None)  # 直接丟棄
```

**當前統計**（全域）:
```python
stats = {
    "total_raw_events": 0,
    "cleaned_events": 0,  # 只有總數，沒有分類
    ...
}
```

### 評估

#### ✅ 優點
1. **異常檢測完整**: 零價、價格限制、價差檢查
2. **處理一致**: 全部直接丟棄（不修補）
3. **全域統計**: 有總清洗數

#### ⚠️ 不足
1. **缺少分類統計**:
   - 不知道丟了多少「零價」
   - 不知道丟了多少「超限價」
   - 不知道丟了多少「價差異常」

2. **無按股票統計**:
   - 無法識別「特定股票異常多」
   - 無法審計異常集中現象

3. **summary.json 不完整**:
   - 當前只有通過/過濾股票數
   - 沒有異常細節

### 建議行動 ✅ **採納**

#### 優先級：**中**（審計需求，易實現）

#### 實現方案

**Step 1: 擴展全域統計**
```python
# preprocess_single_day.py
stats = {
    "total_raw_events": 0,
    "cleaned_events": 0,

    # 新增：異常分類
    "rejected_zero_price": 0,
    "rejected_out_of_limit": 0,
    "rejected_bad_spread": 0,
    "rejected_trial": 0,
    "rejected_out_of_time": 0,
}
```

**Step 2: 修改 parse_line()**
```python
def parse_line(raw: str):
    ...
    # 試撮移除
    if parts[IDX_TRIAL].strip() == "1":
        stats["rejected_trial"] += 1  # 新增
        return (sym, t, None)

    # 時間窗檢查
    if not is_in_trading_window(t):
        stats["rejected_out_of_time"] += 1  # 新增
        return (sym, t, None)

    # 價差檢查
    if not spread_ok(bid1, ask1):
        stats["rejected_bad_spread"] += 1  # 新增
        return (sym, t, None)

    # 零值檢查
    for p, q in zip(...):
        if p == 0.0 and q != 0.0:
            stats["rejected_zero_price"] += 1  # 新增
            return (sym, t, None)

    # 價格限制
    if not all(within_limits(...)):
        stats["rejected_out_of_limit"] += 1  # 新增
        return (sym, t, None)
```

**Step 3: 擴展 summary.json**
```python
# preprocess_single_day.py - generate_daily_summary()
summary = {
    ...
    "data_quality": {
        "total_raw_events": stats['total_raw_events'],
        "cleaned_events": stats['cleaned_events'],
        "rejection_breakdown": {
            "trial": stats['rejected_trial'],
            "out_of_time": stats['rejected_out_of_time'],
            "bad_spread": stats['rejected_bad_spread'],
            "zero_price": stats['rejected_zero_price'],
            "out_of_limit": stats['rejected_out_of_limit'],
        },
        "rejection_rate": {
            "total": 1 - stats['cleaned_events'] / stats['total_raw_events'],
            "zero_price_pct": stats['rejected_zero_price'] / stats['total_raw_events'],
            ...
        }
    }
}
```

**Step 4: 按股票異常統計**（可選，第二階段）
```python
# 若發現異常集中
"abnormal_symbols": [
    {
        "symbol": "3048",
        "zero_price_ratio": 0.25,  # 25% 的事件是零價
        "warning": "high_zero_price_rate"
    }
]
```

#### 時程
- **本週**: 實現 Step 1-3（分類統計和 summary.json）
- **下週**: 觀察 summary.json，決定是否需要 Step 4

---

## E. 標籤與權重的邊界檢查

### ChatGPT 建議
- **問題**: 標籤映射和權重裁剪已實現，但缺少硬性檢查
- **建議**:
  1. Hard assertion：`y ∈ {0, 1, 2}`
  2. Hard assertion：`weights` 全部有限，均值 ≈ 1
  3. Neutral 監控：`Neutral < 15%` 報警
  4. 寫入 metadata 每個 split 的標籤分布

### 現狀檢查

**extract_tw_stock_data_v6.py:543-544**（標籤映射）:
```python
# 4. 轉換標籤
y_tb = tb_df["y"].map({-1: 0, 0: 1, 1: 2})
```

**extract_tw_stock_data_v6.py:547-558**（權重計算）:
```python
if config['sample_weights']['enabled']:
    w = make_sample_weight(...)
else:
    w = pd.Series(np.ones(len(y_tb)), index=y_tb.index)
```

**make_sample_weight() - 權重裁剪**:
```python
w = np.clip(w, 0.1, 5.0)  # ✅ 已裁剪
```

**當前輸出統計**:
```python
# 只有總體分布，沒有驗證
logging.info(f"  標籤分布: Down={...}, Neutral={...}, Up={...}")
```

### 評估

#### ✅ 優點
1. **標籤映射明確**: {-1,0,1} → {0,1,2}
2. **權重已裁剪**: [0.1, 5.0]
3. **有日誌輸出**: 顯示標籤分布

#### ⚠️ 不足
1. **無 assertion 檢查**:
   - 若映射失敗（NaN 標籤），不會報錯
   - 若權重異常（inf），不會報錯

2. **Neutral < 15% 無自動報警**:
   - 雖然文檔要求，但代碼未實現
   - 需要手動檢查日誌

3. **metadata 缺少詳細分布**:
   - 當前只記錄總數，沒有百分比

### 建議行動 ✅ **立即採納**

#### 優先級：**高**（數據完整性關鍵）

#### 實現方案

**Step 1: 標籤邊界檢查**
```python
# extract_tw_stock_data_v6.py - build_split_v6()

# 映射後立即檢查
y_tb = tb_df["y"].map({-1: 0, 0: 1, 1: 2})

# Hard assertion
if y_tb.isna().any():
    raise ValueError(
        f"❌ 標籤映射失敗：發現 {y_tb.isna().sum()} 個 NaN\n"
        f"   原始標籤可能不在 {{-1, 0, 1}} 範圍內"
    )

if not y_tb.isin([0, 1, 2]).all():
    invalid = y_tb[~y_tb.isin([0, 1, 2])]
    raise ValueError(
        f"❌ 標籤值異常：發現非 {{0,1,2}} 的值\n"
        f"   異常值: {invalid.unique()}"
    )
```

**Step 2: 權重邊界檢查**
```python
# 在生成樣本權重後
w = make_sample_weight(...)

# Hard assertion
if not np.isfinite(w).all():
    raise ValueError(
        f"❌ 權重包含 NaN 或 inf：{np.sum(~np.isfinite(w))} 個"
    )

# 檢查均值（應接近 1）
w_mean = w.mean()
if not (0.5 < w_mean < 2.0):
    logging.warning(
        f"⚠️ 權重均值異常: {w_mean:.3f} (預期接近 1.0)\n"
        f"   這可能導致訓練不穩定"
    )
```

**Step 3: Neutral 比例檢查**
```python
# 在每個 split 生成後
label_dist = np.bincount(y_array, minlength=3)
label_pct = label_dist / label_dist.sum() * 100

neutral_pct = label_pct[1]
if neutral_pct < 15.0:
    logging.warning(
        f"⚠️ {split_name.upper()} 集 Neutral 比例過低: {neutral_pct:.1f}%\n"
        f"   目標: 20-45%，當前: {neutral_pct:.1f}%\n"
        f"   → 建議調整 Triple-Barrier 參數或過濾閾值"
    )

if neutral_pct > 60.0:
    logging.warning(
        f"⚠️ {split_name.upper()} 集 Neutral 比例過高: {neutral_pct:.1f}%\n"
        f"   目標: 20-45%，當前: {neutral_pct:.1f}%\n"
        f"   → 建議調整 min_return 閾值"
    )
```

**Step 4: 擴展 metadata**
```python
# normalization_meta.json
"data_split": {
    "results": {
        "train": {
            "samples": 5584553,
            "label_dist": [1234567, 2345678, 2004308],
            "label_pct": [22.1, 42.0, 35.9],  # 新增
            "neutral_warning": false,  # 新增
            "weight_stats": {
                "mean": 1.02,
                "std": 0.85,
                "min": 0.1,
                "max": 5.0,
                "is_finite": true  # 新增
            }
        },
        ...
    }
}
```

#### 時程
- **立即**: 實現 Step 1-3（邊界檢查和報警）
- **本週**: 實現 Step 4（metadata 擴展）

---

## F. 計算資源與可擴展性

### ChatGPT 建議
- **問題**: 大量 symbol-day 串起來滑窗，峰值記憶體可能飆高
- **建議**:
  1. 按 symbol 流式生成、分塊 `np.savez_compressed`
  2. 使用 `float32` 降低體積

### 現狀檢查

**記憶體使用**（當前數據規模）:
```python
# 訓練集: 5,584,553 樣本
# 每樣本: (100, 20) float32 + label + weight
# 記憶體: 5.58M × 100 × 20 × 4 bytes ≈ 44.6 GB（理論峰值）
```

**當前實現**:
```python
# extract_tw_stock_data_v6.py - build_split_v6()
X_windows = []  # 累積所有樣本
y_labels = []
...
X_array = np.stack(X_windows, axis=0)  # 一次性轉換
np.savez_compressed(npz_path, X=X_array, ...)
```

**數據類型**:
```python
# preprocess_single_day.py:629
np.savez_compressed(...,
    features=features.astype(np.float32),  # ✅ 已是 float32
    ...
)
```

### 評估

#### ✅ 優點
1. **已使用 float32**: 預處理階段已轉型
2. **已使用壓縮**: `savez_compressed`
3. **當前規模可承受**: 44.6 GB 峰值，RTX 5090 有 32GB VRAM

#### ⚠️ 潛在問題
1. **未來擴展性**:
   - 若數據增加 10 倍（500 天 × 500 股票）
   - 峰值記憶體 > 400 GB → 超出 RAM 限制

2. **全部載入記憶體**:
   - `X_windows.append()` 累積所有樣本
   - 對極大數據集不適用

#### 🤔 實際需求
- **當前數據**: 10 天 × 250 股票 = 2500 symbol-days
- **預期規模**: 60 天 × 500 股票 = 30000 symbol-days (12x)
- **記憶體估算**: 44.6 GB × 12 ≈ **535 GB** 峰值

**結論**: 未來可能需要優化

### 建議行動 ⏳ **延後**（但準備方案）

#### 優先級：**低**（當前無瓶頸）

#### 實現方案（當需要時）

**方案 1: 分塊保存**（推薦）
```python
# extract_tw_stock_data_v6.py
def build_split_v6_chunked(split_name, chunk_size=1000000):
    """按 100 萬樣本分塊保存"""
    X_windows = []
    y_labels = []
    chunk_id = 0

    for sym, n_points, day_data_sorted in stock_list:
        for date, features, mids in day_data_sorted:
            # ... 生成樣本 ...

            # 達到 chunk_size 時保存
            if len(X_windows) >= chunk_size:
                save_chunk(split_name, chunk_id, X_windows, y_labels, ...)
                X_windows = []
                y_labels = []
                chunk_id += 1

    # 保存最後一塊
    save_chunk(split_name, chunk_id, X_windows, y_labels, ...)

    # 合併所有 chunks（可選）
    merge_chunks(split_name)
```

**方案 2: 流式寫入**（更高級）
```python
# 使用 zarr 或 h5py 流式寫入
import zarr

z = zarr.open('train.zarr', mode='w', shape=(0, 100, 20),
              chunks=(10000, 100, 20), dtype='float32')

for sample in generate_samples():
    z.append(sample)
```

**方案 3: memmap**（內建）
```python
# 預先分配文件映射
X_mmap = np.memmap('train_X.dat', dtype='float32',
                   mode='w+', shape=(estimated_size, 100, 20))

idx = 0
for sample in generate_samples():
    X_mmap[idx] = sample
    idx += 1
```

#### 時程
- **當前**: 不實現（無瓶頸）
- **監控**: 記憶體使用率
- **觸發條件**: 峰值記憶體 > 80% 系統 RAM
- **實現時機**: 數據規模擴大 10 倍時

---

## G. 批次器與目錄健壯性

### ChatGPT 建議
- **問題 1**: Windows batch 與 Python 路徑分隔符差異（\ vs /）
- **問題 2**: 當天無通過過濾的 symbol 時，應優雅跳過

### 現狀檢查

**問題 1: 路徑處理**

**batch_preprocess.bat**:
```bat
python scripts\preprocess_single_day.py ^
    --input data\temp\%filename% ^
    --output-dir data\preprocessed_v5_1hz ^
    --config configs\config_pro_v5_ml_optimal.yaml
```

**Python 代碼**:
```python
# extract_tw_stock_data_v6.py:332
for npz_file in sorted(glob.glob(os.path.join(daily_dir, "*", "*.npz"))):
    # ✅ 使用 os.path.join
```

**問題 2: 空數據處理**

**extract_tw_stock_data_v6.py:832-834**:
```python
if not preprocessed_data:
    logging.error("沒有可用的預處理數據！")
    return 1  # ✅ 優雅退出
```

**preprocess_single_day.py:812-814**:
```python
if not daily_stats:
    logging.warning("當天無有效數據！")
    return stats  # ✅ 返回空統計
```

### 評估

#### ✅ 優點
1. **Python 端路徑處理正確**: 全部使用 `os.path.join`
2. **空數據已處理**: 兩階段都有檢查
3. **優雅退出**: 不會崩潰

#### ⚠️ 潛在問題
1. **batch 腳本硬編碼路徑**:
   - `data\temp\` 使用反斜線
   - 跨平台兼容性差（但僅 Windows 使用）

2. **空數據返回碼不一致**:
   - 預處理階段：返回 `stats`（成功）
   - 訓練數據生成階段：返回 `1`（錯誤）
   - 應該統一為「警告」狀態碼 2

3. **summary.json 生成時機**:
   - 若所有 symbol 被過濾，是否仍生成 summary？
   - 當前：會生成空 summary

### 建議行動 ✅ **採納**

#### 優先級：**中**（Windows 環境重要）

#### 實現方案

**Step 1: 統一路徑變數**（batch 腳本）
```bat
REM batch_preprocess.bat
SET INPUT_DIR=data\temp
SET OUTPUT_DIR=data\preprocessed_v5_1hz
SET CONFIG=configs\config_pro_v5_ml_optimal.yaml

for %%f in (%INPUT_DIR%\*.txt) do (
    python scripts\preprocess_single_day.py ^
        --input "%%f" ^
        --output-dir "%OUTPUT_DIR%" ^
        --config "%CONFIG%"
)
```

**Step 2: 空數據返回碼統一**
```python
# preprocess_single_day.py - process_single_day()
if not daily_stats:
    logging.warning("⚠️ 當天無有效數據！")
    # 仍然生成空 summary
    generate_daily_summary(
        output_dir, date, [], 0.0, 'none',
        {'down': 0.33, 'neutral': 0.34, 'up': 0.33},
        0, 0
    )
    return stats  # 返回統計（視為成功）
```

**Step 3: 空數據報告增強**
```python
# extract_tw_stock_data_v6.py - load_all_preprocessed_data()
if not all_data:
    logging.error("❌ 沒有可用的預處理數據！")
    logging.info("可能原因：")
    logging.info("  1. preprocessed_dir 路徑錯誤")
    logging.info("  2. 所有股票被過濾（pass_filter=false）")
    logging.info("  3. 預處理尚未執行")
    return []
```

**Step 4: 跨平台路徑兼容性**（可選）
```python
# utils/path_helper.py (新增)
import os
from pathlib import Path

def normalize_path(path_str):
    """統一路徑分隔符（跨平台）"""
    return str(Path(path_str))

# 使用範例
input_path = normalize_path("data/temp/20250901.txt")  # Windows: data\temp\20250901.txt
```

#### 時程
- **本週**: 實現 Step 2-3（空數據處理增強）
- **下週**: 實現 Step 1（batch 腳本改進）
- **可選**: Step 4（若需要跨平台）

---

## 總結與優先級排序

### 立即實施（本週完成）✅

| 建議 | 工作量 | 影響 | 行動 |
|------|--------|------|------|
| **C. 切分穩定性** | 小（30 分鐘） | 高 | 記錄 seed 到 metadata + 測試 |
| **E. 標籤邊界檢查** | 中（2 小時） | 高 | 添加 assertion + Neutral 報警 |
| **A. ffill 品質過濾** | 中（3 小時） | 高 | 滑窗品質檢查 + 降低 ffill_limit |
| **G. 批次器健壯性** | 小（1 小時） | 中 | 空數據處理增強 |

### 短期實施（下週）⏳

| 建議 | 工作量 | 影響 | 行動 |
|------|--------|------|------|
| **D. 異常統計** | 中（3 小時） | 中 | 分類統計 + summary.json |
| **A. ffill 摘要** | 小（1 小時） | 中 | 長缺口/高 ffill 報告 |
| **C. 復現性測試** | 中（2 小時） | 高 | 單元測試 + 文檔 |

### 觀察後決定（1-2 個月）🔍

| 建議 | 工作量 | 觸發條件 | 行動 |
|------|--------|---------|------|
| **B. 動態過濾退化** | 小（1 小時） | 出現極端行情日 | 添加 IQR 檢查 |
| **F. 記憶體優化** | 大（1-2 天） | 數據規模 > 10x | 分塊保存或流式寫入 |

### 不採納 ❌

無（所有建議都有價值，只是優先級不同）

---

## 實施檢查清單

### 本週任務（預估 8 小時）

- [ ] **C.1**: 記錄 seed 到 normalization_meta.json（30 分鐘）
- [ ] **E.1**: 標籤映射 assertion（30 分鐘）
- [ ] **E.2**: 權重邊界檢查（30 分鐘）
- [ ] **E.3**: Neutral 比例報警（30 分鐘）
- [ ] **E.4**: metadata 擴展（1 小時）
- [ ] **A.1**: 降低 ffill_limit 120→60 秒（15 分鐘）
- [ ] **A.2**: 滑窗 ffill 品質過濾（2 小時）
- [ ] **G.1**: 空數據返回碼統一（30 分鐘）
- [ ] **G.2**: 空數據報告增強（30 分鐘）
- [ ] **測試**: 運行完整流水線驗證（1 小時）

### 下週任務（預估 6 小時）

- [ ] **D.1**: 擴展全域異常統計（1 小時）
- [ ] **D.2**: 修改 parse_line() 分類計數（1 小時）
- [ ] **D.3**: summary.json 異常細節（1 小時）
- [ ] **A.3**: 長缺口/高 ffill 摘要報告（1 小時）
- [ ] **C.2**: 復現性單元測試（2 小時）

### 觀察指標

- [ ] **B**: 監控每日 IQR 值（每週檢查 summary.json）
- [ ] **F**: 監控記憶體峰值使用率（每次訓練檢查）

---

**分析完成日期**: 2025-10-21
**分析者**: Claude (Sonnet 4.5)
**下一步**: 等待用戶確認採納項目後開始實施
