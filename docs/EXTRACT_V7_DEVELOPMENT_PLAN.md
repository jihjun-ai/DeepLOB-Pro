# extract_tw_stock_data_v7.py 开发改版计划（簡化版）

**版本**: v7.0.0-simplified
**文档日期**: 2025-10-23
**目标**: 簡化數據處理流程，專注數據組織而非計算

---

## 📋 目录

1. [改版背景](#改版背景)
2. [核心理念](#核心理念)
3. [技術設計](#技術設計)
4. [性能預期](#性能預期)
5. [實施風險](#實施風險)
6. [測試策略](#測試策略)

---

## 改版背景

### V6 架構的問題

根據實際分析，V6 存在以下**重複計算問題**：

| 步驟 | V6 做法 | 預處理已完成 | 問題 |
|-----|---------|------------|------|
| **標籤計算** | ⚠️⚠️⚠️ 重新計算 | ✅ `labels` 字段已存在 | **嚴重重複** |
| **波動率計算** | ⚠️ 重新計算 | ✅ 標籤已生成，不需波動率 | **不必要** |
| **權重計算** | ⚠️ 每次計算 | ✅ `weight_strategies` 已有 11 種 | **重複工作** |
| **數據清洗** | ⚠️ Z-Score 標準化 | ✅ `features` 已標準化 | **重複工作** |

### 核心發現

通過檢查實際 NPZ 文件（`data/preprocessed_swing/daily/20250901/0050.npz`）：

```
✅ features (15957, 20)  - 已標準化的 LOB 特徵
✅ labels (15957,)       - 值為 {0.0, 1.0} 已計算完成
✅ metadata.weight_strategies - 包含 5+ 種預計算策略
✅ metadata.label_preview - 包含標籤統計信息
```

**結論**: 預處理階段已完成所有重要計算，V6 的"重新計算"是**完全不必要的重複工作**。

### V6 → V7 演進目標

**核心理念**: "預處理已完成，V7 只做數據組織"

```
預處理階段（preprocess_single_day.py）:
  ✅ 數據清洗與聚合
  ✅ Z-Score 標準化
  ✅ 標籤計算（Triple-Barrier / Trend）
  ✅ 權重策略計算（11 種）
  ✅ 統計信息記錄

V7 階段（extract_tw_stock_data_v7.py）:
  ✅ 讀取預處理 NPZ（直接使用 features, labels）
  ✅ 數據選擇（dataset_selection.json 或配置過濾）
  ✅ 滑動窗口生成（100 timesteps）
  ✅ 按股票劃分（train/val/test = 70/15/15）
  ✅ 輸出 NPZ（與 V6 格式兼容）

  ❌ 不重新計算標籤
  ❌ 不重新計算波動率
  ❌ 不重新計算權重
  ❌ 不重新標準化
```

---

## 核心理念

### ❌ V7 不做的事（已在預處理完成）

1. **❌ 標籤計算**
   - 預處理已完成 Triple-Barrier 或趨勢標籤
   - NPZ 的 `labels` 字段直接可用

2. **❌ 波動率計算**
   - 標籤已生成，不需要再算波動率
   - 如需修改標籤，應重跑預處理（不是 V7）

3. **❌ 權重計算**
   - 預處理已計算 11 種權重策略
   - 直接從 `metadata.weight_strategies` 讀取

4. **❌ 數據清洗**
   - 預處理已完成 Z-Score 標準化
   - `features` 字段已是標準化結果

5. **❌ 參數匹配檢查**
   - 不做"智能重用"邏輯（過於複雜）
   - 如需不同參數，重跑預處理

### ✅ V7 只做的事

1. **讀取預處理 NPZ**
   - 載入 `features`, `labels`, `metadata`
   - 驗證版本（強制 v2.0+）

2. **數據選擇**（新增功能 ⭐⭐⭐⭐⭐）
   - 支持 `dataset_selection.json`（優先級 1）
   - 或使用配置過濾（優先級 2）

3. **滑動窗口生成**
   - 生成 (100, 20) 時間序列窗口
   - 標籤轉換：{-1, 0, 1} → {0, 1, 2}

4. **按股票劃分**
   - train/val/test = 70/15/15

5. **輸出 NPZ**
   - 與 V6 格式完全兼容
   - 確保訓練流程無需修改

---

## 技術設計

### 簡化後的函數清單

#### 新增函數

1. **`read_dataset_selection_json(json_path: str) -> Dict`**
   - 讀取 dataset_selection.json
   - 返回 file_list（每個元素: {date, symbol, file_path}）

2. **`filter_data_by_selection(all_data, config) -> List`**
   - 優先級 1: 使用 dataset_selection.json 過濾
   - 優先級 2: 使用配置過濾（start_date, num_days, symbols）

#### 修改函數

1. **`load_preprocessed_npz(npz_path) -> Tuple`**
   ```
   V6 返回: (features, mids, bucket_mask, meta)
   V7 返回: (features, labels, meta)

   改動:
   - 新增返回 labels
   - 移除 mids（不需要計算標籤）
   - 移除 bucket_mask（不需要）
   ```

2. **`sliding_windows_v6()` → `sliding_windows_v7()`**
   ```
   簡化內容:
   - 移除標籤計算邏輯（tb_labels, trend_labels）
   - 移除波動率計算邏輯（ewma_vol）
   - 直接使用預處理的 labels
   - 從 metadata 讀取權重（不重新計算）
   ```

3. **`load_all_preprocessed_data()`**
   ```
   新增檢查:
   - 強制 NPZ v2.0+（必須有 labels 字段）
   - 舊版 NPZ 直接報錯，提示重跑預處理
   ```

#### 移除函數

以下 V6 函數在 V7 中**完全移除**：

```python
# ❌ 移除: 標籤計算函數（預處理已完成）
def tb_labels(...)  # Triple-Barrier 標籤
def trend_labels(...)  # 趨勢標籤

# ❌ 移除: 波動率計算函數（不需要）
def ewma_vol(...)  # EWMA 波動率

# ❌ 移除: 參數匹配檢查（過於複雜）
def check_label_compatibility(...)
def load_volatility_from_metadata(...)

# ⚪ 保留但簡化: 權重函數（改為讀取 metadata）
def get_sample_weights_from_metadata(...)
```

**代碼量變化**: V6 約 1800 行 → V7 約 1200 行（**減少 33%**）

---

### 配置文件變化

```yaml
# configs/config_pro_v7_optimal.yaml

# V7 新增: 數據選擇（⭐⭐⭐⭐⭐ 核心功能）
data_selection:
  # 選項 1: 使用 JSON 文件（優先級最高）
  json_file: "results/dataset_selection_auto.json"

  # 選項 2: 使用配置過濾（後備方案）
  start_date: "20250901"           # 起始日期（null = 全部）
  end_date: null                   # 結束日期（null = 最晚）
  num_days: 10                     # 處理天數（null = 全部）
  symbols: null                    # 指定股票（null = 全部）
  sample_ratio: 1.0                # 採樣比例

# V7 簡化: 權重策略（直接從 metadata 讀取）
sample_weights:
  enabled: true
  strategy: "effective_num_0999"   # 從預計算的 11 種策略選擇

# V7 移除的配置（不再需要）
# ❌ label_reuse（不做智能重用）
# ❌ triple_barrier 參數（預處理決定）
# ❌ volatility 參數（預處理決定）
# ❌ sample_generation.target_dist（使用 dataset_selection.json 控制）
```

---

### dataset_selection.json 格式

V7 的**核心新功能**是支持精確控制數據集。

**格式**:
```json
{
  "metadata": {
    "creation_date": "2025-10-23T10:30:00",
    "preprocessed_dir": "data/preprocessed_v5",
    "total_samples": 2500000,
    "label_distribution": {
      "down": 750000,
      "neutral": 1000000,
      "up": 750000
    }
  },
  "file_list": [
    {
      "date": "20250901",
      "symbol": "2330",
      "file_path": "data/preprocessed_v5/daily/20250901/2330.npz",
      "samples": 15000,
      "label_dist": {"down": 4500, "neutral": 6000, "up": 4500}
    },
    {
      "date": "20250901",
      "symbol": "2454",
      "file_path": "data/preprocessed_v5/daily/20250901/2454.npz",
      "samples": 12000,
      "label_dist": {"down": 3600, "neutral": 4800, "up": 3600}
    }
  ]
}
```

**生成方式**:
```bash
# 使用 analyze_label_distribution.py 生成
python scripts/analyze_label_distribution.py \
    --preprocessed-dir data/preprocessed_v5 \
    --mode smart_recommend \
    --start-date 20250901 \
    --target-dist "0.30,0.40,0.30" \
    --min-samples 50000 \
    --output results/dataset_selection_auto.json
```

---

### 輸出 Metadata 更新

```json
{
  "format": "deeplob_v7_simplified",
  "version": "7.0.0",
  "creation_date": "2025-10-23T10:30:00",

  "data_source": {
    "preprocessed_version": "v2.0",
    "selection_method": "json_file",
    "json_file": "results/dataset_selection_auto.json",
    "npz_count": 1785,
    "date_range": ["20250901", "20250912"],
    "symbols_count": 195
  },

  "label_source": {
    "method": "preprocessed",
    "note": "直接使用預處理 NPZ 的 labels 字段，未重新計算"
  },

  "sample_weights": {
    "method": "preprocessed",
    "strategy": "effective_num_0999",
    "source": "metadata.weight_strategies"
  },

  "performance": {
    "total_time_seconds": 120,
    "v6_estimated_time": 600,
    "speedup_note": "無標籤/波動率計算，速度提升 5x"
  },

  "data_split": {
    "method": "by_symbol",
    "train": {
      "samples": 5584553,
      "symbols": 137,
      "label_dist": [1678365, 2233111, 1673077],
      "label_pct": [30.1, 40.0, 29.9]
    },
    "val": {...},
    "test": {...}
  }
}
```

---

## 性能預期

### 時間節省

假設處理 **195 檔股票 × 10 天 = 1,950 個 symbol-day**：

| 步驟 | V6 耗時 | V7 耗時 | 節省 | 說明 |
|-----|---------|---------|------|------|
| 數據載入 | 30秒 | 35秒 | -5秒 | 載入 labels 字段 |
| Z-Score 標準化 | 60秒 | 0秒 | **+60秒** | 預處理已完成 |
| **波動率計算** | **60秒** | **0秒** | **+60秒** | 不需要 |
| **Triple-Barrier 標籤** | **300秒** | **0秒** | **+300秒** | 直接使用 NPZ |
| 樣本權重 | 30秒 | 5秒 | +25秒 | 讀取 metadata |
| 滑窗生成 | 120秒 | 100秒 | +20秒 | 簡化邏輯 |
| **總計** | **600秒 (10分鐘)** | **140秒 (2.3分鐘)** | **+460秒** | **節省 77%** |

### 與 V6 對比

| 指標 | V6 | V7 簡化版 | 改善 |
|-----|----|----|------|
| 處理時間 | 10分鐘 | 2.3分鐘 | **-77%** |
| 代碼行數 | 1800 行 | 1200 行 | **-33%** |
| 複雜度 | 高 | 低 | ⬇️ |
| 維護性 | 中 | 高 | ⬆️ |
| 可靠性 | 中 | 高 | ⬆️ |

### 為何更快？

1. **不重新計算標籤**: 節省 300 秒（50%）
2. **不重新計算波動率**: 節省 60 秒（10%）
3. **不重新標準化**: 節省 60 秒（10%）
4. **簡化權重讀取**: 節省 25 秒（4%）
5. **代碼更簡潔**: 節省 15 秒（3%）

**總節省**: **77%**

---

## 實施風險

### 風險 1: NPZ 版本不兼容 ⚠️

**描述**: 用戶使用舊版 NPZ（v1.0，無 labels 字段）

**影響**: 程序報錯，無法運行

**緩解措施**:
```python
# 清晰的錯誤提示
if 'labels' not in npz_data:
    raise ValueError(
        f"❌ NPZ 版本過舊（v1.0）\n"
        f"   V7 要求 v2.0+ NPZ（含 labels 字段）\n"
        f"   解決方法:\n"
        f"   1. 運行: scripts\\batch_preprocess.bat\n"
        f"   2. 確保 NPZ 含有 'labels' 字段"
    )
```

**風險等級**: 中 ⚠️（可控）

### 風險 2: dataset_selection.json 格式錯誤 ⚠️

**描述**: JSON 文件格式不正確或路徑錯誤

**影響**: 無法載入數據

**緩解措施**:
- 詳細格式驗證
- 清晰錯誤提示
- 提供範例文件

**風險等級**: 低 ⚪

### 風險 3: 標籤精度差異 ⚪

**描述**: 直接使用預處理標籤，無法動態調整參數

**影響**: 如需不同標籤參數，必須重跑預處理

**緩解措施**:
- 文檔說明清楚
- 預處理速度快（4 分鐘/天）

**風險等級**: 低 ⚪

---

## 測試策略

### Phase 1: 單元測試

#### 測試 1.1: dataset_selection.json 讀取

**目標**: 驗證 JSON 讀取與驗證

**測試案例**:
- 正常 JSON 文件
- 缺少必要字段
- 文件不存在
- 格式錯誤

#### 測試 1.2: 標籤載入

**目標**: 驗證從 NPZ 讀取 labels

**測試案例**:
- 正常載入（值為 {-1, 0, 1}）
- 長度不匹配
- 異常值檢測
- 缺少 labels 字段（舊版 NPZ）

#### 測試 1.3: 權重讀取

**目標**: 驗證從 metadata 讀取權重策略

**測試案例**:
- 正常讀取（11 種策略之一）
- 策略不存在
- metadata 格式錯誤

### Phase 2: 整合測試

#### 測試 2.1: 小數據集完整流程

**步驟**:
```bash
# 創建測試配置
# config_v7_test.yaml:
#   data_selection:
#     start_date: "20250901"
#     num_days: 2

python scripts/extract_tw_stock_data_v7.py \
    --preprocessed-dir ./data/preprocessed_v5 \
    --output-dir ./data/processed_v7_test \
    --config ./configs/config_v7_test.yaml
```

**驗證**:
- ✅ 程序正常運行（無錯誤）
- ✅ 生成 train/val/test.npz
- ✅ 標籤分布正確
- ✅ 處理時間 < 5 分鐘

#### 測試 2.2: dataset_selection.json 測試

**步驟**:
```bash
# 1. 生成 JSON
python scripts/analyze_label_distribution.py \
    --preprocessed-dir data/preprocessed_v5 \
    --mode smart_recommend \
    --start-date 20250901 \
    --target-dist "0.30,0.40,0.30" \
    --min-samples 50000 \
    --output results/dataset_selection_test.json

# 2. 使用 JSON 運行 V7
python scripts/extract_tw_stock_data_v7.py \
    --preprocessed-dir ./data/preprocessed_v5 \
    --output-dir ./data/processed_v7_json \
    --config ./configs/config_v7_json.yaml
```

**驗證**:
- ✅ 只處理 JSON 指定的文件
- ✅ 標籤分布符合 JSON 預期
- ✅ 日誌輸出正確

#### 測試 2.3: 對比 V6 vs V7 一致性

**步驟**:
```python
# 使用相同預處理數據運行 V6 和 V7
# 對比輸出標籤分布

v6_data = np.load('data/processed_v6/npz/stock_embedding_train.npz')
v7_data = np.load('data/processed_v7/npz/stock_embedding_train.npz')

v6_dist = np.bincount(v6_data['y'], minlength=3) / len(v6_data['y'])
v7_dist = np.bincount(v7_data['y'], minlength=3) / len(v7_data['y'])

# 標籤分布應一致（因為都用預處理的 labels）
assert np.allclose(v6_dist, v7_dist, atol=0.01)
```

**驗證**:
- ✅ 標籤分布一致（誤差 < 1%）
- ✅ 樣本數量一致
- ✅ 特徵維度一致

### Phase 3: 完整測試

#### 測試 3.1: 完整數據集

**步驟**:
```bash
# 完整 195 檔 × 10 天
python scripts/extract_tw_stock_data_v7.py \
    --preprocessed-dir ./data/preprocessed_v5 \
    --output-dir ./data/processed_v7 \
    --config ./configs/config_pro_v7_optimal.yaml
```

**驗證**:
- ✅ 處理時間 < 5 分鐘（V6 需 10 分鐘）
- ✅ 標籤分布符合預期
- ✅ 無錯誤或警告

#### 測試 3.2: 訓練驗證

**步驟**:
```bash
# 使用 V7 數據訓練 DeepLOB
python scripts/train_deeplob_generic.py \
    --data-dir ./data/processed_v7/npz \
    --config ./configs/deeplob_config.yaml
```

**驗證**:
- ✅ 訓練正常運行
- ✅ 準確率 > 65%（基準）
- ✅ 無異常錯誤

---

## 總結

### V7 簡化版核心特性

✅ **代碼簡單**: 移除 500+ 行標籤/波動率計算代碼（-33%）
✅ **速度更快**: 不重複計算，時間節省 77%（10分鐘 → 2.3分鐘）
✅ **更可靠**: 不重複計算，避免不一致
✅ **更靈活**: 支持 dataset_selection.json 精確控制數據集
✅ **易維護**: 邏輯清晰，專注數據組織

### V7 與 V6 對比

| 特性 | V6 | V7 簡化版 |
|-----|----|----|
| 標籤來源 | 重新計算 | 預處理 NPZ ✅ |
| 波動率計算 | 重新計算 | 不需要 ✅ |
| 權重計算 | 每次計算 | 讀取 metadata ✅ |
| 數據選擇 | 配置過濾 | JSON + 配置 ✅ |
| 處理時間 | 10 分鐘 | 2.3 分鐘 ✅ |
| 代碼行數 | 1800 行 | 1200 行 ✅ |
| 複雜度 | 高 | 低 ✅ |

### 預期效益

| 指標 | V6 | V7 簡化版 | 改善 |
|-----|----|----|------|
| 處理時間 | 10分鐘 | 2.3分鐘 | **-77%** |
| 參數調整成本 | 重跑預處理 + V6 | 僅重跑預處理 | 更清晰 |
| 代碼維護 | 複雜 | 簡單 | ⬆️ |
| 可靠性 | 中 | 高 | ⬆️ |

### 下一步

詳見 [EXTRACT_V7_TODO.md](EXTRACT_V7_TODO.md)（簡化版實施計劃）

---

**文檔版本**: v7.0.0-simplified
**最後更新**: 2025-10-23
**狀態**: ✅ 設計完成（簡化版），待實施
