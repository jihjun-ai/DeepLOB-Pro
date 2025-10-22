# V6 雙階段資料處理流程指南

**版本**: 6.0.0
**更新日期**: 2025-10-21
**適用範圍**: DeepLOB-Pro 台股高頻交易系統

---

## 📋 目錄

1. [概述](#概述)
2. [架構設計](#架構設計)
3. [快速開始](#快速開始)
4. [階段1：預處理](#階段1預處理)
5. [階段2：訓練數據生成](#階段2訓練數據生成)
6. [核心特性：動態過濾](#核心特性動態過濾)
7. [輸出格式說明](#輸出格式說明)
8. [常見問題](#常見問題)
9. [性能對比](#性能對比)

---

## 概述

### 問題背景

**V5 單階段架構的痛點**：

❌ **固定過濾閾值**：無法適應每天不同的市場特性
❌ **參數調整成本高**：修改 Triple-Barrier 參數需重新處理全部數據（30-60 分鐘）
❌ **無法增量處理**：新增一天數據需重跑所有歷史
❌ **標籤分布不穩定**：持平標籤比例可能過高（>60%）或過低（<20%）

### V6 解決方案

✅ **動態自適應過濾**：每天自動分析並選擇最佳閾值
✅ **快速參數測試**：調整 TB 參數僅需 5-10 分鐘（vs 原本 45 分鐘）
✅ **增量處理**：新增數據只需處理新日期
✅ **標籤分布穩定**：目標維持在 30% / 40% / 30% (Down/Neutral/Up)

---

## 架構設計

### 雙階段處理流程

```
階段 1: 預處理（Data Preprocessing）
  輸入：原始 TXT 檔案（按日）
  處理：清洗 → 聚合 → 統計 → 動態過濾 → 保存 NPZ
  輸出：中間格式 NPZ + 每日摘要

階段 2: 訓練數據生成（Feature Engineering）
  輸入：預處理 NPZ 檔案
  處理：Z-Score → 波動率 → Triple-Barrier → 滑窗
  輸出：訓練 NPZ (train/val/test)
```

### 關鍵優勢

| 特性 | V5 (單階段) | V6 (雙階段) |
|------|------------|------------|
| **過濾策略** | 固定閾值 | 每天動態決策 |
| **參數調整** | 45 分鐘 | 8 分鐘 (82% ↓) |
| **新增數據** | 45 分鐘 | 4 分鐘 (91% ↓) |
| **標籤分布** | 不穩定 | 目標導向 |
| **可追溯性** | 低 | 高（每天決策記錄） |

---

## 快速開始

### 前置需求

1. **Conda 環境**：`deeplob-pro`
2. **配置文件**：`configs/config_pro_v5_ml_optimal.yaml`
3. **原始數據**：`data/temp/*.txt`

### 完整流程（3 步驟）

```bash
# 步驟 1: 啟動 conda 環境
conda activate deeplob-pro

# 步驟 2: 批次預處理所有歷史數據（階段1）
scripts\batch_preprocess.bat

# 步驟 3: 生成訓練數據（階段2）
python scripts\extract_tw_stock_data_v6.py ^
    --preprocessed-dir .\data\preprocessed_v5 ^
    --output-dir .\data\processed_v6 ^
    --config .\configs\config_pro_v5_ml_optimal.yaml
```

### 輸出結果

```
data/
├── preprocessed_v5/              # 階段1 輸出
│   ├── daily/
│   │   ├── 20250901/
│   │   │   ├── 2330.npz          # 個股預處理數據
│   │   │   ├── 2454.npz
│   │   │   └── summary.json      # 當天摘要
│   │   └── ...
│   └── reports/
│       ├── overall_summary.json   # 整體統計
│       ├── daily_statistics.csv
│       └── filter_decisions.csv
│
└── processed_v6/                 # 階段2 輸出
    └── npz/
        ├── stock_embedding_train.npz
        ├── stock_embedding_val.npz
        ├── stock_embedding_test.npz
        └── normalization_meta.json
```

---

## 階段1：預處理

### 功能說明

**腳本**: `scripts/preprocess_single_day.py`

**核心任務**：
1. 讀取單一天的 TXT 檔案
2. 解析、清洗、去重（繼承 V5 邏輯）
3. 10 事件聚合
4. 計算每個 symbol 的日內統計（震盪幅度、漲跌幅）
5. **動態決定當天的過濾閾值**（基於目標標籤分布）
6. 應用過濾並保存為 NPZ
7. 生成當天摘要報告

### 使用方式

#### 單檔處理（測試用）

```bash
python scripts\preprocess_single_day.py ^
    --input .\data\temp\20250901.txt ^
    --output-dir .\data\preprocessed_v5 ^
    --config .\configs\config_pro_v5_ml_optimal.yaml
```

#### 批次處理（生產用）

```bash
# Windows
scripts\batch_preprocess.bat

# Linux/Mac
bash scripts/batch_preprocess.sh
```

### 輸出說明

#### 個股 NPZ 檔案

**路徑**: `data/preprocessed_v5/daily/20250901/2330.npz`

**內容**:
```python
{
    "features": np.ndarray,  # (T, 20) - LOB 特徵矩陣
    "mids": np.ndarray,      # (T,) - 中間價序列
    "metadata": {
        # 基本資訊
        "symbol": "2330",
        "date": "20250901",
        "n_points": 2850,

        # 日內統計
        "range_pct": 0.0234,      # 震盪幅度
        "return_pct": 0.0089,     # 漲跌幅
        "high": 587.5,
        "low": 574.2,
        "open": 575.0,
        "close": 580.1,

        # 過濾資訊
        "pass_filter": true,           # 是否通過過濾
        "filter_threshold": 0.0050,    # 當天閾值
        "filter_method": "adaptive_P25", # 閾值選擇方法

        # 處理資訊
        "processed_at": "2025-10-21T10:30:00",
        "raw_events": 28500,
        "aggregated_points": 2850
    }
}
```

#### 每日摘要

**路徑**: `data/preprocessed_v5/daily/20250901/summary.json`

**內容**:
```json
{
    "date": "20250901",
    "total_symbols": 195,
    "passed_filter": 156,
    "filtered_out": 39,
    "filter_threshold": 0.0050,
    "filter_method": "adaptive_P25",

    "volatility_distribution": {
        "min": 0.0012,
        "max": 0.0892,
        "mean": 0.0234,
        "P10": 0.0045,
        "P25": 0.0050,
        "P50": 0.0078,
        "P75": 0.0123
    },

    "predicted_label_dist": {
        "down": 0.32,
        "neutral": 0.38,
        "up": 0.30
    },

    "top_volatile": [
        {"symbol": "2454", "range_pct": 0.0892},
        {"symbol": "3008", "range_pct": 0.0765}
    ]
}
```

---

## 階段2：訓練數據生成

### 功能說明

**腳本**: `scripts/extract_tw_stock_data_v6.py`

**核心任務**：
1. 讀取所有預處理 NPZ 檔案
2. 過濾掉未通過過濾的 symbol-day
3. Z-Score 正規化（基於訓練集）
4. EWMA 波動率估計
5. Triple-Barrier 標籤生成
6. 樣本權重計算
7. 滑窗生成樣本
8. 按股票切分 70/15/15
9. 保存訓練 NPZ

### 使用方式

#### 基本使用

```bash
python scripts\extract_tw_stock_data_v6.py ^
    --preprocessed-dir .\data\preprocessed_v5 ^
    --output-dir .\data\processed_v6 ^
    --config .\configs\config_pro_v5_ml_optimal.yaml
```

#### 測試不同 TB 參數（快速）

修改配置文件後直接重新執行：

```bash
# 1. 修改 config（例如調整 pt_multiplier）
# configs/config_pro_v5_test.yaml:
#   triple_barrier:
#     pt_multiplier: 3.0  # 從 3.5 降到 3.0
#     min_return: 0.002   # 從 0.0015 提高到 0.002

# 2. 快速生成新數據（僅需 5-10 分鐘）
python scripts\extract_tw_stock_data_v6.py ^
    --preprocessed-dir .\data\preprocessed_v5 ^
    --output-dir .\data\processed_v6_test ^
    --config .\configs\config_pro_v5_test.yaml
```

### 輸出說明

#### 訓練 NPZ 檔案

**格式**: 與 V5 完全相同，向後兼容

```python
# stock_embedding_train.npz
{
    "X": np.ndarray,      # (N, 100, 20) - 特徵矩陣
    "y": np.ndarray,      # (N,) - 標籤 {0, 1, 2}
    "weights": np.ndarray, # (N,) - 樣本權重
    "stock_ids": np.ndarray # (N,) - 股票代碼
}
```

#### Metadata

**路徑**: `data/processed_v6/npz/normalization_meta.json`

**關鍵欄位**:
```json
{
    "format": "deeplob_v6",
    "version": "6.0.0",

    "data_split": {
        "results": {
            "train": {
                "samples": 5584553,
                "label_dist": [1678365, 2233111, 1673077],
                "label_pct": [30.1%, 40.0%, 29.9%]  // ✅ 接近目標
            }
        }
    }
}
```

---

## 核心特性：動態過濾

### 動態閾值決策演算法

**函數**: `determine_adaptive_threshold()`

**策略**：

```
For 當天所有 symbol:
    計算波動率分位數: P10, P25, P50, P75

For each 候選閾值:
    模擬過濾: 保留 range_pct >= 閾值 的 symbol
    估計標籤分布: estimate_label_distribution()
    計算與目標分布的距離: MSE(predicted, target)

選擇距離最小的閾值
```

### 標籤分布估計

**簡化規則**（實際由 Triple-Barrier 決定）：

```python
if abs(return_pct) < min_return:
    label = "neutral"  # 持平
elif return_pct > 0:
    label = "up"       # 上漲
else:
    label = "down"     # 下跌
```

### 目標分布

```yaml
target_label_dist:
  down: 0.30      # 30%
  neutral: 0.40   # 40%
  up: 0.30        # 30%
```

### 實際案例

**20250901 的決策過程**：

```
候選閾值測試:
  P10 (0.0045): 預測分布 [25%, 55%, 20%] → 距離 0.035 ❌
  P25 (0.0050): 預測分布 [32%, 38%, 30%] → 距離 0.001 ✅
  P50 (0.0078): 預測分布 [40%, 20%, 40%] → 距離 0.080 ❌
  P75 (0.0123): 預測分布 [42%, 15%, 43%] → 距離 0.125 ❌

選擇: P25 = 0.0050 (0.50%)
原因: 預測分布最接近目標 (30/40/30)
```

---

## 輸出格式說明

### 目錄結構

```
DeepLOB-Pro/
├── data/
│   ├── temp/                        # 原始 TXT
│   │   ├── 20250901.txt
│   │   └── ...
│   │
│   ├── preprocessed_v5/             # 階段1 輸出
│   │   ├── daily/
│   │   │   ├── 20250901/
│   │   │   │   ├── 2330.npz
│   │   │   │   ├── 2454.npz
│   │   │   │   ├── ...
│   │   │   │   └── summary.json
│   │   │   └── ...
│   │   └── reports/
│   │       ├── overall_summary.json
│   │       ├── daily_statistics.csv
│   │       └── filter_decisions.csv
│   │
│   └── processed_v6/                # 階段2 輸出
│       └── npz/
│           ├── stock_embedding_train.npz
│           ├── stock_embedding_val.npz
│           ├── stock_embedding_test.npz
│           └── normalization_meta.json
│
└── scripts/
    ├── preprocess_single_day.py     # 階段1 腳本
    ├── batch_preprocess.bat         # 批次處理
    ├── extract_tw_stock_data_v6.py  # 階段2 腳本
    ├── generate_preprocessing_report.py
    └── test_preprocess.bat          # 測試腳本
```

### 磁碟空間需求

| 類型 | 大小估算 | 說明 |
|------|---------|------|
| 原始 TXT (10 天) | 12 GB | 每天 ~1.2 GB |
| 預處理 NPZ | 2.9 GB | 壓縮後約 30% |
| 訓練 NPZ | 1.5 GB | train/val/test 共 3 檔 |
| **總計** | **16.4 GB** | 增加 22%，換取 82% 時間節省 |

---

## 常見問題

### Q1: 如何測試單一檔案？

```bash
# 使用測試腳本
scripts\test_preprocess.bat

# 或手動執行
conda activate deeplob-pro
python scripts\preprocess_single_day.py ^
    --input .\data\temp\20250901.txt ^
    --output-dir .\data\preprocessed_v5_test ^
    --config .\configs\config_pro_v5_ml_optimal.yaml
```

### Q2: 如何檢查預處理結果？

```bash
# 查看每日摘要
type data\preprocessed_v5\daily\20250901\summary.json

# 查看整體報告
type data\preprocessed_v5\reports\overall_summary.json
```

### Q3: 如何快速調整 Triple-Barrier 參數？

```bash
# 1. 修改配置文件（僅需改 triple_barrier 區塊）
# 2. 重新執行階段2（無需重跑階段1）
python scripts\extract_tw_stock_data_v6.py ^
    --preprocessed-dir .\data\preprocessed_v5 ^
    --output-dir .\data\processed_v6_new ^
    --config .\configs\config_new.yaml

# 預計時間：5-10 分鐘（vs 原本 45 分鐘）
```

### Q4: 如何處理新增數據？

```bash
# 1. 僅預處理新日期（例如 20250913）
conda activate deeplob-pro
python scripts\preprocess_single_day.py ^
    --input .\data\temp\20250913.txt ^
    --output-dir .\data\preprocessed_v5 ^
    --config .\configs\config_pro_v5_ml_optimal.yaml

# 2. 重新執行階段2（會自動載入所有預處理數據，包含新日期）
python scripts\extract_tw_stock_data_v6.py ^
    --preprocessed-dir .\data\preprocessed_v5 ^
    --output-dir .\data\processed_v6 ^
    --config .\configs\config_pro_v5_ml_optimal.yaml
```

### Q5: 標籤分布仍不理想怎麼辦？

**檢查報告**:
```bash
type data\preprocessed_v5\reports\overall_summary.json
```

**調整策略**:

| 問題 | 解決方案 |
|------|---------|
| Neutral > 50% | 提高 `min_return` 或降低 `pt_multiplier` |
| Neutral < 20% | 降低 `min_return` 或提高 `pt_multiplier` |
| 過濾太多 (>60%) | 檢查 `intraday_volatility_filter` 設定 |
| 過濾太少 (<10%) | 動態過濾已自動調整，無需干預 |

### Q6: V6 與 V5 的訓練數據兼容嗎？

**完全兼容**！V6 輸出格式與 V5 完全相同：

```python
# V5 訓練腳本無需修改
python scripts/train_deeplob_generic.py \
    --data-dir ./data/processed_v6/npz \
    --config configs/deeplob_config.yaml
```

---

## 性能對比

### 時間效率

| 場景 | V5 | V6 | 改善 |
|------|----|----|------|
| 首次處理 (10 天) | 45 min | 30 + 8 min | +15% |
| 調整 TB 參數 | 45 min | **8 min** | **82% ↓** |
| 新增 1 天數據 | 45 min | **4 min** | **91% ↓** |
| 測試 5 組參數 | 225 min | **70 min** | **69% ↓** |

### 標籤分布穩定性

**V5（固定閾值 0.0005）**:
- 20250901: [28%, 52%, 20%] ❌ Neutral 過高
- 20250902: [35%, 25%, 40%] ⚠️ Neutral 過低
- 波動：±15%

**V6（動態閾值）**:
- 20250901: [32%, 38%, 30%] ✅
- 20250902: [31%, 41%, 28%] ✅
- 波動：±3%

### 可追溯性

**V5**: 無法查詢為何某天過濾這麼多股票

**V6**: 完整記錄每天的決策過程
```bash
type data\preprocessed_v5\reports\filter_decisions.csv
# 可查看每天的閾值選擇理由
```

---

## 總結

### V6 核心優勢

1. **動態適應** → 每天自動調整過濾參數
2. **效率提升** → 參數調整快 82%
3. **穩定標籤** → 維持目標分布 30/40/30
4. **完全兼容** → 無需修改訓練代碼
5. **可追溯** → 每天決策過程可查

### 推薦工作流程

```bash
# 步驟 1: 首次批次預處理（30 分鐘）
conda activate deeplob-pro
scripts\batch_preprocess.bat

# 步驟 2: 生成訓練數據（8 分鐘）
python scripts\extract_tw_stock_data_v6.py ^
    --preprocessed-dir .\data\preprocessed_v5 ^
    --output-dir .\data\processed_v6 ^
    --config .\configs\config_pro_v5_ml_optimal.yaml

# 步驟 3: 檢查標籤分布
type data\processed_v6\npz\normalization_meta.json

# 步驟 4: 開始訓練
python scripts\train_deeplob_generic.py ^
    --data-dir .\data\processed_v6\npz ^
    --config .\configs\deeplob_config.yaml
```

### 下一步

- 閱讀: [VOLATILITY_FILTER_GUIDE.md](VOLATILITY_FILTER_GUIDE.md)
- 訓練: [1.DeepLOB 台股模型訓練最終報告.md](1.DeepLOB 台股模型訓練最終報告.md)
- SB3: [SB3_IMPLEMENTATION_REPORT.md](SB3_IMPLEMENTATION_REPORT.md)

---

**文檔版本**: 1.0
**最後更新**: 2025-10-21
**維護者**: DeepLOB-Pro Team
