# extract_tw_stock_data_v5.py 使用說明

## 📋 概述

`extract_tw_stock_data_v5.py` 是 DeepLOB-Pro 專案的**台股數據預處理工具**（V5 專業版），負責將原始台股 LOB 數據轉換為 DeepLOB 模型可用的訓練格式。

**版本**：v5.0.0
**更新日期**：2025-10-19
**專案文檔**：[CLAUDE.md](../CLAUDE.md)

---

## 🎯 核心功能

### V5 相較於 V4 的重大升級

| 功能模組 | V4 | V5 |
|---------|----|----|
| **標籤生成** | 固定 k 步價格變動（alpha=0.002） | Triple-Barrier 自適應標籤 ✅ |
| **波動率估計** | 簡單 std/mean | EWMA / Yang-Zhang / GARCH ✅ |
| **樣本權重** | 無 | 收益加權 + 時間衰減 + 類別平衡 ✅ |
| **震盪統計** | 無 | 完整震盪分布報告 ✅ **（NEW）** |
| **輸出格式** | (X, y) | (X, y, weights, metadata) ✅ |

---

## 📦 輸入與輸出

### 輸入

**原始數據格式**：台股逐筆交易數據（.txt 文件）

```
數據目錄結構：
data/temp/
├── 20241001_stock_data.txt
├── 20241002_stock_data.txt
├── ...
└── 20241050_stock_data.txt
```

**數據欄位要求**（34 欄，|| 分隔）：
- 索引 1: 股票代碼（Symbol）
- 索引 3-5: 參考價、漲停價、跌停價
- 索引 9-11: 最後成交價、最後成交量、總成交量
- 索引 12-21: 五檔買價、買量
- 索引 22-31: 五檔賣價、賣量
- 索引 32: 時間（HHMMSS）
- 索引 33: 試撮標記（IsTrialMatch）

### 輸出

**標準輸出結構**：

```
data/processed_v5/
├── npz/                                    # 訓練數據
│   ├── stock_embedding_train.npz          # 訓練集（70%）
│   ├── stock_embedding_val.npz            # 驗證集（15%）
│   ├── stock_embedding_test.npz           # 測試集（15%）
│   └── normalization_meta.json            # 元數據
├── volatility_stats.csv                   # 震盪統計（完整數據）⭐ NEW
└── volatility_summary.json                # 震盪統計（摘要）⭐ NEW
```

**NPZ 檔案內容**：

```python
import numpy as np

data = np.load('stock_embedding_train.npz')
X = data['X']           # (N, 100, 20) - LOB 特徵序列
y = data['y']           # (N,) - 標籤 {0: 下跌, 1: 持平, 2: 上漲}
weights = data['weights']    # (N,) - 樣本權重（NEW in V5）
stock_ids = data['stock_ids'] # (N,) - 股票代碼
```

---

## 🚀 基本使用

### 1. 環境準備

```bash
# 激活環境
conda activate deeplob-pro

# 確認依賴已安裝
pip install triple-barrier arch ruamel.yaml pandas scikit-learn numpy
```

### 2. 快速開始（使用預設配置）

```bash
python scripts/extract_tw_stock_data_v5.py \
    --input-dir ./data/temp \
    --output-dir ./data/processed_v5 \
    --config configs/config_pro_v5.yaml \
    --make-npz
```

**參數說明**：
- `--input-dir`: 原始數據目錄（包含 .txt 文件）
- `--output-dir`: 輸出目錄
- `--config`: V5 配置文件路徑
- `--make-npz`: 生成 NPZ 訓練文件（默認啟用）

### 3. 執行流程

```
開始執行
  ↓
載入配置文件（config_pro_v5.yaml）
  ↓
讀取原始數據（.txt 文件）
  ↓
數據清洗（移除試撮、價差檢查、時間窗口過濾）
  ↓
10 事件聚合（10 筆快照 → 1 時間點）
  ↓
計算震盪統計（每個 symbol-day）⭐ NEW
  ↓
按股票串接數據
  ↓
計算波動率（EWMA/GARCH）
  ↓
生成 Triple-Barrier 標籤
  ↓
計算樣本權重
  ↓
產生滑窗樣本（100 timesteps）
  ↓
70/15/15 切分（按股票數）
  ↓
Z-Score 標準化（基於訓練集）
  ↓
保存 NPZ 文件 + Metadata + 震盪統計報告 ⭐ NEW
  ↓
完成
```

---

## ⚙️ 配置文件說明

### 配置文件位置

`configs/config_pro_v5.yaml`

### 關鍵配置項

```yaml
# ============================================================
# V5 Pro 配置文件
# ============================================================

version: "5.0.0"

# ------------------------------------------------------------
# 1. 波動率估計（Volatility Estimation）
# ------------------------------------------------------------
volatility:
  method: "ewma"        # 可選: "ewma", "garch"
  halflife: 60          # EWMA 半衰期（僅當 method=ewma 時使用）

# ------------------------------------------------------------
# 2. Triple-Barrier 標籤（Labeling）
# ------------------------------------------------------------
triple_barrier:
  pt_multiplier: 2.0    # 止盈倍數（profit-taking）
  sl_multiplier: 2.0    # 止損倍數（stop-loss）
  max_holding: 200      # 最大持有期（bars）
  min_return: 0.0001    # 最小報酬閾值（0.01%）

# ------------------------------------------------------------
# 3. 樣本權重（Sample Weighting）
# ------------------------------------------------------------
sample_weights:
  enabled: true         # 是否啟用樣本權重
  tau: 100.0           # 時間衰減參數
  return_scaling: 10.0 # 收益縮放係數
  balance_classes: true # 是否啟用類別平衡

# ------------------------------------------------------------
# 4. 數據切分（Train/Val/Test Split）
# ------------------------------------------------------------
split:
  train_ratio: 0.70    # 訓練集比例
  val_ratio: 0.15      # 驗證集比例
  test_ratio: 0.15     # 測試集比例
  seed: 42             # 隨機種子（確保可重現）

# ------------------------------------------------------------
# 5. 震盪篩選（Intraday Volatility Filter）⭐ 待實作
# ------------------------------------------------------------
# intraday_volatility_filter:
#   enabled: false      # 是否啟用震盪篩選
#   min_range_pct: 0.02 # 最小震盪幅度（2%）
```

### 配置參數詳解

#### 波動率估計方法

| 方法 | 優點 | 缺點 | 建議場景 |
|------|------|------|---------|
| **ewma** | 快速、穩定、輕量 | 精度略低 | 日常使用、快速實驗 ⭐ |
| **garch** | 專業、精確 | 計算慢、可能失敗 | 最終訓練、研究用途 |

#### Triple-Barrier 參數調優

| 參數 | 預設值 | 建議範圍 | 說明 |
|------|--------|---------|------|
| **pt_multiplier** | 2.0 | 1.5 ~ 3.0 | 越大越保守（需更大波動才標記） |
| **sl_multiplier** | 2.0 | 1.5 ~ 3.0 | 越大越寬鬆（允許更大虧損） |
| **max_holding** | 200 | 100 ~ 300 | 高頻交易建議 100-150 |
| **min_return** | 0.0001 | 0.0001 ~ 0.001 | 過濾微小波動 |

#### 樣本權重策略

**權重計算公式**：
```
weight = |return| × scale × exp(-tt / tau) × class_weight
```

- `|return|`: 絕對報酬（獎勵大波動樣本）
- `scale`: 縮放係數（避免權重過小）
- `exp(-tt / tau)`: 時間衰減（獎勵快速觸發）
- `class_weight`: 類別平衡（處理類別不平衡）

---

## 📊 新功能：震盪統計報告（V5.0 新增）

### 功能說明

自動計算每個 **symbol-day** 的震盪幅度，生成詳細報告，用於：
- 了解數據波動特性
- 決定是否需要篩選低波動股票
- 優化 Triple-Barrier 參數

### 輸出檔案

#### 1. volatility_stats.csv（完整數據）

**範例內容**：

| date | symbol | range_pct | return_pct | high | low | open | close | n_points |
|------|--------|-----------|------------|------|-----|------|-------|----------|
| 20241018 | 2454 | 0.0345 | 0.0123 | 102.5 | 99.0 | 100.0 | 101.23 | 245 |
| 20241018 | 1216 | 0.0082 | -0.0015 | 50.5 | 50.1 | 50.3 | 50.22 | 187 |

**欄位說明**：
- `range_pct`: 震盪幅度（(最高-最低)/開盤）⭐
- `return_pct`: 漲跌幅（(收盤-開盤)/開盤）
- `n_points`: 數據點數

#### 2. volatility_summary.json（統計摘要）

**範例內容**：

```json
{
  "total_samples": 9750,
  "n_stocks": 195,
  "n_dates": 50,
  "range_pct": {
    "mean": 2.34,
    "median": 2.10,
    "percentiles": {
      "P50": 2.10,
      "P75": 3.12,
      "P90": 4.25
    }
  },
  "threshold_stats": [
    {"threshold_pct": 1.0, "count": 7800, "percentage": 80.0},
    {"threshold_pct": 2.0, "count": 3900, "percentage": 40.0}
  ]
}
```

#### 3. 控制台報告（自動顯示）

執行時會自動輸出：

```
============================================================
📊 震盪幅度統計報告（Intraday Range Analysis）
============================================================
總樣本數: 9,750 個 symbol-day 組合
股票數: 195 檔
交易日數: 50 天

震盪幅度 (Range %) 分布:
  平均值: 2.34%
  中位數: 2.10%

閾值篩選統計（震盪 ≥ X% 的樣本數）:
  閾值 | 樣本數 |  佔比
------+--------+------
  1.0% |  7,800 | 80.0%
  2.0% |  3,900 | 40.0%  ← 建議閾值
  3.0% |  1,463 | 15.0%
============================================================
```

---

## 🔧 進階使用

### 使用案例 1：更改波動率方法為 GARCH

**步驟 1**：修改配置文件

```yaml
volatility:
  method: "garch"  # 改為 GARCH
```

**步驟 2**：執行

```bash
python scripts/extract_tw_stock_data_v5.py \
    --config configs/config_pro_v5_garch.yaml \
    --output-dir ./data/processed_v5_garch
```

### 使用案例 2：調整 Triple-Barrier 參數（更保守）

**步驟 1**：創建新配置文件 `config_pro_v5_conservative.yaml`

```yaml
triple_barrier:
  pt_multiplier: 3.0    # 更大的止盈倍數（更保守）
  sl_multiplier: 3.0    # 更大的止損倍數（更寬鬆）
  max_holding: 150      # 更短的持有期（高頻交易）
  min_return: 0.0005    # 更大的最小報酬閾值
```

**步驟 2**：執行

```bash
python scripts/extract_tw_stock_data_v5.py \
    --config configs/config_pro_v5_conservative.yaml \
    --output-dir ./data/processed_v5_conservative
```

### 使用案例 3：禁用樣本權重

```yaml
sample_weights:
  enabled: false  # 禁用權重（所有樣本權重為 1）
```

### 使用案例 4：只產生震盪統計（不生成訓練數據）⭐ NEW

**目的**：快速分析數據波動特性，不需等待完整訓練數據生成

**方法 A：使用主程式 + --stats-only 參數**

```bash
python scripts/extract_tw_stock_data_v5.py \
    --input-dir ./data/temp \
    --output-dir ./data/volatility_only \
    --config configs/config_pro_v5.yaml \
    --stats-only
```

**方法 B：使用快速統計工具**（推薦）

```bash
python scripts/quick_stats_only.py \
    --input-dir ./data/temp \
    --output-dir ./data/volatility_stats
```

**方法 C：Windows 批次腳本**（最簡單）

```bash
# 雙擊執行，或在命令列執行：
quick_stats.bat
```

**優點**：
- ⚡ **速度快**：只需 5-10% 的時間（相比完整流程）
  - 完整流程：30-60 分鐘
  - 快速模式：3-5 分鐘
- 💾 **節省空間**：不生成大型 NPZ 檔案（可節省數 GB 空間）
- 📊 **完整報告**：仍然產生 CSV + JSON + 控制台報告

**輸出檔案**：

```
data/volatility_stats/
├── volatility_stats.csv      # 完整震盪數據
└── volatility_summary.json   # 統計摘要
```

**執行結果範例**：

```
============================================================
⚡ 快速模式：已完成震盪統計，跳過訓練數據生成
============================================================
如需生成訓練數據，請移除 --stats-only 參數
============================================================

============================================================
📊 震盪幅度統計報告（Intraday Range Analysis）
============================================================
總樣本數: 9,750 個 symbol-day 組合
股票數: 195 檔

閾值篩選統計（震盪 ≥ X% 的樣本數）:
  閾值 | 樣本數 |  佔比
------+--------+------
  2.0% |  3,900 | 40.0%  ← 建議閾值
============================================================

============================================================
[完成] 震盪統計成功，輸出資料夾: ./data/volatility_stats
============================================================
統計資料:
  原始事件數: 125,487,352
  清洗後: 98,234,521
  聚合後時間點: 9,823,452
  震盪統計樣本: 9,750
============================================================
```

**使用場景**：
1. **初步數據探索**：快速了解數據波動特性
2. **決策篩選閾值**：根據統計決定是否需要篩選
3. **數據品質驗證**：確認數據源是否符合預期
4. **快速實驗**：測試不同數據源或時間範圍

---

## 📈 輸出數據統計資訊

### 執行完成後的統計報告

```
============================================================
[完成] V5 轉換成功，輸出資料夾: ./data/processed_v5
============================================================
統計資料:
  原始事件數: 125,487,352
  清洗後: 98,234,521
  聚合後時間點: 9,823,452
  有效窗口: 7,549,123
  Triple-Barrier 成功: 195
  震盪統計樣本: 9,750 ⭐ NEW
============================================================
```

### Metadata 檔案內容

`normalization_meta.json` 包含完整的處理資訊：

```json
{
  "format": "deeplob_v5_pro",
  "version": "5.0.0",
  "creation_date": "2025-10-19T14:30:00",
  "seq_len": 100,
  "feature_dim": 20,

  "volatility": {
    "method": "ewma",
    "halflife": 60
  },

  "triple_barrier": {
    "pt_multiplier": 2.0,
    "sl_multiplier": 2.0,
    "max_holding": 200,
    "min_return": 0.0001
  },

  "data_split": {
    "train_stocks": 136,
    "val_stocks": 29,
    "test_stocks": 30,
    "results": {
      "train": {
        "samples": 5584553,
        "label_dist": [397396, 506030, 345993]
      }
    }
  }
}
```

---

## ⚠️ 常見問題與解決方案

### Q1: 執行時出現 `FileNotFoundError: 配置文件不存在`

**原因**：配置文件路徑錯誤

**解決**：
```bash
# 確認配置文件存在
ls configs/config_pro_v5.yaml

# 使用絕對路徑
python scripts/extract_tw_stock_data_v5.py \
    --config "D:/Case-New/python/DeepLOB-Pro/configs/config_pro_v5.yaml"
```

### Q2: GARCH 波動率計算失敗

**錯誤訊息**：
```
GARCH 失敗: ..., 回退到 EWMA
```

**原因**：數據點不足或數據異常

**解決**：
1. 檢查數據品質（是否有足夠的時間點）
2. 改用 EWMA 方法
3. 增加數據量（使用更多交易日）

### Q3: 樣本數量太少

**現象**：訓練集樣本數 < 100 萬

**可能原因**：
- 原始數據太少
- Triple-Barrier 參數過於嚴格
- 股票序列太短被過濾

**解決**：
1. 增加原始數據（更多交易日）
2. 降低 `min_return` 閾值
3. 減少 `pt_multiplier` 和 `sl_multiplier`
4. 檢查 `MIN_POINTS` 設定（預設 150）

### Q4: 類別嚴重不平衡

**現象**：
```
標籤分布: 上漲=100,000, 持平=800,000, 下跌=50,000
```

**解決**：
1. 啟用樣本權重（`sample_weights.enabled: true`）
2. 調整 Triple-Barrier 參數
3. 增加 `min_return` 閾值（減少持平樣本）

### Q5: 執行速度太慢

**優化方向**：
1. 使用 EWMA 而非 GARCH（快 10 倍）
2. 減少原始數據量（測試時）
3. 使用 SSD 儲存數據
4. 增加記憶體（減少 swap）

### Q6: 震盪統計顯示所有股票波動都很低（< 1%）

**可能原因**：
- 數據源問題（只包含穩定股票）
- 數據時間範圍太短
- 使用了非交易時段數據

**解決**：
1. 檢查數據源（確保包含活躍股票）
2. 延長數據時間範圍
3. 確認時間過濾正確（09:00-13:30）

---

## 🧪 測試與驗證

### 1. 快速驗證功能

```bash
# 使用模擬數據測試震盪統計功能
python scripts/quick_verify_volatility.py
```

**預期輸出**：
```
🎉 所有測試通過！
```

### 2. 小規模數據測試

```bash
# 使用少量數據測試完整流程（5-10 分鐘）
python scripts/extract_tw_stock_data_v5.py \
    --input-dir ./data/temp_sample \
    --output-dir ./data/processed_v5_test
```

### 3. 驗證輸出數據

```python
import numpy as np
import json

# 載入訓練數據
data = np.load('data/processed_v5/npz/stock_embedding_train.npz')

print(f"X shape: {data['X'].shape}")        # 應為 (N, 100, 20)
print(f"y shape: {data['y'].shape}")        # 應為 (N,)
print(f"weights shape: {data['weights'].shape}")  # 應為 (N,)
print(f"Label distribution: {np.bincount(data['y'])}")

# 載入 metadata
with open('data/processed_v5/npz/normalization_meta.json', 'r') as f:
    meta = json.load(f)
    print(f"Version: {meta['version']}")
    print(f"Method: {meta['volatility']['method']}")
```

---

## 📚 相關文檔

### 專案文檔

- [CLAUDE.md](../CLAUDE.md) - 專案總覽
- [VOLATILITY_ANALYSIS_GUIDE.md](../VOLATILITY_ANALYSIS_GUIDE.md) - 震盪分析完整指南
- [QUICK_START_VOLATILITY.md](../QUICK_START_VOLATILITY.md) - 震盪分析快速開始
- [V5_Pro_NoMLFinLab_Guide.md](./V5_Pro_NoMLFinLab_Guide.md) - V5 設計文檔

### 配置文件

- [configs/config_pro_v5.yaml](../configs/config_pro_v5.yaml) - 標準配置

### 相關腳本

- [scripts/train_deeplob_generic.py](../scripts/train_deeplob_generic.py) - DeepLOB 訓練腳本
- [scripts/quick_verify_volatility.py](../scripts/quick_verify_volatility.py) - 功能驗證腳本

---

## 🔄 版本歷史

### v5.0.0 (2025-10-19)

**重大更新**：
- ✅ 新增震盪統計功能
  - `calculate_intraday_volatility()` 函數
  - `generate_volatility_report()` 函數
  - 自動生成 CSV + JSON 報告
- ✅ Triple-Barrier 標籤生成
- ✅ 專業波動率估計（EWMA/GARCH）
- ✅ 樣本權重計算

### v4.0.0 (之前版本)

- 基礎數據處理
- 固定 k 步標籤
- Z-Score 標準化

---

## 💡 最佳實踐

### 1. 數據處理流程建議

```bash
# 步驟 1: 快速測試（小數據）
python scripts/extract_tw_stock_data_v5.py \
    --input-dir ./data/temp_sample \
    --output-dir ./data/test

# 步驟 2: 查看震盪報告，決定是否篩選
cat data/test/volatility_summary.json

# 步驟 3: 調整配置（如有需要）
# 編輯 configs/config_pro_v5.yaml

# 步驟 4: 完整處理
python scripts/extract_tw_stock_data_v5.py \
    --input-dir ./data/temp \
    --output-dir ./data/processed_v5_final
```

### 2. 參數調優策略

**第一次執行**：使用預設參數
```yaml
triple_barrier:
  pt_multiplier: 2.0
  sl_multiplier: 2.0
```

**根據結果調整**：
- 如果持平樣本 > 50%：增加 `min_return` 或減少 `pt/sl_multiplier`
- 如果時間觸發 > 70%：減少 `max_holding` 或增加 `pt/sl_multiplier`

### 3. 數據品質檢查清單

執行後檢查：
- ✅ 樣本數 > 100 萬（訓練集）
- ✅ 類別分布相對平衡（30% ~ 40% 每類）
- ✅ Triple-Barrier 有效觸發率 > 30%
- ✅ 震盪分布合理（平均 > 1%）

---

## 📞 技術支援

### 報告問題

如遇到問題，請提供：
1. 完整錯誤訊息
2. 使用的配置文件
3. 數據規模（檔案數、大小）
4. 執行環境（Python 版本、作業系統）

### 參考資源

- **Triple-Barrier 標籤**：《Advances in Financial Machine Learning》（Marcos López de Prado）
- **GARCH 模型**：`arch` 庫文檔
- **DeepLOB 論文**：Zhang et al., 2019

---

**文檔版本**：v1.0
**最後更新**：2025-10-19
**維護者**：DeepLOB-Pro Team
