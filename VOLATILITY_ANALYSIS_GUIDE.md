# 震盪幅度分析指南（Intraday Volatility Analysis）

## 📋 概述

本指南說明如何使用震盪幅度統計功能，來決定最佳的數據篩選閾值。

**目標**：找出「震盪幅度 ≥ X%」的最佳閾值，提高訓練數據質量

---

## 🚀 階段 1：執行統計分析（不篩選）

### 方法 A：完整執行

```bash
conda activate deeplob-pro

python scripts/extract_tw_stock_data_v5.py \
    --input-dir ./data/temp \
    --output-dir ./data/processed_v5_stats \
    --config configs/config_pro_v5.yaml \
    --make-npz
```

**預期輸出**：
```
data/processed_v5_stats/
├── volatility_stats.csv          # 完整震盪數據（每個 symbol-day）
├── volatility_summary.json       # 統計摘要
└── npz/                          # 訓練數據（未篩選）
    ├── stock_embedding_train.npz
    ├── stock_embedding_val.npz
    └── stock_embedding_test.npz
```

### 方法 B：快速測試（推薦）

```bash
python scripts/test_volatility_stats.py \
    --input-dir ./data/temp \
    --output-dir ./data/volatility_test
```

**優點**：快速驗證統計功能是否正常

---

## 📊 階段 2：分析統計報告

### 2.1 控制台報告範例

執行完成後，你會看到類似以下報告：

```
============================================================
📊 震盪幅度統計報告（Intraday Range Analysis）
============================================================
總樣本數: 9,750 個 symbol-day 組合
股票數: 195 檔
交易日數: 50 天

震盪幅度 (Range %) 分布:
  最小值: 0.12%
  最大值: 8.45%
  平均值: 2.34%
  中位數: 2.10%
  標準差: 1.23%

分位數分布:
  P10:   0.85%
  P25:   1.45%
  P50:   2.10%
  P75:   3.12%
  P90:   4.25%
  P95:   5.10%
  P99:   6.80%

閾值篩選統計（震盪 ≥ X% 的樣本數）:
  閾值 | 樣本數 |  佔比
------+--------+------
  0.5% |  9,234 | 94.7%
  1.0% |  7,800 | 80.0%
  1.5% |  5,850 | 60.0%
  2.0% |  3,900 | 40.0%  ← 建議閾值
  2.5% |  2,438 | 25.0%
  3.0% |  1,463 | 15.0%
  4.0% |    585 |  6.0%
  5.0% |    195 |  2.0%

漲跌幅 (Return %) 分布:
  平均值: 0.05%
  中位數: 0.02%
  標準差: 1.85%

震盪最大的 10 個樣本:
  2454 @ 20241018: 震盪 8.45%, 報酬 3.12%
  3008 @ 20241015: 震盪 7.23%, 報酬 -2.45%
  ...
============================================================
```

### 2.2 閱讀 JSON 摘要

```bash
# Linux/Mac
cat data/processed_v5_stats/volatility_summary.json | python -m json.tool

# Windows
type data\processed_v5_stats\volatility_summary.json
```

### 2.3 分析 CSV 數據

使用 Excel 或 Python 分析：

```python
import pandas as pd
import matplotlib.pyplot as plt

# 讀取數據
df = pd.read_csv('data/processed_v5_stats/volatility_stats.csv')

# 繪製震盪分布圖
plt.figure(figsize=(12, 6))
plt.hist(df['range_pct'] * 100, bins=50, edgecolor='black')
plt.xlabel('震盪幅度 (%)')
plt.ylabel('樣本數')
plt.title('震盪幅度分布')
plt.axvline(x=2.0, color='r', linestyle='--', label='閾值 2%')
plt.legend()
plt.show()

# 分析震盪 vs 報酬關係
plt.figure(figsize=(10, 6))
plt.scatter(df['range_pct'] * 100, df['return_pct'] * 100, alpha=0.5)
plt.xlabel('震盪幅度 (%)')
plt.ylabel('報酬率 (%)')
plt.title('震盪幅度 vs 報酬率')
plt.grid(True, alpha=0.3)
plt.show()
```

---

## 🎯 階段 3：決定最佳閾值

### 決策矩陣

| 閾值 | 保留樣本 | 適用場景 | 風險 |
|------|---------|---------|------|
| **0.5%** | 95% | 保守策略，保留大部分數據 | 包含太多低波動樣本 |
| **1.0%** | 80% | 平衡策略（輕度篩選） | 仍有較多橫盤樣本 |
| **1.5%** | 60% | 平衡策略（中度篩選） | 數據量減少明顯 |
| **2.0%** | 40% | **推薦**：高頻交易最佳平衡 | 樣本偏向高波動 ⭐ |
| **2.5%** | 25% | 激進策略（只要活躍股） | 泛化能力下降 |
| **3.0%** | 15% | 極端策略（只要異常波動） | 過擬合風險高 ⚠️ |

### 推薦閾值

根據**高頻交易（HFT）**目標，建議：

- **第一選擇**：**2.0%**（保留 40% 高質量樣本）
- **第二選擇**：**1.5%**（保留 60%，較穩健）
- **實驗組**：測試 1%, 2%, 3% 三組，比較模型準確率

---

## 🛠️ 階段 4：實作篩選邏輯（待完成）

確定閾值後，修改 `config_pro_v5.yaml`：

```yaml
# 新增震盪篩選配置
intraday_volatility_filter:
  enabled: true
  min_range_pct: 0.02  # 2% 閾值

  # 進階選項（可選）
  max_range_pct: 0.10  # 10% 上限（避免極端異常）
  apply_to_splits: ['train', 'val']  # 只篩選訓練集，測試集保持原樣
```

**修改程式位置**：
- 檔案：`scripts/extract_tw_stock_data_v5.py`
- 函數：`main()` 中的 `per_day_symbol_points.append()` 前

**修改邏輯**：
```python
# 在 per_day_symbol_points.append() 之前
if config.get('intraday_volatility_filter', {}).get('enabled', False):
    min_range = config['intraday_volatility_filter']['min_range_pct']

    if vol_stats is not None and vol_stats['range_pct'] < min_range:
        logging.debug(f"  {sym} @ {day}: 震盪過小 ({vol_stats['range_pct']*100:.2f}%)，跳過")
        continue
```

---

## 📈 階段 5：訓練 3 組實驗模型

### 5.1 準備 3 組數據集

```bash
# 無篩選（基準組）
python scripts/extract_tw_stock_data_v5.py \
    --output-dir ./data/processed_v5_baseline \
    --config configs/config_pro_v5.yaml

# 閾值 1% （溫和組）
python scripts/extract_tw_stock_data_v5.py \
    --output-dir ./data/processed_v5_filter1 \
    --config configs/config_pro_v5_filter1.yaml

# 閾值 2% （推薦組）
python scripts/extract_tw_stock_data_v5.py \
    --output-dir ./data/processed_v5_filter2 \
    --config configs/config_pro_v5_filter2.yaml

# 閾值 3% （激進組）
python scripts/extract_tw_stock_data_v5.py \
    --output-dir ./data/processed_v5_filter3 \
    --config configs/config_pro_v5_filter3.yaml
```

### 5.2 訓練 4 組模型

```bash
# 基準組（無篩選）
python scripts/train_deeplob_generic.py \
    --data-dir ./data/processed_v5_baseline/npz \
    --checkpoint-dir ./checkpoints/deeplob_baseline

# 閾值 1%
python scripts/train_deeplob_generic.py \
    --data-dir ./data/processed_v5_filter1/npz \
    --checkpoint-dir ./checkpoints/deeplob_filter1

# 閾值 2% （推薦）
python scripts/train_deeplob_generic.py \
    --data-dir ./data/processed_v5_filter2/npz \
    --checkpoint-dir ./checkpoints/deeplob_filter2

# 閾值 3%
python scripts/train_deeplob_generic.py \
    --data-dir ./data/processed_v5_filter3/npz \
    --checkpoint-dir ./checkpoints/deeplob_filter3
```

### 5.3 比較實驗結果

| 組別 | 閾值 | 訓練樣本數 | 準確率 | F1 Score | Triple-Barrier 有效率 |
|------|------|-----------|--------|----------|---------------------|
| 基準 | 無 | 5,584,553 | 72.98% | 73.24% | 35% |
| 組1 | 1% | ? | ? | ? | ? |
| 組2 | 2% | ? | ? | ? | ? |
| 組3 | 3% | ? | ? | ? | ? |

**驗收標準**：
- ✅ 準確率提升 > 2%（目標：75%+）
- ✅ F1 Score 提升 > 1%
- ✅ Triple-Barrier 有效觸發率 > 50%

---

## 📝 注意事項

### ⚠️ 風險提醒

1. **樣本選擇偏差**：
   - 問題：只學高波動 → 無法應對橫盤市場
   - 解法：測試集保持原始分布（不篩選）

2. **數據量不足**：
   - 問題：閾值太高（≥3%）→ 樣本數 <100 萬
   - 解法：監控訓練樣本數，至少保留 200 萬樣本

3. **過度優化**：
   - 問題：針對特定閾值過擬合
   - 解法：使用 Cross-Validation 驗證

### ✅ 最佳實踐

1. **先統計，後決策**：不要盲目設定閾值
2. **多組實驗**：測試至少 3 組閾值
3. **保留對照組**：永遠保留無篩選的基準組
4. **記錄完整**：保存所有統計報告和配置文件

---

## 🔗 相關文件

- [CLAUDE.md](CLAUDE.md) - 專案總覽
- [configs/config_pro_v5.yaml](configs/config_pro_v5.yaml) - V5 配置文件
- [scripts/extract_tw_stock_data_v5.py](scripts/extract_tw_stock_data_v5.py) - 數據處理腳本

---

**更新日期**：2025-10-19
**版本**：v1.0
**狀態**：階段 1-2 已完成，階段 3-5 待執行
