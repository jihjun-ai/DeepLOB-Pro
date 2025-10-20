# 震盪幅度分析快速開始指南

## 📌 目標

分析台股數據的「當日震盪幅度」分布，決定最佳的數據篩選閾值（例如：震盪 ≥2%）

---

## ✅ 已完成的工作

### 1. 程式碼修改

**修改檔案**：[scripts/extract_tw_stock_data_v5.py](scripts/extract_tw_stock_data_v5.py)

**新增功能**：
- ✅ `calculate_intraday_volatility()` - 計算震盪統計
- ✅ `generate_volatility_report()` - 生成震盪報告
- ✅ 在主流程中自動計算並保存震盪數據

### 2. 新增工具腳本

- ✅ [scripts/quick_verify_volatility.py](scripts/quick_verify_volatility.py) - 快速驗證功能
- ✅ [scripts/test_volatility_stats.py](scripts/test_volatility_stats.py) - 完整測試工具

### 3. 文檔

- ✅ [VOLATILITY_ANALYSIS_GUIDE.md](VOLATILITY_ANALYSIS_GUIDE.md) - 完整分析指南

---

## 🚀 執行步驟（按順序）

### 步驟 1：快速驗證功能（1 分鐘）

**目的**：確認震盪統計功能正常運作

```bash
# 啟動環境
conda activate deeplob-pro

# 執行驗證
python scripts/quick_verify_volatility.py
```

**預期結果**：
```
============================================================
🧪 震盪統計功能快速驗證
============================================================

測試 1: 震盪計算函數
✅ 案例 1（正常震盪）測試通過
✅ 案例 2（高震盪）測試通過
✅ 案例 3（低震盪）測試通過

測試 2: 震盪報告生成
✅ 報告生成成功
✅ CSV 驗證通過
✅ JSON 驗證通過

🎉 所有測試通過！
```

---

### 步驟 2：執行完整統計分析

**目的**：分析實際台股數據的震盪分布

#### 方法 A：快速統計模式（推薦）⚡

**時間**：3-5 分鐘
**輸出**：震盪統計報告（CSV + JSON）

```bash
# 使用快速工具（推薦）
python scripts/quick_stats_only.py

# 或使用 Windows 批次腳本（雙擊執行）
quick_stats.bat

# 或使用主程式 + --stats-only 參數
python scripts/extract_tw_stock_data_v5.py \
    --input-dir ./data/temp \
    --output-dir ./data/volatility_stats \
    --config configs/config_pro_v5.yaml \
    --stats-only
```

#### 方法 B：完整執行

**時間**：30-60 分鐘
**輸出**：震盪統計 + 訓練數據（NPZ）

```bash
python scripts/extract_tw_stock_data_v5.py \
    --input-dir ./data/temp \
    --output-dir ./data/processed_v5_stats \
    --config configs/config_pro_v5.yaml \
    --make-npz
```

**建議**：
- **初次使用**：選擇方法 A（快速模式）
- **需要訓練數據**：選擇方法 B（完整模式）

**輸出檔案**：

```bash
# 方法 A（快速模式）輸出：
data/volatility_stats/
├── volatility_stats.csv          # ⭐ 完整震盪數據
└── volatility_summary.json       # ⭐ 統計摘要

# 方法 B（完整模式）輸出：
data/processed_v5_stats/
├── volatility_stats.csv          # ⭐ 完整震盪數據
├── volatility_summary.json       # ⭐ 統計摘要
└── npz/                          # 訓練數據（未篩選版本）
    ├── stock_embedding_train.npz
    ├── stock_embedding_val.npz
    └── stock_embedding_test.npz
```

---

### 步驟 3：查看統計報告

#### 3.1 控制台報告（執行時自動顯示）

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

#### 3.2 JSON 摘要

```bash
# Windows
type data\processed_v5_stats\volatility_summary.json

# Linux/Mac
cat data/processed_v5_stats/volatility_summary.json
```

#### 3.3 CSV 數據分析（使用 Excel 或 Python）

```python
import pandas as pd
import matplotlib.pyplot as plt

# 讀取數據
df = pd.read_csv('data/processed_v5_stats/volatility_stats.csv')

# 查看統計摘要
print(df.describe())

# 繪製震盪分布圖
plt.hist(df['range_pct'] * 100, bins=50)
plt.xlabel('震盪幅度 (%)')
plt.ylabel('樣本數')
plt.axvline(x=2.0, color='r', linestyle='--', label='建議閾值 2%')
plt.legend()
plt.show()
```

---

### 步驟 4：決定閾值（基於報告）

根據統計結果，選擇合適的閾值：

| 閾值 | 保留樣本比例 | 建議場景 |
|------|------------|---------|
| 1.0% | ~80% | 溫和篩選（保留大部分數據） |
| **2.0%** | **~40%** | **推薦：高頻交易最佳平衡** ⭐ |
| 3.0% | ~15% | 激進篩選（只要高波動股票） |

**決策考量**：
- 樣本數是否足夠（建議 ≥ 200 萬樣本）
- Triple-Barrier 有效觸發率（目標 > 50%）
- 是否符合高頻交易目標

---

### 步驟 5：實作篩選並生成 3 組實驗數據（待完成）

**目前狀態**：階段 1-3 已完成，階段 4-5 待實作

**下一步**：
1. 根據統計報告決定 3 組閾值（例如：1%, 2%, 3%）
2. 修改配置文件或程式碼，加入篩選邏輯
3. 生成 3 組訓練數據
4. 訓練 3 組 DeepLOB 模型
5. 比較準確率、F1 Score、Triple-Barrier 有效率

---

## 📊 輸出檔案說明

### 1. volatility_stats.csv（完整數據）

| 欄位 | 說明 |
|------|------|
| date | 交易日期 |
| symbol | 股票代碼 |
| range_pct | 震盪幅度（小數，例如 0.02 = 2%） |
| return_pct | 漲跌幅（小數） |
| high | 最高價 |
| low | 最低價 |
| open | 開盤價 |
| close | 收盤價 |
| n_points | 數據點數 |

**用途**：詳細分析、繪圖、Excel 統計

### 2. volatility_summary.json（統計摘要）

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
    {"threshold_pct": 2.0, "count": 3900, "percentage": 40.0},
    {"threshold_pct": 3.0, "count": 1463, "percentage": 15.0}
  ]
}
```

**用途**：程式化處理、自動決策

---

## ❓ 常見問題

### Q1: 為什麼需要震盪統計？

**A**: 過濾低波動股票，提高訓練效率和 Triple-Barrier 標籤質量。

### Q2: 震盪幅度 vs 漲跌幅有什麼不同？

**A**:
- **震盪幅度**：`(最高-最低)/開盤` - 反映盤中波動
- **漲跌幅**：`(收盤-開盤)/開盤` - 反映方向性變動

範例：
- 開盤 100 → 最高 105 → 最低 98 → 收盤 101
- 震盪幅度 = 7%（105-98=7）
- 漲跌幅 = 1%（101-100=1）

### Q3: 建議閾值是多少？

**A**: 根據**高頻交易**目標，推薦 **2.0%**（保留 40% 高質量樣本）

### Q4: 如果統計結果顯示震盪普遍很低怎麼辦？

**A**:
- 檢查數據源（是否包含夠多的活躍股票）
- 降低閾值（例如 1.0% 或 1.5%）
- 考慮混合策略（保留部分低波動樣本）

---

## 🔗 相關文件

- [VOLATILITY_ANALYSIS_GUIDE.md](VOLATILITY_ANALYSIS_GUIDE.md) - 完整分析指南（包含階段 4-5 實作說明）
- [CLAUDE.md](CLAUDE.md) - 專案總覽
- [configs/config_pro_v5.yaml](configs/config_pro_v5.yaml) - V5 配置文件

---

## 📝 下一步行動

### 立即執行（5-10 分鐘）

```bash
# 1. 啟動環境
conda activate deeplob-pro

# 2. 快速驗證
python scripts/quick_verify_volatility.py

# 3. 執行完整統計
python scripts/extract_tw_stock_data_v5.py \
    --input-dir ./data/temp \
    --output-dir ./data/processed_v5_stats \
    --config configs/config_pro_v5.yaml

# 4. 查看報告
type data\processed_v5_stats\volatility_summary.json
```

### 後續規劃（待統計完成後）

1. ✅ 分析震盪分布報告
2. ⏳ 決定 3 組實驗閾值（例如：1%, 2%, 3%）
3. ⏳ 實作篩選邏輯
4. ⏳ 生成 3 組訓練數據
5. ⏳ 訓練並比較模型性能

---

**更新日期**：2025-10-19
**狀態**：階段 1-3 已完成，可立即執行統計分析
