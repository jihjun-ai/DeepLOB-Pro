# 波動率過濾功能使用指南

## 📋 目錄
1. [功能概述](#功能概述)
2. [實作原理](#實作原理)
3. [完整使用流程](#完整使用流程)
4. [參數調整建議](#參數調整建議)
5. [常見問題](#常見問題)

---

## 功能概述

### 問題背景
在分析 Triple-Barrier 標籤時，發現**波動率過低的股票日**會導致：
- PT/SL barrier 難以觸及（例如 3.5σ 太寬）
- 大量 time trigger → 「持平」標籤過多（>50%）
- 標籤分布不均衡，影響模型訓練

### 解決方案
在數據處理階段（`sliding_windows_v5`）直接過濾掉波動率過低的股票日：
- ✅ **源頭解決**：不生成低質量標籤
- ✅ **保持邏輯清晰**：不需修改 `tb_labels` 核心邏輯
- ✅ **標籤分布均衡**：保留的數據標籤分布更合理
- ✅ **靈活可調**：可根據實際數據分布調整閾值

---

## 實作原理

### 過濾邏輯
```python
# 在 sliding_windows_v5 → build_split_v5 中
# 計算波動率後，立即檢查
if vol_filter_enabled:
    vol_clean = vol.replace([np.inf, -np.inf], np.nan).dropna()
    daily_vol_mean = vol_clean.mean()
    
    if daily_vol_mean < min_vol_threshold:
        # 跳過這個股票日
        vol_filtered_days += 1
        continue
```

### 過濾位置
- **階段**：在 `sliding_windows_v5` 的逐日處理循環中
- **時機**：計算完 EWMA 波動率之後，生成 Triple-Barrier 標籤之前
- **效果**：被過濾的股票日不會產生任何訓練樣本

### 與現有震盪過濾的關係
系統有兩層過濾：

| 過濾類型 | 階段 | 指標 | 配置項 |
|---------|------|------|--------|
| **震盪過濾** | 主程式（資料聚合後） | (high-low)/open | `intraday_volatility_filter` |
| **波動率過濾** ⭐ | sliding_windows_v5 | EWMA vol 平均值 | `volatility_filter` |

兩者可同時啟用，互補使用。

---

## 完整使用流程

### Step 1：分析當前數據的波動率分布

```bash
python scripts/extract_tw_stock_data_v5.py \
    --input-dir ./data/temp \
    --output-dir ./data/volatility_analysis \
    --config configs/config_pro_v5_ml_optimal.yaml \
    --stats-only
```

**查看報告**：
```bash
# 查看波動率統計
cat ./data/volatility_analysis/volatility_summary.json

# 或查看詳細 CSV
# ./data/volatility_analysis/volatility_stats.csv
```

**關鍵資訊**：
- `range_pct.percentiles` - 找 P10、P25、P50
- `n_samples` - 總樣本數
- `top_10_volatile` - 最高波動率的股票日

### Step 2：決定過濾閾值

根據 Step 1 的分析結果：

```yaml
# 保守方案（過濾少，保留更多數據）
volatility_filter:
  enabled: true
  min_daily_vol: 0.0003  # P10 的 50%

# 中等方案（推薦，平衡過濾與數據量）
volatility_filter:
  enabled: true
  min_daily_vol: 0.0005  # 約 P25

# 激進方案（過濾多，只保留高波動率）
volatility_filter:
  enabled: true
  min_daily_vol: 0.001   # 約 P50
```

**決策參考**：
- 如果當前「持平」標籤 >60% → 使用激進方案
- 如果當前「持平」標籤 40-60% → 使用中等方案
- 如果當前「持平」標籤 <40% → 不需過濾或使用保守方案

### Step 3：生成過濾後的數據

```bash
python scripts/extract_tw_stock_data_v5.py \
    --input-dir ./data/temp \
    --output-dir ./data/processed_v5_vol_filtered \
    --config configs/config_pro_v5_vol_filtered.yaml
```

**預期輸出**：
```
============================================================
V5 滑窗流程开始（按日处理模式），共 XXX 个 symbol-day 组合
日界线保护: 启用
波动率过滤: 启用 (阈值: 0.050%)  ← 確認已啟用
============================================================

...

TRAIN 总计: XXX 个样本 (来自 YYY 个 symbol-day)
触发原因分布: {'up': A, 'down': B, 'time': C}
波动率过滤: ZZZ 个股票日被移除 (XX.X%)  ← 過濾統計
```

### Step 4：檢查過濾效果

```bash
# 查看 metadata
cat ./data/processed_v5_vol_filtered/npz/normalization_meta.json
```

**關鍵指標**：
```json
{
  "data_quality": {
    "volatility_filtered": 123,  // 過濾的股票日數
    "valid_windows": 45678       // 有效訓練樣本數
  },
  "data_split": {
    "results": {
      "train": {
        "label_dist": [10234, 12345, 11234],  // [下跌, 持平, 上漲]
        // 計算比例：持平% = 12345 / (10234+12345+11234) * 100
      }
    }
  }
}
```

**判斷標準**：
- ✅ **成功**：持平標籤 30-45%，下跌/上漲各 25-35%
- ⚠️ **需調整**：持平仍 >50% → 提高 `min_daily_vol`
- ⚠️ **過濾太多**：持平 <20% → 降低 `min_daily_vol`

### Step 5：迭代優化（可選）

如果 Step 4 的結果不理想：

1. **調整 `min_daily_vol`**
   ```yaml
   # 增加/減少閾值
   min_daily_vol: 0.0007  # 從 0.0005 提高到 0.0007
   ```

2. **配合調整 Triple-Barrier 參數**
   ```yaml
   triple_barrier:
     pt_multiplier: 3.0    # 從 3.5 降到 3.0（更容易觸發）
     sl_multiplier: 3.0
     min_return: 0.002     # 從 0.0015 提高到 0.002（更嚴格）
   ```

3. **重新生成數據**（使用新配置）

---

## 參數調整建議

### `min_daily_vol` 設定指南

| 閾值 | 百分比 | 適用場景 | 預期過濾比例 |
|------|--------|---------|-------------|
| 0.0002 | 0.02% | 極保守，幾乎不過濾 | <5% |
| 0.0003 | 0.03% | 保守，只過濾極低波動率 | 5-15% |
| 0.0005 | 0.05% | **中等（推薦）** | 15-30% |
| 0.0007 | 0.07% | 積極 | 30-50% |
| 0.001 | 0.1% | 激進，只保留高波動率 | 50-70% |

### 與其他參數的配合

#### 場景 1：持平標籤過多（>60%）
```yaml
volatility_filter:
  min_daily_vol: 0.0007  # 積極過濾

triple_barrier:
  pt_multiplier: 3.0     # 縮小 barrier
  sl_multiplier: 3.0
  min_return: 0.002      # 提高閾值
```

#### 場景 2：持平標籤適中（40-50%）
```yaml
volatility_filter:
  min_daily_vol: 0.0005  # 中等過濾（推薦）

triple_barrier:
  pt_multiplier: 3.5     # 維持原設定
  sl_multiplier: 3.5
  min_return: 0.0015
```

#### 場景 3：持平標籤已偏低（<30%）
```yaml
volatility_filter:
  enabled: false         # 關閉過濾

# 或使用極保守設定
volatility_filter:
  min_daily_vol: 0.0002
```

---

## 常見問題

### Q1：過濾會損失多少數據？
**A**：取決於 `min_daily_vol` 設定：
- 保守設定（0.0003）：損失 5-15% 股票日
- 中等設定（0.0005）：損失 15-30% 股票日
- 激進設定（0.001）：損失 50%+ 股票日

**重要**：損失的是**低質量**的股票日（標籤分布極度不均），對模型訓練反而有利。

### Q2：如何確認過濾是否生效？
**A**：檢查三個地方：
1. **運行日誌**：看到 "波动率过滤: X 个股票日被移除"
2. **normalization_meta.json**：`data_quality.volatility_filtered` 數值
3. **標籤分布**：`data_split.results.*.label_dist` 比例

### Q3：過濾後標籤分布沒改善？
**A**：可能原因：
1. **閾值太低**：提高 `min_daily_vol`
2. **PT/SL 太寬**：降低 `pt_multiplier` / `sl_multiplier`
3. **min_return 太小**：提高 `min_return`

建議：同時調整三個參數，多次實驗找最佳組合。

### Q4：可以只用震盪過濾嗎？
**A**：可以，但兩者目的不同：
- **震盪過濾**（`intraday_volatility_filter`）：基於價格範圍，在主程式階段
- **波動率過濾**（`volatility_filter`）：基於 EWMA 波動率，在 sliding_windows 階段

建議：兩者配合使用效果最佳。

### Q5：訓練集和測試集的過濾會不一致嗎？
**A**：不會。過濾是在**按股票切分之後**進行的，所有 split 使用相同的閾值和邏輯。

### Q6：過濾後如何評估模型性能？
**A**：
1. **標籤分布**：應更均衡（30-40-30）
2. **訓練穩定性**：loss 下降更平滑
3. **驗證集表現**：準確率/F1-score 提升
4. **類別 Recall**：各類別 Recall 更均衡（避免只預測一類）

---

## 實驗記錄模板

建議記錄每次實驗的參數和結果，便於對比：

```markdown
## 實驗 1：中等過濾
- 配置：`config_pro_v5_vol_filtered.yaml`
- `min_daily_vol`: 0.0005
- `pt_multiplier`: 3.5
- `min_return`: 0.0015

**結果**：
- 過濾股票日：123 個（15.2%）
- 訓練樣本：45,678 個
- 標籤分布：[10234, 12345, 11234] → [30.2%, 36.4%, 33.4%] ✅
- 訓練效果：準確率 62.3%，F1 0.58

---

## 實驗 2：積極過濾
- 配置：同上
- `min_daily_vol`: 0.0007 ← 提高
- 其他參數同實驗 1

**結果**：
- 過濾股票日：245 個（30.3%）
- 訓練樣本：32,456 個
- 標籤分布：[11234, 9876, 11346] → [34.6%, 30.4%, 35.0%] ✅✅
- 訓練效果：準確率 65.1%，F1 0.61 ← 改善！
```

---

## 下一步建議

1. **立即行動**：
   ```bash
   # 先分析當前數據
   python scripts/extract_tw_stock_data_v5.py \
       --input-dir ./data/temp \
       --output-dir ./data/vol_analysis \
       --config configs/config_pro_v5_ml_optimal.yaml \
       --stats-only
   ```

2. **查看報告**，決定閾值

3. **生成過濾後的數據**

4. **對比訓練效果**（原始 vs 過濾後）

5. **記錄實驗結果**，持續優化

---

## 總結

波動率過濾功能是解決「持平」標籤過多問題的**最佳實踐方案**：

✅ **優點**：
- 源頭解決問題
- 邏輯清晰簡單
- 靈活可調
- 提升數據質量

⚠️ **注意**：
- 會減少樣本數（但提升質量）
- 需要實驗找最佳閾值
- 建議配合 TB 參數調整

🎯 **推薦流程**：
分析 → 設閾值 → 生成數據 → 檢查效果 → 迭代優化

