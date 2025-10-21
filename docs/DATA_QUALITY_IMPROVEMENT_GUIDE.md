# 數據質量改進指南 - 5 個必要條件完整檢查

**目標**: 確保 `batch_preprocess.bat + extract_tw_stock_data_v6.py` 產生的數據滿足「具有學習價值、且可以學起來」的標準

**版本**: v1.0
**日期**: 2025-10-21
**狀態**: 已完成診斷，部分修復，待進一步改進

---

## 📋 一句話定義

**「具有學習價值、且可以學起來」= 這批資料對你的任務含有穩定、可重複的統計結構（訊號），而且在避免洩漏、正確切分與適當模型/訓練下，能在看不見的資料上持續打敗天真基準，並轉化成實際效益。**

---

## 🎯 5 個必要條件檢查表

| # | 必要條件 | 當前狀態 | 嚴重度 | 已修復 |
|---|---------|---------|--------|--------|
| 1 | **有明確任務與標籤** | ✅ 通過 | - | ✅ |
| 2 | **不含未來資訊（無洩漏）** | ⚠️ 部分問題 | 🔴 高 | 🟡 部分 |
| 3 | **足量且涵蓋多樣情境** | ⚠️ 樣本偏斜 | 🟡 中 | ❌ |
| 4 | **訊號非瞬時幻覺（穩定性）** | ❌ 缺少驗證 | 🔴 高 | ❌ |
| 5 | **噪聲可被管理** | ✅ 通過 | - | ✅ |

---

## 🔍 詳細診斷結果

### ✅ 必要條件 1: 有明確任務與標籤

#### 當前實作

**任務定義** (清晰 ✅):
```
輸入: t 時刻的五檔 LOB (100 timesteps × 20 features)
輸出: 預測 t+H (H ≤ 200 秒) 的價格方向 {0: 下跌, 1: 持平, 2: 上漲}
方法: Triple-Barrier 標籤（止盈/止損/到期）
```

**標籤可行動性** (✅):
- 在 t 時刻可以決策（只用 ≤ t 的資訊）
- max_holding=200 秒 ≈ 3.3 分鐘，可實際執行
- Triple-Barrier 避免了「事後才知道」的標籤定義

#### 評估: ✅ **完全通過**

---

### ⚠️ 必要條件 2: 不含未來資訊（無洩漏）

#### 發現的地雷（已修復 2/4）

| 地雷 | 位置 | 嚴重度 | 狀態 | 說明 |
|------|------|--------|------|------|
| **1. bfill() 洩漏** | `ewma_vol()` | 🔴 高 | ✅ **已修復** | 使用未來波動率填充過去 |
| **2. day_end_idx 風險** | `tb_labels()` | 🟡 中 | ✅ **已修復** | 可能跨日掃描 |
| **3. 動態閾值洩漏** | `determine_adaptive_threshold()` | 🟡 中 | ⚠️ **已註記** | 使用當日全天資訊決策 |
| **4. Z-Score 時間假設** | `zscore_fit/apply()` | 🟡 中 | ⚠️ **待改進** | 假設統計量在訓練期穩定 |

---

#### 地雷 1: bfill() 洩漏 ✅ **已修復**

**問題**:
```python
# 舊版 (extract_tw_stock_data_v6.py:90)
def ewma_vol(close, halflife=60):
    vol = np.sqrt(var).bfill()  # ❌ 用未來值填充過去
```

**影響**:
- 前 halflife 個時間點的波動率會使用「未來」的數據回填
- 這些時間點的 Triple-Barrier 閾值會使用「未來」的波動率
- **明顯的未來資訊洩漏**

**修復** ([extract_tw_stock_data_v6.py:86-108](scripts/extract_tw_stock_data_v6.py:86-108)):
```python
def ewma_vol(close, halflife=60):
    vol = np.sqrt(var)
    # 修復：使用前向數據的平均值（保守估計）
    if vol.isna().any():
        valid_vols = vol.dropna()
        initial_vol = valid_vols.iloc[:min(100, len(valid_vols))].mean()
        vol = vol.fillna(initial_vol)
    return vol
```

---

#### 地雷 2: day_end_idx 風險 ✅ **已修復**

**問題**:
```python
# 舊版 (extract_tw_stock_data_v6.py:470)
day_end_idx = len(close) - 1 if respect_day_boundary else None
```

- 如果 `respect_day_boundary=False`，會跨日掃描
- 可能使用「明日開盤價」作為 Triple-Barrier 觸發點

**修復** ([extract_tw_stock_data_v6.py:471](scripts/extract_tw_stock_data_v6.py:471)):
```python
# 強制開啟日界保護，避免跨日洩漏
day_end_idx = len(close) - 1  # 始終啟用
```

---

#### 地雷 3: 動態閾值洩漏 ⚠️ **已註記，待改進**

**問題** ([preprocess_single_day.py:478-550](scripts/preprocess_single_day.py:478-550)):
```python
def determine_adaptive_threshold(daily_stats, config):
    """使用當日收盤後的完整統計量來決定過濾閾值"""
    # 計算當天所有股票的波動率分布
    candidates = {
        'P10': np.percentile(range_values, 10),
        'P25': np.percentile(range_values, 25),
        'P50': np.percentile(range_values, 50),
        'P75': np.percentile(range_values, 75),
    }
    # 選擇最接近目標標籤分布的閾值
```

**為什麼是洩漏？**
- 這是在**處理當天所有股票後**，基於當天的完整數據決定閾值
- 在實盤時，你在盤中**無法知道當天所有股票的波動率分布**
- 這是一種「**後見之明洩漏**」(hindsight bias)

**實盤無法複製**:
- 訓練時看起來「聰明地」過濾了低質量股票
- 實盤時需要等收盤才知道分布，無法在盤中決策

**修復方案**（待實作）:

**方案 A: 使用前 N 天滾動統計量**
```python
def determine_threshold_with_lookback(current_date, historical_data, lookback=5):
    """使用前 N 天的統計量決定今天的閾值"""
    past_days = get_past_n_days(current_date, lookback)
    past_range_values = [s['range_pct'] for day in past_days for s in day['stats']]

    # 使用歷史分布的 P50
    threshold = np.percentile(past_range_values, 50)
    return threshold
```

**方案 B: 固定閾值策略**
```python
# 最簡單：直接用固定閾值
FIXED_THRESHOLD = 1.005  # 約等於 P50
```

**方案 C: 使用盤中可得資訊**
```python
def determine_threshold_intraday(current_date, first_hour_stats):
    """使用盤中前 1 小時的統計量"""
    # 可在盤中 10:00 獲得前 1 小時數據
    range_values = [s['range_pct'] for s in first_hour_stats]
    return np.percentile(range_values, 50)
```

**當前狀態**:
- 已在函數中添加⚠️ 警告註解 ([preprocess_single_day.py:486-497](scripts/preprocess_single_day.py:486-497))
- 標註「當前僅用於離線回測，需改進後才可用於實盤」

---

#### 地雷 4: Z-Score 時間假設 ⚠️ **待改進**

**問題** ([extract_tw_stock_data_v6.py:391-409](scripts/extract_tw_stock_data_v6.py:391-409)):
```python
# 計算 Z-Score 參數（訓練集）
train_X_list = []
for sym, n_points, day_data_sorted in train_stocks:
    for date, features, mids in day_data_sorted:
        train_X_list.append(features)  # 包含所有訓練期的數據

Xtr = np.concatenate(train_X_list, axis=0)
mu, sd = zscore_fit(Xtr)  # 計算全訓練集的 μ, σ

# 然後在建立滑窗時
for date, features, mids in day_data_sorted:
    Xn = zscore_apply(features, mu, sd)  # 用全訓練集統計量正規化
```

**這是洩漏嗎？**

**部分洩漏** - 取決於實盤場景：
- ✅ **可接受**: 如果你在實盤時會**定期重新計算**訓練集統計量（如每週更新）
- ✅ **可接受**: 如果你在實盤時使用**固定的歷史統計量**（如只用前 3 個月訓練集）
- ❌ **不可接受**: 用了訓練期間「未來」的統計量（如用第 90 天的數據影響第 10 天的正規化）

**風險分析**:
- 當前實作隱含假設這些統計量在整個訓練期間是穩定的
- 但在金融數據中，分布會漂移（regime change）
- **風險等級**: 🟡 **中等風險**

**改進方案**（待實作）:

**方案 A: 滾動視窗正規化**
```python
def rolling_zscore(features, window=5000):
    """使用滾動視窗計算 Z-Score"""
    mu = pd.Series(features.flatten()).rolling(window, min_periods=100).mean()
    sd = pd.Series(features.flatten()).rolling(window, min_periods=100).std()
    return (features - mu) / sd
```

**方案 B: 按日正規化（更保守）**
```python
def daily_zscore(features):
    """每天獨立正規化（只用當天數據）"""
    mu = features.mean(axis=0)
    sd = features.std(axis=0)
    return (features - mu) / sd
```

**方案 C: Expanding Window（推薦）**
```python
def expanding_zscore(features_by_day):
    """使用 Expanding Window（只用過去數據）"""
    normalized = []
    for i, day_features in enumerate(features_by_day):
        if i == 0:
            mu = day_features.mean(axis=0)
            sd = day_features.std(axis=0)
        else:
            # 只用 ≤ i-1 天的統計量
            past_features = np.concatenate(features_by_day[:i], axis=0)
            mu = past_features.mean(axis=0)
            sd = past_features.std(axis=0)

        normalized.append((day_features - mu) / sd)
    return normalized
```

**當前狀態**: ⚠️ **保留現狀，建議監控 PSI**

---

#### 評估: ⚠️ **部分通過**

- ✅ 已修復 2 個嚴重洩漏（bfill, day_boundary）
- ⚠️ 2 個中等風險待改進（動態閾值, Z-Score）

---

### ⚠️ 必要條件 3: 足量且涵蓋多樣情境

#### 當前狀況

從之前的分析 ([summary.json](data/preprocessed_v5_1hz/daily/20250901/summary.json)):
```json
{
  "total_symbols": 223,
  "passed_filter": 56,
  "filtered_out": 167,
  "filter_method": "P75",
  "predicted_label_dist": {
    "down": 0.5536,
    "neutral": 0.0536,
    "up": 0.3929
  }
}
```

#### 問題分析

| 維度 | 當前 | 理想 | 差距 |
|------|------|------|------|
| **股票數** | 56 檔 | 100+ 檔 | ⚠️ 少 |
| **流動性覆蓋** | 高流動性為主 | 高中低均有 | ⚠️ 偏高 |
| **標籤分布** | 55/5/40 | 30/40/30 | ❌ 嚴重偏斜 |
| **Neutral%** | 5.4% | 30-45% | ❌ 過少 |

#### 根本原因

**P75 過濾太激進** → 只保留 25% 最活躍股票

結果：
- ❌ 缺少中低流動性股票
- ❌ 缺少盤整行情數據（高波動股票上漲/下跌為主）
- ❌ 模型只見過「高波動、高活躍」場景

**實盤風險**:
- 遇到盤整、低流動性時表現可能崩潰
- **樣本選擇偏差** (selection bias)

#### 改進方案

**方案 A: 放寬過濾閾值** (推薦)
```yaml
# configs/config_pro_v5_ml_optimal.yaml

# 當前（激進）
filter_method: P75  # 只保留 25% 股票

# 改為（平衡）
filter_method: P50  # 保留 50% 股票

# 或（保守）
filter_method: P25  # 保留 75% 股票
```

**預期效果**:
- P50: Neutral 預計提升至 15-25%
- P25: Neutral 預計提升至 25-35%

**方案 B: 分層採樣** (最佳)

修改 `preprocess_single_day.py` 實作分層過濾：

```python
def stratified_filtering(daily_stats, target_dist, n_stocks_per_tier=3):
    """按流動性分層，確保各層級都有代表"""

    # 按波動率分成 3 層
    sorted_stats = sorted(daily_stats, key=lambda x: x['range_pct'])
    n = len(sorted_stats)

    tier_1 = sorted_stats[:n//3]       # 低流動性
    tier_2 = sorted_stats[n//3:2*n//3] # 中流動性
    tier_3 = sorted_stats[2*n//3:]     # 高流動性

    # 每層選取 top N
    selected = []
    for tier in [tier_1, tier_2, tier_3]:
        tier_sorted = sorted(tier, key=lambda x: x['range_pct'], reverse=True)
        selected.extend(tier_sorted[:n_stocks_per_tier])

    return selected
```

**方案 C: 多天覆蓋檢查**

確保訓練集跨越不同市場狀態：

```python
def check_market_regime_coverage(daily_summaries):
    """檢查是否涵蓋不同行情類型"""

    regimes = {
        'high_vol': 0,    # 高波動日 (range > 1.5%)
        'low_vol': 0,     # 低波動日 (range < 0.5%)
        'trending_up': 0, # 上漲日 (up > 50%)
        'trending_dn': 0, # 下跌日 (down > 50%)
        'ranging': 0      # 盤整日 (neutral > 40%)
    }

    for day in daily_summaries:
        if day['range_mean'] > 0.015:
            regimes['high_vol'] += 1
        elif day['range_mean'] < 0.005:
            regimes['low_vol'] += 1

        if day['label_dist']['up'] > 0.5:
            regimes['trending_up'] += 1
        elif day['label_dist']['down'] > 0.5:
            regimes['trending_dn'] += 1
        elif day['label_dist']['neutral'] > 0.4:
            regimes['ranging'] += 1

    return regimes
```

#### 評估: ⚠️ **不通過** - 需立即改進

**建議**: 使用方案 A（改為 P50）+ 方案 C（檢查多樣性）

---

### ❌ 必要條件 4: 訊號非瞬時幻覺（穩定性）

#### 當前狀況

**完全沒有穩定性驗證！**

當前流程只產生一次切分的訓練數據，沒有：
- ❌ 滾動回測
- ❌ 多時期驗證
- ❌ PSI/特徵漂移檢測
- ❌ IC 穩定性分析
- ❌ 學習曲線

**風險**:
- 可能在單一時期過擬合
- 無法證明訊號在時間上穩定
- **這是最嚴重的缺失**

#### 改進方案

**新增模組**: 時間穩定性驗證

**檔案**: `scripts/stability_check.py`

```python
def rolling_backtest(
    data_dir: str,
    train_window: int = 30,  # 30 天訓練
    test_window: int = 7,    # 7 天測試
    step: int = 7            # 每 7 天滾動一次
):
    """
    滾動回測驗證穩定性

    輸出：
    - 每個測試窗口的 AUC, IC, Sharpe
    - 穩定性指標（均值、標準差、最小值）
    """

    results = []

    for start_day in range(0, total_days - train_window - test_window, step):
        train_days = days[start_day:start_day + train_window]
        test_days = days[start_day + train_window:start_day + train_window + test_window]

        # 訓練簡單模型
        clf = train_simple_model(train_days)

        # 測試
        metrics = evaluate_on_window(clf, test_days)

        results.append({
            'window_start': start_day,
            'auc': metrics['auc'],
            'ic': metrics['ic'],
            'sharpe': metrics['sharpe']
        })

    # 穩定性統計
    auc_list = [r['auc'] for r in results]
    ic_list = [r['ic'] for r in results]

    stability_report = {
        'auc': {
            'mean': np.mean(auc_list),
            'std': np.std(auc_list),
            'min': np.min(auc_list),
            'positive_ratio': sum(np.array(auc_list) > 0.5) / len(auc_list)
        },
        'ic': {
            'mean': np.mean(ic_list),
            'std': np.std(ic_list),
            'min': np.min(ic_list),
            'positive_ratio': sum(np.array(ic_list) > 0) / len(ic_list)
        }
    }

    return stability_report
```

**判斷標準**:
- ✅ **穩定**: AUC 平均 > 0.52，標準差 < 0.05，正比例 > 80%
- ⚠️ **勉強**: AUC 平均 > 0.50，正比例 > 60%
- ❌ **不穩定**: AUC 經常低於 0.5，或波動劇烈

#### 評估: ❌ **完全缺失** - 需立即實作

---

### ✅ 必要條件 5: 噪聲可被管理

#### 當前實作

✅ **1Hz 時間聚合** ([preprocess_single_day.py:200-398](scripts/preprocess_single_day.py:200-398)):
```python
def aggregate_to_1hz(seq, reducer='last', ffill_limit=120):
    # 將微結構噪聲聚合到秒級
    # 使用 last reducer（減少噪聲）
    # ffill 機制保持連續性
```

✅ **波動率過濾** ([preprocess_single_day.py:478-550](scripts/preprocess_single_day.py:478-550)):
- 移除低波動率、高噪聲股票日

✅ **Triple-Barrier 標籤** ([extract_tw_stock_data_v6.py:94-175](scripts/extract_tw_stock_data_v6.py:94-175)):
- 使用波動率倍數設定閾值（自適應噪聲）
- min_return 過濾微小波動

#### 評估: ✅ **通過**

---

## 🛠️ 改進行動方案

### 立即執行（優先級 🔴 高）

#### 1. 修復未來資訊洩漏 ✅ **已完成**

- [x] 修復 `ewma_vol()` 的 bfill() 問題
- [x] 強制開啟 day_end_idx 保護
- [x] 在動態閾值函數添加警告註解

#### 2. 放寬過濾閾值 ⚠️ **待執行**

**行動**:
```bash
# 修改配置文件
# 將 determine_adaptive_threshold() 的候選閾值限制為 P50

# 或直接使用固定閾值
# 修改 preprocess_single_day.py，跳過動態決策，直接用固定值
```

**預期效果**:
- 保留 50% 股票（而非 25%）
- Neutral 標籤提升至 15-25%
- 涵蓋更多行情類型

#### 3. 實作穩定性驗證 ⚠️ **待執行**

**行動**:
```bash
# 創建 scripts/stability_check.py
# 實作滾動回測模組
# 運行並生成穩定性報告
```

### 中期改進（優先級 🟡 中）

#### 4. 改進 Z-Score 正規化

**行動**: 實作 Expanding Window 正規化

#### 5. 實作分層採樣

**行動**: 修改過濾邏輯，按流動性分層

### 長期優化（優先級 🟢 低）

#### 6. 完整的數據健檢自動化

**行動**: 將所有檢查整合到 CI/CD 流程

---

## 📊 使用數據健檢腳本

### 快速開始

```bash
# 1. 先生成訓練數據
python scripts/extract_tw_stock_data_v6.py \
    --preprocessed-dir ./data/preprocessed_v5_1hz \
    --output-dir ./data/processed_v6 \
    --config configs/config_pro_v5_ml_optimal.yaml

# 2. 運行健檢
python scripts/data_health_check.py \
    --train-npz ./data/processed_v6/npz/stock_embedding_train.npz \
    --val-npz ./data/processed_v6/npz/stock_embedding_val.npz \
    --test-npz ./data/processed_v6/npz/stock_embedding_test.npz \
    --output-dir ./data/processed_v6/health_check/

# 3. 查看報告
cat ./data/processed_v6/health_check/health_check_report.txt
```

### 解讀報告

**健檢項目**:
1. ✅ **明確任務與標籤**: 類別分布、熵、失衡度
2. ✅ **未來資訊洩漏**: 時間打亂測試、未對齊測試
3. ✅ **足量且多樣**: 總樣本數、Neff、股票覆蓋
4. ✅ **穩定性**: PSI、訓練/測試性能差距
5. ✅ **基準對比**: 與天真基準、簡單模型對比

**通過標準**:
- 所有 5 項檢查都標記為 ✅
- 簡單模型（Logistic Regression）AUC > 0.52
- 平衡準確率 > 0.40（三分類隨機 = 0.333）

---

## 📈 判斷口訣（超精簡）

* **能定義** ✅（清楚任務/標籤）
* **能對齊** 🟡（部分洩漏已修復，待改進）
* **能覆蓋** ⚠️（量足但多樣性不足）
* **能外推** ❌（缺少穩定性驗證）
* **能打贏** ❓（待健檢腳本驗證）

---

## 🎯 下一步

### 立即行動（今天）

1. **執行數據健檢腳本**
   ```bash
   python scripts/data_health_check.py --train-npz ... --val-npz ... --test-npz ...
   ```

2. **根據健檢報告決定**:
   - 如果 5 項全通過 → 可以開始訓練
   - 如果未通過 → 按優先級逐項修復

### 本週行動

1. **放寬過濾閾值** → P50 或 P25
2. **重新生成訓練數據**
3. **實作滾動回測模組**
4. **再次運行健檢**

### 本月目標

1. **所有 5 項必要條件通過**
2. **簡單模型 AUC > 0.55**
3. **滾動回測 IC 穩定正值**
4. **開始訓練 DeepLOB**

---

**最後更新**: 2025-10-21
**版本**: v1.0
**狀態**: 已完成診斷，部分修復，待進一步改進
