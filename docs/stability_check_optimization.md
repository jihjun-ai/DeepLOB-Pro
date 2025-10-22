# Stability Check 效能優化報告

## 問題描述

執行 `scripts/stability_check.py` 時，腳本在提取訓練特徵階段停滯無進度：

```
窗口 1:
訓練期: 20250901 ~ 20250908
測試期: 20250909 ~ 20250910
```

## 根本原因分析

### 計算瓶頸

**原始計算量**（未優化）：
- 6-8 個訓練日 × ~200 檔股票/日 = ~1,200-1,600 個 stock-day
- 每個 stock 約 15,000 個資料點（1Hz 採樣，~4 小時交易時間）
- 滑窗採樣：15,000 個時間點，每個點都創建一個樣本
- Triple-Barrier 標籤：每個點向後檢查 max_holding=200 個點
- **總計算量**：~1,500 stocks × 15,000 points × 200 holding = **45億次操作**

### 關鍵問題

1. **無進度顯示**：用戶無法判斷是當機還是正在運行
2. **過密採樣**：每個時間點都取樣，大量重疊
3. **過長持倉期**：max_holding=200（200秒 ≈ 3.3分鐘）對 1Hz 數據過長
4. **單股樣本無上限**：某些股票可能產生數萬個樣本

---

## 優化方案

### 1. 進度日誌（可見性）⭐⭐⭐⭐⭐

**位置**：`extract_features_labels()` 函數（Line 212-217）

```python
for symbol, features, mids, meta in day_data:
    processed += 1

    # 每處理 20 個股票顯示一次進度
    if processed % 20 == 0 or processed == total_stocks:
        logging.info(f"    處理進度: {processed}/{total_stocks} 股票 ({processed/total_stocks*100:.1f}%)")
```

**效果**：
- ✅ 用戶可實時看到進度
- ✅ 可估計剩餘時間
- ✅ 避免誤判為當機

---

### 2. 滑窗降採樣（效率）⭐⭐⭐⭐⭐

**位置**：`extract_features_labels()` 函數（Line 242）

```python
# 原始（未優化）
for t in range(seq_len - 1, max_t):  # 每個點都取樣
    window = features[t - seq_len + 1:t + 1, :]
    # ...

# 優化後（stride=10）
for t in range(seq_len - 1, max_t, stride):  # 每 10 個點取一個樣本
    window = features[t - seq_len + 1:t + 1, :]
    # ...
```

**效果**：
- 樣本數減少：15,000 → 1,500 (**降低 90%**)
- 計算時間減少：**~10倍加速**
- 訊號保留：每 10 秒取一個樣本，對於穩定性檢測足夠

**函數簽名**：
```python
def extract_features_labels(day_data: List[Tuple], seq_len: int = 100,
                           stride: int = 10, max_samples_per_stock: int = 500)
```

---

### 3. 單股樣本上限（記憶體）⭐⭐⭐⭐

**位置**：`extract_features_labels()` 函數（Line 243-244）

```python
stock_samples = 0

for t in range(seq_len - 1, max_t, stride):
    if stock_samples >= max_samples_per_stock:  # 限制每股最多 500 個樣本
        break

    # ... 處理樣本 ...
    stock_samples += 1
```

**效果**：
- ✅ 避免個別股票產生過多樣本（某些股票可能有 >10,000 個樣本）
- ✅ 記憶體使用可控
- ✅ 平衡各股權重

---

### 4. 降低持倉期（準確性）⭐⭐⭐⭐

**位置**：`simple_tb_labels()` 函數（Line 229）

```python
# 原始
y_labels = simple_tb_labels(close, vol, max_holding=200)  # 200 秒 ≈ 3.3 分鐘

# 優化後
y_labels = simple_tb_labels(close, vol, max_holding=40)  # 40 秒（更合理）
```

**理由**：
- 1Hz 數據，max_holding=200 意味著向後看 200 秒（3.3 分鐘）
- 對於高頻交易訊號，40 秒（40 個 tick）更合理
- 計算量減少：**200 → 40 (5倍加速)**

---

### 5. 詳細日誌（可追蹤性）⭐⭐⭐

**位置**：`rolling_backtest()` 函數（Line 336, 348）

```python
logging.info(f"  提取訓練特徵（{len(train_data)} 個 stock-day）...")
X_train, y_train, stock_train = extract_features_labels(train_data, seq_len)

logging.info(f"  提取測試特徵（{len(test_data)} 個 stock-day）...")
X_test, y_test, stock_test = extract_features_labels(test_data, seq_len)
```

**效果**：
- ✅ 清楚顯示當前階段
- ✅ 顯示待處理數據量
- ✅ 配合進度日誌，完整追蹤

---

## 優化效果總結

| 項目 | 優化前 | 優化後 | 提升 |
|------|-------|-------|------|
| **樣本密度** | 每個點 | 每 10 個點 | 10x |
| **單股上限** | 無限制 | 500 個 | 可控 |
| **持倉期** | 200 | 40 | 5x |
| **進度可見性** | ❌ 無 | ✅ 每 20 股 | - |
| **估計加速** | - | - | **~30-50x** |

### 計算量對比

**優化前**：
- 1,500 stocks × 15,000 points × 200 holding = **45億次操作**

**優化後**：
- 1,500 stocks × (min(1,500, 500) samples) × 40 holding = **9,000萬次操作**
- **減少 ~98%**

---

## 使用方式

### 執行命令（無需修改）

```bash
python scripts\stability_check.py ^
    --preprocessed-dir data\preprocessed_v5_1hz ^
    --output-dir data\stability_check ^
    --train-window 6 ^
    --test-window 2 ^
    --step 4
```

### 預期輸出（優化後）

```
============================================================
時間穩定性驗證開始
============================================================

...

============================================================
滾動回測開始
============================================================
訓練窗口: 6 天
測試窗口: 2 天
滾動步長: 4 天
總交易日: 10

窗口 1:
  訓練期: 20250901 ~ 20250908
  測試期: 20250909 ~ 20250910
  提取訓練特徵（1234 個 stock-day）...
    處理進度: 20/1234 股票 (1.6%)     ← 新增進度顯示
    處理進度: 40/1234 股票 (3.2%)
    處理進度: 60/1234 股票 (4.9%)
    ...
    處理進度: 1234/1234 股票 (100.0%)
  提取測試特徵（456 個 stock-day）...
    處理進度: 20/456 股票 (4.4%)
    ...
```

---

## 參數調整建議

### stride（採樣步長）

```python
# 預設：stride=10（每 10 個點取一個）
stride = 10  # 適合大多數情況

# 更快但可能損失訊號：
stride = 20  # 每 20 個點，更快但樣本更少

# 更精細但更慢：
stride = 5   # 每 5 個點，更多樣本
```

### max_samples_per_stock（單股上限）

```python
# 預設：500 個/股
max_samples_per_stock = 500  # 平衡性能與代表性

# 更多樣本（更慢）：
max_samples_per_stock = 1000

# 更少樣本（更快）：
max_samples_per_stock = 200
```

### max_holding（持倉期）

```python
# 預設：40（適合 1Hz 數據）
max_holding = 40  # 40 秒

# 更長持倉（更慢，適合低頻訊號）：
max_holding = 100  # 100 秒

# 更短持倉（更快，適合超高頻）：
max_holding = 20  # 20 秒
```

---

## 後續建議

### 1. 監控實際效能

運行優化版本後，記錄：
- 每個窗口的實際耗時
- 總耗時
- 記憶體使用峰值

### 2. 驗證訊號質量

確認優化後的穩定性指標是否合理：
- AUC 平均應 > 0.50
- IC 正比例應 > 60%
- 與未優化版本（如果有）對比結果

### 3. 進一步優化（如需要）

如果仍然太慢，可考慮：
- 增加 stride（例如 stride=20）
- 降低 max_samples_per_stock（例如 200）
- 使用更少的訓練窗口（例如 train_window=4）
- 使用多進程並行處理（需修改代碼）

---

## 版本歷史

- **v1.0** (2025-10-21)：初始版本，無優化
- **v1.1** (2025-10-22)：效能優化版本
  - 新增進度日誌
  - 新增滑窗降採樣（stride=10）
  - 新增單股樣本上限（500）
  - 降低持倉期（40）
  - 詳細階段日誌

---

**優化完成日期**：2025-10-22
**優化版本**：v1.1
**預期加速比**：30-50倍
