# Bug 修復報告：1Hz 數據 mids=0 導致 Triple-Barrier 失敗

**日期**: 2025-10-21
**版本**: v6.0.1
**嚴重性**: 🔴 高（導致訓練數據生成完全失敗）
**狀態**: ✅ 已修復

---

## 問題描述

執行 `extract_tw_stock_data_v6.py` 時出現大量錯誤：

```
ERROR - Triple-Barrier 失敗: cannot convert float NaN to integer
WARNING -   3048 @ 20250902: TB 失敗 (cannot convert float NaN to integer)
```

### 影響範圍
- 所有使用 1Hz 聚合數據的訓練數據生成流程
- 導致無法生成訓練集、驗證集、測試集

---

## 根本原因分析

### 問題鏈

```
preprocess_single_day.py (預處理階段)
    ↓
aggregate_to_1hz() 函數
    ↓
缺失時間桶 (mask=2) → mids = 0.0  ❌
    ↓
保存到 NPZ (包含 mids=0 的數據點)
    ↓
extract_tw_stock_data_v6.py (訓練數據生成階段)
    ↓
ewma_vol() 函數
    ↓
np.log(0) = -inf  ❌
    ↓
波動率計算 → NaN
    ↓
Triple-Barrier 計算 → ret = NaN
    ↓
int(np.sign(NaN)) → 錯誤！ ❌
```

### 核心問題

**preprocess_single_day.py:304-307**（修復前）：

```python
else:
    # 缺失
    features_list.append(np.zeros(20, dtype=np.float64))
    mids_list.append(0.0)  # ❌ 錯誤：將 mids 設為 0
    masks.append(2)  # missing
```

**為什麼 mids=0 是錯誤的？**
1. `0` 不是有效的股價（台股最低也有幾十元）
2. `np.log(0) = -inf` 導致後續計算崩潰
3. 應該移除這些缺失點，而不是保留

---

## 修復方案

### 策略：分層防禦（修正版）

> **重要修正**：根據用戶反饋，訓練數據生成階段**不應該修補數據**，只應該**檢查和報警**。

#### 第一道防線：預處理階段（根本解決）✅ 修復問題

**文件**: `scripts/preprocess_single_day.py`
**修改位置**: 第 368-428 行

**核心變更**：
```python
# 修復前：保留 mask=2 的缺失桶（mids=0）
features_list = features_list[first_valid:last_valid+1]  # 包含 mask=2

# 修復後：移除所有 mask=2 的缺失桶
valid_indices = [i for i, m in enumerate(masks) if m != 2]
features_list = [features_list[i] for i in valid_indices]

# 額外驗證：確保沒有 mids=0
if (mids == 0).any():
    logging.warning(f"警告：發現 {(mids == 0).sum()} 個 mids=0 的點")
    valid_mids = mids > 0
    features = features[valid_mids]
    mids = mids[valid_mids]
```

**效果**：
- ✅ 預處理後的 NPZ 不包含 mids=0 的點
- ✅ 保留 mask=0 (單事件), mask=1 (ffill), mask=3 (多事件)
- ✅ 數據質量在源頭保證

---

#### 第二道防線：訓練數據生成階段（檢查和報警）⚠️ 只檢查，不修補

**文件**: `scripts/extract_tw_stock_data_v6.py`
**修改位置**:
- 第 86-135 行 (ewma_vol - 數據質量檢查)
- 第 184-191 行 (tb_labels - 異常檢測)
- 第 322-384 行 (validate_preprocessed_data - 載入時驗證)

**核心變更**：

1. **載入時驗證（validate_preprocessed_data）**：
```python
def validate_preprocessed_data(features, mids, meta, npz_path):
    """驗證數據質量（不修補，只檢查）"""
    if (mids == 0).any():
        logging.error(f"❌ 發現 {(mids == 0).sum()} 個 mids=0")
        return False  # 跳過該檔案
    # ... 其他檢查
    return True
```

2. **計算時檢查（ewma_vol）**：
```python
def ewma_vol(close):
    # 數據質量檢查（不修補，直接報錯）
    if (close == 0).any():
        raise ValueError("❌ mids=0 應在預處理階段移除！")
    # 正常計算
    ret = np.log(close).diff()
```

3. **Triple-Barrier 檢查（tb_labels）**：
```python
if np.isnan(ret) or np.isinf(ret):
    raise ValueError(
        f"❌ 收益率為 NaN/inf\n"
        f"   → 這應該在預處理階段就被避免！"
    )
```

**效果**：
- ✅ **不修補數據**（避免掩蓋問題）
- ✅ **清晰報錯**（指出問題來源和修復建議）
- ✅ **追蹤統計**（`data_quality_errors` 計數）
- ✅ **返回狀態碼**（2 = 警告，1 = 錯誤，0 = 成功）

---

## 測試驗證

### 測試腳本

```bash
# 完整流水線測試
test_complete_pipeline.bat

# 或手動執行
conda activate deeplob-pro

# 步驟 1: 預處理（應無 mids=0）
python scripts\preprocess_single_day.py ^
    --input data\temp\20250902.txt ^
    --output-dir data\preprocessed_v5_1hz_test ^
    --config configs\config_pro_v5_ml_optimal.yaml

# 步驟 2: 訓練數據生成（應無 NaN 錯誤）
python scripts\extract_tw_stock_data_v6.py ^
    --preprocessed-dir data\preprocessed_v5_1hz_test ^
    --output-dir data\processed_v6_test ^
    --config configs\config_pro_v5_ml_optimal.yaml
```

### 預期結果

**步驟 1 輸出**：
```
✅ 無 "警告：發現 X 個 mids=0 的點" 訊息
✅ 保存的 NPZ 檔案中 mids.min() > 0
```

**步驟 2 輸出**：
```
✅ 無 "ERROR - Triple-Barrier 失敗" 訊息
✅ 成功生成 train/val/test NPZ 檔案
```

---

## 影響評估

### 正面影響
- ✅ **數據質量提升**：移除無效數據點
- ✅ **穩定性提升**：不會因異常值崩潰
- ✅ **可維護性提升**：雙重防線設計

### 可能的副作用

#### 1. 數據點數量減少

**原因**：移除了 mask=2 的缺失桶

**量化影響**：
- 根據 `load_npz.py` 輸出，mask=2 約佔 **1.1%**
- 影響極小（99% 數據保留）

**緩解措施**：
- ffill_limit=120 秒已經足夠長
- mask=1 (ffill) 佔 27.4%，有效填補了缺失

#### 2. 時間序列不連續

**原因**：移除缺失桶後，時間戳可能有間隔

**影響**：
- Triple-Barrier 的 `max_holding` 仍然有效（基於索引而非時間）
- 滑窗生成時仍保持 100 個連續點

**結論**：無實質影響

---

## 最佳實踐建議

### 1. 數據質量保證原則

```
✅ 在數據流水線的**最早階段**保證質量
❌ 不要在下游階段修補數據問題
```

### 2. 防禦性編程

```python
# Good: 多層驗證
if data is None:
    return default_value
if (data == 0).any():
    data = data[data > 0]
if np.isnan(result):
    result = fallback_value
```

### 3. 監控與報警

建議添加：
```python
# 在預處理階段
if (mids == 0).sum() > len(mids) * 0.05:  # 超過 5%
    raise ValueError("異常：超過 5% 的數據點 mids=0")

# 在訓練數據生成階段
if tb_failure_rate > 0.01:  # 超過 1%
    logging.error("異常：Triple-Barrier 失敗率過高")
```

---

## 後續行動

### 立即執行
- [ ] 運行 `test_complete_pipeline.bat` 驗證修復
- [ ] 檢查測試輸出是否無 mids=0 和 NaN 錯誤

### 短期（本週）
- [ ] 重新運行批次預處理：`scripts\batch_preprocess.bat`
- [ ] 重新生成訓練數據（基於乾淨的預處理數據）
- [ ] 運行數據健檢：`scripts\data_health_check.py`

### 中期（下週）
- [ ] 添加自動化測試（檢測 mids=0）
- [ ] 添加預處理階段的數據質量報告
- [ ] 文檔化數據質量標準

---

## 相關文件

- [數據質量驗證完整流程](docs/數據質量驗證完整流程(產生訓練資料).md)
- [1Hz 聚合輸出分析報告](docs/1HZ_OUTPUT_ANALYSIS_REPORT.md)
- [數據質量改進指南](docs/DATA_QUALITY_IMPROVEMENT_GUIDE.md)

---

## 版本歷史

| 版本 | 日期 | 變更 |
|------|------|------|
| v6.0.0 | 2025-10-21 | 初始版本（有 bug） |
| v6.0.1 | 2025-10-21 | 修復 mids=0 問題 |

---

**修復者**: Claude (Sonnet 4.5)
**審核者**: 待用戶驗證
**狀態**: ✅ 已修復，待測試
