# 修改總結：數據質量保證策略改進

**日期**: 2025-10-21
**版本**: v6.0.3
**狀態**: ✅ 完成，待測試

---

## 問題概述

執行 `extract_tw_stock_data_v6.py` 時出現大量錯誤：
```
ERROR - Triple-Barrier 失敗: cannot convert float NaN to integer
```

**根本原因**：預處理階段的 1Hz 聚合函數將缺失的時間桶填充為 `mids=0`，導致下游計算失敗。

---

## 修改文件

### 1. `scripts/preprocess_single_day.py` ✅ 根本修復

**位置**: 第 368-428 行

**修改內容**：
- 移除所有 `mask=2` (缺失) 的時間桶
- 添加 `mids=0` 驗證並自動清除
- 確保輸出的 NPZ 數據質量

**修改前**：
```python
# 保留缺失桶，mids=0
mids_list.append(0.0)  # ❌ 問題
```

**修改後**：
```python
# 移除缺失桶
valid_indices = [i for i, m in enumerate(masks) if m != 2]
features_list = [features_list[i] for i in valid_indices]

# 額外驗證
if (mids == 0).any():
    logging.warning(f"發現 {(mids == 0).sum()} 個 mids=0")
    valid_mids = mids > 0
    features = features[valid_mids]
```

---

### 2. `scripts/extract_tw_stock_data_v6.py` ✅ 檢查和報警

**新增函數**: `validate_preprocessed_data()` (第 322-384 行)

**功能**：
- ✅ 載入 NPZ 時驗證數據質量
- ✅ 檢查 mids=0、NaN、負值、形狀不匹配
- ✅ **不修補數據**，只報錯和跳過
- ✅ 記錄到 `data_quality_errors` 統計

**修改函數**: `ewma_vol()` (第 86-135 行)

**修改前**（v6.0.1 - 錯誤版本）：
```python
# ❌ 靜默修補數據
close_clean = close.replace(0, np.nan)
close_clean = close_clean.ffill()
```

**修改後**（v6.0.2 - 正確版本）：
```python
# ✅ 檢查並報錯
if (close == 0).any():
    raise ValueError(
        f"❌ 發現 {(close == 0).sum()} 個 mids=0\n"
        f"   → 這應該在預處理階段移除！"
    )
```

**修改函數**: `tb_labels()` (第 184-191 行)

**修改前**（v6.0.1）：
```python
# ❌ 靜默修補
if np.isnan(ret) or np.isinf(ret):
    label = 0
    ret = 0.0
```

**修改後**（v6.0.2）：
```python
# ✅ 拋出異常
if np.isnan(ret) or np.isinf(ret):
    raise ValueError(
        f"❌ 收益率為 NaN/inf\n"
        f"   entry_price={entry_price}, exit_price={exit_price}\n"
        f"   → 這應該在預處理階段避免！"
    )
```

**新增統計**：
```python
global_stats = {
    ...
    "data_quality_errors": 0,  # 新增
}
```

**新增退出狀態碼**：
```python
if global_stats['data_quality_errors'] > 0:
    logging.warning("⚠️ 發現數據質量問題")
    return 2  # 警告狀態
```

---

## 設計原則變更

### ❌ 舊設計（v6.0.1 - 錯誤）

```
預處理階段：修復問題 ✅
     ↓
訓練數據生成階段：修補問題 ❌（掩蓋上游錯誤）
```

### ✅ 新設計（v6.0.2 - 正確）

```
預處理階段：修復問題 ✅
     ↓
訓練數據生成階段：檢查和報警 ✅（暴露上游錯誤）
```

**核心原則**：
1. **在源頭保證質量**（預處理階段）
2. **在下游檢查質量**（訓練數據生成階段）
3. **不在下游修補問題**（避免掩蓋上游錯誤）

---

## 新增文檔

### 1. `docs/DATA_QUALITY_VALIDATION_STRATEGY.md` 📄 NEW

**內容**：
- 數據質量驗證策略完整說明
- 分層防禦設計原則
- 數據質量檢查清單
- 錯誤訊息設計指南
- 修復流程

### 2. `docs/BUG_FIX_MIDS_ZERO.md` 📄 UPDATED

**更新**：
- 修正「雙重防線」為「分層防禦」
- 明確第二道防線**不修補數據**
- 更新修復方案說明

---

## 數據質量檢查機制

### 載入時檢查（5 項）

| 檢查項 | 條件 | 行動 |
|--------|------|------|
| mids=0 | `(mids == 0).any()` | 記錄錯誤，跳過檔案 |
| NaN 值 | `np.isnan(mids).any()` | 記錄錯誤，跳過檔案 |
| 負價格 | `(mids < 0).any()` | 記錄錯誤，跳過檔案 |
| NaN 特徵 | `np.isnan(features).any()` | 記錄錯誤，跳過檔案 |
| 長度不匹配 | `features.shape[0] != len(mids)` | 記錄錯誤，跳過檔案 |

### 計算時檢查（2 項）

| 檢查項 | 位置 | 行動 |
|--------|------|------|
| 異常價格 | `ewma_vol()` | 拋出 ValueError |
| NaN 收益率 | `tb_labels()` | 拋出 ValueError |

---

## 錯誤訊息範例

### 良好的錯誤訊息 ✅

```
❌ 數據質量錯誤 [3048 @ 20250902]
   發現 15 個 mids=0 (0.1%)
   檔案: data/preprocessed_v5_1hz/daily/20250902/3048.npz
   → 預處理階段 (preprocess_single_day.py) 應該移除這些點！
```

**優點**：
- 清楚說明問題
- 提供定量資訊
- 指出問題來源
- 建議修復方法

---

## 退出狀態碼

| 狀態碼 | 意義 | 後續行動 |
|--------|------|----------|
| 0 | ✅ 成功，無問題 | 繼續訓練 |
| 1 | ❌ 嚴重錯誤 | 檢查配置 |
| 2 | ⚠️ 警告（有問題但已跳過） | 重新預處理 |

---

## 測試步驟

### 1. 測試完整流水線

```bash
# 一鍵測試
test_complete_pipeline.bat
```

### 2. 預期行為

#### 情境 A：數據質量良好 ✅

```
[預處理]
✅ 已保存: data/preprocessed_v5_1hz_test/daily/20250902/...
✅ 無 mids=0 警告

[訓練數據生成]
✅ 數據質量檢查: 全部通過
✅ 成功生成訓練集

退出碼: 0
```

#### 情境 B：發現舊版數據（有 mids=0）⚠️

```
[訓練數據生成]
❌ 數據質量錯誤 [3048 @ 20250902]
   發現 15 個 mids=0 (0.1%)
   → 預處理階段應該移除這些點！
⚠️ 跳過有問題的數據

統計:
  ⚠️ 數據質量錯誤: 1 個檔案
     → 請重新運行預處理！

退出碼: 2
```

---

## 後續行動

### 立即執行 ✅

- [ ] 運行 `test_complete_pipeline.bat` 驗證修復
- [ ] 檢查測試輸出是否無錯誤

### 短期（本週）

- [ ] 重新預處理所有數據：`scripts\batch_preprocess.bat`
- [ ] 生成新的訓練數據（基於乾淨的 NPZ）
- [ ] 運行數據健檢：`scripts\data_health_check.py`

### 中期（下週）

- [ ] 添加自動化測試（CI/CD）
- [ ] 監控數據質量趨勢
- [ ] 更新訓練文檔

---

## 相關資源

### 文檔
- [數據質量驗證策略](docs/DATA_QUALITY_VALIDATION_STRATEGY.md) ⭐ NEW
- [Bug 修復報告](docs/BUG_FIX_MIDS_ZERO.md)
- [數據質量驗證完整流程](docs/數據質量驗證完整流程(產生訓練資料).md)

### 腳本
- `scripts/preprocess_single_day.py` - 預處理（修復問題）
- `scripts/extract_tw_stock_data_v6.py` - 訓練數據生成（檢查問題）
- `test_complete_pipeline.bat` - 完整流水線測試

---

## 版本歷史

| 版本 | 日期 | 變更 | 狀態 |
|------|------|------|------|
| v6.0.0 | 2025-10-21 早 | 初始版本 | ❌ 有 bug |
| v6.0.1 | 2025-10-21 中 | 修復 mids=0（但下游靜默修補） | ⚠️ 設計錯誤 |
| v6.0.2 | 2025-10-21 中 | 改為檢查和報警（不修補） | ✅ 正確設計 |
| v6.0.3 | 2025-10-21 晚 | 新增 5 個高優先級改進 | ✅ 功能增強 |

### v6.0.3 新增功能

基於 ChatGPT 建議和數據質量驗證策略，實作了 5 個改進：

1. **C.1 - 隨機種子記錄** ✅
   - 在 metadata 中記錄 `SPLIT_SEED`
   - 確保數據切分可復現

2. **E.1 - 標籤邊界檢查** ✅
   - 檢查標籤是否只包含 [0, 1, 2]
   - 檢查標籤是否含 NaN

3. **E.2 - 權重邊界檢查** ✅
   - 檢查權重是否包含 NaN/inf/負值/全零
   - 輸出權重統計信息

4. **E.3 - Neutral 比例警告** ✅
   - 自動檢測 Class 1 比例是否異常
   - 比例 < 15% 或 > 60% 時發出警告
   - 提供調整建議

5. **A.1 - 降低 ffill_limit** ✅
   - 從 120 秒降至 60 秒
   - 減少長期 ffill 造成的假訊號

6. **A.2 - 滑窗品質過濾** ✅
   - 新增 `ffill_quality_threshold` 配置參數
   - 過濾 ffill 占比 > 50% 的窗口
   - 提升訓練樣本質量

7. **G.1 - 增強錯誤報告** ✅
   - 空數據時提供詳細診斷步驟
   - 清晰的錯誤原因和修復建議

**詳細文檔**: [docs/V6_IMPLEMENTATION_CHANGELOG.md](docs/V6_IMPLEMENTATION_CHANGELOG.md)

---

**修改者**: Claude (Sonnet 4.5)
**審核者**: User (正確指出設計問題)
**狀態**: ✅ 實作完成，待測試
**下一步**: 執行 `test_v6_pipeline.bat` 驗證所有新功能
