# 數據質量驗證策略

**日期**: 2025-10-21
**版本**: v1.0
**原則**: **在源頭保證質量，在下游檢查質量**

---

## 核心原則

### ✅ 正確做法：分層防禦

```
階段 1: 預處理（preprocess_single_day.py）
   ↓
   角色: 數據清洗與質量保證
   責任: 移除異常值、填充缺失值、驗證數據完整性
   輸出: 乾淨的 NPZ 檔案（無 mids=0、無 NaN）

階段 2: 訓練數據生成（extract_tw_stock_data_v6.py）
   ↓
   角色: 數據質量檢查與報警
   責任: 驗證輸入數據質量、報告問題、拒絕處理異常數據
   輸出: 訓練集 NPZ（或報錯）
```

### ❌ 錯誤做法：下游修補

```python
# 錯誤示範（不要這樣做）
def process_data(data):
    # 靜默修補數據問題
    data = data.replace(0, np.nan)
    data = data.ffill()
    # ... 繼續處理
```

**問題**：
1. ❌ 掩蓋了上游的數據質量問題
2. ❌ 無法追溯問題根源
3. ❌ 可能引入新的偏差

---

## 實作策略

### 階段 1: 預處理階段（修復問題）

**文件**: `scripts/preprocess_single_day.py`

#### 責任
1. **清洗原始數據**
   - 移除試撮（trial=1）
   - 移除交易時間外數據
   - 移除價差異常數據

2. **1Hz 聚合**
   - 按秒聚合事件
   - 前向填充（ffill, 最多 120 秒）
   - **移除無法填充的缺失桶**（mask=2, mids=0）

3. **驗證輸出質量**
   ```python
   # 確保沒有 mids=0
   if (mids == 0).any():
       logging.warning(f"發現 {(mids == 0).sum()} 個 mids=0")
       valid_mids = mids > 0
       features = features[valid_mids]
       mids = mids[valid_mids]
   ```

#### 輸出保證
- ✅ `mids` 全部 > 0
- ✅ 無 NaN 值
- ✅ `features` 與 `mids` 長度匹配

---

### 階段 2: 訓練數據生成階段（檢查問題）

**文件**: `scripts/extract_tw_stock_data_v6.py`

#### 責任
1. **驗證輸入數據**（載入時）
   ```python
   def validate_preprocessed_data(features, mids, meta, npz_path):
       """不修補，只檢查"""
       # 檢查 1: mids=0
       if (mids == 0).any():
           logging.error(f"❌ 發現 mids=0，檔案: {npz_path}")
           return False

       # 檢查 2: NaN
       if np.isnan(mids).any():
           logging.error(f"❌ 發現 NaN，檔案: {npz_path}")
           return False

       # ... 其他檢查
       return True
   ```

2. **驗證計算結果**（處理時）
   ```python
   def ewma_vol(close):
       """假設輸入已清洗"""
       # 數據質量檢查（不修補）
       if (close == 0).any():
           raise ValueError("❌ mids=0 應在預處理階段移除！")

       # 正常計算
       ret = np.log(close).diff()
       ...
   ```

3. **報告數據質量問題**
   ```python
   logging.warning(f"⚠️ 跳過有問題的數據: {npz_path}")
   global_stats["data_quality_errors"] += 1
   ```

#### 輸出行為
- ✅ 如果數據質量通過：正常生成訓練集
- ⚠️ 如果發現問題：記錄錯誤、跳過該文件、返回狀態碼 2
- ❌ 如果問題嚴重：拋出異常、停止執行

---

## 數據質量檢查清單

### 載入時檢查（validate_preprocessed_data）

| 檢查項 | 條件 | 行動 |
|--------|------|------|
| mids=0 | `(mids == 0).any()` | ❌ 記錄錯誤，跳過檔案 |
| NaN 值 | `np.isnan(mids).any()` | ❌ 記錄錯誤，跳過檔案 |
| 負價格 | `(mids < 0).any()` | ❌ 記錄錯誤，跳過檔案 |
| NaN 特徵 | `np.isnan(features).any()` | ❌ 記錄錯誤，跳過檔案 |
| 長度匹配 | `features.shape[0] != len(mids)` | ❌ 記錄錯誤，跳過檔案 |

### 計算時檢查（ewma_vol, tb_labels）

| 檢查項 | 位置 | 行動 |
|--------|------|------|
| mids=0 | ewma_vol 開頭 | 🚨 拋出 ValueError |
| NaN 價格 | ewma_vol 開頭 | 🚨 拋出 ValueError |
| 負價格 | ewma_vol 開頭 | 🚨 拋出 ValueError |
| NaN 收益率 | tb_labels 計算後 | 🚨 拋出 ValueError |

---

## 錯誤訊息設計

### 良好的錯誤訊息範例

```python
logging.error(
    f"❌ 數據質量錯誤 [{symbol} @ {date}]\n"
    f"   發現 {zero_count} 個 mids=0 ({zero_pct:.1f}%)\n"
    f"   檔案: {npz_path}\n"
    f"   → 預處理階段 (preprocess_single_day.py) 應該移除這些點！"
)
```

**優點**：
- ✅ 清楚說明問題（mids=0）
- ✅ 提供定量資訊（數量、百分比）
- ✅ 指出問題來源（哪個檔案）
- ✅ 建議修復方法（重新預處理）

### 糟糕的錯誤訊息範例

```python
# ❌ 不要這樣做
logging.error("數據有問題")  # 太模糊
logging.error(f"錯誤: {e}")  # 沒有上下文
# 或更糟：完全不報錯，靜默修補
```

---

## 測試與驗證

### 測試腳本

```bash
# 完整流水線測試
test_complete_pipeline.bat
```

### 預期行為

#### 情境 1: 數據質量良好 ✅

```
[預處理階段]
✅ 聚合後: 16,197 時間點，223 個股票
✅ 通過過濾: 112 檔股票

[訓練數據生成階段]
✅ 載入 NPZ: 112 個檔案
✅ 數據質量檢查: 全部通過
✅ 成功生成訓練集: 5,584,553 樣本
```

#### 情境 2: 發現數據質量問題 ⚠️

```
[預處理階段]
⚠️ 警告：發現 15 個 mids=0 的點（可能是數據問題）
✅ 已移除，剩餘 16,182 時間點

[訓練數據生成階段]
✅ 載入 NPZ: 112 個檔案
✅ 數據質量檢查: 全部通過
✅ 成功生成訓練集
```

#### 情境 3: 嚴重數據質量問題 ❌

```
[預處理階段]
（假設有 bug，沒有移除 mids=0）

[訓練數據生成階段]
❌ 數據質量錯誤 [3048 @ 20250902]
   發現 15 個 mids=0 (0.1%)
   檔案: data/preprocessed_v5_1hz/daily/20250902/3048.npz
   → 預處理階段應該移除這些點！
⚠️ 跳過有問題的數據

最終統計:
  ⚠️ 數據質量錯誤: 1 個檔案
     → 請檢查上方的錯誤訊息，並重新運行預處理！

退出碼: 2（警告）
```

---

## 修復流程

### 如果發現數據質量問題

#### 步驟 1: 檢查錯誤日誌

```bash
# 查找錯誤訊息
grep "❌ 數據質量錯誤" logs/extract_v6.log
```

#### 步驟 2: 重新預處理問題檔案

```bash
# 單檔重新預處理
python scripts\preprocess_single_day.py ^
    --input data\temp\20250902.txt ^
    --output-dir data\preprocessed_v5_1hz ^
    --config configs\config_pro_v5_ml_optimal.yaml
```

#### 步驟 3: 驗證修復

```bash
# 重新檢查該檔案
python -c "
import numpy as np
data = np.load('data/preprocessed_v5_1hz/daily/20250902/3048.npz')
mids = data['mids']
print(f'mids min: {mids.min():.2f}')
print(f'mids max: {mids.max():.2f}')
print(f'零值數量: {(mids == 0).sum()}')
"
```

#### 步驟 4: 重新生成訓練數據

```bash
python scripts\extract_tw_stock_data_v6.py ^
    --preprocessed-dir data\preprocessed_v5_1hz ^
    --output-dir data\processed_v6 ^
    --config configs\config_pro_v5_ml_optimal.yaml
```

---

## 退出狀態碼

| 狀態碼 | 意義 | 行動 |
|--------|------|------|
| 0 | ✅ 成功，無問題 | 繼續下一步 |
| 1 | ❌ 嚴重錯誤（無法繼續） | 檢查配置或輸入 |
| 2 | ⚠️ 警告（有問題但已跳過） | 重新預處理問題檔案 |

### 使用範例

```bash
python scripts\extract_tw_stock_data_v6.py ...
if %ERRORLEVEL% EQU 2 (
    echo ⚠️ 發現數據質量問題，建議重新預處理
)
```

---

## 相關文件

- [Bug 修復報告: mids=0](docs/BUG_FIX_MIDS_ZERO.md)
- [數據質量驗證完整流程](docs/數據質量驗證完整流程(產生訓練資料).md)
- [1Hz 聚合輸出分析](docs/1HZ_OUTPUT_ANALYSIS_REPORT.md)

---

## 最佳實踐總結

### ✅ Do（推薦做法）

1. **在最早階段保證質量**
   ```python
   # 預處理階段：移除問題
   if (mids == 0).any():
       mids = mids[mids > 0]
   ```

2. **在下游檢查質量**
   ```python
   # 訓練數據生成階段：檢查並報警
   if (mids == 0).any():
       raise ValueError("數據應該已清洗！")
   ```

3. **提供清晰的錯誤訊息**
   ```python
   logging.error(
       f"❌ 問題描述\n"
       f"   詳細資訊\n"
       f"   → 修復建議"
   )
   ```

4. **使用退出狀態碼**
   ```python
   if has_warnings:
       return 2  # 警告狀態
   ```

### ❌ Don't（避免做法）

1. **不要在下游靜默修補**
   ```python
   # ❌ 不要這樣做
   data = data.fillna(0)  # 掩蓋問題
   ```

2. **不要使用模糊的錯誤訊息**
   ```python
   # ❌ 不要這樣做
   logging.error("有問題")
   ```

3. **不要忽略數據質量問題**
   ```python
   # ❌ 不要這樣做
   try:
       process_data(data)
   except:
       pass  # 忽略錯誤
   ```

---

**作者**: Claude (Sonnet 4.5)
**審核**: 待用戶確認
**版本**: v1.0
**狀態**: ✅ 已實作
