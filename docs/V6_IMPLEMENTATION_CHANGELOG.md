# V6 實作變更日誌

**日期**: 2025-10-21
**版本**: v6.0.3
**狀態**: ✅ 實作完成，待測試

---

## 變更概述

基於 ChatGPT 建議和數據質量驗證策略，實作了 5 個高優先級改進：

| 改進編號 | 類別 | 說明 | 優先級 | 狀態 |
|---------|------|------|--------|------|
| C.1 | 可復現性 | 記錄隨機種子到 metadata | 高 | ✅ |
| E.1-3 | 驗證 | 標籤/權重邊界檢查 + Neutral 比例警告 | 高 | ✅ |
| A.1 | 數據質量 | 降低 ffill_limit (120→60秒) | 高 | ✅ |
| A.2 | 數據質量 | 滑窗品質過濾 (ffill 占比閾值) | 高 | ✅ |
| G.1 | 錯誤處理 | 增強空數據錯誤報告 | 高 | ✅ |

**核心原則**: **在源頭保證質量，在下游檢查質量**（不修補）

---

## 詳細變更

### C.1: 記錄隨機種子 ✅

**檔案**: `scripts/extract_tw_stock_data_v6.py`

**修改位置**: Line 769

**變更內容**:
```python
# 保存 normalization metadata（含隨機種子）
normalization_meta = {
    "global_mean": global_mean.tolist(),
    "global_std": global_std.tolist(),
    "normalization_method": "z-score (global)",
    "seed": SPLIT_SEED,  # ✅ 新增：記錄隨機種子
}
```

**效果**:
- ✅ 可復現數據切分 (train/val/test)
- ✅ Metadata 包含完整實驗配置
- ✅ 符合科學研究標準

**測試方法**:
```python
import json
meta = json.load(open('data/processed_v6/npz/normalization_meta.json'))
print(f"Seed: {meta['seed']}")  # 應顯示 42
```

---

### E.1: 標籤邊界檢查 ✅

**檔案**: `scripts/extract_tw_stock_data_v6.py`

**修改位置**: Line 627-638

**變更內容**:
```python
# E.1: 標籤邊界檢查（映射後應只有 0,1,2）
if y_tb.isna().any():
    nan_count = y_tb.isna().sum()
    raise ValueError(
        f"❌ 標籤映射失敗：發現 {nan_count} 個 NaN\n"
        f"   可能原因：Triple-Barrier 返回了未定義的值"
    )
if not y_tb.isin([0, 1, 2]).all():
    invalid_labels = y_tb[~y_tb.isin([0, 1, 2])].unique()
    raise ValueError(
        f"❌ 標籤值異常：發現非 [0,1,2] 的值 {invalid_labels}\n"
        f"   → 請檢查 tb_labels() 函數的輸出"
    )
```

**效果**:
- ✅ 捕捉標籤映射錯誤
- ✅ 提供清晰的錯誤診斷
- ✅ 防止無效標籤進入訓練集

**預期行為**:
- 正常情況：靜默通過
- 異常情況：拋出 `ValueError`，顯示具體問題

---

### E.2: 權重邊界檢查 ✅

**檔案**: `scripts/extract_tw_stock_data_v6.py`

**修改位置**: Line 640-674

**變更內容**:
```python
# E.2: 權重邊界檢查（應為有限正值）
if not np.isfinite(w).all():
    inf_count = np.isinf(w).sum()
    nan_count = np.isnan(w).sum()
    raise ValueError(
        f"❌ 樣本權重包含 NaN 或 inf\n"
        f"   NaN: {nan_count}, inf: {inf_count}\n"
        f"   → 請檢查 sample_weights 計算邏輯"
    )

if (w < 0).any():
    neg_count = (w < 0).sum()
    raise ValueError(
        f"❌ 樣本權重包含負值: {neg_count} 個\n"
        f"   → 權重應為非負數"
    )

if (w == 0).all():
    raise ValueError(
        f"❌ 所有樣本權重為 0\n"
        f"   → 請檢查 sample_weights.enabled 和計算參數"
    )

# 正常情況：顯示權重統計
logging.info(
    f"權重統計 - "
    f"最小: {w.min():.6f}, "
    f"最大: {w.max():.6f}, "
    f"平均: {w.mean():.6f}, "
    f"標準差: {w.std():.6f}"
)
```

**效果**:
- ✅ 檢測 NaN/inf/負值/全零權重
- ✅ 提供詳細統計信息
- ✅ 防止無效權重導致訓練失敗

**預期輸出**（正常情況）:
```
權重統計 - 最小: 0.000123, 最大: 2.345678, 平均: 1.000000, 標準差: 0.456789
```

---

### E.3: Neutral 比例警告 ✅

**檔案**: `scripts/extract_tw_stock_data_v6.py`

**修改位置**: Line 737-758

**變更內容**:
```python
# E.3: Neutral (Class 1) 比例檢查
neutral_pct = label_pct[1]
if neutral_pct < 15.0:
    logging.warning(
        f"\n⚠️ {split_name.upper()} 集 Neutral (Class 1) 比例過低！\n"
        f"   當前: {neutral_pct:.1f}%\n"
        f"   目標: 20-45% (config 設計值: 30-40%)\n"
        f"   → 可能原因：\n"
        f"      1. min_return 閾值過小（當前: {config['triple_barrier']['min_return']}）\n"
        f"      2. max_holding 過長（當前: {config['triple_barrier']['max_holding']}）\n"
        f"      3. 市場波動極端（請檢查原始數據）\n"
        f"   → 建議：檢查配置參數或原始數據質量"
    )
elif neutral_pct > 60.0:
    logging.warning(
        f"\n⚠️ {split_name.upper()} 集 Neutral (Class 1) 比例過高！\n"
        f"   當前: {neutral_pct:.1f}%\n"
        f"   目標: 20-45%\n"
        f"   → 可能原因：\n"
        f"      1. min_return 閾值過大\n"
        f"      2. max_holding 過短\n"
        f"      3. 市場橫盤整理期\n"
        f"   → 建議：調整 Triple-Barrier 參數"
    )
```

**效果**:
- ✅ 自動檢測標籤分布異常
- ✅ 提供具體原因分析
- ✅ 建議調整方向

**預期行為**:
- `neutral_pct` ∈ [15%, 60%]: 靜默通過
- `neutral_pct` < 15%: 顯示「過低」警告
- `neutral_pct` > 60%: 顯示「過高」警告

---

### A.1: 降低 ffill_limit ✅

**檔案**: `scripts/preprocess_single_day.py`

**修改位置**: Line 786

**變更內容**:
```python
# A.1: 降低 ffill_limit（避免長期 ffill 造成假訊號）
features, mids, bucket_mask, meta = aggregate_to_1hz(
    events_filtered,
    ffill_limit=60,  # 從 120 秒降至 60 秒
)
```

**理由**:
- ❌ 120 秒 ffill：可能產生 2 分鐘的假訊號
- ✅ 60 秒 ffill：在保留數據和質量間取得平衡
- ✅ 配合 A.2 的品質過濾，進一步提升數據質量

**影響評估**:
- 預期減少 5-15% 的時間點（移除長期缺失的桶）
- 提升訓練數據真實性

**測試方法**:
```bash
# 比較 ffill_limit=120 vs 60 的輸出
python scripts/preprocess_single_day.py ... > log_60.txt
# 檢查 meta['total_aggregated_points']
```

---

### A.2: 滑窗品質過濾 ✅

**檔案**:
1. `scripts/extract_tw_stock_data_v6.py` (Line 705-711)
2. `configs/config_pro_v5_ml_optimal.yaml` (Line 38-43)

**變更內容**:

**1. 配置新增**:
```yaml
# A.2: 滑窗品質過濾（避免過度 ffill 的假訊號）
ffill_quality_threshold: 0.5  # 滑窗內 ffill 占比 > 50% 則跳過
# 說明：
#   - ffill (mask=1) 是前向填充的值，非真實報價
#   - 過多 ffill 表示該時段市場不活躍，可能產生假訊號
#   - 建議值：0.5（50%），可根據實際情況調整（0.3-0.7）
```

**2. 代碼實作**:
```python
# A.2: 滑窗品質過濾
window_mask = bucket_mask[window_start:t + 1]
ffill_ratio = (window_mask == 1).sum() / len(window_mask)
if ffill_ratio > ffill_quality_threshold:
    continue  # 跳過低品質滑窗
```

**3. bucket_mask 載入支援**:
```python
# 修改 load_preprocessed_npz() 以支援 bucket_mask
bucket_mask = data.get('bucket_mask', np.zeros(len(mids), dtype=np.int32))
```

**效果**:
- ✅ 過濾掉「假活躍」時段（大量 ffill）
- ✅ 提升訓練樣本真實性
- ✅ 可配置閾值（靈活調整）

**預期行為**:
- `ffill_ratio` ≤ 50%: 保留樣本
- `ffill_ratio` > 50%: 跳過樣本（不記錄到訓練集）

**統計輸出**（建議新增）:
```python
# 可選：記錄過濾統計
filtered_by_quality = 0
if ffill_ratio > ffill_quality_threshold:
    filtered_by_quality += 1
    continue

logging.info(f"品質過濾: 移除 {filtered_by_quality} 個低品質窗口")
```

---

### G.1: 增強空數據錯誤報告 ✅

**檔案**: `scripts/extract_tw_stock_data_v6.py`

**修改位置**: Line 454-472

**變更內容**:
```python
# G.1: 增強錯誤報告（空數據診斷）
if len(all_data) == 0:
    logging.error(f"\n❌ 沒有可用的預處理數據！")
    logging.error(f"可能原因:")
    logging.error(f"  1. preprocessed_dir 路徑錯誤: {preprocessed_dir}")
    logging.error(f"  2. 沒有運行預處理腳本 (preprocess_single_day.py)")
    logging.error(f"  3. 預處理輸出目錄為空")
    logging.error(f"  4. 所有數據都被 validate_preprocessed_data() 過濾掉")
    logging.error(f"\n診斷步驟:")
    logging.error(f"  1. 檢查路徑: ls {preprocessed_dir}")
    logging.error(f"  2. 檢查子目錄: ls {preprocessed_dir}/daily/YYYYMMDD/")
    logging.error(f"  3. 檢查 NPZ 文件: ls {preprocessed_dir}/daily/YYYYMMDD/*.npz")
    logging.error(f"  4. 查看預處理日誌確認是否有錯誤")
    logging.error(f"\n修復建議:")
    logging.error(f"  → 運行: python scripts/preprocess_single_day.py ...")
    logging.error(f"  → 或使用批次腳本: scripts\\batch_preprocess.bat")
    return 1
```

**效果**:
- ✅ 清晰的錯誤診斷流程
- ✅ 具體的修復建議
- ✅ 縮短排錯時間

**預期輸出**（空數據情況）:
```
❌ 沒有可用的預處理數據！
可能原因:
  1. preprocessed_dir 路徑錯誤: data/wrong_path
  2. 沒有運行預處理腳本 (preprocess_single_day.py)
  ...
診斷步驟:
  1. 檢查路徑: ls data/wrong_path
  ...
修復建議:
  → 運行: python scripts/preprocess_single_day.py ...
```

---

## 測試計劃

### 1. 單元測試（建議新增）

```python
# tests/test_v6_improvements.py
def test_seed_in_metadata():
    """測試 metadata 包含 seed"""
    meta = json.load(open('data/processed_v6/npz/normalization_meta.json'))
    assert 'seed' in meta
    assert meta['seed'] == 42

def test_label_boundary_check():
    """測試標籤邊界檢查"""
    # 正常情況
    y_tb = pd.Series([0, 1, 2, 0, 1])
    # 應不拋出異常

    # 異常情況
    y_tb_invalid = pd.Series([0, 1, 3])  # 包含無效值 3
    with pytest.raises(ValueError, match="標籤值異常"):
        # 觸發邊界檢查
        pass

def test_weight_boundary_check():
    """測試權重邊界檢查"""
    # 正常權重
    w_valid = np.array([0.5, 1.0, 1.5])
    # 應不拋出異常

    # 異常權重（含 NaN）
    w_invalid = np.array([0.5, np.nan, 1.5])
    with pytest.raises(ValueError, match="權重包含 NaN"):
        pass

def test_ffill_quality_filtering():
    """測試 ffill 品質過濾"""
    bucket_mask = np.array([0, 1, 1, 1, 1, 0])  # 66% ffill
    ffill_ratio = (bucket_mask == 1).sum() / len(bucket_mask)
    assert ffill_ratio > 0.5  # 應被過濾
```

### 2. 整合測試

**測試腳本**: `test_v6_pipeline.bat`

```batch
@echo off
echo ========================================
echo V6 完整流水線測試
echo ========================================

echo.
echo [階段 1] 預處理測試（ffill_limit=60）
echo ----------------------------------------
conda activate deeplob-pro
python scripts\preprocess_single_day.py ^
    --input data\temp\20250902.txt ^
    --output-dir data\preprocessed_v6_test ^
    --config configs\config_pro_v5_ml_optimal.yaml

if %ERRORLEVEL% NEQ 0 (
    echo ❌ 預處理失敗！
    exit /b 1
)

echo.
echo [階段 2] 訓練數據生成測試（含所有新功能）
echo ----------------------------------------
python scripts\extract_tw_stock_data_v6.py ^
    --preprocessed-dir data\preprocessed_v6_test ^
    --output-dir data\processed_v6_test ^
    --config configs\config_pro_v5_ml_optimal.yaml

if %ERRORLEVEL% NEQ 0 (
    echo ❌ 訓練數據生成失敗！
    exit /b 1
)

echo.
echo [階段 3] 驗證測試
echo ----------------------------------------
echo 檢查 1: Seed 記錄
python -c "import json; meta=json.load(open('data/processed_v6_test/npz/normalization_meta.json')); print(f'✅ Seed: {meta[\"seed\"]}')"

echo 檢查 2: 標籤分布
python -c "import json; meta=json.load(open('data/processed_v6_test/npz/normalization_meta.json')); print(f'標籤分布: {meta[\"train_label_distribution\"]}')"

echo 檢查 3: Neutral 比例
python -c "import json; meta=json.load(open('data/processed_v6_test/npz/normalization_meta.json')); neutral=meta['train_label_distribution']['1']; print(f'Neutral: {neutral:.1f}%'); assert 15 <= neutral <= 60, '❌ Neutral 比例異常'"

echo.
echo ========================================
echo ✅ V6 測試完成！
echo ========================================
```

### 3. 回歸測試

**確認不破壞現有功能**:

```bash
# 1. 比較 V5 vs V6 輸出格式
python -c "
import numpy as np
v5 = np.load('data/processed_v5/npz/train.npz')
v6 = np.load('data/processed_v6_test/npz/train.npz')
assert v5['X'].shape[1:] == v6['X'].shape[1:]  # 特徵維度一致
assert v5['y'].shape[1:] == v6['y'].shape[1:]  # 標籤維度一致
print('✅ 輸出格式兼容')
"

# 2. 驗證標籤值域
python -c "
import numpy as np
data = np.load('data/processed_v6_test/npz/train.npz')
y = data['y'][:, 0]  # horizon=1
assert set(np.unique(y)).issubset({0, 1, 2})
print('✅ 標籤值域正確')
"

# 3. 驗證權重合法性
python -c "
import numpy as np
data = np.load('data/processed_v6_test/npz/train.npz')
w = data['sample_weight']
assert np.isfinite(w).all()
assert (w >= 0).all()
print(f'✅ 權重範圍: [{w.min():.6f}, {w.max():.6f}]')
"
```

---

## 效果評估

### 預期改進

| 指標 | V5 (舊版) | V6 (新版) | 改進 |
|------|----------|----------|------|
| 數據質量 | 可能包含長期 ffill | 移除低品質窗口 | ⬆️ 提升 |
| 可復現性 | 無 seed 記錄 | metadata 含 seed | ✅ 完全可復現 |
| 錯誤診斷 | 模糊 | 詳細診斷步驟 | ⬆️ 顯著提升 |
| 標籤安全性 | 無邊界檢查 | 嚴格驗證 | ✅ 防止無效標籤 |
| Neutral 監控 | 手動檢查 | 自動警告 | ✅ 主動提示 |

### 性能影響

| 項目 | 預期影響 | 說明 |
|------|---------|------|
| 預處理速度 | ⬇️ 微降 (< 5%) | ffill_limit 降低，少量計算減少 |
| 訓練數據生成 | ⬇️ 微降 (< 10%) | 新增品質過濾步驟 |
| 訓練集大小 | ⬇️ 減少 5-15% | 移除低品質窗口 |
| 模型準確率 | ⬆️ 預期提升 1-3% | 數據質量提升 |

---

## 配置文件變更摘要

**檔案**: `configs/config_pro_v5_ml_optimal.yaml`

**新增參數**:
```yaml
# A.2: 滑窗品質過濾
ffill_quality_threshold: 0.5
```

**修改建議**（可選）:
```yaml
# 根據實際情況調整
ffill_quality_threshold: 0.3  # 更嚴格（數據量充足時）
ffill_quality_threshold: 0.7  # 更寬鬆（數據量不足時）
```

---

## 向後兼容性

### ✅ 完全兼容

1. **配置文件**:
   - 新增參數有默認值（0.5）
   - 舊配置文件可直接使用

2. **輸出格式**:
   - NPZ 結構不變
   - 僅 metadata 新增 `seed` 字段

3. **API 接口**:
   - 函數簽名不變
   - 僅內部邏輯增強

### ⚠️ 可能影響

1. **訓練集大小**:
   - 可能減少 5-15%（品質過濾效果）
   - 需重新評估 batch_size

2. **標籤分布**:
   - Neutral 比例可能變化
   - 需重新校準類別權重

---

## 下一步

### 立即執行

- [ ] 運行 `test_v6_pipeline.bat` 驗證所有改進
- [ ] 檢查測試輸出，確認無錯誤
- [ ] 查看 Neutral 比例警告是否觸發

### 短期（本週）

- [ ] 比較 V5 vs V6 的模型訓練結果
- [ ] 評估品質過濾對準確率的影響
- [ ] 調整 `ffill_quality_threshold` 至最優值

### 中期（下週）

- [ ] 實作 D.1-3: 異常統計擴展
- [ ] 實作 A.3: 長間隙股票報告
- [ ] 實作 C.2: 可復現性單元測試

---

## 相關文檔

- [數據質量驗證策略](DATA_QUALITY_VALIDATION_STRATEGY.md) - 設計原則
- [ChatGPT 建議分析](CHATGPT_SUGGESTIONS_ANALYSIS.md) - 完整建議評估
- [Bug 修復報告](BUG_FIX_MIDS_ZERO.md) - mids=0 問題修復
- [修改總結](../CHANGES_SUMMARY.md) - V6.0.2 總結

---

**作者**: Claude (Sonnet 4.5)
**審核**: 待用戶確認
**版本**: v1.0
**狀態**: ✅ 實作完成，待測試
