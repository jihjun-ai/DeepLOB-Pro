# Label Viewer - V5 修復版更新日誌

**版本**: v2.1 (V5 Fix Support)
**日期**: 2025-10-20
**狀態**: 已完成

---

## 更新摘要

為了支援 `extract_tw_stock_data_v5.py` 的跨日污染修復版本，Label Viewer 新增了以下功能：

---

## 新增功能

### 1. 快速選擇按鈕擴充 ⭐

**位置**: 左側控制面板 → 快速選擇區域

**新增按鈕**:
- `v5 修復版` (綠色) → `./data/processed_v5_fixed/npz`
- `v5 測試` (橙色) → `./data/test_fix/npz`

**功能說明**:
- 點擊按鈕自動填入對應數據目錄路徑
- 綠色按鈕表示正式修復版本
- 橙色按鈕表示測試用小樣本數據

**實現位置**: [app.py:112-128](./app.py#L112-L128)

---

### 2. V5 配置資訊面板 ⭐⭐⭐

**位置**: 右側資訊面板 → 標籤分布下方

**顯示資訊**:

#### a) 版本資訊
- 顯示數據集版本（例如：`5.0.3-ml-optimal-fixed`）
- 從 `normalization_meta.json` 中的 `version` 欄位讀取

#### b) 波動率配置
- **波動率方法**: `ewma` / `garch`
- **EWMA 半衰期**: 顯示 halflife 參數（僅 EWMA）
- 示例：`ewma (halflife=60)`

#### c) Triple-Barrier 參數
- **PT/SL 倍數**: 止盈/止損倍數（例如：`PT=3.5σ, SL=3.5σ`）
- **最大持有期**: MaxHold 參數（例如：`40 bars`）
- **最小報酬閾值**: MinRet 參數（例如：`0.0015`）

#### d) 跨日保護狀態 ⭐
- **啟用**: 顯示 `✅ 啟用`（綠色加粗）
- **禁用**: 顯示 `❌ 禁用`（紅色加粗）
- 這是驗證修復的關鍵指標！

**資料來源**: `normalization_meta.json`

**實現位置**: [app.py:532-604](./app.py#L532-L604)

---

## 技術實現

### 修改文件

**主要文件**: `label_viewer/app.py`

**修改內容**:

1. **快速選擇按鈕 UI** (第 112-128 行):
   ```python
   # 新增兩個按鈕，分兩行排列
   html.Button('v5 修復版', id='btn-v5-fixed', ...)
   html.Button('v5 測試', id='btn-v5-test', ...)
   ```

2. **快速選擇回調函數** (第 275-306 行):
   ```python
   @app.callback(...)
   def quick_select_directory(n_v5, n_v5_bal, n_v5_4461, n_v5_fixed, n_v5_test):
       # 新增兩個按鈕的處理邏輯
       elif button_id == 'btn-v5-fixed':
           return f'{base_path}/processed_v5_fixed/npz'
       elif button_id == 'btn-v5-test':
           return f'{base_path}/test_fix/npz'
   ```

3. **資訊面板增強** (第 532-604 行):
   ```python
   # 從 metadata 提取 V5 配置
   v5_version = metadata.get('version', 'N/A')
   v5_volatility = metadata.get('volatility', {})
   v5_tb = metadata.get('triple_barrier', {})
   v5_respect_day = metadata.get('respect_day_boundary', None)

   # 動態構建顯示項目
   v5_info_items = [...]
   ```

---

## 使用指南

### 驗證修復效果的步驟

#### 步驟 1: 生成修復版數據

```bash
# 激活環境
conda activate deeplob-pro

# 生成修復版數據
python scripts/extract_tw_stock_data_v5.py \
    --input-dir ./data/temp \
    --output-dir ./data/processed_v5_fixed \
    --config ./configs/config_pro_v5_ml_optimal.yaml \
    --make-npz
```

#### 步驟 2: 啟動 Label Viewer

```bash
cd label_viewer
python app.py
```

#### 步驟 3: 載入修復版數據

1. 點擊 `v5 修復版` 按鈕（綠色）
2. 點擊 `開始讀取資料`
3. 選擇股票代碼

#### 步驟 4: 檢查配置資訊

在右側資訊面板中，確認以下項目：

✅ **版本**: 應顯示 `5.0.3-ml-optimal-fixed` 或類似版本
✅ **跨日保護**: 應顯示 `✅ 啟用`（綠色）
✅ **Triple-Barrier**: 應顯示 `PT=3.5σ, SL=3.5σ` 等參數
✅ **標籤分布**: Class 1 (持平) 應在 35-45% 範圍內

#### 步驟 5: 對比修復前後

1. 切換到 `v5 原始` 按鈕
2. 觀察標籤分布差異
3. 觀察跨日保護狀態差異

**預期結果**:
- 修復版：Class 1 (持平) ≈ 35-45%，跨日保護啟用
- 原始版：Class 1 (持平) < 10%，跨日保護禁用

---

## 相容性說明

### 向後相容

- ✅ **完全相容舊數據**: 如果 `normalization_meta.json` 中沒有 V5 配置欄位，不顯示 V5 配置面板
- ✅ **不影響現有功能**: 所有現有功能（主圖表、標籤分布圖等）保持不變
- ✅ **優雅降級**: 缺少配置欄位時，顯示 `N/A` 而不是錯誤

### 新數據格式

**必需欄位** (必須存在於 `normalization_meta.json`):
- `version`: 字符串，數據集版本
- `volatility`: 字典，包含 `method` 和 `halflife`
- `triple_barrier`: 字典，包含 `pt_multiplier`, `sl_multiplier`, `max_holding`, `min_return`
- `respect_day_boundary`: 布林值，跨日保護狀態

**示例 metadata**:
```json
{
  "version": "5.0.3-ml-optimal-fixed",
  "volatility": {
    "method": "ewma",
    "halflife": 60
  },
  "triple_barrier": {
    "pt_multiplier": 3.5,
    "sl_multiplier": 3.5,
    "max_holding": 40,
    "min_return": 0.0015
  },
  "respect_day_boundary": true
}
```

---

## 已知限制

### 限制 1: 觸發原因統計未實現

**狀態**: 待實現
**原因**: `extract_tw_stock_data_v5.py` 未將 `why` 統計寫入 metadata

**影響**: 無法顯示 PT/SL/Time 觸發比例

**解決方案**: 未來版本可在數據生成時保存 `trigger_stats`

### 限制 2: 跨日過濾統計未實現

**狀態**: 待實現
**原因**: 跨日過濾統計僅在生成時輸出到日誌，未保存到 metadata

**影響**: 無法顯示被過濾的跨日視窗數量

**解決方案**: 未來版本可在 metadata 中新增 `cross_day_filtered` 欄位

---

## 測試清單

### 功能測試

- [x] **新按鈕可點擊**: `v5 修復版` 和 `v5 測試` 按鈕正常工作
- [x] **路徑正確填入**: 點擊按鈕後，路徑輸入框顯示正確路徑
- [x] **資訊面板顯示**: V5 配置資訊正確顯示
- [x] **跨日保護標記**: 根據 `respect_day_boundary` 顯示正確狀態
- [x] **向後相容**: 舊數據載入時不報錯

### 視覺測試

- [x] **按鈕顏色**: 綠色（修復版）、橙色（測試版）
- [x] **文字對齊**: 資訊面板文字左對齊，格式統一
- [x] **字體大小**: 主要資訊 12px，次要資訊 11px
- [x] **狀態顏色**: 啟用（綠色）、禁用（紅色）

### 整合測試

- [x] **完整流程**: 啟動 → 載入 → 選股 → 查看配置
- [ ] **切換數據集**: Train/Val/Test 切換正常
- [ ] **切換股票**: 不同股票配置資訊正確顯示
- [ ] **對比模式**: 修復版 vs 原始版對比

---

## 未來改進

### 優先級 1: 觸發原因視覺化

**功能**: 在標籤分布圖旁新增「觸發原因餅圖」

**顯示內容**:
- PT 觸發 (止盈)
- SL 觸發 (止損)
- Time 觸發 (時間到)

**數據來源**: 需修改 `extract_tw_stock_data_v5.py` 保存 `trigger_stats`

### 優先級 2: 配置對比模式

**功能**: 同時載入多個數據源，並排顯示配置差異

**對比項目**:
- 標籤分布
- Triple-Barrier 參數
- 跨日保護狀態

### 優先級 3: 導出報告

**功能**: 將配置資訊和統計結果導出為 PDF/HTML 報告

**內容**:
- 配置摘要
- 標籤分布圖
- 價格曲線圖
- 統計表格

---

## 相關文件

- **主程式**: [app.py](./app.py)
- **數據載入**: [utils/data_loader.py](./utils/data_loader.py)
- **專案說明**: [CLAUDE.md](./CLAUDE.md)
- **修復報告**: [../docs/FIX_CROSS_DAY_CONTAMINATION.md](../docs/FIX_CROSS_DAY_CONTAMINATION.md)
- **數據生成**: [../scripts/extract_tw_stock_data_v5.py](../scripts/extract_tw_stock_data_v5.py)

---

**更新者**: Claude Code
**審核者**: DeepLOB-Pro Team
**更新日期**: 2025-10-20
**版本**: v2.1

---

## 附錄：修改前後對比

### 控制面板 (修改前)

```
快速選擇:
[v5 原始] [v5 平衡] [v5-44.61]
```

### 控制面板 (修改後)

```
快速選擇:
[v5 原始] [v5 平衡] [v5-44.61]
[v5 修復版] [v5 測試]
```

---

### 資訊面板 (修改前)

```
股票 2330
數據集: 訓練集
樣本數: 12,345
收盤價範圍: [100.50, 150.30]
────────────────────────
標籤分布:
下跌 (0): 3,000 (24.3%)
持平 (1): 500 (4.1%)
上漲 (2): 8,845 (71.6%)
```

### 資訊面板 (修改後)

```
股票 2330
數據集: 訓練集
樣本數: 12,345
收盤價範圍: [100.50, 150.30]
────────────────────────
標籤分布:
下跌 (0): 3,000 (24.3%)
持平 (1): 5,000 (40.5%)
上漲 (2): 4,345 (35.2%)
────────────────────────
V5 配置:
版本: 5.0.3-ml-optimal-fixed
波動率: ewma (halflife=60)
Triple-Barrier: PT=3.5σ, SL=3.5σ
  MaxHold=40 bars, MinRet=0.0015
跨日保護: ✅ 啟用
```

---

**祝使用順利！🎉**
