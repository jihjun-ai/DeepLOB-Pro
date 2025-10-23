# 快速開始 - 查看預處理數據標籤

**目標**: 查看 `preprocess_single_day.py` 產生的 NPZ 數據和標籤

**狀態**: ✅ 功能完整可用

---

## 🚀 最快啟動方式

```batch
# 1. 進入 label_viewer 目錄
cd label_viewer

# 2. 執行啟動腳本
start_preprocessed_viewer.bat

# 3. 瀏覽器訪問
http://localhost:8051
```

**就這麼簡單！** 🎉

---

## 📖 使用步驟

### 1. 啟動應用

```batch
cd label_viewer
start_preprocessed_viewer.bat
```

### 2. 在瀏覽器中操作

1. **輸入日期目錄路徑**
   ```
   範例: data/preprocessed_swing/daily/20250901
   ```
   或者
   ```
   範例: ../data/preprocessed_swing/daily/20250901
   ```

2. **點擊「載入目錄」按鈕**
   - 系統會掃描該目錄下的所有 NPZ 文件
   - 顯示可用的股票列表

3. **選擇查看方式**
   - **📊 全部股票** - 查看整體標籤統計
   - **個別股票**（如 0050, 2330） - 查看詳細資料

4. **勾選顯示項目**
   - ☑️ 中間價折線圖 - 時序價格圖（含標籤疊加）
   - ☑️ 標籤預覽分布 - 柱狀圖顯示標籤統計
   - ☑️ 元數據表格 - 股票詳細資訊

---

## 🎨 界面說明

### 中間價折線圖（含標籤疊加）

```
價格 ▲
    │     🟢
    │   /    \   🔴
    │  /  ⚪  \ /
    │ /        V
    └─────────────► 時間

圖例:
🔴 紅色點 = Down (-1) - 價格下跌
⚪ 灰色點 = Neutral (0) - 價格持平
🟢 綠色點 = Up (1) - 價格上漲
```

### 標籤分布柱狀圖

```
數量
    ▲
    │  ████     ████
    │  ████     ████  ████
    │  ████     ████  ████
    │  ████     ████  ████
    └─────────────────────►
      Down   Neutral   Up
     (31.5%) (35.7%) (32.8%)
```

### 元數據表格

顯示以下資訊：
- 股票代碼、日期
- 數據點數（例如 15,957 個時間點）
- 波動範圍（例如 2.34%）
- 日內收益率
- 開高低收價格
- 是否通過過濾
- 過濾閾值和方法
- 1Hz 聚合統計（前向填充、缺失比例等）

---

## 📂 數據目錄範例

### 可用的預處理數據目錄

```bash
# 趨勢標籤（波段交易）- 推薦
data/preprocessed_swing/daily/20250901

# Triple-Barrier 標籤（高頻交易）
data/preprocessed_v5_1hz/daily/20250901

# 測試數據
data/preprocessed_v5_test/daily/20240930
```

### 目錄結構

```
data/preprocessed_swing/daily/20250901/
├── 0050.npz          # NPZ 文件
├── 2330.npz
├── 2317.npz
├── ...
└── summary.json      # 當天摘要（可選）
```

---

## 🔍 檢查標籤正確性

### ✅ 正確標籤的特徵

在「中間價折線圖」中檢查：

1. **價格上升階段** → 主要是綠色點（Up）
2. **價格下降階段** → 主要是紅色點（Down）
3. **價格橫盤階段** → 主要是灰色點（Neutral）

### ❌ 異常標籤的特徵

1. **標籤與趨勢相反**
   - 價格上升卻標記為 Down（紅色）
   - 價格下降卻標記為 Up（綠色）

2. **標籤過於集中**
   - Neutral > 70%（大部分是灰色）
   - 原因: `min_return` 閾值設置過高

3. **標籤過於跳動**
   - 每個時間點標籤都不同（像彩虹）
   - 原因: 參數設置不當

---

## 📊 標籤方法識別

應用會自動識別使用的標籤方法（從 metadata 中讀取）：

| 標籤方法 | 適用場景 | 交易頻率 |
|---------|---------|---------|
| **triple_barrier** | 高頻交易 | 10-20次/天 |
| **trend_adaptive** | 日內波段 | 1-2次/天 |
| **trend_stable** | 日內波段（穩定版）| 1-2次/天（推薦）|

---

## 🔧 常見問題

### Q1: 找不到目錄

**錯誤訊息**: `❌ 目錄不存在: data/preprocessed_swing/daily/20250901`

**解決方法**:
1. 檢查路徑是否正確（使用 `/` 而非 `\`）
2. 確保已執行 `preprocess_single_day.py` 或 `batch_preprocess.bat`
3. 使用絕對路徑或從 `label_viewer` 目錄的相對路徑

### Q2: 沒有顯示標籤點

**可能原因**:
1. NPZ 中沒有 `labels` 鍵
2. `metadata` 中沒有 `label_preview`

**解決方法**:
- 確保使用最新版本的 `preprocess_single_day.py`
- 重新執行預處理（確保計算標籤）

### Q3: 應用無法啟動

**可能原因**:
1. 埠號 8051 被佔用
2. Conda 環境未啟動

**解決方法**:
```batch
# 手動啟動環境
conda activate deeplob-pro

# 檢查埠號
netstat -ano | findstr "8051"

# 或修改埠號（編輯 app_preprocessed.py）
```

---

## 💡 使用建議

### 1. 每次預處理後立即檢查

```batch
# 1. 預處理
python scripts\preprocess_single_day.py ^
    --input data\temp\20250901.txt ^
    --output-dir data\preprocessed_v5_1hz ^
    --config configs\config_pro_v5_ml_optimal.yaml

# 2. 立即檢查
cd label_viewer
start_preprocessed_viewer.bat

# 3. 在瀏覽器輸入路徑
# data/preprocessed_v5_1hz/daily/20250901
```

### 2. 比較不同標籤方法

```
# 1. 在 viewer 中載入 triple_barrier 數據
data/preprocessed_v5_1hz/daily/20250901

# 2. 記錄標籤分布

# 3. 載入 trend_stable 數據
data/preprocessed_swing/daily/20250901

# 4. 比較兩種方法的標籤分布差異
```

### 3. 檢查整體標籤分布

```
# 1. 載入目錄後選擇「📊 全部股票」
# 2. 查看整體標籤分布柱狀圖
# 3. 確認標籤分布是否符合預期（例如 30/40/30）
```

---

## 📚 相關文檔

### 在 label_viewer 專案內

- [README.md](README.md) - 專案總覽
- [TEST_PREPROCESSED_VIEWER.md](TEST_PREPROCESSED_VIEWER.md) - 測試報告
- [QUICKSTART.md](QUICKSTART.md) - 快速開始（app.py 版本）

### 在主專案中

- [../docs/LABEL_VIEWER_GUIDE.md](../docs/LABEL_VIEWER_GUIDE.md) - 完整使用指南
- [../scripts/README_LABEL_VIEWER.md](../scripts/README_LABEL_VIEWER.md) - 腳本說明
- [../QUICK_START_LABEL_VIEWER.md](../QUICK_START_LABEL_VIEWER.md) - 主專案快速開始

---

## 🎯 對比說明

### label_viewer 專案內有兩個應用

| 應用 | 數據來源 | 用途 | 啟動方式 |
|-----|---------|------|---------|
| **app.py** | extract_tw_stock_data_v5.py | 查看訓練數據標籤 | start_viewer.bat |
| **app_preprocessed.py** | preprocess_single_day.py | 查看預處理數據標籤 | start_preprocessed_viewer.bat |

**本文檔說明**: app_preprocessed.py（查看預處理數據）

---

## ✅ 確認清單

開始使用前，確認以下事項：

- [ ] 已啟動 Conda 環境（deeplob-pro）
- [ ] 已執行預處理腳本（preprocess_single_day.py 或 batch_preprocess.bat）
- [ ] 有可用的預處理數據目錄
- [ ] 瀏覽器可以訪問 localhost:8051

---

## 🎉 開始使用

```batch
cd label_viewer
start_preprocessed_viewer.bat
```

**祝你使用愉快！** 🚀

---

**更新日期**: 2025-10-23
**版本**: v3.0
**作者**: DeepLOB-Pro Team
