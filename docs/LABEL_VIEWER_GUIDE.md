# Label Viewer 使用指南

**版本**: v3.0
**更新日期**: 2025-10-23
**適用範圍**: 查看 `preprocess_single_day.py` 產生的預處理數據和標籤

---

## 📖 簡介

Label Viewer 是一個基於 Dash 的互動式 Web 應用程式，用於視覺化和分析 `preprocess_single_day.py` 產生的預處理數據。

### 主要功能

1. ✅ **日期目錄掃描**: 自動掃描指定日期目錄下的所有股票 NPZ 文件
2. ✅ **股票數據視覺化**: 查看中間價時序圖（含標籤疊加）
3. ✅ **標籤分布分析**: 視覺化 Triple-Barrier 或趨勢標籤的分布
4. ✅ **元數據檢視**: 查看股票的詳細元數據（波動率、過濾資訊等）
5. ✅ **整體統計**: 查看當天所有股票的標籤分布統計
6. ✅ **LRU 快取**: 自動快取已載入的數據，提升效能

---

## 🚀 快速開始

### 方法一：使用批次檔（Windows，推薦）

```batch
# 1. 執行啟動腳本
scripts\run_label_viewer.bat

# 2. 在瀏覽器打開
http://localhost:8051
```

### 方法二：手動啟動

```bash
# 1. 啟動 Conda 環境
conda activate deeplob-pro

# 2. 執行應用程式
python label_viewer/app_preprocessed.py

# 3. 在瀏覽器打開
http://localhost:8051
```

---

## 📂 數據目錄結構

Label Viewer 需要以下目錄結構：

```
data/
└── preprocessed_v5_1hz/          # 或其他預處理目錄名稱
    └── daily/
        ├── 20250901/              # 日期目錄
        │   ├── 0050.npz           # 股票 NPZ 文件
        │   ├── 2330.npz
        │   ├── ...
        │   └── summary.json       # 當天摘要（可選）
        ├── 20250902/
        │   ├── ...
        └── ...
```

### NPZ 文件內容

每個 NPZ 文件應包含以下鍵：

- **必要鍵**:
  - `features`: (T, 20) LOB 特徵陣列
  - `mids`: (T,) 中間價陣列
  - `metadata`: JSON 字符串，包含元數據

- **可選鍵**:
  - `labels`: (T,) 標籤陣列（如果已預先計算）
  - `bucket_event_count`: 每秒事件數
  - `bucket_mask`: 時間桶遮罩

### Metadata 結構

`metadata` 應包含以下欄位：

```json
{
  "symbol": "2330",
  "date": "20250901",
  "n_points": 14400,
  "range_pct": 0.0456,
  "return_pct": 0.0123,
  "pass_filter": true,
  "filter_threshold": 0.005,
  "filter_method": "P50",
  "label_preview": {
    "total_labels": 14278,
    "down_count": 4512,
    "neutral_count": 2145,
    "up_count": 7621,
    "down_pct": 0.316,
    "neutral_pct": 0.150,
    "up_pct": 0.534,
    "labeling_method": "trend_stable"  // 或 "triple_barrier"
  },
  "weight_strategies": {
    "uniform": {...},
    "balanced": {...},
    ...
  }
}
```

---

## 🎨 使用者介面說明

### 控制面板（左側）

#### 1. 日期目錄路徑

- **輸入框**: 輸入預處理數據的日期目錄路徑
  - 範例: `data/preprocessed_v5_1hz/daily/20250901`
  - 範例: `data/preprocessed_swing/daily/20250901`

- **載入目錄按鈕**: 點擊後掃描目錄並載入股票列表

#### 2. 股票選擇

- **下拉選單**: 選擇要查看的股票
  - `📊 全部股票（整體統計）`: 查看當天所有股票的標籤分布
  - `2330`, `2317`, ... : 查看個別股票的詳細資料

#### 3. 顯示選項

- ☑️ **中間價折線圖**: 顯示時序價格圖（含標籤疊加）
- ☑️ **標籤預覽分布**: 顯示標籤分布柱狀圖/圓餅圖
- ☑️ **元數據表格**: 顯示詳細元數據資訊

#### 4. 快取資訊

- 顯示當前快取命中率和大小

### 圖表區域（右側）

#### 單一股票檢視

1. **中間價時序圖**
   - 折線圖顯示中間價變化
   - 疊加標籤點（紅色=下跌，灰色=持平，綠色=上漲）
   - 懸停顯示詳細資訊

2. **標籤預覽分布**
   - 柱狀圖顯示標籤數量和比例
   - 使用顏色區分三種標籤類別

3. **元數據表格**
   - 顯示股票基本資訊（開高低收、波動率等）
   - 顯示過濾資訊和採樣模式

#### 全部股票檢視

1. **整體標籤分布**
   - 顯示所有股票的總標籤分布
   - 標註有標籤的股票數量

2. **前 10 檔股票堆疊柱狀圖**
   - 按標籤數量排序
   - 分組顯示各股票的標籤分布

3. **Summary.json 統計資訊**
   - 顯示當天摘要統計（如果存在）

---

## 📊 標籤顏色說明

| 標籤類別 | 數值 | 顏色 | 說明 |
|---------|------|------|------|
| 下跌 (Down) | -1 | 🔴 紅色 | 價格下跌超過閾值 |
| 持平 (Neutral) | 0 | ⚪ 灰色 | 價格波動在閾值內 |
| 上漲 (Up) | 1 | 🟢 綠色 | 價格上漲超過閾值 |

---

## 🔧 進階功能

### 1. 標籤來源

Label Viewer 支援兩種標籤來源：

- **預先計算** (推薦): 如果 NPZ 中包含 `labels` 鍵，直接使用
- **實時計算**: 如果 NPZ 中沒有 `labels`，從 `mids` 實時計算 Triple-Barrier 標籤

### 2. 標籤方法識別

應用會自動識別標籤方法（從 `metadata.label_preview.labeling_method`）：

- `triple_barrier`: Triple-Barrier 方法（高頻交易）
- `trend_adaptive`: 趨勢標籤（自適應版）
- `trend_stable`: 趨勢標籤（穩定版，遲滯+平滑）

### 3. 快取機制

- **快取大小**: 預設快取 10 個股票數據
- **快取策略**: LRU (Least Recently Used)
- **快取效益**: 重複查看同一股票時，載入速度大幅提升

---

## 🐛 常見問題

### 問題 1: 找不到目錄

**錯誤訊息**: `❌ 目錄不存在: data/preprocessed_v5_1hz/daily/20250901`

**解決方法**:
1. 檢查目錄路徑是否正確
2. 確保已執行 `preprocess_single_day.py` 或 `batch_preprocess.bat`
3. 使用正確的預處理目錄名稱（例如 `preprocessed_swing`）

### 問題 2: 沒有 NPZ 文件

**錯誤訊息**: `❌ 目錄中沒有找到 NPZ 文件`

**解決方法**:
1. 檢查目錄中是否有 `.npz` 文件
2. 確認預處理腳本是否成功執行
3. 檢查日誌查看是否有股票通過過濾

### 問題 3: 無法載入標籤

**現象**: 圖表中沒有顯示標籤點

**解決方法**:
1. 檢查 `metadata` 中是否有 `label_preview` 欄位
2. 檢查 NPZ 中是否有 `labels` 鍵
3. 如果都沒有，應用會嘗試實時計算（需要 `mids` 數據）

### 問題 4: 應用啟動失敗

**錯誤訊息**: `ModuleNotFoundError: No module named 'dash'`

**解決方法**:
```bash
# 安裝必要套件
conda activate deeplob-pro
pip install dash plotly pandas numpy
```

### 問題 5: 埠號衝突

**錯誤訊息**: `OSError: [Errno 48] Address already in use`

**解決方法**:
1. 停止其他使用 8051 埠的應用
2. 或修改 `app_preprocessed.py` 的埠號：
   ```python
   app.run(debug=False, port=8052, host='0.0.0.0')  # 改為 8052
   ```

---

## 📝 使用範例

### 範例 1: 查看單一股票的標籤分布

```
1. 啟動應用：scripts\run_label_viewer.bat
2. 輸入路徑：data/preprocessed_swing/daily/20250901
3. 點擊「載入目錄」
4. 選擇股票：2330
5. 查看中間價圖（含標籤疊加）和標籤分布柱狀圖
```

### 範例 2: 查看整體標籤分布

```
1. 啟動應用
2. 輸入路徑：data/preprocessed_swing/daily/20250901
3. 點擊「載入目錄」
4. 選擇：📊 全部股票（整體統計）
5. 查看整體標籤分布和前 10 檔股票堆疊圖
```

### 範例 3: 比較不同日期的標籤分布

```
1. 載入第一個日期（例如 20250901）
2. 記錄整體標籤分布
3. 清空輸入框，輸入第二個日期（例如 20250902）
4. 點擊「載入目錄」
5. 比較兩個日期的標籤分布差異
```

---

## 🔗 相關文檔

- [preprocess_single_day.py 腳本說明](../scripts/preprocess_single_day.py) - 查看標頭註解
- [V6_TWO_STAGE_PIPELINE_GUIDE.md](V6_TWO_STAGE_PIPELINE_GUIDE.md) - V6 雙階段處理流程
- [TREND_LABELING_IMPLEMENTATION.md](TREND_LABELING_IMPLEMENTATION.md) - 趨勢標籤方法說明
- [PREPROCESSED_DATA_SPECIFICATION.md](PREPROCESSED_DATA_SPECIFICATION.md) - 預處理數據格式規範

---

## 📞 技術支援

如有問題，請：

1. 檢查本指南的「常見問題」章節
2. 查看應用終端的錯誤訊息
3. 檢查 `preprocess_single_day.py` 的輸出日誌
4. 聯繫開發團隊

---

**版本歷史**:

- **v3.0** (2025-10-23): 完整文檔，支援趨勢標籤和 Triple-Barrier
- **v2.0** (2025-10-22): 新增整體統計檢視
- **v1.0** (2025-10-15): 初始版本

---

**作者**: DeepLOB-Pro Team
**授權**: MIT License
