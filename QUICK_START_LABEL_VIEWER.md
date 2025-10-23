# Label Viewer 快速開始指南

**版本**: v3.0
**更新日期**: 2025-10-23

---

## 🚀 最快啟動方式（推薦）

```batch
# 1. 啟動統合選單
scripts\label_viewer_menu.bat

# 2. 在選單中選擇 [1] 啟動 Label Viewer

# 3. 瀏覽器訪問
http://localhost:8051
```

**就這麼簡單！** 🎉

---

## 📖 什麼是 Label Viewer？

Label Viewer 是一個互動式 Web 應用程式，用於視覺化和分析 `preprocess_single_day.py` 產生的預處理數據和標籤。

### 主要功能

- ✅ 查看中間價時序圖（含標籤疊加）
- ✅ 分析標籤分布（Triple-Barrier 或趨勢標籤）
- ✅ 查看股票元數據（波動率、過濾資訊等）
- ✅ 整體統計（查看所有股票的標籤分布）

### 支持的標籤方法

| 標籤方法 | 適用場景 | 交易頻率 |
|---------|---------|---------|
| **triple_barrier** | 高頻交易 | 10-20次/天 |
| **trend_adaptive** | 日內波段 | 1-2次/天 |
| **trend_stable** | 日內波段（穩定版）| 1-2次/天（推薦）|

---

## 🎯 三種使用情境

### 情境 1: 我是新手，第一次使用

```batch
# 步驟 1: 啟動選單
scripts\label_viewer_menu.bat

# 步驟 2: 選擇 [2] 快速測試
# 系統會自動：
#   - 預處理一天測試數據
#   - 啟動 Label Viewer
#   - 自動填入數據路徑

# 步驟 3: 在瀏覽器查看
# 點擊「載入目錄」→ 選擇股票 → 查看標籤
```

**所需時間**: 5-10 分鐘

---

### 情境 2: 我已經有預處理數據

```batch
# 步驟 1: 直接啟動
scripts\run_label_viewer.bat

# 步驟 2: 在瀏覽器輸入日期目錄路徑
# 例如: data/preprocessed_swing/daily/20250901

# 步驟 3: 點擊「載入目錄」

# 步驟 4: 選擇股票查看
#   - 選擇「全部股票」查看整體統計
#   - 選擇單一股票查看詳細資料
```

**所需時間**: 1-2 分鐘

---

### 情境 3: 我需要批次處理所有數據

```batch
# 步驟 1: 批次預處理（首次執行，需時較長）
scripts\batch_preprocess.bat

# 步驟 2: 啟動 Label Viewer
scripts\run_label_viewer.bat

# 步驟 3: 查看任意日期的數據
# 在瀏覽器輸入: data/preprocessed_v5_1hz/daily/YYYYMMDD
```

**首次預處理時間**: 30-60 分鐘（取決於數據量）
**後續查看時間**: 1-2 分鐘

---

## 🎨 界面說明

### 控制面板（左側）

1. **日期目錄路徑輸入**
   - 輸入格式: `data/preprocessed_*/daily/YYYYMMDD`
   - 範例: `data/preprocessed_swing/daily/20250901`

2. **載入目錄按鈕**
   - 點擊後掃描並載入股票列表

3. **股票選擇下拉選單**
   - `📊 全部股票`: 查看整體統計
   - `2330`, `2317`, ...: 查看個別股票

4. **顯示選項（勾選框）**
   - ☑️ 中間價折線圖
   - ☑️ 標籤預覽分布
   - ☑️ 元數據表格

### 圖表區域（右側）

#### 單一股票檢視

1. **中間價時序圖**
   - 折線顯示價格變化
   - 疊加標籤點：
     - 🔴 紅色 = 下跌 (-1)
     - ⚪ 灰色 = 持平 (0)
     - 🟢 綠色 = 上漲 (1)

2. **標籤分布柱狀圖**
   - 顯示三種標籤的數量和比例

3. **元數據表格**
   - 股票代碼、日期
   - 開高低收、波動率
   - 過濾資訊、採樣模式

#### 全部股票檢視

1. **整體標籤分布**
   - 所有股票的總標籤分布

2. **前 10 檔股票堆疊圖**
   - 按標籤數量排序

3. **Summary.json 統計**
   - 當天摘要資訊

---

## 📂 數據來源

Label Viewer 讀取 `preprocess_single_day.py` 產生的 NPZ 文件。

### NPZ 文件結構

每個 NPZ 文件包含：

- **features**: (T, 20) LOB 特徵陣列
- **mids**: (T,) 中間價陣列
- **labels**: (T,) 標籤陣列（-1, 0, 1）
- **metadata**: JSON 格式的元數據
  - `label_preview`: 標籤統計
  - `labeling_method`: 使用的標籤方法
  - `weight_strategies`: 權重策略（5 種）

### 如何生成數據？

```batch
# 方法 1: 批次處理所有數據（推薦）
scripts\batch_preprocess.bat

# 方法 2: 處理單一天數據
python scripts\preprocess_single_day.py ^
    --input data\temp\20250901.txt ^
    --output-dir data\preprocessed_v5_1hz ^
    --config configs\config_pro_v5_ml_optimal.yaml
```

---

## 🔧 故障排除

### 問題 1: 找不到數據目錄

**錯誤訊息**: `❌ 目錄不存在`

**解決方法**:
```batch
# 檢查可用的數據目錄
dir /s /b data\preprocessed_*\daily

# 如果沒有數據，執行預處理
scripts\batch_preprocess.bat
```

---

### 問題 2: 無法啟動 Conda 環境

**錯誤訊息**: `無法啟動 conda 環境 deeplob-pro`

**解決方法**:
```batch
# 檢查環境
conda info --envs | findstr "deeplob-pro"

# 如果不存在，創建環境
conda create -n deeplob-pro python=3.11
conda activate deeplob-pro
pip install dash plotly pandas numpy
```

---

### 問題 3: 沒有顯示標籤

**可能原因**:
- NPZ 中沒有 `labels` 鍵
- `metadata` 中沒有 `label_preview`

**解決方法**:
```batch
# 重新預處理數據（確保計算標籤）
python scripts\preprocess_single_day.py ^
    --input data\temp\20250901.txt ^
    --output-dir data\preprocessed_v5_1hz ^
    --config configs\config_pro_v5_ml_optimal.yaml
```

---

### 問題 4: 埠號衝突

**錯誤訊息**: `Address already in use`

**解決方法**:
1. 停止其他使用 8051 埠的應用
2. 或修改埠號（編輯 `label_viewer/app_preprocessed.py`）

---

## 📚 進階功能

### 1. 比較不同標籤方法

```batch
# 1. 使用 triple_barrier 預處理
python scripts\preprocess_single_day.py ^
    --input data\temp\20250901.txt ^
    --output-dir data\preprocessed_tb ^
    --config configs\config_tb.yaml

# 2. 使用 trend_stable 預處理
python scripts\preprocess_single_day.py ^
    --input data\temp\20250901.txt ^
    --output-dir data\preprocessed_trend ^
    --config configs\config_trend.yaml

# 3. 在 Label Viewer 中分別查看
# data/preprocessed_tb/daily/20250901
# data/preprocessed_trend/daily/20250901
```

### 2. 查看權重策略

Label Viewer 會自動顯示 5 種權重策略：

1. **uniform**: 無權重（全部 1.0）
2. **balanced**: sklearn 標準平衡權重
3. **balanced_sqrt**: 平衡權重的平方根
4. **inverse_freq**: 反頻率權重
5. **focal_alpha**: Focal Loss 風格權重

這些權重策略可用於訓練時的類別平衡。

---

## 📞 獲取幫助

### 1. 查看完整文檔

```batch
# 打開完整使用指南
docs\LABEL_VIEWER_GUIDE.md

# 打開腳本說明
scripts\README_LABEL_VIEWER.md
```

### 2. 使用內建檢查功能

```batch
# 啟動選單
scripts\label_viewer_menu.bat

# 選擇 [3] 檢查環境與數據
# 或選擇 [4] 查看使用說明
```

### 3. 聯繫技術支援

如果問題仍未解決，請提供：
- 錯誤訊息截圖
- 執行的命令
- 系統環境資訊

---

## 🎉 開始使用

現在你已經準備好了！

```batch
# 最簡單的開始方式
scripts\label_viewer_menu.bat
```

選擇選項，享受視覺化標籤的樂趣！

---

**祝你使用愉快！** 🚀

---

**更新日期**: 2025-10-23
**版本**: v3.0
**作者**: DeepLOB-Pro Team
