# Label Viewer 腳本說明

本目錄包含多個與 Label Viewer 相關的腳本，以下說明各腳本的用途和使用方式。

---

## 📋 腳本總覽

| 腳本名稱 | 用途 | 推薦使用 |
|---------|------|----------|
| **label_viewer_menu.bat** | 統合工具選單（包含所有功能） | ⭐⭐⭐⭐⭐ |
| run_label_viewer.bat | 直接啟動 Label Viewer | ⭐⭐⭐ |
| quick_test_label_viewer.bat | 預處理測試數據並啟動 viewer | ⭐⭐⭐ |
| test_label_viewer_simple.bat | 簡單環境檢查 | ⭐⭐ |

---

## 🚀 推薦使用方式

### 方法一：使用統合選單（最簡單）⭐⭐⭐⭐⭐

```batch
# 啟動統合選單
scripts\label_viewer_menu.bat

# 選單提供以下選項：
# [1] 啟動 Label Viewer (查看已有數據)
# [2] 快速測試 (預處理+查看) - 使用 trend_stable
# [3] 檢查環境與數據
# [4] 查看使用說明
# [0] 退出
```

**優點**：
- 一鍵式操作，無需記憶命令
- 自動檢測可用數據目錄
- 包含環境檢查和使用說明
- 適合所有用戶

---

### 方法二：直接啟動（已有數據時使用）

```batch
# 直接啟動 Label Viewer
scripts\run_label_viewer.bat

# 瀏覽器訪問
http://localhost:8051
```

**優點**：
- 快速啟動，適合已熟悉操作的用戶
- 適合已有預處理數據的情況

---

### 方法三：快速測試（首次使用或測試新功能）

```batch
# 預處理測試數據並啟動 viewer
scripts\quick_test_label_viewer.bat
```

**功能**：
1. 預處理單天測試數據（20240930）
2. 使用 trend_stable 標籤方法
3. 自動啟動 Label Viewer

**適用場景**：
- 首次使用，想快速體驗
- 測試新的標籤方法
- 驗證系統是否正常運作

---

## 📂 數據目錄結構

Label Viewer 需要以下目錄結構：

```
data/
└── preprocessed_**/              # 預處理目錄（可有多個）
    └── daily/
        ├── 20250901/              # 日期目錄
        │   ├── 2330.npz           # 股票 NPZ 文件
        │   ├── 2317.npz
        │   └── summary.json       # 當天摘要（可選）
        ├── 20250902/
        └── ...
```

### 常見數據目錄

1. **preprocessed_swing**: 使用趨勢標籤（波段交易）
2. **preprocessed_v5_1hz**: 使用 Triple-Barrier（高頻交易）
3. **preprocessed_v5_test**: 快速測試數據

---

## 🔧 環境檢查

如果遇到問題，可以使用以下方法檢查環境：

### 方法 1: 使用選單的檢查功能

```batch
scripts\label_viewer_menu.bat
# 選擇 [3] 檢查環境與數據
```

### 方法 2: 使用簡易測試腳本

```batch
scripts\test_label_viewer_simple.bat
```

### 方法 3: 手動檢查

```batch
# 1. 檢查 Conda 環境
conda info --envs | findstr "deeplob-pro"

# 2. 檢查必要文件
dir label_viewer\app_preprocessed.py
dir label_viewer\utils\preprocessed_loader.py

# 3. 檢查數據目錄
dir /s /b data\preprocessed_*\daily\*.npz | more
```

---

## 📊 使用流程示例

### 情境 1: 首次使用

```batch
# 步驟 1: 啟動選單
scripts\label_viewer_menu.bat

# 步驟 2: 選擇 [3] 檢查環境
# 確認環境和數據正常

# 步驟 3: 選擇 [2] 快速測試
# 預處理測試數據並查看

# 步驟 4: 在瀏覽器查看
# http://localhost:8051
# 路徑自動填入: data/preprocessed_v5_test/daily/20240930
```

### 情境 2: 查看已有數據

```batch
# 步驟 1: 確保已執行預處理
scripts\batch_preprocess.bat

# 步驟 2: 啟動選單
scripts\label_viewer_menu.bat

# 步驟 3: 選擇 [1] 啟動 Label Viewer

# 步驟 4: 在瀏覽器輸入日期目錄
# 例如: data/preprocessed_swing/daily/20250901
```

### 情境 3: 快速查看特定日期

```batch
# 直接啟動
scripts\run_label_viewer.bat

# 在瀏覽器輸入路徑
# data/preprocessed_swing/daily/20250901
# 點擊「載入目錄」
# 選擇股票查看
```

---

## 🐛 常見問題

### 問題 1: 找不到預處理數據

**解決方法**:
```batch
# 方法 1: 批次預處理所有數據（推薦）
scripts\batch_preprocess.bat

# 方法 2: 快速測試單天數據
scripts\label_viewer_menu.bat
# 選擇 [2] 快速測試
```

### 問題 2: 無法啟動 Conda 環境

**錯誤訊息**: `無法啟動 conda 環境 deeplob-pro`

**解決方法**:
```batch
# 檢查環境是否存在
conda info --envs

# 創建環境（如果不存在）
conda create -n deeplob-pro python=3.11

# 安裝必要套件
conda activate deeplob-pro
pip install dash plotly pandas numpy
```

### 問題 3: 埠號衝突

**錯誤訊息**: `Address already in use`

**解決方法**:
1. 停止其他使用 8051 埠的應用
2. 或修改 `label_viewer/app_preprocessed.py` 的埠號

### 問題 4: 沒有顯示標籤

**可能原因**:
- 預處理時沒有計算標籤
- NPZ 文件格式不正確

**解決方法**:
```batch
# 重新預處理數據（確保計算標籤）
python scripts\preprocess_single_day.py ^
    --input data\temp\20250901.txt ^
    --output-dir data\preprocessed_v5_1hz ^
    --config configs\config_pro_v5_ml_optimal.yaml
```

---

## 📝 腳本詳細說明

### label_viewer_menu.bat

**功能**: 統合工具選單

**包含選項**:
1. 啟動 Label Viewer
2. 快速測試（預處理+查看）
3. 環境檢查
4. 查看使用說明

**適用場景**: 所有情況（推薦）

---

### run_label_viewer.bat

**功能**: 直接啟動 Label Viewer

**執行流程**:
1. 啟動 Conda 環境
2. 檢查必要文件
3. 啟動 Web 應用（埠號 8051）

**適用場景**: 已有數據，快速查看

---

### quick_test_label_viewer.bat

**功能**: 預處理測試數據並啟動 viewer

**執行流程**:
1. 預處理 20240930.txt
2. 使用 trend_stable 標籤方法
3. 生成 NPZ 文件到 data/preprocessed_v5_test
4. 自動啟動 Label Viewer

**適用場景**: 首次使用、功能測試

**注意事項**:
- 需要測試文件: data/temp/20240930.txt
- 輸出到 data/preprocessed_v5_test（避免覆蓋正式數據）

---

### test_label_viewer_simple.bat

**功能**: 簡單環境檢查

**檢查項目**:
1. 必要文件是否存在
2. 數據目錄是否存在
3. 啟動腳本是否正常

**適用場景**: 環境驗證、問題排查

---

## 🔗 相關文檔

- [LABEL_VIEWER_GUIDE.md](../docs/LABEL_VIEWER_GUIDE.md) - 完整使用指南
- [V6_TWO_STAGE_PIPELINE_GUIDE.md](../docs/V6_TWO_STAGE_PIPELINE_GUIDE.md) - V6 數據處理流程
- [TREND_LABELING_IMPLEMENTATION.md](../docs/TREND_LABELING_IMPLEMENTATION.md) - 趨勢標籤說明

---

## 📞 技術支援

如有問題，請：

1. 查看本文檔的「常見問題」章節
2. 使用選單的 [3] 環境檢查功能
3. 查看終端錯誤訊息
4. 聯繫開發團隊

---

**更新日期**: 2025-10-23
**版本**: v1.0
**作者**: DeepLOB-Pro Team
