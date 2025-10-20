# Label Viewer - 互動式訓練數據標籤查看器

**版本**：v2.0 (開發中)
**技術棧**：Plotly + Dash + Python
**狀態**：🚧 30% 完成（規劃 + 基礎建設完成）
**最後更新**：2025-10-20

---

## 📋 專案概述

互動式 Web 工具，用於查看和分析 `extract_tw_stock_data_v5.py` 產生的訓練數據標籤正確性，取代靜態 PNG 圖表。

### 核心功能
- ✅ **即時互動**：下拉選單即時切換股票
- ✅ **深入分析**：單一股票詳細標籤分析
- ✅ **多維度對比**：數據集切換、時間範圍篩選
- ✅ **輕量部署**：本地 Web 服務器，瀏覽器訪問

### 與靜態 PNG 版本對比
| 特性 | 靜態 PNG | 互動 Web (本專案) |
|------|----------|------------------|
| 股票選擇 | 預先選定 | 即時切換 ⭐ |
| 圖表互動 | 無 | 縮放、懸停 ⭐ |
| 數據集切換 | 需重新執行 | 即時切換 ⭐ |
| 時間範圍 | 固定 | 動態調整 ⭐ |
| 磁碟佔用 | 大（數百 MB） | 小（只有代碼）⭐ |

---

## 🚀 快速開始（開發完成後）

```bash
# 進入專案目錄
cd label_viewer

# 安裝依賴
pip install -r requirements.txt

# 啟動應用
python app.py

# 瀏覽器訪問
# http://localhost:8050
```

---

## 📁 專案結構

```
label_viewer/
├── README.md              # 專案說明（本文件）
├── requirements.txt       # Python 依賴套件
├── app.py                 # Dash 主應用（待開發）
│
├── components/            # UI 組件模組
│   ├── __init__.py
│   ├── main_chart.py      # 主圖表（待開發）
│   ├── label_timeline.py  # 標籤時間軸（待開發）
│   ├── label_dist.py      # 標籤分布圖（待開發）
│   └── weight_dist.py     # 權重分布圖（待開發）
│
├── utils/                 # 工具函數模組
│   ├── __init__.py
│   ├── config.py          # 配置管理 ✅ 已完成
│   ├── price_builder.py   # 收盤價重建 ✅ 已完成
│   └── data_loader.py     # 數據載入（待開發）
│
├── assets/                # 靜態資源
│   └── style.css          # CSS 樣式（待開發）
│
├── docs/                  # 📚 完整文檔（過去/現在/未來）
│   ├── INDEX.md           # 主導覽 ⭐ 從這裡開始
│   ├── planning/          # 過去：規劃階段文檔
│   ├── development/       # 現在：開發進度文檔
│   └── roadmap/           # 未來：功能路線圖
│
└── tests/                 # 測試（待開發）
```

---

## 📚 文檔導覽

### 🤖 給其他 AI 團隊成員（優先閱讀）⭐
**如果您是接手開發的 AI**，請先閱讀以下文件：
1. **[CLAUDE.md](CLAUDE.md)** - 專案配置與開發指南 ⭐⭐⭐
2. **[docs/development/TODOLIST.md](docs/development/TODOLIST.md)** - 待辦任務清單（20 個任務）⭐⭐⭐
3. [docs/development/STATUS.md](docs/development/STATUS.md) - 當前狀態摘要

這三份文件包含：
- 專案背景與技術選型
- 完整目錄結構說明
- 核心技術要點與實作指南
- 詳細的待辦任務清單（含時間估算）
- 下一步開發建議

### 🎯 快速了解專案（給人類）
**推薦閱讀順序**：
1. [主導覽](docs/INDEX.md) - 文檔結構總覽（過去/現在/未來）⭐
2. [規劃總結](docs/planning/03_summary.md) - 快速了解專案概況 ⭐
3. [當前狀態](docs/development/STATUS.md) - 當前進度與下一步 ⭐

### 📖 深入了解
- [功能規劃書](docs/planning/01_specification.md) - 完整功能設計
- [目錄結構規劃](docs/planning/02_structure.md) - 專案結構設計
- [進度追蹤](docs/development/PROGRESS.md) - 詳細任務清單
- [變更日誌](docs/development/CHANGELOG.md) - 開發記錄
- [完整路線圖](docs/roadmap/ROADMAP.md) - 三階段功能計劃
- [下一步指南](docs/roadmap/NEXT_STEPS.md) - 審查與行動建議

---

## 🎯 開發狀態

### 當前進度：30% 完成

```
整體進度: [████████░░░░░░░░░░░░░░░░░░░░] 30%

規劃階段:  [████████████████████████] 100% ✅
基礎建設:  [████████████████████████] 100% ✅
MVP 開發:  [█████░░░░░░░░░░░░░░░░░░░]  20% 🚧
進階功能:  [░░░░░░░░░░░░░░░░░░░░░░░░]   0% ⏳
測試文檔:  [░░░░░░░░░░░░░░░░░░░░░░░░]   0% ⏳
```

### 已完成工作（2025-10-20）

**1. 規劃階段（100%）✅**
- ✅ 功能規劃書（15 個功能點，三階段開發）
- ✅ 目錄結構規劃（三種方案對比，推薦方案 A）
- ✅ 規劃總結（開發範圍確認）
- ✅ 開發進度追蹤文件
- ✅ 開發狀態摘要

**2. 基礎建設（100%）✅**
- ✅ 完整目錄結構（components/, utils/, assets/, docs/, tests/）
- ✅ 配置文件（`utils/config.py`, 85 行）
- ✅ 收盤價重建模組（`utils/price_builder.py`, 145 行）
- ✅ 初始化文件（`__init__.py`）
- ✅ 依賴清單（`requirements.txt`）

**3. 文檔結構重組（100%）✅**
- ✅ 主導覽文件（`docs/INDEX.md`）
- ✅ 過去：規劃階段文檔（`docs/planning/`）
- ✅ 現在：開發進度文檔（`docs/development/`）
- ✅ 未來：功能路線圖（`docs/roadmap/`）
- ✅ 變更日誌（`docs/development/CHANGELOG.md`）

**程式碼統計**：
- 已完成：230 行（17%）
- 預估總計：1,350 行

### 待完成工作

**階段一：MVP（70%）⏳**
- ⏳ `utils/data_loader.py` - 數據載入與快取（30 分鐘）
- ⏳ `components/main_chart.py` - 主圖表（40 分鐘）
- ⏳ `components/label_dist.py` - 標籤分布圖（20 分鐘）
- ⏳ `app.py` - Dash 應用 MVP 版本（30 分鐘）
- ⏳ 測試基礎功能（15 分鐘）

**階段二：進階功能（0%）⏳**
- ⏳ 數據集切換（train/val/test）
- ⏳ 時間範圍篩選（滑桿）
- ⏳ 標籤時間軸圖
- ⏳ 樣本權重分布圖
- ⏳ 資訊面板
- ⏳ CSS 美化

**測試與文檔（0%）⏳**
- ⏳ 完整功能測試
- ⏳ 使用文檔撰寫
- ⏳ 啟動腳本建立

**預計剩餘時間**：4-5 小時

---

## 🎯 下一步行動

**當前狀態**：⏸️ 暫停中，等待審查

### 建議行動
1. **審查文檔**（30-60 分鐘）
   - 閱讀 [主導覽](docs/INDEX.md)
   - 閱讀 [規劃總結](docs/planning/03_summary.md)
   - 閱讀 [當前狀態](docs/development/STATUS.md)
   - 閱讀 [下一步指南](docs/roadmap/NEXT_STEPS.md)

2. **審查代碼**（15 分鐘）
   - 檢查 `utils/config.py`
   - 檢查 `utils/price_builder.py`

3. **決定是否繼續開發**
   - 選項 A：繼續開發 MVP（1.5-2 小時）
   - 選項 B：提出調整建議
   - 選項 C：暫停，稍後繼續

詳細說明請參閱：[下一步指南](docs/roadmap/NEXT_STEPS.md)

---

## 🛠️ 技術細節

### 技術選型
- **Web 框架**：Dash (Plotly)
- **圖表庫**：Plotly
- **數據處理**：Pandas, NumPy
- **Python 版本**：3.11+
- **服務器**：內建開發服務器（localhost:8050）

### 核心設計
- **模組化架構**：UI 組件與工具函數分離
- **LRU 快取**：快取最近訪問的 3 個數據集
- **互動式圖表**：Plotly 提供縮放、懸停等功能
- **響應式布局**：自適應不同螢幕尺寸

---

## 📊 功能規劃

### 階段一：MVP（必須）
- 數據載入與快取
- 股票選擇器
- 主圖表（收盤價 + 標籤背景）
- 標籤分布圖
- Web 服務器啟動

### 階段二：進階功能（建議）
- 數據集切換（train/val/test）
- 時間範圍篩選
- 標籤時間軸圖
- 樣本權重分布圖
- 資訊面板

### 階段三：高級功能（可選）
- 標籤轉換點高亮
- 圖表下載功能
- 多股票對比
- 配置版本對比
- 搜尋與篩選

詳細規劃請參閱：[完整路線圖](docs/roadmap/ROADMAP.md)

---

## 🤝 如何貢獻

**當前階段**：規劃完成，開發進行中

如需參與開發：
1. 閱讀 [主導覽](docs/INDEX.md) 了解專案結構
2. 閱讀 [進度追蹤](docs/development/PROGRESS.md) 了解待開發任務
3. 閱讀 [功能規劃書](docs/planning/01_specification.md) 了解技術細節
4. 選擇一個待開發任務開始實作

---

## 📝 授權

本專案為 DeepLOB-Pro 的配套工具，遵循主專案授權。

---

## 🔗 相關連結

### 主專案
- [DeepLOB-Pro README](../README.md)
- [CLAUDE.md 專案配置](../CLAUDE.md)

### 文檔
- [📚 完整文檔導覽](docs/INDEX.md) ⭐
- [📋 規劃總結](docs/planning/03_summary.md)
- [🚧 當前狀態](docs/development/STATUS.md)
- [🎯 功能路線圖](docs/roadmap/ROADMAP.md)
- [➡️ 下一步指南](docs/roadmap/NEXT_STEPS.md)

### 技術參考
- [Plotly 官方文檔](https://plotly.com/python/)
- [Dash 官方文檔](https://dash.plotly.com/)
- [Dash Gallery](https://dash-gallery.plotly.host/Portal/)

---

## 📞 聯絡資訊

**專案維護者**：DeepLOB-Pro Team
**最後更新**：2025-10-20 15:25
**版本**：v2.0 (開發中)

---

**⭐ 快速導航：從 [docs/INDEX.md](docs/INDEX.md) 開始探索！**
