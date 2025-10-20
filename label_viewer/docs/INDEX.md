# Label Viewer 文檔導覽

**專案名稱**：Interactive Label Viewer（互動式訓練數據標籤查看器）
**版本**：v2.0
**最後更新**：2025-10-20 15:15

---

## 📚 文檔結構總覽

本目錄採用「**過去 → 現在 → 未來**」的時間軸結構，幫助您快速了解專案從規劃到實作的完整歷程。

```
docs/
├── INDEX.md                    # 📍 您在這裡（主導覽）
│
├── 📋 planning/                # 過去：規劃階段文檔
│   ├── 01_specification.md    # 功能規劃書
│   ├── 02_structure.md         # 目錄結構規劃
│   └── 03_summary.md           # 規劃總結與確認
│
├── 🚧 development/             # 現在：開發進度文檔
│   ├── TODOLIST.md             # 待辦事項清單（任務追蹤）
│   ├── PROGRESS.md             # 詳細任務進度追蹤
│   ├── STATUS.md               # 當前狀態摘要
│   └── CHANGELOG.md            # 開發變更日誌
│
├── 🎯 roadmap/                 # 未來：功能路線圖
│   ├── ROADMAP.md              # 完整路線圖（階段 1-3）
│   └── NEXT_STEPS.md           # 下一步行動指南
│
└── 📖 usage/                   # 使用文檔（待開發完成後撰寫）
    ├── QUICKSTART.md           # 快速上手
    ├── USER_GUIDE.md           # 使用者指南
    └── API.md                  # API 參考
```

---

## 🕐 時間軸導覽

### 📋 過去：規劃階段（2025-10-20 14:00-14:45）

**目標**：確定專案範圍、技術選型、目錄結構

| 文檔 | 說明 | 狀態 | 連結 |
|------|------|------|------|
| **功能規劃書** | 核心需求分析、UI 設計、功能模組、三階段開發計劃 | ✅ 完成 | [01_specification.md](planning/01_specification.md) |
| **目錄結構規劃** | 三種方案對比、推薦方案 A、模組化設計、整合方式 | ✅ 完成 | [02_structure.md](planning/02_structure.md) |
| **規劃總結** | 最終確認方案、開發範圍（階段一+二）、待確認事項 | ✅ 完成 | [03_summary.md](planning/03_summary.md) |

**關鍵決策**：
- ✅ 技術選型：Plotly + Dash
- ✅ 目錄結構：方案 A（獨立子目錄 `label_viewer/`）
- ✅ 開發範圍：階段一（MVP）+ 階段二（進階功能）
- ✅ 用戶確認：2025-10-20 14:45

---

### 🚧 現在：開發進行中（2025-10-20 14:45-至今）

**目標**：實作核心功能，建立可運行的互動式查看器

| 文檔 | 說明 | 狀態 | 連結 |
|------|------|------|------|
| **進度追蹤** | 15 個任務清單、預估時間、實際時間、檢查點 | 🚧 即時更新 | [PROGRESS.md](development/PROGRESS.md) |
| **狀態摘要** | 已完成工作、待完成工作、風險追蹤、下一步建議 | 🚧 即時更新 | [STATUS.md](development/STATUS.md) |
| **變更日誌** | 每個模組的開發記錄、問題與解決方案、程式碼變更 | 🚧 即時更新 | [CHANGELOG.md](development/CHANGELOG.md) |

**當前狀態**（2025-10-20 15:15）：
- ✅ 目錄結構建立完成
- ✅ 配置文件完成（`utils/config.py`）
- ✅ 收盤價重建模組完成（`utils/price_builder.py`）
- ⏳ 數據載入模組待開發（`utils/data_loader.py`）
- ⏳ UI 組件待開發（4 個圖表組件）
- ⏳ Dash 應用待開發（`app.py`）

**整體進度**：30% 完成（3/10 核心模組）

---

### 🎯 未來：功能路線圖（待開發）

**目標**：完整功能規劃與後續擴展方向

| 文檔 | 說明 | 狀態 | 連結 |
|------|------|------|------|
| **完整路線圖** | 階段一（MVP）、階段二（進階）、階段三（高級）詳細計劃 | ✅ 完成 | [ROADMAP.md](roadmap/ROADMAP.md) |
| **下一步指南** | MVP 完成後的下一步、功能優先級、優化方向 | ⏳ 待撰寫 | [NEXT_STEPS.md](roadmap/NEXT_STEPS.md) |

**階段規劃**：

**階段一：MVP**（預計 1.5-2 小時）
- 數據載入與快取
- 股票選擇器
- 主圖表（收盤價 + 標籤背景）
- 標籤分布圖
- Web 服務器啟動

**階段二：進階功能**（預計 1-1.5 小時）
- 數據集切換（train/val/test）
- 時間範圍篩選
- 標籤時間軸圖
- 樣本權重分布圖
- 資訊面板

**階段三：高級功能**（可選，預計 2-3 小時）
- 標籤轉換點高亮
- 圖表下載功能
- 多股票對比
- 配置版本對比
- 搜尋與篩選

---

## 📖 使用文檔（待開發完成後撰寫）

這些文檔將在 MVP 完成後撰寫：

| 文檔 | 說明 | 預計撰寫時間 |
|------|------|-------------|
| **快速上手** | 5 分鐘快速啟動指南 | 階段一完成後 |
| **使用者指南** | 完整功能介紹、操作說明、常見問題 | 階段二完成後 |
| **API 參考** | 模組 API 文檔、函數參考 | 階段二完成後 |

---

## 🔍 文檔閱讀建議

### 初次了解專案
**建議順序**：
1. 先看 [規劃總結](planning/03_summary.md) - 快速了解專案概況
2. 再看 [狀態摘要](development/STATUS.md) - 了解當前進度
3. 最後看 [功能規劃書](planning/01_specification.md) - 深入了解技術細節

### 追蹤開發進度
**建議順序**：
1. 查看 [狀態摘要](development/STATUS.md) - 整體進度
2. 查看 [進度追蹤](development/PROGRESS.md) - 任務清單
3. 查看 [變更日誌](development/CHANGELOG.md) - 最新變更

### 了解未來規劃
**建議順序**：
1. 查看 [完整路線圖](roadmap/ROADMAP.md) - 三階段計劃
2. 查看 [下一步指南](roadmap/NEXT_STEPS.md) - 後續行動

### 參與開發
**建議順序**：
1. 閱讀 [目錄結構規劃](planning/02_structure.md) - 了解專案結構
2. 閱讀 [進度追蹤](development/PROGRESS.md) - 確認待開發任務
3. 閱讀 [功能規劃書](planning/01_specification.md) - 了解技術實作細節

---

## 📊 文檔統計

### 規劃階段文檔
- 總文檔數：3 個
- 總字數：~8,000 字
- 總行數：~800 行
- 完成度：100%

### 開發階段文檔
- 總文檔數：3 個（含本導覽）
- 總字數：~5,000 字
- 總行數：~500 行
- 更新頻率：即時更新

### 已撰寫代碼
- 配置文件：85 行
- 工具模組：145 行
- 總計：230 行（預估最終 ~1,350 行）

---

## 🔗 快速連結

### 🤖 給 AI 團隊成員（優先閱讀）
- 🤖 **[CLAUDE.md](../CLAUDE.md)** - Label Viewer 專案配置與開發指南 ⭐⭐⭐
- 📋 **[TODOLIST.md](development/TODOLIST.md)** - 待辦任務清單（20 個任務）⭐⭐⭐
- 🚧 [當前狀態](development/STATUS.md) - 最新進度

### 重要文檔
- 📋 [功能規劃書](planning/01_specification.md) - 完整功能設計
- 🚧 [當前狀態](development/STATUS.md) - 最新進度
- 🎯 [路線圖](roadmap/ROADMAP.md) - 未來計劃

### 外部參考
- 🏠 [主專案 README](../../README.md)
- 📘 [主專案 CLAUDE.md](../../CLAUDE.md) - DeepLOB-Pro 專案配置
- 🔧 [Plotly 官方文檔](https://plotly.com/python/)
- 🌐 [Dash 官方文檔](https://dash.plotly.com/)

### 源碼目錄
- 📁 [components/](../components/) - UI 組件
- 📁 [utils/](../utils/) - 工具函數
- 📁 [assets/](../assets/) - 靜態資源

---

## ⚡ 快速指令

### 查看當前狀態
```bash
cd label_viewer/docs
cat development/STATUS.md
```

### 查看詳細進度
```bash
cd label_viewer/docs
cat development/PROGRESS.md
```

### 查看規劃總結
```bash
cd label_viewer/docs
cat planning/03_summary.md
```

---

## 🎯 下一步行動

根據當前狀態，建議的下一步行動：

1. **審查已完成文檔**（當前階段）
   - ✅ 檢視規劃階段文檔
   - ✅ 檢視開發進度文檔
   - ✅ 確認開發範圍與優先級

2. **繼續開發 MVP**（下一階段）
   - ⏳ 開發 `utils/data_loader.py`
   - ⏳ 開發 `components/main_chart.py`
   - ⏳ 開發 `components/label_dist.py`
   - ⏳ 開發 `app.py` MVP 版本
   - ⏳ 測試基礎功能

3. **完成進階功能**（後續階段）
   - ⏳ 開發其餘組件
   - ⏳ 整合所有功能
   - ⏳ 撰寫使用文檔

---

## 📝 維護說明

### 文檔更新規範

1. **即時更新文檔**：
   - `development/PROGRESS.md` - 每完成一個任務
   - `development/STATUS.md` - 每達成一個檢查點
   - `development/CHANGELOG.md` - 每次程式碼變更

2. **版本控制**：
   - 規劃階段文檔：不再修改（已鎖定）
   - 開發階段文檔：持續更新
   - 未來規劃文檔：根據實際進度調整

3. **文檔命名規則**：
   - 規劃文檔：`01_`, `02_`, `03_` 前綴（按時間順序）
   - 開發文檔：大寫全名（`PROGRESS.md`, `STATUS.md`）
   - 使用文檔：大寫全名（`QUICKSTART.md`, `USER_GUIDE.md`）

---

**文檔維護者**：DeepLOB-Pro Team
**最後更新**：2025-10-20 15:15
**下次更新**：完成下一個模組時（預計 `utils/data_loader.py`）
