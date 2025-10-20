# 標籤查看器專案 - 獨立目錄結構規劃

## 📋 目錄結構設計

### 方案 A: 在 DeepLOB-Pro 下建立子目錄（推薦）

```
DeepLOB-Pro/
├── configs/                    # 主專案配置
├── src/                        # 主專案源碼
├── scripts/                    # 主專案腳本
├── data/                       # 共用數據目錄
├── docs/                       # 主專案文檔
│
├── label_viewer/              # 🆕 標籤查看器子專案（獨立）
│   ├── README.md              # 子專案說明
│   ├── requirements.txt       # 獨立依賴（dash, plotly）
│   ├── app.py                 # Dash 主應用程式
│   ├── start_viewer.bat       # Windows 啟動腳本
│   ├── start_viewer.sh        # Linux/Mac 啟動腳本
│   │
│   ├── components/            # UI 組件模組
│   │   ├── __init__.py
│   │   ├── main_chart.py      # 主圖表生成
│   │   ├── label_timeline.py  # 標籤時間軸
│   │   ├── label_dist.py      # 標籤分布圖
│   │   └── weight_dist.py     # 權重分布圖
│   │
│   ├── utils/                 # 工具模組
│   │   ├── __init__.py
│   │   ├── data_loader.py     # 數據載入與快取
│   │   ├── price_builder.py   # 收盤價重建
│   │   └── config.py          # 配置管理
│   │
│   ├── assets/                # 靜態資源（CSS, 圖片）
│   │   ├── style.css          # 自定義樣式
│   │   └── logo.png           # Logo（可選）
│   │
│   ├── docs/                  # 子專案文檔
│   │   ├── usage.md           # 使用說明
│   │   ├── api.md             # API 文檔
│   │   └── screenshots/       # 截圖
│   │
│   └── tests/                 # 測試（可選）
│       └── test_data_loader.py
│
└── LABEL_VIEWER_README.md     # 子專案快速入口（指向 label_viewer/）
```

**優點**：
- ✅ 保持在同一個 Git 倉庫，方便版本控制
- ✅ 可共用 `data/` 目錄，避免數據重複
- ✅ 獨立的依賴管理（不影響主專案）
- ✅ 結構清晰，易於維護

**缺點**：
- ⚠️ 專案根目錄稍微複雜一點（但可接受）

---

### 方案 B: 完全獨立專案（最乾淨）

```
# 主專案
DeepLOB-Pro/
├── configs/
├── src/
├── scripts/
├── data/
└── docs/

# 獨立專案（平行目錄）
DeepLOB-LabelViewer/           # 🆕 完全獨立的專案
├── README.md
├── requirements.txt
├── app.py
├── start_viewer.bat
├── components/
├── utils/
├── assets/
├── docs/
└── data/                      # 🔗 符號連結到 ../DeepLOB-Pro/data/
```

**優點**：
- ✅ 完全獨立，DeepLOB-Pro 保持純淨
- ✅ 可單獨發佈到 GitHub（獨立倉庫）
- ✅ 依賴完全隔離

**缺點**：
- ❌ 需要管理兩個 Git 倉庫
- ❌ 數據需要符號連結或複製
- ❌ 版本同步較複雜

---

### 方案 C: 混合方案（工具目錄）

```
DeepLOB-Pro/
├── configs/
├── src/
├── scripts/
├── data/
├── docs/
│
└── tools/                     # 🆕 工具集目錄
    ├── label_viewer/          # 標籤查看器
    │   ├── app.py
    │   ├── components/
    │   └── utils/
    │
    ├── data_inspector/        # 其他工具（未來擴展）
    └── performance_monitor/
```

**優點**：
- ✅ 可擴展（未來其他工具也放在 `tools/`）
- ✅ 結構清晰
- ✅ 共用數據與配置

**缺點**：
- ⚠️ 需要規劃 `tools/` 的統一管理方式

---

## 🎯 推薦方案：**方案 A**（子目錄）

### 理由
1. **實務性最佳**：標籤查看器是 DeepLOB-Pro 的配套工具，放在同一倉庫便於管理
2. **數據共用方便**：直接讀取 `../data/processed_v5/npz`，無需符號連結
3. **版本控制統一**：一個 Git 倉庫，提交記錄清晰
4. **未來擴展性**：如需獨立，可輕鬆拆分成方案 B

---

## 📦 詳細目錄說明（方案 A）

### 核心文件

#### 1. `label_viewer/README.md`
```markdown
# 訓練數據標籤互動查看器

DeepLOB-Pro 配套工具 - 互動式 Web 介面查看訓練數據標籤

## 快速開始
\`\`\`bash
cd label_viewer
pip install -r requirements.txt
python app.py
\`\`\`

瀏覽器訪問: http://localhost:8050
```

#### 2. `label_viewer/requirements.txt`
```
dash>=2.14.0
plotly>=5.18.0
pandas>=2.0.0
numpy>=1.24.0
```

#### 3. `label_viewer/app.py`（主應用）
```python
"""
訓練數據標籤互動查看器 - Dash 主應用
使用方式: python app.py
訪問: http://localhost:8050
"""
import dash
from dash import dcc, html
from components import main_chart, label_timeline, label_dist, weight_dist
from utils import data_loader

app = dash.Dash(__name__, suppress_callback_exceptions=True)
# ... 應用邏輯
```

#### 4. `label_viewer/start_viewer.bat`（快速啟動）
```batch
@echo off
cd /d "%~dp0"
echo 啟動標籤查看器...
conda activate deeplob-pro
pip install -r requirements.txt --quiet
python app.py
```

---

### 模組化設計

#### `components/` - UI 組件
- **`main_chart.py`**: 收盤價與標籤趨勢圖
  ```python
  def create_main_chart(close, labels, weights, options):
      # 返回 Plotly Figure 對象
      return fig
  ```

- **`label_timeline.py`**: 標籤時間軸
- **`label_dist.py`**: 標籤分布圓餅圖/柱狀圖
- **`weight_dist.py`**: 樣本權重直方圖

#### `utils/` - 工具函數
- **`data_loader.py`**: 數據載入與快取
  ```python
  @functools.lru_cache(maxsize=3)
  def load_split_data(data_dir, split):
      # 載入並處理數據
      return stock_data
  ```

- **`price_builder.py`**: 收盤價重建
  ```python
  def reconstruct_close_price(X, metadata):
      # 反向 Z-Score
      return close_prices
  ```

- **`config.py`**: 配置管理
  ```python
  DEFAULT_DATA_DIR = "../data/processed_v5/npz"
  LABEL_COLORS = {0: '#e74c3c', 1: '#95a5a6', 2: '#2ecc71'}
  ```

---

## 🔗 與主專案的整合

### 數據路徑引用
```python
# label_viewer/utils/config.py
import os
from pathlib import Path

# 自動檢測主專案數據目錄
PROJECT_ROOT = Path(__file__).parent.parent.parent  # DeepLOB-Pro/
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "processed_v5" / "npz"

# 支援多個數據源
DATA_SOURCES = {
    "v5": PROJECT_ROOT / "data" / "processed_v5" / "npz",
    "v5_balanced": PROJECT_ROOT / "data" / "processed_v5_balanced" / "npz",
    "v5-44.61": PROJECT_ROOT / "data" / "processed_v5-44.61" / "npz",
}
```

### 主專案入口連結
在 `DeepLOB-Pro/` 根目錄建立快速入口：

**`LABEL_VIEWER_README.md`**:
```markdown
# 🔍 標籤查看器

互動式 Web 工具，用於查看訓練數據標籤正確性。

## 快速啟動
\`\`\`bash
cd label_viewer
python app.py
\`\`\`

詳細說明: [label_viewer/README.md](label_viewer/README.md)
```

---

## 🚀 啟動方式

### Windows
```batch
# 方式 1: 使用批次腳本
label_viewer\start_viewer.bat

# 方式 2: 手動啟動
cd label_viewer
conda activate deeplob-pro
python app.py
```

### Linux/Mac
```bash
# 方式 1: 使用 Shell 腳本
cd label_viewer
./start_viewer.sh

# 方式 2: 手動啟動
cd label_viewer
conda activate deeplob-pro
python app.py
```

---

## 📊 目錄大小預估

```
label_viewer/
├── app.py                    # ~300 行
├── components/               # ~600 行（4 個組件）
│   ├── main_chart.py         # ~200 行
│   ├── label_timeline.py     # ~100 行
│   ├── label_dist.py         # ~150 行
│   └── weight_dist.py        # ~150 行
├── utils/                    # ~400 行
│   ├── data_loader.py        # ~200 行
│   ├── price_builder.py      # ~100 行
│   └── config.py             # ~100 行
├── assets/                   # ~50 行 CSS
└── docs/                     # 文檔

總計: ~1,350 行代碼（精簡設計）
```

---

## 🎯 優勢總結

### 1. 結構清晰
```
主專案（DeepLOB-Pro）專注於模型訓練
    ↓
工具專案（label_viewer）專注於數據檢查
```

### 2. 獨立依賴
```
主專案: PyTorch, Stable-Baselines3, arch, ...（重量級）
工具專案: Dash, Plotly（輕量級，約 50 MB）
```

### 3. 易於維護
- 主專案更新不影響工具
- 工具更新不影響主專案
- 可單獨發佈工具到 PyPI（未來）

### 4. 開發靈活
- 可在工具目錄內快速迭代
- 可輕鬆添加新工具（`tools/` 概念）
- 可獨立測試與除錯

---

## ❓ 待確認事項

### 1. 目錄命名
- [ ] `label_viewer/` （推薦，簡潔）
- [ ] `tools/label_viewer/` （可擴展）
- [ ] `interactive_label_viewer/` （描述性強）

### 2. 啟動方式
- [ ] 獨立腳本 `label_viewer/app.py`（推薦）
- [ ] 整合到 `scripts/` 目錄（不建議，混亂）

### 3. 文檔位置
- [ ] `label_viewer/docs/` （推薦，獨立）
- [ ] 仍放在主專案 `docs/` （不建議）

### 4. 是否需要獨立 Git 子模組？
- [ ] 否，保持在同一倉庫（推薦，方案 A）
- [ ] 是，使用 Git Submodule（複雜，方案 B）

---

## 📝 下一步行動

**請確認**：
1. ✅ 我同意使用**方案 A**（`label_viewer/` 子目錄）
2. ✅ 目錄命名使用 `label_viewer/`
3. ✅ 獨立的 `requirements.txt` 和啟動腳本

**確認後我將**：
1. 建立 `label_viewer/` 目錄結構
2. 開發模組化的 Dash 應用
3. 提供獨立的使用說明

---

**版本**: v1.0（目錄結構規劃）
**日期**: 2025-10-20
**狀態**: ⏳ 等待確認
