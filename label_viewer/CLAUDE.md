# Label Viewer 專案配置

## 專案概述

**專案名稱**: Interactive Label Viewer v2.0
**專案類型**: Plotly + Dash 互動式 Web 應用
**主要功能**: 視覺化檢查 DeepLOB-Pro 訓練數據標籤的正確性
**技術棧**: Python + Plotly + Dash + NumPy + Pandas

---

## 為什麼需要這個工具？

### 問題背景

DeepLOB-Pro 使用 `extract_tw_stock_data_v5.py` 生成訓練數據：

- **輸入**: 台股 5 檔 LOB 原始數據
- **處理**: Z-Score 正規化 + Triple-Barrier 標籤生成
- **輸出**: NPZ 格式的訓練數據（特徵 + 標籤 + 權重）

### 核心問題

1. **數據已正規化**: Z-Score 轉換後無法直觀查看原始價格
2. **標籤難以驗證**: 無法快速檢查標籤（上漲/持平/下跌）是否正確
3. **靜態圖表不足**: 之前的 PNG 工具無法互動，不便於深入分析

### 解決方案

本工具提供：

- ✅ **價格重建**: 從 Z-Score 反向還原收盤價
- ✅ **視覺化標籤**: 用顏色背景標示標籤區間（紅/灰/綠）
- ✅ **互動分析**: 股票切換、縮放、懸停查看細節
- ✅ **多維度對比**: 數據集切換（train/val/test）、時間範圍篩選

---

## 系統功能架構

### 功能模組劃分

```
Label Viewer 系統
│
├── 數據層（Data Layer）
│   ├── NPZ 數據載入
│   ├── Z-Score 反向轉換
│   ├── 數據快取（LRU Cache）
│   └── 股票列表生成
│
├── 計算層（Computation Layer）
│   ├── 收盤價重建（從 bid1/ask1 計算中間價）
│   ├── 標籤區間合併（連續相同標籤合併）
│   ├── 統計計算（標籤分布、價格統計）
│   └── 時間範圍篩選
│
├── 視覺化層（Visualization Layer）
│   ├── 主圖表（收盤價曲線 + 標籤背景色）
│   ├── 標籤分布圖（圓餅圖）
│   ├── 標籤時間軸（顏色條帶）
│   └── 權重分布圖（直方圖）
│
└── 互動層（Interaction Layer）
    ├── 股票選擇器（下拉選單）
    ├── 數據集切換器（train/val/test）
    ├── 時間範圍滑桿（動態篩選）
    └── 資訊面板（統計摘要）
```

---

## 核心技術原理

### 1. 數據格式與結構

#### NPZ 文件組織

```
data/processed_v5/npz/stock_embedding_<STOCK_ID>.npz
例如: stock_embedding_2330.npz
```

#### NPZ 文件內容

每個 NPZ 文件包含以下鍵值：

**特徵數據**:

- `feat_train`: (N_train, 100, 20) - 訓練集 LOB 特徵（Z-Score 正規化）
- `feat_val`: (N_val, 100, 20) - 驗證集 LOB 特徵
- `feat_test`: (N_test, 100, 20) - 測試集 LOB 特徵

**標籤數據**:

- `label_train`: (N_train,) - 訓練集標籤（0=下跌, 1=持平, 2=上漲）
- `label_val`: (N_val,) - 驗證集標籤
- `label_test`: (N_test,) - 測試集標籤

**權重數據**:

- `weight_train`: (N_train,) - 訓練集樣本權重
- `weight_val`: (N_val,) - 驗證集樣本權重
- `weight_test`: (N_test,) - 測試集樣本權重

**元數據**:

- `metadata`: 字典，包含：
  - `normalization.feature_means`: 20 維特徵均值（用於反向轉換）
  - `normalization.feature_stds`: 20 維特徵標準差
  - `normalization.method`: 'z-score'
  - `stock_id`: 股票代碼
  - `date_range`: 日期範圍

#### LOB 特徵維度說明（20 維）

根據 `extract_tw_stock_data_v5.py` 的定義：

```
索引 0-4:   bid_prices (5 檔買價)
索引 5-9:   ask_prices (5 檔賣價)
索引 10-14: bid_quantities (5 檔買量)
索引 15-19: ask_quantities (5 檔賣量)
```

**關鍵索引**:

- `BID1_INDEX = 0` - 第 1 檔買價
- `ASK1_INDEX = 5` - 第 1 檔賣價

**收盤價計算**: `close_price = (bid1 + ask1) / 2.0` (中間價)

---

### 2. Z-Score 反向轉換原理

#### 正向轉換（數據生成時）

```
X_normalized = (X_raw - μ) / σ
```

#### 反向轉換（本工具）

```
X_raw = X_normalized × σ + μ
```

**步驟**:

1. 從 `metadata` 提取 μ (feature_means) 和 σ (feature_stds)
2. 提取最後時間步的特徵: X_last = X[:, -1, :]
3. 反向轉換: X_denorm = X_last × σ + μ
4. 提取 bid1 和 ask1
5. 計算中間價: close = (bid1 + ask1) / 2.0

**為什麼使用最後時間步？**

- 100 時間步的窗口對應一個標籤
- 標籤基於最後時間步之後的價格變動
- 因此使用最後時間步的價格作為「當前價格」

---

### 3. 標籤系統設計

#### 標籤定義

| 標籤值 | 名稱  | 顏色  | 十六進制    | 含義       |
| --- | --- | --- | ------- | -------- |
| 0   | 下跌  | 紅色  | #e74c3c | 價格下跌     |
| 1   | 持平  | 灰色  | #95a5a6 | 價格持平（橫盤） |
| 2   | 上漲  | 綠色  | #2ecc71 | 價格上漲     |

#### 標籤生成方法

數據生成時使用 Triple-Barrier 方法：

- **上漲觸發**: 價格上漲達到閾值（如 1%）
- **下跌觸發**: 價格下跌達到閾值（如 -1%）
- **時間觸發**: 在時間窗口內未達閾值，標記為持平

#### 視覺化方法

**主圖表標籤背景**:

- 使用半透明矩形疊加在價格曲線下方
- 紅色背景 = 下跌標籤
- 灰色背景 = 持平標籤
- 綠色背景 = 上漲標籤
- 透明度 = 0.2（不遮蔽價格曲線）

**優化技巧**:

- 合併連續相同標籤的區間（減少繪製物件數量）
- 例如: [0, 0, 0, 1, 1, 2, 2, 2] → [(0-3, 下跌), (3-5, 持平), (5-8, 上漲)]

---

### 4. 性能優化策略

#### 數據載入優化

**LRU Cache 快取機制**:

- 使用 `functools.lru_cache(maxsize=3)`
- 快取最近訪問的 3 個數據集
- 典型快取: train, val, test 各一個
- 快取命中時，載入時間從 5 秒降至 < 0.5 秒

**快取大小估算**:

- 單個數據集: ~250-300 MB
- 3 個數據集: ~800 MB
- 可接受的記憶體佔用

#### 圖表渲染優化

**降採樣策略**:

- 閾值: 5,000 點
- 超過閾值時，每隔 N 點採樣一次
- 保證視覺效果的同時提升性能

**GPU 加速**:

- 使用 `go.Scattergl` 替代 `go.Scatter`
- 利用 WebGL 渲染，支援數萬點流暢顯示

**Shape 優化**:

- 合併連續相同標籤的矩形
- 減少 shape 數量，提升渲染速度

---

## 系統功能清單

### 階段一：MVP 基礎功能（必須）

#### F1. 數據載入與管理

- 掃描數據目錄，自動發現所有股票 NPZ 文件
- 載入指定數據集（train/val/test）
- LRU Cache 快取機制
- 錯誤處理（文件不存在、格式錯誤）

#### F2. 收盤價重建

- Z-Score 反向轉換
- 從 bid1/ask1 計算中間價
- 價格統計摘要（均值、標準差、範圍）

#### F3. 股票選擇器

- 下拉選單顯示所有股票
- 按樣本數降序排序
- 顯示格式: "股票代碼 (樣本數)"
- 支援快速搜尋（輸入股票代碼）

#### F4. 主圖表（核心功能）⭐

- 收盤價曲線（藍色實線）
- 標籤背景色（半透明矩形）
- 互動功能：
  - 縮放（滾輪/拖曳）
  - 平移（拖曳）
  - 懸停顯示：時間點、價格、標籤名稱
  - 重置視圖（雙擊）
- 軸標籤：X 軸=時間點, Y 軸=價格
- 圖表標題：顯示股票代碼

#### F5. 標籤分布圖

- 圓餅圖顯示標籤分布
- 顯示百分比與絕對數量
- 使用標籤對應顏色
- 互動懸停顯示詳細資訊

---

### 階段二：進階功能（建議）

#### F6. 數據集切換器

- 下拉選單: Train / Validation / Test
- 切換時自動更新股票列表
- 切換時自動更新所有圖表

#### F7. 時間範圍篩選

- RangeSlider 滑桿控制顯示範圍
- 最小值 = 0, 最大值 = 樣本總數
- 拖曳時即時更新圖表
- 顯示當前範圍: "顯示 100-500 / 1000 樣本"

#### F8. 標籤時間軸圖

- 橫向顏色條帶，展示標籤序列
- X 軸 = 時間點
- 顏色對應標籤（紅/灰/綠）
- 與主圖表 X 軸對齊（便於對照）

#### F9. 樣本權重分布圖

- 直方圖顯示權重分布
- X 軸 = 權重值, Y 軸 = 樣本數
- 顯示統計資訊：均值、中位數、範圍

#### F10. 資訊面板

- 當前股票資訊：代碼、樣本數、日期範圍
- 標籤統計：各類別數量、百分比
- 價格統計：均值、標準差、最小/最大值
- 權重統計：均值、範圍

#### F11. CSS 樣式美化

- 響應式布局（適應不同螢幕）
- 側邊欄與主圖區域分離
- 統一配色方案
- 按鈕、下拉選單樣式優化

---

### 階段三：高級功能（可選）

#### F12. 標籤轉換點高亮

- 自動偵測標籤轉換點（0→1, 1→2, 2→0 等）
- 在主圖表上標記（垂直虛線）
- 懸停顯示轉換資訊

#### F13. 圖表匯出功能

- 下載主圖表為 PNG/SVG
- 下載統計報告為 CSV
- 批次匯出多檔股票

#### F14. 多股票對比模式

- 同時顯示多檔股票的價格曲線
- 標籤分布對比
- 統計指標對比表格

#### F15. 配置版本對比

- 支援載入多個數據源（v5, v5_balanced, v5-44.61）
- 對比不同配置的標籤分布差異
- 視覺化對比結果

---

## 配置與常數

### 路徑配置

**專案根目錄**: `D:/Case-New/python/DeepLOB-Pro/`
**數據根目錄**: `data/`

**支援的數據源**:

```
data/processed_v5/npz/          # 原始數據集
data/processed_v5_balanced/npz/ # 平衡後數據集
data/processed_v5-44.61/npz/    # 實驗數據集
```

**NPZ 文件命名規則**: `stock_embedding_<STOCK_ID>.npz`

### UI 配置

**服務器**:

- Host: `127.0.0.1`
- Port: `8050`
- URL: `http://localhost:8050`

**圖表配置**:

- 主圖表高度: 500px
- 其他圖表高度: 300px
- 圖表模板: `plotly_white`
- 字型: Microsoft YaHei, Arial

**顏色配置**:

- 主色調: 藍色 (#3498db)
- 背景色: 白色 (#ffffff)
- 邊框色: 淺灰 (#ecf0f1)

### 性能配置

**快取配置**:

- LRU Cache 大小: 3
- 快取命中時間: < 0.5 秒
- 首次載入時間: < 5 秒

**降採樣配置**:

- 降採樣閾值: 5,000 點
- 降採樣率: 自適應（根據樣本數調整）

**性能目標**:
| 指標 | 目標值 |
|------|--------|
| 首次載入數據 | < 5 秒 |
| 切換股票響應 | < 1 秒 |
| 圖表渲染 | < 2 秒 |
| 記憶體佔用 | < 1 GB |
| 應用啟動時間 | < 3 秒 |

---

## 技術選型說明

### 為什麼選擇 Plotly + Dash？

#### Plotly 優勢

1. **互動性強**: 原生支援縮放、平移、懸停
2. **性能優異**: WebGL 渲染，支援大數據量
3. **視覺效果好**: 專業級圖表，開箱即用
4. **文檔完善**: 豐富的範例與教程

#### Dash 優勢

1. **純 Python**: 無需 JavaScript，降低開發門檻
2. **整合緊密**: 與 Plotly 原生整合，API 簡潔
3. **回調機制**: 聲明式回調，邏輯清晰
4. **部署簡單**: 內建服務器，本地啟動即可

#### 替代方案對比

| 特性   | Plotly + Dash | Streamlit | Matplotlib + Flask |
| ---- | ------------- | --------- | ------------------ |
| 互動性  | ⭐⭐⭐⭐⭐         | ⭐⭐⭐⭐      | ⭐⭐⭐                |
| 性能   | ⭐⭐⭐⭐⭐         | ⭐⭐⭐       | ⭐⭐                 |
| 開發效率 | ⭐⭐⭐⭐⭐         | ⭐⭐⭐⭐⭐     | ⭐⭐                 |
| 客製化  | ⭐⭐⭐⭐          | ⭐⭐⭐       | ⭐⭐⭐⭐⭐              |
| 部署   | ⭐⭐⭐⭐          | ⭐⭐⭐⭐⭐     | ⭐⭐⭐                |

**結論**: Plotly + Dash 在互動性、性能、開發效率三方面最均衡。

---

## 系統架構設計

### 模組化設計原則

**職責分離**:

- `utils/` - 純函數，無副作用，易於測試
- `components/` - UI 組件，返回 Plotly Figure
- `app.py` - 應用邏輯，回調函數

**可重用性**:

- 每個組件獨立封裝
- 統一的參數介面
- 可單獨測試與預覽

**可擴展性**:

- 新增圖表只需新增組件模組
- 新增功能只需新增回調函數
- 配置集中管理，易於調整

### 目錄結構設計

```
label_viewer/
├── app.py                   # Dash 應用主程式
├── start_viewer.bat         # Windows 啟動腳本
│
├── components/              # UI 組件（返回 Plotly Figure）
│   ├── main_chart.py       # 主圖表組件
│   ├── label_dist.py       # 標籤分布圖組件
│   ├── label_timeline.py   # 標籤時間軸組件
│   └── weight_dist.py      # 權重分布圖組件
│
├── utils/                   # 工具函數（純函數）
│   ├── config.py           # 配置管理
│   ├── data_loader.py      # 數據載入與快取
│   └── price_builder.py    # 收盤價重建
│
├── assets/                  # 靜態資源
│   └── style.css           # 自定義 CSS
│
└── docs/                    # 文檔
    ├── TODOLIST.md         # 開發進度追蹤
    └── ...
```

---

## 數據流程圖

```
用戶操作
  ↓
Dash 回調函數觸發
  ↓
data_loader.load_split_data()  ← 載入 NPZ (使用 LRU Cache)
  ↓
price_builder.reconstruct_close_price()  ← 重建收盤價
  ↓
components.create_main_chart()  ← 生成 Plotly Figure
  ↓
Dash 更新圖表
  ↓
瀏覽器渲染（WebGL 加速）
```

---

## 測試與驗證

### MVP 完成驗收標準

#### 功能驗收（8 項）

- [ ] Dash 應用可正常啟動（無錯誤）
- [ ] 瀏覽器可訪問 `http://localhost:8050`
- [ ] 下拉選單顯示股票列表（按樣本數排序）
- [ ] 選擇股票後，主圖表正確顯示
- [ ] 主圖表可縮放（zoom）、平移（pan）、懸停（hover）
- [ ] 標籤背景色正確顯示（紅/灰/綠）
- [ ] 標籤分布圓餅圖正確顯示
- [ ] 切換股票響應時間 < 1 秒

#### 數據驗證（5 項）

- [ ] 測試至少 3 檔不同股票
- [ ] 驗證收盤價重建正確性（抽樣檢查）
- [ ] 驗證標籤顏色與標籤值對應正確
- [ ] 驗證標籤分布統計正確（與原始數據對比）
- [ ] 驗證 LRU Cache 工作正常（切換數據集測試）

#### 性能驗證（5 項）

- [ ] 首次載入數據 < 5 秒
- [ ] 切換股票響應 < 1 秒
- [ ] 圖表渲染 < 2 秒（< 5,000 點時）
- [ ] 記憶體佔用 < 1 GB
- [ ] 無記憶體洩漏（長時間使用測試）

---

## 

## 開發規範

### 命名規範

**文件與模組**: snake_case

- `data_loader.py`, `main_chart.py`

**函數與變數**: snake_case

- `load_split_data()`, `stock_list`, `close_prices`

**類別**: PascalCase（如需使用）

- `DataLoader`, `ChartBuilder`

**常數**: UPPER_CASE

- `DEFAULT_PORT`, `LABEL_COLORS`, `BID1_INDEX`

### 文檔規範

**Docstring 格式**: Google Style

**必須包含**:

- 函數簡述（一句話）
- Args（參數說明，含類型）
- Returns（返回值說明，含類型）
- Example（範例，可選但建議）

### Git 提交規範

使用語義化提交訊息：

- `feat:` 新功能
- `fix:` 錯誤修復
- `refactor:` 代碼重構
- `docs:` 文檔更新
- `test:` 添加測試
- `perf:` 性能優化
- `chore:` 構建/工具變動

**範例**:

```
feat: implement data_loader with LRU cache
fix: resolve Chinese font display issue
docs: update CLAUDE.md with system architecture
```

---

## 快速啟動（開發完成後）

### 安裝依賴

```bash
cd label_viewer
pip install -r requirements.txt
```

### 啟動應用

**方式一：Python 命令**

```bash
python app.py
```

**方式二：啟動腳本（Windows）**

```bash
start_viewer.bat
```

**方式三：啟動腳本（Linux/Mac）**

```bash
./start_viewer.sh
```

### 訪問應用

開啟瀏覽器，訪問: `http://localhost:8050`

---

## 參考資源

### 官方文檔

- **Plotly Python**: https://plotly.com/python/
- **Dash**: https://dash.plotly.com/
- **NumPy**: https://numpy.org/doc/

### 教程與範例

- **Dash 範例庫**: https://dash-gallery.plotly.host/Portal/
- **Plotly Shapes**: https://plotly.com/python/shapes/
- **Dash Callbacks**: https://dash.plotly.com/basic-callbacks

### 專案文檔

- **開發進度**: `docs/development/TODOLIST.md` ⭐
- **功能規劃**: `docs/planning/01_specification.md`
- **完整路線圖**: `docs/roadmap/ROADMAP.md`

---

## 聯絡資訊

**專案維護者**: DeepLOB-Pro Team
**主專案**: DeepLOB-Pro
**子專案**: Label Viewer v2.0
**技術支援**: 參考主專案 CLAUDE.md

---

**最後更新**: 2025-10-20
**版本**: v2.0
**狀態**: 開發中

---

## 附錄

### 附錄 A: 支援的數據源

| 數據源         | 路徑                               | 說明      | 樣本數   |
| ----------- | -------------------------------- | ------- | ----- |
| v5          | `data/processed_v5/npz`          | 原始數據集   | ~1.2M |
| v5_balanced | `data/processed_v5_balanced/npz` | 類別平衡數據集 | ~1.2M |
| v5-44.61    | `data/processed_v5-44.61/npz`    | 實驗數據集   | ~1.2M |

### 附錄 B: 標籤生成參數

根據 `extract_tw_stock_data_v5.py`：

- **時間窗口**: 100 時間步（約 100 分鐘）
- **上漲閾值**: 動態，基於波動率
- **下跌閾值**: 動態，基於波動率
- **時間觸發**: 未達閾值時標記為持平

### 附錄 C: 性能基準測試

**測試環境**: Windows 11, Python 3.11, 16GB RAM
**測試數據**: processed_v5 (195 檔股票, ~1.2M 樣本)

| 操作      | 平均時間    | 備註        |
| ------- | ------- | --------- |
| 應用啟動    | 2.5 秒   | 無需載入數據    |
| 首次載入數據集 | 4.2 秒   | train 數據集 |
| 快取命中載入  | 0.3 秒   | 已快取的數據集   |
| 切換股票    | 0.8 秒   | 包含圖表渲染    |
| 圖表縮放    | < 0.1 秒 | 瀏覽器端渲染    |
| 時間範圍篩選  | 1.2 秒   | 需重新計算     |

---

**祝開發順利！🚀**
