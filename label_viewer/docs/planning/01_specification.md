# 互動式標籤查看器 - 功能規劃書

## 📋 專案概述

**專案名稱**：Interactive Label Viewer（互動式訓練數據標籤查看器）
**目的**：取代靜態 PNG 圖表，提供互動式 Web 介面深入探索訓練數據標籤
**技術選型**：Plotly + Dash（Python Web 框架）
**版本**：v2.0（相較於 v1.0 靜態版本）
**開發日期**：2025-10-20

---

## 🎯 核心需求分析

### 問題點（v1.0 靜態版本的限制）

1. **彈性不足**
   - 只能查看預先選定的前 N 檔股票
   - 無法動態切換股票
   - 無法即時調整顯示範圍

2. **互動性差**
   - 靜態 PNG 無法縮放、平移
   - 無法懸停查看詳細數值
   - 無法快速對比不同股票

3. **單一股票深入分析困難**
   - 需要重新執行腳本才能查看其他股票
   - 無法同時對比多檔股票
   - 無法即時篩選特定時間範圍

4. **批次處理不便**
   - 需要產生大量 PNG 文件
   - 無法快速導航
   - 佔用大量磁碟空間

### 目標（v2.0 互動式版本）

1. **即時互動**
   - 下拉選單即時切換股票
   - 滑鼠縮放、拖曳平移
   - 懸停顯示詳細資訊

2. **單一股票深入分析**
   - 選擇任意股票查看詳細標籤
   - 時間範圍篩選（顯示指定區間）
   - 標籤觸發點高亮顯示

3. **多維度對比**
   - 多股票疊加對比
   - 不同數據集（train/val/test）切換
   - 不同配置版本（v5/v5_balanced）對比

4. **輕量化部署**
   - 本地 Web 服務器（不需要額外安裝）
   - 瀏覽器訪問（跨平台）
   - 資源佔用小

---

## 🛠️ 技術選型

### 方案對比

| 方案 | 優點 | 缺點 | 評分 |
|------|------|------|------|
| **Plotly + Dash** | Python 原生、互動性強、部署簡單 | 需要額外套件 | ⭐⭐⭐⭐⭐ |
| Streamlit | 更簡單、UI 美觀 | 彈性較差、不適合複雜互動 | ⭐⭐⭐⭐ |
| Flask + D3.js | 完全自定義 | 開發複雜、需要寫 JavaScript | ⭐⭐⭐ |
| PyQt/Tkinter | 原生桌面應用 | 跨平台差、部署困難 | ⭐⭐ |

### 最終選擇：**Plotly + Dash**

**理由**：
1. ✅ **Python 原生**：與現有代碼無縫整合
2. ✅ **互動性強**：Plotly 提供豐富的互動功能（縮放、懸停、選擇）
3. ✅ **部署簡單**：內建 Web 服務器，瀏覽器訪問即可
4. ✅ **學習曲線平緩**：類似 Flask，易於上手
5. ✅ **社群活躍**：文檔完善、範例豐富

### 依賴套件

```bash
pip install dash plotly pandas numpy
```

---

## 📐 UI 設計（頁面布局）

### 整體布局

```
+----------------------------------------------------------+
|                    標題列 (Header)                        |
|  訓練數據標籤互動查看器 - DeepLOB-Pro v2.0                |
+----------------------------------------------------------+
|                    控制面板 (Sidebar)                     |
|  +----------------------------------------------------+  |
|  | [下拉] 選擇數據集: [train ▼] [val] [test]         |  |
|  | [下拉] 選擇股票:   [2330 ▼] (顯示前 50 檔)        |  |
|  | [滑桿] 時間範圍:   [0 ----●●●●---- 5000]          |  |
|  | [核取方塊] 顯示選項:                               |  |
|  |   ☑ 顯示標籤背景色                                  |  |
|  |   ☑ 顯示標籤轉換點                                  |  |
|  |   ☐ 顯示樣本權重                                    |  |
|  | [按鈕] 重新整理數據                                 |  |
|  | [按鈕] 下載當前圖表 (PNG)                          |  |
|  +----------------------------------------------------+  |
+----------------------------------------------------------+
|                  主圖表區域 (Main Chart)                  |
|  +----------------------------------------------------+  |
|  |  圖 1: 收盤價與標籤趨勢（可縮放、拖曳）             |  |
|  |  - 收盤價曲線（藍色線）                            |  |
|  |  - 標籤背景色（紅/灰/綠）                          |  |
|  |  - 懸停顯示：時間點、價格、標籤、權重              |  |
|  +----------------------------------------------------+  |
+----------------------------------------------------------+
|                 次級圖表區域 (Secondary Charts)           |
|  +------------------------+  +-------------------------+ |
|  | 圖 2: 標籤序列（時間軸）|  | 圖 3: 標籤分布統計      | |
|  | - 顏色條帶顯示          |  | - 圓餅圖或柱狀圖        | |
|  +------------------------+  +-------------------------+ |
|  +----------------------------------------------------+  |
|  | 圖 4: 樣本權重分布（直方圖）                        |  |
|  +----------------------------------------------------+  |
+----------------------------------------------------------+
|                    資訊面板 (Info Panel)                  |
|  當前股票: 2330 | 樣本數: 8,543 | 標籤分布: 下跌 28.5%,  |
|  持平 45.2%, 上漲 26.3% | 權重均值: 0.98             |
+----------------------------------------------------------+
```

---

## 🎨 功能模組設計

### 模組 1: 數據載入與快取 (Data Loader)

**功能**：
- 載入 NPZ 數據與 metadata
- 重建收盤價（反向 Z-Score）
- 快取數據（避免重複載入）
- 支援多數據集切換

**輸入**：
- `data_dir`: 數據目錄路徑
- `split`: 數據集劃分（train/val/test）

**輸出**：
- `stock_data`: 股票數據字典
  ```python
  {
      '2330': {
          'close': np.array([...]),
          'labels': np.array([...]),
          'weights': np.array([...]),
          'indices': np.array([...])  # 原始索引
      },
      ...
  }
  ```

**技術細節**：
```python
import functools

@functools.lru_cache(maxsize=3)  # 快取最多 3 個數據集
def load_split_data(data_dir, split):
    # 載入並處理數據
    return stock_data
```

---

### 模組 2: 主圖表生成 (Main Chart Generator)

**功能**：
- 生成收盤價與標籤趨勢的互動圖表
- 支援縮放、平移、懸停
- 標籤背景色疊加
- 標籤轉換點標記

**輸入**：
- `stock_id`: 股票代碼
- `close`: 收盤價序列
- `labels`: 標籤序列
- `weights`: 樣本權重序列
- `time_range`: 顯示的時間範圍 (start, end)
- `show_options`: 顯示選項字典

**輸出**：
- `fig`: Plotly 圖表對象

**技術細節**：
```python
import plotly.graph_objects as go

def create_main_chart(stock_id, close, labels, weights, time_range, show_options):
    fig = go.Figure()

    # 添加收盤價曲線
    fig.add_trace(go.Scatter(
        x=list(range(len(close))),
        y=close,
        mode='lines',
        name='收盤價',
        line=dict(color='#3498db', width=2),
        hovertemplate='<b>時間點</b>: %{x}<br><b>價格</b>: %{y:.2f}<extra></extra>'
    ))

    # 添加標籤背景色（使用 shapes）
    if show_options.get('show_label_background', True):
        for i in range(len(labels)):
            label = int(labels[i])
            color = LABEL_COLORS[label]
            fig.add_shape(
                type='rect',
                x0=i-0.5, x1=i+0.5,
                y0=0, y1=1,
                yref='paper',
                fillcolor=color,
                opacity=0.2,
                layer='below',
                line_width=0
            )

    # 配置布局
    fig.update_layout(
        title=f'{stock_id} - 收盤價與標籤趨勢',
        xaxis_title='時間點',
        yaxis_title='價格',
        hovermode='x unified',
        template='plotly_white',
        height=500
    )

    return fig
```

---

### 模組 3: 次級圖表生成 (Secondary Charts Generator)

**功能**：
- 標籤序列（時間軸顏色條帶）
- 標籤分布統計（圓餅圖或柱狀圖）
- 樣本權重分布（直方圖）

**輸入**：
- `labels`: 標籤序列
- `weights`: 樣本權重序列

**輸出**：
- `label_timeline_fig`: 標籤時間軸圖表
- `label_dist_fig`: 標籤分布圖表
- `weight_dist_fig`: 權重分布圖表

**技術細節**：
```python
# 標籤分布圓餅圖
def create_label_distribution_chart(labels):
    label_counts = pd.Series(labels).value_counts().sort_index()

    fig = go.Figure(data=[go.Pie(
        labels=[LABEL_NAMES[i] for i in label_counts.index],
        values=label_counts.values,
        marker=dict(colors=[LABEL_COLORS[i] for i in label_counts.index]),
        textinfo='label+percent',
        hovertemplate='<b>%{label}</b><br>數量: %{value}<br>佔比: %{percent}<extra></extra>'
    )])

    fig.update_layout(title='標籤分布', height=300)
    return fig
```

---

### 模組 4: Dash 應用框架 (Dash App)

**功能**：
- Web 服務器啟動
- UI 組件布局
- 回調函數（Callback）處理互動

**核心結構**：
```python
import dash
from dash import dcc, html
from dash.dependencies import Input, Output

app = dash.Dash(__name__)

# 布局定義
app.layout = html.Div([
    # 標題列
    html.H1('訓練數據標籤互動查看器 - DeepLOB-Pro v2.0'),

    # 控制面板
    html.Div([
        # 數據集選擇
        dcc.Dropdown(
            id='split-selector',
            options=[
                {'label': '訓練集 (Train)', 'value': 'train'},
                {'label': '驗證集 (Val)', 'value': 'val'},
                {'label': '測試集 (Test)', 'value': 'test'}
            ],
            value='train'
        ),

        # 股票選擇
        dcc.Dropdown(id='stock-selector'),

        # 時間範圍滑桿
        dcc.RangeSlider(id='time-range-slider'),

        # 顯示選項
        dcc.Checklist(
            id='display-options',
            options=[
                {'label': '顯示標籤背景色', 'value': 'show_bg'},
                {'label': '顯示標籤轉換點', 'value': 'show_transition'},
                {'label': '顯示樣本權重', 'value': 'show_weight'}
            ],
            value=['show_bg']
        )
    ], style={'width': '20%', 'float': 'left'}),

    # 主圖表區域
    html.Div([
        dcc.Graph(id='main-chart'),
        html.Div([
            dcc.Graph(id='label-timeline', style={'width': '50%', 'display': 'inline-block'}),
            dcc.Graph(id='label-distribution', style={'width': '50%', 'display': 'inline-block'})
        ]),
        dcc.Graph(id='weight-distribution')
    ], style={'width': '80%', 'float': 'right'}),

    # 資訊面板
    html.Div(id='info-panel')
])

# 回調函數：更新股票列表
@app.callback(
    Output('stock-selector', 'options'),
    Input('split-selector', 'value')
)
def update_stock_options(split):
    # 載入數據並返回股票列表
    pass

# 回調函數：更新圖表
@app.callback(
    [Output('main-chart', 'figure'),
     Output('label-timeline', 'figure'),
     Output('label-distribution', 'figure'),
     Output('weight-distribution', 'figure'),
     Output('info-panel', 'children')],
    [Input('stock-selector', 'value'),
     Input('time-range-slider', 'value'),
     Input('display-options', 'value')]
)
def update_charts(stock_id, time_range, display_options):
    # 生成並返回所有圖表
    pass

# 啟動服務器
if __name__ == '__main__':
    app.run_server(debug=True, port=8050)
```

---

## 🌟 核心功能清單

### 階段一：基礎功能（MVP）

- [ ] **F1.1** 數據載入與快取
  - 載入 NPZ 數據
  - 重建收盤價
  - LRU 快取避免重複載入

- [ ] **F1.2** 股票選擇器
  - 下拉選單顯示前 50 檔股票（按樣本數排序）
  - 即時切換股票

- [ ] **F1.3** 主圖表（收盤價與標籤趨勢）
  - 收盤價曲線
  - 標籤背景色疊加
  - 懸停顯示詳細資訊
  - 縮放、平移功能

- [ ] **F1.4** 標籤分布圖
  - 圓餅圖或柱狀圖
  - 顯示百分比與數量

- [ ] **F1.5** Web 服務器啟動
  - 本地啟動（localhost:8050）
  - 瀏覽器訪問

### 階段二：進階功能

- [ ] **F2.1** 數據集切換
  - train/val/test 快速切換
  - 自動更新股票列表

- [ ] **F2.2** 時間範圍篩選
  - 滑桿選擇顯示區間
  - 即時更新圖表

- [ ] **F2.3** 標籤序列時間軸圖
  - 顏色條帶顯示標籤
  - 與主圖同步縮放

- [ ] **F2.4** 樣本權重分布圖
  - 直方圖顯示權重分配
  - 統計摘要（均值、標準差）

- [ ] **F2.5** 資訊面板
  - 當前股票統計摘要
  - 標籤分布百分比
  - 權重統計

### 階段三：高級功能（可選）

- [ ] **F3.1** 多股票對比
  - 選擇多檔股票同時顯示
  - 不同顏色區分

- [ ] **F3.2** 標籤轉換點高亮
  - 標記標籤切換位置
  - 顯示觸發原因（up/down/time）

- [ ] **F3.3** 配置版本對比
  - 同時載入多個配置版本
  - 並排對比圖表

- [ ] **F3.4** 圖表下載
  - 下載當前圖表為 PNG
  - 批次下載所有股票圖表

- [ ] **F3.5** 搜尋與篩選
  - 股票代碼搜尋
  - 按標籤分布篩選（例如：只顯示上漲類別 > 30% 的股票）

---

## 📊 性能考量

### 數據量評估

- 訓練集：1,249,419 個樣本，256 檔股票
- 單檔股票平均：~4,880 個樣本
- 記憶體佔用：
  - NPZ 載入：~500 MB
  - 處理後數據：~200 MB
  - Dash 快取：~100 MB
  - **總計**：~800 MB（可接受）

### 優化策略

1. **懶加載（Lazy Loading）**
   - 只載入當前選擇的股票數據
   - 其他股票按需載入

2. **LRU 快取**
   - 快取最近訪問的 3 個數據集
   - 避免重複載入

3. **降採樣（Downsampling）**
   - 當樣本數 > 5000 時，進行降採樣顯示
   - 保留完整數據供下載

4. **圖表優化**
   - 使用 Plotly 的 `Scattergl`（WebGL 加速）
   - 限制顯示點數（最多 10,000 點）

---

## 🚀 開發計劃

### 里程碑

| 階段 | 功能 | 預估時間 | 狀態 |
|------|------|----------|------|
| 階段一 | MVP（基礎功能 F1.1-F1.5） | 2-3 小時 | ⏳ 待開發 |
| 階段二 | 進階功能（F2.1-F2.5） | 1-2 小時 | ⏳ 待開發 |
| 階段三 | 高級功能（F3.1-F3.5，可選） | 2-3 小時 | ⏳ 可選 |
| 測試 | 完整測試與除錯 | 1 小時 | ⏳ 待執行 |
| 文檔 | 使用說明文檔 | 30 分鐘 | ⏳ 待撰寫 |

### 開發優先級

**必須完成（P0）**：
- F1.1 ~ F1.5（MVP 基礎功能）
- F2.1 ~ F2.5（進階功能）

**建議完成（P1）**：
- F3.2（標籤轉換點高亮）
- F3.4（圖表下載）

**可選完成（P2）**：
- F3.1（多股票對比）
- F3.3（配置版本對比）
- F3.5（搜尋與篩選）

---

## 🎯 成功標準

### 功能完整性
- ✅ 可即時切換股票查看標籤
- ✅ 圖表可縮放、平移、懸停
- ✅ 數據集（train/val/test）可切換
- ✅ 顯示標籤分布與樣本權重統計

### 性能標準
- ✅ 載入數據 < 5 秒
- ✅ 切換股票響應 < 1 秒
- ✅ 圖表渲染 < 2 秒
- ✅ 記憶體佔用 < 1 GB

### 用戶體驗
- ✅ UI 簡潔直觀
- ✅ 操作流暢無卡頓
- ✅ 錯誤提示清晰
- ✅ 使用說明完整

---

## 📦 交付物清單

### 代碼文件
1. **`scripts/interactive_label_viewer.py`** - 主程式
2. **`scripts/label_viewer_utils.py`** - 工具函數模組（可選）

### 說明文檔
1. **`docs/interactive_label_viewer_usage.md`** - 使用說明
2. **`INTERACTIVE_VIEWER_README.md`** - 快速上手指南

### 啟動腳本
1. **`start_label_viewer.bat`** - Windows 啟動腳本
2. **`start_label_viewer.sh`** - Linux/Mac 啟動腳本（可選）

### 範例截圖
1. **`docs/screenshots/viewer_main.png`** - 主介面截圖
2. **`docs/screenshots/viewer_interaction.gif`** - 互動示範（可選）

---

## 🔄 與 v1.0 對比

| 特性 | v1.0 (靜態 PNG) | v2.0 (互動 Web) |
|------|----------------|----------------|
| 股票選擇 | 預先選定前 N 檔 | 下拉選單即時切換 ⭐ |
| 圖表互動 | 無（靜態圖） | 縮放、平移、懸停 ⭐ |
| 數據集切換 | 需重新執行 | 即時切換 ⭐ |
| 時間範圍 | 固定或降採樣 | 滑桿動態調整 ⭐ |
| 部署方式 | 生成 PNG 文件 | Web 瀏覽器訪問 ⭐ |
| 磁碟佔用 | 大（數百 MB PNG） | 小（只有代碼） ⭐ |
| 使用門檻 | 低（開啟圖片） | 中（需啟動服務器） |
| 適用場景 | 報告、文檔 | 探索性分析、除錯 ⭐ |

**結論**：v2.0 在互動性、彈性、資源佔用方面全面優於 v1.0，建議開發。

---

## ❓ 待確認問題

### 技術問題
1. **是否需要支援多數據目錄對比？**
   - 例如同時比較 `processed_v5` 和 `processed_v5_balanced`
   - 增加開發複雜度，但實用性高

2. **是否需要支援自定義時間範圍精確選擇？**
   - 除了滑桿，是否需要輸入框精確指定 [start, end]
   - 提高精確度，但增加 UI 複雜度

3. **是否需要標籤轉換點的詳細分析？**
   - 顯示每個轉換點的觸發原因（up/down/time）
   - 顯示觸發時的價格變化
   - 增加分析深度，但需要額外數據處理

### 功能優先級
4. **階段三的高級功能（F3.1-F3.5）是否需要實作？**
   - 多股票對比
   - 配置版本對比
   - 搜尋與篩選
   - 根據用戶需求決定

5. **是否需要支援匯出報告？**
   - HTML 報告（包含所有圖表）
   - PDF 報告（靜態版本）
   - Excel 數據表（原始數據）

---

## 📝 下一步行動

**請確認以下事項，我將開始開發**：

### 確認項目
- [ ] **功能範圍**：階段一 + 階段二（必須），階段三（可選）
- [ ] **技術選型**：Plotly + Dash（已確認）
- [ ] **優先級**：P0（必須）+ P1（建議）
- [ ] **待確認問題 1-5** 的答案

### 開始開發前
- [ ] 用戶已閱讀並確認本規劃書
- [ ] 用戶已決定是否需要階段三功能
- [ ] 用戶已決定待確認問題的答案

---

**版本**：v1.0（規劃書）
**撰寫日期**：2025-10-20
**狀態**：⏳ 待用戶確認
