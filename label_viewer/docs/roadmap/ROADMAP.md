# Label Viewer 功能路線圖

**專案名稱**：Interactive Label Viewer v2.0
**版本計劃**：v2.0 (MVP) → v2.1 (進階) → v2.2 (高級)
**最後更新**：2025-10-20

---

## 🎯 總體目標

打造一個**互動式 Web 工具**，用於查看和分析訓練數據標籤正確性，取代靜態 PNG 圖表，提供更好的探索性分析體驗。

### 核心價值主張
1. **即時互動**：下拉選單即時切換股票，無需重新執行腳本
2. **深入分析**：單一股票詳細分析，標籤與趨勢對照
3. **輕量部署**：本地 Web 服務器，瀏覽器訪問
4. **易於擴展**：模組化設計，未來可輕鬆添加新功能

---

## 📅 開發時間軸

```
2025-10-20 (Day 1)
├── ✅ 規劃階段 (1 小時)
├── ✅ 基礎設施 (0.5 小時)
└── ⏳ 階段一 (MVP) - 預計 1.5-2 小時

2025-10-21 (Day 2, 可選)
├── ⏳ 階段二 (進階功能) - 預計 1-1.5 小時
└── ⏳ 測試與文檔 - 預計 0.5 小時

未來擴展 (Day 3+, 可選)
└── ⏳ 階段三 (高級功能) - 預計 2-3 小時
```

---

## 🚀 階段一：MVP（最小可行產品）

**目標**：建立可運行的基礎版本，實現核心功能
**預計時間**：1.5-2 小時
**優先級**：P0（必須完成）
**當前狀態**：🚧 進行中（30% 完成）

### 功能清單

| # | 功能 | 說明 | 預估時間 | 狀態 | 優先級 |
|---|------|------|----------|------|--------|
| **F1.1** | 數據載入與快取 | 載入 NPZ、LRU 快取、元數據解析 | 30 分鐘 | ⏳ 待開發 | P0 |
| **F1.2** | 收盤價重建 | 反向 Z-Score、中間價計算 | - | ✅ 完成 | P0 |
| **F1.3** | 股票選擇器 | 下拉選單、按樣本數排序 | 包含 F1.5 | ⏳ 待開發 | P0 |
| **F1.4** | 主圖表 | 收盤價曲線 + 標籤背景色 | 40 分鐘 | ⏳ 待開發 | P0 |
| **F1.5** | 標籤分布圖 | 圓餅圖或柱狀圖 | 20 分鐘 | ⏳ 待開發 | P0 |
| **F1.6** | Dash 應用骨架 | 布局、路由、回調 | 30 分鐘 | ⏳ 待開發 | P0 |
| **F1.7** | Web 服務器啟動 | 本地 localhost:8050 | 包含 F1.6 | ⏳ 待開發 | P0 |

### 技術實作細節

#### F1.1: 數據載入與快取
```python
@functools.lru_cache(maxsize=3)
def load_split_data(data_dir: str, split: str):
    """載入並快取數據集"""
    # 載入 NPZ
    data = np.load(f"{data_dir}/stock_embedding_{split}.npz")
    X, y, weights, stock_ids = data['X'], data['y'], data['weights'], data['stock_ids']

    # 載入 metadata
    with open(f"{data_dir}/normalization_meta.json") as f:
        metadata = json.load(f)

    # 重建收盤價
    close = reconstruct_close_price(X, metadata)

    # 組織為股票字典
    stock_data = {}
    for stock in np.unique(stock_ids):
        mask = stock_ids == stock
        stock_data[stock] = {
            'close': close[mask],
            'labels': y[mask],
            'weights': weights[mask],
            'indices': np.where(mask)[0]
        }

    return stock_data, metadata
```

#### F1.4: 主圖表
```python
def create_main_chart(stock_id, close, labels, weights, options):
    """生成主圖表"""
    fig = go.Figure()

    # 添加收盤價曲線
    fig.add_trace(go.Scatter(
        x=list(range(len(close))),
        y=close,
        mode='lines',
        name='收盤價',
        line=dict(color='#3498db', width=2),
        hovertemplate='<b>時間</b>: %{x}<br><b>價格</b>: %{y:.2f}<extra></extra>'
    ))

    # 添加標籤背景色
    if options.get('show_label_background', True):
        for i, label in enumerate(labels):
            color = LABEL_COLORS[int(label)]
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

#### F1.6: Dash 應用骨架
```python
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1('訓練數據標籤互動查看器'),

    # 側邊欄
    html.Div([
        html.Label('選擇股票:'),
        dcc.Dropdown(id='stock-selector', options=[], value=None),
    ], style={'width': '20%', 'float': 'left'}),

    # 主圖區
    html.Div([
        dcc.Graph(id='main-chart'),
        dcc.Graph(id='label-distribution'),
    ], style={'width': '80%', 'float': 'right'}),
])

@app.callback(
    [Output('main-chart', 'figure'),
     Output('label-distribution', 'figure')],
    Input('stock-selector', 'value')
)
def update_charts(stock_id):
    # 生成並返回圖表
    pass
```

### 檢查點（Milestone 1）

**定義**：MVP 完成標準

**檢查清單**：
- [ ] Dash 應用可成功啟動（`python app.py`）
- [ ] 可在瀏覽器訪問 `http://localhost:8050`
- [ ] 下拉選單顯示至少 10 檔股票
- [ ] 可選擇股票並顯示主圖表
- [ ] 主圖表顯示收盤價曲線
- [ ] 主圖表顯示標籤背景色（紅/灰/綠）
- [ ] 主圖表可縮放、平移、懸停
- [ ] 標籤分布圖正確顯示

**驗收測試**：
1. 啟動應用無錯誤
2. 選擇 3 檔不同股票，圖表正確更新
3. 縮放主圖表，背景色不變形
4. 懸停顯示正確的時間點與價格

**預期成果截圖**：
```
[主圖表]
- 藍色收盤價曲線
- 紅/灰/綠背景色塊（半透明）
- 懸停顯示：時間點 123, 價格 150.25

[標籤分布圖]
- 圓餅圖：下跌 28%, 持平 45%, 上漲 27%
```

---

## 🎨 階段二：進階功能

**目標**：完整互動功能，支援多維度分析
**預計時間**：1-1.5 小時
**優先級**：P1（建議完成）
**當前狀態**：⏳ 待開始

### 功能清單

| # | 功能 | 說明 | 預估時間 | 狀態 | 優先級 |
|---|------|------|----------|------|--------|
| **F2.1** | 數據集切換 | train/val/test 下拉選單 | 20 分鐘 | ⏳ 待開發 | P1 |
| **F2.2** | 時間範圍篩選 | RangeSlider 動態調整顯示區間 | 20 分鐘 | ⏳ 待開發 | P1 |
| **F2.3** | 標籤時間軸圖 | 顏色條帶顯示標籤序列 | 25 分鐘 | ⏳ 待開發 | P1 |
| **F2.4** | 樣本權重分布圖 | 直方圖 + 統計摘要 | 20 分鐘 | ⏳ 待開發 | P1 |
| **F2.5** | 資訊面板 | 統計摘要（樣本數、分布、權重） | 15 分鐘 | ⏳ 待開發 | P1 |
| **F2.6** | 圖表聯動 | 所有圖表同步更新 | 包含回調 | ⏳ 待開發 | P1 |
| **F2.7** | CSS 美化 | 自定義樣式、響應式布局 | 15 分鐘 | ⏳ 待開發 | P1 |

### 技術實作細節

#### F2.2: 時間範圍篩選
```python
# 布局添加滑桿
dcc.RangeSlider(
    id='time-range-slider',
    min=0,
    max=5000,  # 動態設置
    value=[0, 500],
    marks={i: str(i) for i in range(0, 5001, 500)}
)

# 回調處理
@app.callback(
    Output('main-chart', 'figure'),
    [Input('stock-selector', 'value'),
     Input('time-range-slider', 'value')]
)
def update_main_chart(stock_id, time_range):
    start, end = time_range
    close_sliced = close[start:end]
    labels_sliced = labels[start:end]
    # 生成圖表
    return create_main_chart(stock_id, close_sliced, labels_sliced, ...)
```

#### F2.5: 資訊面板
```python
html.Div([
    html.H4('統計摘要'),
    html.P(id='info-samples', children='樣本數: -'),
    html.P(id='info-label-dist', children='標籤分布: -'),
    html.P(id='info-weight-stats', children='權重統計: -'),
], style={'padding': '10px', 'background-color': '#f0f0f0'})

@app.callback(
    [Output('info-samples', 'children'),
     Output('info-label-dist', 'children'),
     Output('info-weight-stats', 'children')],
    Input('stock-selector', 'value')
)
def update_info_panel(stock_id):
    # 計算統計
    n_samples = len(close)
    label_dist = f"下跌 {pct_down:.1f}%, 持平 {pct_hold:.1f}%, 上漲 {pct_up:.1f}%"
    weight_stats = f"均值 {w_mean:.2f}, 標準差 {w_std:.2f}"

    return f"樣本數: {n_samples}", f"標籤分布: {label_dist}", f"權重統計: {weight_stats}"
```

### 檢查點（Milestone 2）

**定義**：進階功能完成標準

**檢查清單**：
- [ ] 可切換 train/val/test 數據集
- [ ] 切換數據集後股票列表自動更新
- [ ] 時間範圍滑桿可調整顯示區間
- [ ] 調整滑桿後圖表即時更新
- [ ] 標籤時間軸圖正確顯示
- [ ] 樣本權重分布圖正確顯示
- [ ] 資訊面板顯示正確統計
- [ ] 所有圖表（4 個）聯動更新
- [ ] UI 美觀、響應式布局

**驗收測試**：
1. 切換數據集（train → val → test），圖表正確更新
2. 調整時間範圍（0-500 → 200-800），圖表正確更新
3. 所有統計數字與圖表一致
4. 縮小瀏覽器窗口，布局自動調整

---

## ⭐ 階段三：高級功能（可選）

**目標**：增強功能，提供更深入的分析能力
**預計時間**：2-3 小時
**優先級**：P2（可選）
**當前狀態**：⏳ 待規劃

### 功能清單

| # | 功能 | 說明 | 預估時間 | 狀態 | 優先級 |
|---|------|------|----------|------|--------|
| **F3.1** | 多股票對比 | 同時顯示多檔股票（最多 3 檔） | 40 分鐘 | ⏳ 待開發 | P2 |
| **F3.2** | 標籤轉換點高亮 | 標記標籤切換位置與原因 | 30 分鐘 | ⏳ 待開發 | P1 ⭐ |
| **F3.3** | 配置版本對比 | 同時載入多個配置版本對比 | 50 分鐘 | ⏳ 待開發 | P2 |
| **F3.4** | 圖表下載 | 下載為 PNG/HTML | 20 分鐘 | ⏳ 待開發 | P1 ⭐ |
| **F3.5** | 搜尋與篩選 | 股票代碼搜尋、按條件篩選 | 40 分鐘 | ⏳ 待開發 | P2 |
| **F3.6** | 標籤修正工具 | 手動標記錯誤標籤（進階） | 1 小時 | ⏳ 待規劃 | P3 |

**推薦優先開發**：
- F3.2（標籤轉換點高亮）- 對除錯很有幫助 ⭐
- F3.4（圖表下載）- 方便分享報告 ⭐

### F3.2: 標籤轉換點高亮（推薦）

**功能描述**：
- 在主圖表上標記標籤切換的位置
- 顯示觸發原因（up/down/time）
- 顯示觸發時的價格變化

**實作方案**：
```python
# 檢測標籤轉換點
transitions = []
for i in range(1, len(labels)):
    if labels[i] != labels[i-1]:
        transitions.append({
            'index': i,
            'from': labels[i-1],
            'to': labels[i],
            'price_change': close[i] - close[i-1]
        })

# 在圖表上添加標記
for t in transitions:
    fig.add_trace(go.Scatter(
        x=[t['index']],
        y=[close[t['index']]],
        mode='markers',
        marker=dict(size=10, color='red', symbol='x'),
        name='標籤轉換',
        hovertemplate=f'<b>轉換</b>: {LABEL_NAMES[t["from"]]} → {LABEL_NAMES[t["to"]]}<br><b>價格變化</b>: {t["price_change"]:.2f}<extra></extra>'
    ))
```

**預期效果**：
- 標籤切換位置顯示紅色 ✕ 標記
- 懸停顯示：「下跌 → 上漲，價格變化 +2.50」

### F3.4: 圖表下載（推薦）

**功能描述**：
- 下載當前圖表為 PNG（高解析度）
- 下載當前圖表為 HTML（互動式）
- 批次下載所有股票圖表（進階）

**實作方案**：
```python
# 添加下載按鈕
html.Button('下載 PNG', id='download-png-btn'),
html.Button('下載 HTML', id='download-html-btn'),
dcc.Download(id='download-data')

# PNG 下載回調
@app.callback(
    Output('download-data', 'data'),
    Input('download-png-btn', 'n_clicks'),
    State('main-chart', 'figure')
)
def download_png(n_clicks, figure):
    if n_clicks:
        fig = go.Figure(figure)
        img_bytes = fig.to_image(format='png', width=1920, height=1080)
        return dcc.send_bytes(img_bytes, f'label_chart_{stock_id}.png')
```

**依賴套件**：
```bash
pip install kaleido  # 用於 PNG 匯出
```

---

## 📊 功能優先級矩陣

### 價值 vs 工作量矩陣

```
高價值 ↑
        │ F3.2 標籤轉換    │ F2.1 數據集切換
        │ F3.4 圖表下載    │ F2.2 時間篩選
        │                  │ F1.4 主圖表 ⭐
        │                  │ F1.1 數據載入 ⭐
────────┼──────────────────┼────────────────────→ 低工作量
        │ F3.3 版本對比    │ F2.7 CSS 美化
        │ F3.1 多股票對比  │ F2.5 資訊面板
低價值 ↓│                  │ 高工作量
```

**建議開發順序**：
1. **階段一（MVP）**：F1.1 → F1.4 → F1.5 → F1.6（必須）
2. **階段二（進階）**：F2.1 → F2.2 → F2.3 → F2.4 → F2.5（建議）
3. **階段三（高級）**：F3.2 → F3.4（推薦）→ F3.1/F3.3/F3.5（可選）

---

## 🎯 成功標準

### MVP 成功標準（Milestone 1）
- ✅ 應用可正常啟動
- ✅ 可選擇並查看至少 10 檔股票
- ✅ 主圖表顯示正確（收盤價 + 標籤背景）
- ✅ 標籤分布圖顯示正確
- ✅ 圖表可互動（縮放、懸停）

### 進階功能成功標準（Milestone 2）
- ✅ Milestone 1 所有標準
- ✅ 可切換 train/val/test 數據集
- ✅ 可調整時間範圍
- ✅ 所有 4 個圖表正確顯示
- ✅ 資訊面板顯示正確
- ✅ UI 美觀流暢

### 完整版成功標準（Milestone 3，可選）
- ✅ Milestone 2 所有標準
- ✅ 標籤轉換點高亮功能
- ✅ 圖表下載功能
- ✅ 可選：多股票對比、版本對比、搜尋篩選

---

## 🔮 未來展望（v3.0+）

### 長期規劃（待評估）

**v3.0: 進階分析功能**（預估 5-10 小時）
- 標籤正確性自動評估
- 異常標籤檢測（標籤與趨勢不一致）
- 標籤建議（基於規則的標籤修正建議）
- 批次報告生成（PDF/HTML）

**v4.0: 協作功能**（預估 10-20 小時）
- 用戶標註系統（標記錯誤標籤）
- 多用戶協作（共享標註）
- 標註統計與一致性分析
- 匯出標註結果

**v5.0: 整合主專案**（預估 20-40 小時）
- 整合到 DeepLOB 訓練流程
- 即時標籤品質監控
- 訓練數據品質儀表板
- 自動化標籤品質檢查

---

## 📝 備註

### 關於階段三
- 階段三功能為**可選**，根據實際需求決定是否實作
- 建議先完成 MVP 和進階功能，確認可用性後再決定
- F3.2（標籤轉換點）和 F3.4（圖表下載）實用性高，建議優先考慮

### 關於未來展望
- v3.0+ 為長期規劃，目前僅為構想
- 需根據 v2.0 的使用反饋調整
- 可能根據實際需求大幅調整

---

**維護者**：DeepLOB-Pro Team
**最後更新**：2025-10-20 15:15
**版本**：v1.0（路線圖）
