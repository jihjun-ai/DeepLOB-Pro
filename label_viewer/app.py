"""
Label Viewer - Interactive Label Visualization Tool

這是一個基於 Plotly + Dash 的互動式標籤視覺化工具。
用於檢查 DeepLOB-Pro 訓練數據標籤的正確性。

主要功能：
1. 載入 NPZ 格式的股票數據
2. 重建收盤價（從 Z-Score 反向轉換）
3. 視覺化標籤（用顏色背景標示）
4. 互動分析（縮放、懸停、切換股票）

作者：DeepLOB-Pro Team
版本：v2.0 進階版
最後更新：2025-10-20
"""

import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go
import numpy as np
from pathlib import Path
import sys

# 添加當前目錄到 Python 路徑
sys.path.append(str(Path(__file__).parent))

# 導入自定義模組
from utils.config import DEFAULT_DATA_DIR, DEFAULT_PORT
from utils.data_loader import get_stock_list, load_stock_data, get_cache_info
from utils.price_builder import reconstruct_close_price
from components.main_chart import create_main_chart, add_label_legend_annotation
from components.label_dist import create_label_distribution_pie
from components.label_timeline import create_label_timeline_with_legend
from components.weight_dist import create_weight_histogram


# ============================================================================
# 初始化 Dash 應用
# ============================================================================

app = dash.Dash(
    __name__,
    title='Label Viewer - DeepLOB-Pro',
    update_title='載入中...',
    suppress_callback_exceptions=True
)


# ============================================================================
# 應用布局（MVP 版本）
# ============================================================================

app.layout = html.Div([
    # 標題區域
    html.Div([
        html.H1(
            'Interactive Label Viewer',
            style={
                'textAlign': 'center',
                'color': '#2c3e50',
                'marginBottom': '10px',
                'fontFamily': 'Microsoft YaHei, Arial'
            }
        ),
        html.P(
            'DeepLOB-Pro 訓練數據標籤視覺化工具 v2.0',
            style={
                'textAlign': 'center',
                'color': '#7f8c8d',
                'fontSize': '14px',
                'marginTop': '0',
                'fontFamily': 'Microsoft YaHei, Arial'
            }
        ),
    ], style={'padding': '20px', 'backgroundColor': '#ecf0f1'}),

    # 主控制區域
    html.Div([
        # 左側控制面板
        html.Div([
            html.H3(
                '控制面板',
                style={
                    'color': '#2c3e50',
                    'marginTop': '0',
                    'fontFamily': 'Microsoft YaHei, Arial'
                }
            ),

            # 資料目錄選擇
            html.Label(
                '資料目錄 (NPZ 檔案目錄):',
                style={'fontWeight': 'bold', 'marginTop': '10px', 'fontFamily': 'Microsoft YaHei, Arial'}
            ),
            html.Div([
                dcc.Input(
                    id='data-dir-input',
                    type='text',
                    placeholder='請輸入或貼上目錄路徑...',
                    value='D:/Case-New/python/DeepLOB-Pro/data/processed_v5/npz',
                    style={
                        'width': '100%',
                        'padding': '8px',
                        'borderRadius': '4px',
                        'border': '1px solid #bdc3c7',
                        'fontFamily': 'Microsoft YaHei, Arial',
                        'fontSize': '13px'
                    }
                ),
            ], style={'marginBottom': '10px'}),

            # 快速選擇預設目錄
            html.Div([
                html.Label('快速選擇:', style={'fontSize': '12px', 'color': '#7f8c8d', 'marginBottom': '5px'}),
                html.Div([
                    html.Button('v5 原始', id='btn-v5', n_clicks=0,
                               style={'fontSize': '11px', 'padding': '4px 8px', 'marginRight': '5px', 'cursor': 'pointer'}),
                    html.Button('v5 平衡', id='btn-v5-bal', n_clicks=0,
                               style={'fontSize': '11px', 'padding': '4px 8px', 'marginRight': '5px', 'cursor': 'pointer'}),
                    html.Button('v5-44.61', id='btn-v5-4461', n_clicks=0,
                               style={'fontSize': '11px', 'padding': '4px 8px', 'marginRight': '5px', 'cursor': 'pointer'}),
                ], style={'display': 'flex', 'gap': '5px', 'marginBottom': '5px'}),
                html.Div([
                    html.Button('v5 修復版', id='btn-v5-fixed', n_clicks=0,
                               style={'fontSize': '11px', 'padding': '4px 8px', 'marginRight': '5px', 'cursor': 'pointer', 'backgroundColor': '#27ae60', 'color': 'white', 'border': 'none', 'borderRadius': '3px'}),
                    html.Button('v5 測試', id='btn-v5-test', n_clicks=0,
                               style={'fontSize': '11px', 'padding': '4px 8px', 'cursor': 'pointer', 'backgroundColor': '#f39c12', 'color': 'white', 'border': 'none', 'borderRadius': '3px'}),
                ], style={'display': 'flex', 'gap': '5px'}),
            ], style={'marginBottom': '15px'}),

            # 開始讀取資料按鈕
            html.Button(
                '開始讀取資料',
                id='load-data-button',
                n_clicks=0,
                style={
                    'width': '100%',
                    'padding': '10px',
                    'backgroundColor': '#3498db',
                    'color': 'white',
                    'border': 'none',
                    'borderRadius': '5px',
                    'cursor': 'pointer',
                    'fontSize': '14px',
                    'fontWeight': 'bold',
                    'fontFamily': 'Microsoft YaHei, Arial',
                    'marginBottom': '20px'
                }
            ),

            # 載入狀態提示
            html.Div(id='load-status', style={'marginBottom': '20px', 'fontSize': '12px'}),

            html.Hr(style={'margin': '20px 0'}),

            # 數據集選擇
            html.Label(
                '數據集:',
                style={'fontWeight': 'bold', 'fontFamily': 'Microsoft YaHei, Arial'}
            ),
            dcc.Dropdown(
                id='dataset-dropdown',
                options=[
                    {'label': 'Train（訓練集）', 'value': 'train'},
                    {'label': 'Validation（驗證集）', 'value': 'val'},
                    {'label': 'Test（測試集）', 'value': 'test'}
                ],
                value='train',  # 預設選擇訓練集
                clearable=False,
                disabled=True,  # 初始禁用，載入數據後啟用
                style={'marginBottom': '20px', 'fontFamily': 'Microsoft YaHei, Arial'}
            ),

            # 股票選擇
            html.Label(
                '股票代碼:',
                style={'fontWeight': 'bold', 'fontFamily': 'Microsoft YaHei, Arial'}
            ),
            dcc.Dropdown(
                id='stock-dropdown',
                options=[],  # 動態載入
                value=None,
                placeholder='請先載入資料...',
                clearable=False,
                disabled=True,  # 初始禁用，載入數據後啟用
                style={'marginBottom': '20px', 'fontFamily': 'Microsoft YaHei, Arial'}
            ),

            # 資訊面板
            html.Div(id='info-panel', style={'marginTop': '30px'}),

            # 快取資訊（隱藏開發資訊）
            html.Div(id='cache-info', style={
                'marginTop': '30px',
                'padding': '10px',
                'backgroundColor': '#f8f9fa',
                'borderRadius': '5px',
                'fontSize': '12px',
                'fontFamily': 'Microsoft YaHei, Arial',
                'display': 'none'  # 預設隱藏
            }),

        ], style={
            'width': '20%',
            'display': 'inline-block',
            'verticalAlign': 'top',
            'padding': '20px',
            'backgroundColor': '#ffffff',
            'borderRight': '2px solid #ecf0f1',
            'minHeight': '600px'
        }),

        # 右側圖表區域
        html.Div([
            # 主圖表
            html.Div([
                dcc.Graph(
                    id='main-chart',
                    config={'displayModeBar': True, 'displaylogo': False},
                    style={'height': '500px'}
                )
            ]),

            # 標籤分布圖
            html.Div([
                dcc.Graph(
                    id='label-dist-chart',
                    config={'displayModeBar': False, 'displaylogo': False},
                    style={'height': '300px'}
                )
            ], style={'marginTop': '20px'}),

            # 標籤時間軸圖
            html.Div([
                dcc.Graph(
                    id='label-timeline-chart',
                    config={'displayModeBar': False, 'displaylogo': False},
                    style={'height': '150px'}
                )
            ], style={'marginTop': '20px'}),

            # 樣本權重分布圖
            html.Div([
                dcc.Graph(
                    id='weight-dist-chart',
                    config={'displayModeBar': False, 'displaylogo': False},
                    style={'height': '300px'}
                )
            ], style={'marginTop': '20px'}),

        ], style={
            'width': '78%',
            'display': 'inline-block',
            'verticalAlign': 'top',
            'padding': '20px',
            'backgroundColor': '#ffffff'
        }),

    ], style={
        'maxWidth': '1400px',
        'margin': '0 auto',
        'backgroundColor': '#ffffff',
        'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'
    }),

    # 頁尾
    html.Div([
        html.P(
            'DeepLOB-Pro Team © 2025 | Interactive Label Viewer v2.0',
            style={
                'textAlign': 'center',
                'color': '#95a5a6',
                'fontSize': '12px',
                'margin': '20px 0',
                'fontFamily': 'Microsoft YaHei, Arial'
            }
        )
    ]),

], style={
    'fontFamily': 'Microsoft YaHei, Arial, sans-serif',
    'backgroundColor': '#f5f6fa',
    'minHeight': '100vh'
})


# ============================================================================
# 回調函數
# ============================================================================

# 全局變數：儲存當前使用的資料目錄
current_data_dir = DEFAULT_DATA_DIR

# 快速選擇按鈕回調
@app.callback(
    Output('data-dir-input', 'value'),
    Input('btn-v5', 'n_clicks'),
    Input('btn-v5-bal', 'n_clicks'),
    Input('btn-v5-4461', 'n_clicks'),
    Input('btn-v5-fixed', 'n_clicks'),
    Input('btn-v5-test', 'n_clicks'),
    prevent_initial_call=True
)
def quick_select_directory(n_v5, n_v5_bal, n_v5_4461, n_v5_fixed, n_v5_test):
    """快速選擇預設目錄"""
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update

    button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    base_path = 'D:/Case-New/python/DeepLOB-Pro/data'

    if button_id == 'btn-v5':
        return f'{base_path}/processed_v5/npz'
    elif button_id == 'btn-v5-bal':
        return f'{base_path}/processed_v5_balanced/npz'
    elif button_id == 'btn-v5-4461':
        return f'{base_path}/processed_v5-44.61/npz'
    elif button_id == 'btn-v5-fixed':
        return f'{base_path}/processed_v5_fixed/npz'
    elif button_id == 'btn-v5-test':
        return f'{base_path}/test_fix/npz'

    return dash.no_update


@app.callback(
    Output('load-status', 'children'),
    Output('dataset-dropdown', 'disabled'),
    Output('stock-dropdown', 'disabled'),
    Output('stock-dropdown', 'options'),
    Output('stock-dropdown', 'value'),
    Input('load-data-button', 'n_clicks'),
    State('data-dir-input', 'value'),
    State('dataset-dropdown', 'value'),
    prevent_initial_call=True
)
def load_data_on_button_click(n_clicks, data_dir_path, selected_dataset):
    """
    當點擊「開始讀取資料」按鈕時，載入數據並啟用控制項

    Args:
        n_clicks: 按鈕點擊次數
        data_dir_path: 用戶輸入的資料目錄路徑
        selected_dataset: 選擇的數據集

    Returns:
        status_msg: 載入狀態訊息
        dataset_disabled: 數據集下拉選單是否禁用
        stock_disabled: 股票下拉選單是否禁用
        stock_options: 股票選項列表
        default_stock: 預設選擇的股票
    """
    global current_data_dir

    if n_clicks == 0:
        return '', True, True, [], None

    try:
        # 檢查目錄路徑是否有效
        from pathlib import Path
        data_path = Path(data_dir_path)

        if not data_path.exists():
            error_msg = html.Div([
                html.Span('❌ ', style={'color': '#e74c3c'}),
                html.Span(f'目錄不存在: {data_dir_path}',
                         style={'color': '#e74c3c', 'fontSize': '11px'})
            ])
            return error_msg, True, True, [], None

        if not data_path.is_dir():
            error_msg = html.Div([
                html.Span('❌ ', style={'color': '#e74c3c'}),
                html.Span(f'路徑不是目錄: {data_dir_path}',
                         style={'color': '#e74c3c', 'fontSize': '11px'})
            ])
            return error_msg, True, True, [], None

        # 更新當前資料目錄
        current_data_dir = str(data_path)

        # 載入股票列表（按股票代碼排序）
        stock_list = get_stock_list(current_data_dir, selected_dataset, top_n=None, sort_by='code')

        # 生成下拉選單選項
        options = [
            {'label': f'{stock_id} ({n_samples:,} 樣本)', 'value': stock_id}
            for stock_id, n_samples in stock_list
        ]

        # 預設選擇第一個
        default_stock = stock_list[0][0] if stock_list else None

        # 成功訊息
        status_msg = html.Div([
            html.Div([
                html.Span('✅ ', style={'color': '#27ae60'}),
                html.Span(f'成功載入 {len(stock_list)} 檔股票',
                         style={'color': '#27ae60', 'fontWeight': 'bold'})
            ]),
            html.Div([
                html.Span(f'目錄: {data_path.name}',
                         style={'color': '#7f8c8d', 'fontSize': '11px'})
            ], style={'marginTop': '5px'})
        ])

        return status_msg, False, False, options, default_stock

    except Exception as e:
        # 錯誤訊息
        error_msg = html.Div([
            html.Span('❌ ', style={'color': '#e74c3c'}),
            html.Span(f'載入失敗: {str(e)}',
                     style={'color': '#e74c3c', 'fontWeight': 'bold'})
        ])
        return error_msg, True, True, [], None


@app.callback(
    Output('stock-dropdown', 'options', allow_duplicate=True),
    Output('stock-dropdown', 'value', allow_duplicate=True),
    Input('dataset-dropdown', 'value'),
    prevent_initial_call=True
)
def update_stock_list(selected_dataset):
    """
    當數據集切換時，更新股票列表

    Args:
        selected_dataset: 選擇的數據集 ('train', 'val', 'test')

    Returns:
        options: 股票下拉選單選項
        value: 預設選擇的股票（樣本數最多的股票）
    """
    global current_data_dir
    try:
        # 載入股票列表（按股票代碼排序）
        stock_list = get_stock_list(current_data_dir, selected_dataset, top_n=None, sort_by='code')

        # 生成下拉選單選項
        options = [
            {'label': f'{stock_id} ({n_samples:,} 樣本)', 'value': stock_id}
            for stock_id, n_samples in stock_list
        ]

        # 預設選擇第一個（按股票代碼排序後的第一個）
        default_stock = stock_list[0][0] if stock_list else None

        return options, default_stock

    except Exception as e:
        print(f"錯誤：載入股票列表失敗 - {e}")
        return [], None


@app.callback(
    Output('main-chart', 'figure'),
    Output('label-dist-chart', 'figure'),
    Output('label-timeline-chart', 'figure'),
    Output('weight-dist-chart', 'figure'),
    Output('info-panel', 'children'),
    Input('stock-dropdown', 'value'),
    State('dataset-dropdown', 'value')
)
def update_charts(selected_stock, selected_dataset):
    """
    當股票切換時，更新所有圖表

    Args:
        selected_stock: 選擇的股票 ID
        selected_dataset: 選擇的數據集

    Returns:
        main_fig: 主圖表 Figure
        dist_fig: 標籤分布圖 Figure
        timeline_fig: 標籤時間軸圖 Figure
        weight_fig: 權重分布圖 Figure
        info_panel: 資訊面板內容
    """
    # 如果未選擇股票，返回空圖表
    if selected_stock is None:
        empty_fig = go.Figure()
        empty_fig.update_layout(
            title='請選擇股票',
            template='plotly_white',
            font=dict(family='Microsoft YaHei, Arial')
        )
        info_content = html.Div([
            html.P('請先選擇數據集和股票', style={'color': '#7f8c8d'})
        ])
        return empty_fig, empty_fig, empty_fig, empty_fig, info_content

    global current_data_dir
    try:
        # 載入股票數據
        stock_data = load_stock_data(current_data_dir, selected_stock, selected_dataset)

        # 提取數據
        features = stock_data['features']  # (N, 100, 20)
        labels = stock_data['labels']      # (N,)
        weights = stock_data['weights']    # (N,)
        metadata = stock_data['metadata']
        n_samples = stock_data['n_samples']

        # 重建收盤價
        close_prices = reconstruct_close_price(features, metadata)

        # 創建主圖表
        main_fig = create_main_chart(
            stock_id=selected_stock,
            close_prices=close_prices,
            labels=labels,
            chart_options={'height': 500}
        )

        # 添加標籤圖例
        main_fig = add_label_legend_annotation(main_fig)

        # 創建標籤分布圖
        dist_fig = create_label_distribution_pie(
            labels=labels,
            chart_options={'height': 300}
        )

        # 創建標籤時間軸圖
        timeline_fig = create_label_timeline_with_legend(
            labels=labels,
            chart_options={'height': 150}
        )

        # 創建權重分布圖
        weight_fig = create_weight_histogram(
            weights=weights,
            chart_options={'height': 300, 'nbins': 30}
        )

        # 生成資訊面板內容
        label_counts = np.bincount(labels.astype(int))

        # 建立標籤分布列表
        label_dist_items = []
        if len(label_counts) > 0:
            label_dist_items.append(
                html.P([
                    html.B('下跌 (0): '),
                    f"{label_counts[0]:,} ({label_counts[0]/n_samples*100:.1f}%)"
                ], style={'margin': '3px 0'})
            )
        if len(label_counts) > 1:
            label_dist_items.append(
                html.P([
                    html.B('持平 (1): '),
                    f"{label_counts[1]:,} ({label_counts[1]/n_samples*100:.1f}%)"
                ], style={'margin': '3px 0'})
            )
        if len(label_counts) > 2:
            label_dist_items.append(
                html.P([
                    html.B('上漲 (2): '),
                    f"{label_counts[2]:,} ({label_counts[2]/n_samples*100:.1f}%)"
                ], style={'margin': '3px 0'})
            )

        # 提取 V5 配置資訊
        v5_version = metadata.get('version', 'N/A')
        v5_volatility = metadata.get('volatility', {})
        v5_tb = metadata.get('triple_barrier', {})
        v5_respect_day = metadata.get('respect_day_boundary', None)

        # 構建 V5 配置顯示
        v5_info_items = []

        # 版本資訊
        if v5_version != 'N/A':
            v5_info_items.append(
                html.P([html.B('版本: '), v5_version],
                      style={'margin': '3px 0', 'fontSize': '12px'})
            )

        # 波動率方法
        if v5_volatility:
            vol_method = v5_volatility.get('method', 'N/A')
            vol_halflife = v5_volatility.get('halflife', '')
            vol_text = f"{vol_method}"
            if vol_halflife and vol_method == 'ewma':
                vol_text += f" (halflife={vol_halflife})"
            v5_info_items.append(
                html.P([html.B('波動率: '), vol_text],
                      style={'margin': '3px 0', 'fontSize': '12px'})
            )

        # Triple-Barrier 參數
        if v5_tb:
            pt = v5_tb.get('pt_multiplier', 'N/A')
            sl = v5_tb.get('sl_multiplier', 'N/A')
            max_hold = v5_tb.get('max_holding', 'N/A')
            min_ret = v5_tb.get('min_return', 'N/A')

            v5_info_items.append(
                html.P([html.B('Triple-Barrier: '), f"PT={pt}σ, SL={sl}σ"],
                      style={'margin': '3px 0', 'fontSize': '12px'})
            )
            v5_info_items.append(
                html.P([html.B('  '), f"MaxHold={max_hold} bars, MinRet={min_ret}"],
                      style={'margin': '3px 0', 'fontSize': '11px', 'color': '#7f8c8d'})
            )

        # 跨日保護狀態
        if v5_respect_day is not None:
            if v5_respect_day:
                day_boundary_text = html.Span('✅ 啟用', style={'color': '#27ae60', 'fontWeight': 'bold'})
            else:
                day_boundary_text = html.Span('❌ 禁用', style={'color': '#e74c3c', 'fontWeight': 'bold'})

            v5_info_items.append(
                html.P([html.B('跨日保護: '), day_boundary_text],
                      style={'margin': '3px 0', 'fontSize': '12px'})
            )

        info_content = html.Div([
            html.H4(f'股票 {selected_stock}', style={'color': '#2c3e50', 'marginTop': '0'}),
            html.P([
                html.B('數據集: '),
                {'train': '訓練集', 'val': '驗證集', 'test': '測試集'}.get(selected_dataset, selected_dataset)
            ], style={'margin': '5px 0'}),
            html.P([html.B('樣本數: '), f'{n_samples:,}'], style={'margin': '5px 0'}),
            html.P([html.B('收盤價範圍: '), f'[{close_prices.min():.2f}, {close_prices.max():.2f}]'], style={'margin': '5px 0'}),
            html.Hr(),
            html.H5('標籤分布:', style={'color': '#34495e', 'marginBottom': '10px'}),
            html.Div(label_dist_items, style={'fontSize': '13px'}),

            # V5 配置資訊（如果有）
            html.Hr() if v5_info_items else html.Div(),
            html.H5('V5 配置:', style={'color': '#34495e', 'marginBottom': '5px'}) if v5_info_items else html.Div(),
            html.Div(v5_info_items, style={'fontSize': '13px'}) if v5_info_items else html.Div(),
        ], style={'fontFamily': 'Microsoft YaHei, Arial', 'fontSize': '14px'})

        return main_fig, dist_fig, timeline_fig, weight_fig, info_content

    except Exception as e:
        print(f"錯誤：載入股票數據失敗 - {e}")
        import traceback
        traceback.print_exc()

        # 返回錯誤訊息
        error_fig = go.Figure()
        error_fig.update_layout(
            title=f'載入失敗：{str(e)}',
            template='plotly_white',
            font=dict(family='Microsoft YaHei, Arial')
        )

        error_content = html.Div([
            html.P(f'載入股票 {selected_stock} 時發生錯誤', style={'color': '#e74c3c'}),
            html.P(f'錯誤訊息：{str(e)}', style={'color': '#7f8c8d', 'fontSize': '12px'})
        ])

        return error_fig, error_fig, error_fig, error_fig, error_content


# ============================================================================
# 啟動應用
# ============================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("Label Viewer - Interactive Label Visualization Tool")
    print("=" * 70)
    print(f"版本: v2.0 MVP")
    print(f"數據目錄: {DEFAULT_DATA_DIR}")
    print(f"啟動服務器: http://localhost:{DEFAULT_PORT}")
    print("=" * 70)
    print("\n按 Ctrl+C 停止服務器\n")

    # 啟動應用
    app.run(
        debug=True,           # 開發模式
        host='0.0.0.0',       # 允許外部訪問
        port=DEFAULT_PORT,    # 預設端口 8050
        use_reloader=False    # 關閉自動重載（避免雙重啟動）
    )
