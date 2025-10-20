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
版本：v2.0 MVP
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
                placeholder='請選擇股票...',
                clearable=False,
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

@app.callback(
    Output('stock-dropdown', 'options'),
    Output('stock-dropdown', 'value'),
    Input('dataset-dropdown', 'value')
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
    try:
        # 載入股票列表（按樣本數降序）
        stock_list = get_stock_list(DEFAULT_DATA_DIR, selected_dataset, top_n=50)

        # 生成下拉選單選項
        options = [
            {'label': f'{stock_id} ({n_samples:,} 樣本)', 'value': stock_id}
            for stock_id, n_samples in stock_list
        ]

        # 預設選擇第一個（樣本數最多的）
        default_stock = stock_list[0][0] if stock_list else None

        return options, default_stock

    except Exception as e:
        print(f"錯誤：載入股票列表失敗 - {e}")
        return [], None


@app.callback(
    Output('main-chart', 'figure'),
    Output('label-dist-chart', 'figure'),
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
        return empty_fig, empty_fig, info_content

    try:
        # 載入股票數據
        stock_data = load_stock_data(DEFAULT_DATA_DIR, selected_stock, selected_dataset)

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

        # 生成資訊面板內容
        label_counts = np.bincount(labels.astype(int))
        label_dist_text = '<br>'.join([
            f"<b>下跌 (0)</b>: {label_counts[0]:,} ({label_counts[0]/n_samples*100:.1f}%)" if len(label_counts) > 0 else "",
            f"<b>持平 (1)</b>: {label_counts[1]:,} ({label_counts[1]/n_samples*100:.1f}%)" if len(label_counts) > 1 else "",
            f"<b>上漲 (2)</b>: {label_counts[2]:,} ({label_counts[2]/n_samples*100:.1f}%)" if len(label_counts) > 2 else ""
        ])

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
            html.Div([html.P(html.Span(label_dist_text, dangerouslySetInnerHTML=False), style={'margin': '5px 0'})],
                     style={'fontSize': '13px'}),
        ], style={'fontFamily': 'Microsoft YaHei, Arial', 'fontSize': '14px'})

        return main_fig, dist_fig, info_content

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

        return error_fig, error_fig, error_content


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
    app.run_server(
        debug=True,           # 開發模式
        host='0.0.0.0',       # 允許外部訪問
        port=DEFAULT_PORT,    # 預設端口 8050
        use_reloader=False    # 關閉自動重載（避免雙重啟動）
    )
