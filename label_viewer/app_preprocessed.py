"""
Label Viewer - 預處理數據模式（完整版）

用於查看 preprocess_single_day.py 產生的預處理數據和標籤預覽。

主要功能：
1. 輸入日期目錄路徑（例如：data/preprocessed_v5_1hz/daily/20250901）
2. 查看該目錄下所有股票列表
3. 選擇單一股票或「全部股票」查看整體統計
4. 視覺化所有 NPZ 數據（可通過開關選擇）：
   - 中間價折線圖 (mids)
   - LOB 特徵矩陣熱圖 (features)
   - 標籤陣列視覺化 (labels)
   - 事件數量圖 (bucket_event_count)
   - 時間桶遮罩圖 (bucket_mask)
   - 標籤預覽分布
   - 元數據表格

支援 PREPROCESSED_DATA_SPECIFICATION.md 所有數據欄位。

作者：DeepLOB-Pro Team
版本：v4.0 完整版
最後更新：2025-10-23
"""

import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go
import numpy as np
from pathlib import Path
from typing import Optional
import sys
import json

# 添加當前目錄到 Python 路徑
sys.path.append(str(Path(__file__).parent))

# 導入自定義模組
from utils.preprocessed_loader import (
    load_preprocessed_stock,
    get_cache_info
)
from components.label_preview_panel import (
    create_label_preview_comparison,
    create_label_preview_bar,
    create_metadata_table,
    create_overall_label_stats_bar
)


# ============================================================================
# 初始化 Dash 應用
# ============================================================================

app = dash.Dash(
    __name__,
    title='Label Viewer - 預處理數據',
    update_title='載入中...',
    suppress_callback_exceptions=True
)


# ============================================================================
# 輔助函數
# ============================================================================

def scan_daily_directory(daily_dir: str):
    """掃描日期目錄，返回所有股票 NPZ 文件"""
    daily_path = Path(daily_dir)

    if not daily_path.exists():
        return None, f"目錄不存在: {daily_dir}"

    if not daily_path.is_dir():
        return None, f"路徑不是目錄: {daily_dir}"

    # 查找所有 .npz 文件（排除 summary.json）
    npz_files = list(daily_path.glob("*.npz"))

    if not npz_files:
        return None, f"目錄中沒有找到 NPZ 文件: {daily_dir}"

    # 提取股票代碼（移除 .npz 後綴）
    stocks = {}
    for npz_file in npz_files:
        symbol = npz_file.stem  # 檔名不含副檔名
        stocks[symbol] = str(npz_file)

    # 載入 summary.json（如果存在）
    summary_path = daily_path / "summary.json"
    summary = None
    if summary_path.exists():
        try:
            with open(summary_path, 'r', encoding='utf-8') as f:
                summary = json.load(f)
        except Exception as e:
            print(f"載入 summary.json 失敗: {e}")

    return {
        'stocks': stocks,
        'summary': summary,
        'date': daily_path.name,
        'total_stocks': len(stocks)
    }, None


def compute_labels_from_mids(mids: np.ndarray, metadata: dict) -> Optional[np.ndarray]:
    """
    從中間價計算 Triple-Barrier 標籤

    Args:
        mids: 中間價序列
        metadata: 元數據（可能包含 triple_barrier_config）

    Returns:
        標籤陣列 (-1, 0, 1) 或 None（如果計算失敗）
    """
    try:
        # 從 metadata 獲取 triple_barrier 配置（如果有）
        tb_config = metadata.get('triple_barrier_config')

        if tb_config:
            # 使用 metadata 中的配置
            threshold = tb_config.get('threshold', 0.002)
            horizon = tb_config.get('horizon', 10)
            print(f"[INFO] 使用 metadata 中的 TB 配置: threshold={threshold}, horizon={horizon}")
        else:
            # 使用默認配置（與 config_pro_v5_ml_optimal.yaml 一致）
            threshold = 0.002
            horizon = 10
            print(f"[INFO] 使用默認 TB 配置: threshold={threshold}, horizon={horizon}")

        T = len(mids)
        labels = np.zeros(T, dtype=np.int8)

        # 對每個時間點計算標籤
        for t in range(T):
            # 檢查是否有足夠的未來數據
            if t + horizon >= T:
                labels[t] = 0  # 邊界設為 Neutral
                continue

            # 當前價格
            current_price = mids[t]
            if current_price == 0:
                labels[t] = 0
                continue

            # 未來價格序列
            future_prices = mids[t+1:t+1+horizon]

            # 計算收益率
            returns = (future_prices - current_price) / current_price

            # Triple-Barrier 邏輯
            up_barrier = threshold
            down_barrier = -threshold

            # 檢查是否觸碰邊界
            hit_up = np.where(returns >= up_barrier)[0]
            hit_down = np.where(returns <= down_barrier)[0]

            if len(hit_up) == 0 and len(hit_down) == 0:
                # 沒有觸碰邊界
                labels[t] = 0  # Neutral
            elif len(hit_up) > 0 and len(hit_down) == 0:
                # 只觸碰上邊界
                labels[t] = 1  # Up
            elif len(hit_down) > 0 and len(hit_up) == 0:
                # 只觸碰下邊界
                labels[t] = -1  # Down
            else:
                # 同時觸碰兩個邊界，取先觸碰的
                first_up = hit_up[0]
                first_down = hit_down[0]

                if first_down < first_up:
                    labels[t] = -1  # Down 先觸碰
                else:
                    labels[t] = 1   # Up 先觸碰

        return labels

    except Exception as e:
        print(f"[WARNING] 計算標籤失敗: {e}")
        return None


def get_overall_label_stats(daily_dir: str):
    """統計目錄下所有股票的標籤分布"""
    daily_path = Path(daily_dir)
    npz_files = list(daily_path.glob("*.npz"))

    overall_counts = {-1: 0, 0: 0, 1: 0}
    stocks_with_labels = 0
    stock_details = []

    for npz_file in npz_files:
        try:
            data = load_preprocessed_stock(str(npz_file))
            label_preview = data.get('metadata', {}).get('label_preview')

            if label_preview and 'label_counts' in label_preview:
                stocks_with_labels += 1
                counts = label_preview['label_counts']

                # 累加計數
                for label, count in counts.items():
                    label_key = int(label)
                    overall_counts[label_key] += count

                # 記錄個股資訊
                stock_details.append({
                    'symbol': npz_file.stem,
                    'label_preview': label_preview
                })
        except Exception as e:
            print(f"載入 {npz_file.name} 失敗: {e}")

    # 計算整體分布
    total = sum(overall_counts.values())
    overall_dist = {}
    if total > 0:
        overall_dist = {
            str(label): count / total
            for label, count in overall_counts.items()
        }

    return {
        'total_stocks': len(npz_files),
        'stocks_with_labels': stocks_with_labels,
        'overall_counts': overall_counts,
        'overall_dist': overall_dist,
        'stock_details': stock_details
    }


# ============================================================================
# 應用布局
# ============================================================================

app.layout = html.Div([
    # 標題區域
    html.Div([
        html.H1(
            'Label Viewer - 預處理數據模式',
            style={
                'textAlign': 'center',
                'color': '#2c3e50',
                'marginBottom': '10px',
                'fontFamily': 'Microsoft YaHei, Arial'
            }
        ),
        html.P(
            '查看 preprocess_single_day.py 產生的數據和標籤預覽 v4.0 完整版（支援所有 NPZ 欄位）',
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

            # 日期目錄路徑輸入
            html.Label(
                '日期目錄路徑:',
                style={'fontWeight': 'bold', 'marginTop': '10px', 'fontFamily': 'Microsoft YaHei, Arial'}
            ),
            html.Div([
                dcc.Input(
                    id='daily-dir-input',
                    type='text',
                    placeholder='例如: data/preprocessed_v5_1hz/daily/20250901',
                    value='data/preprocessed_v5_1hz/daily/20250901',
                    style={
                        'width': '100%',
                        'padding': '8px',
                        'border': '1px solid #bdc3c7',
                        'borderRadius': '4px',
                        'fontFamily': 'Consolas, monospace',
                        'fontSize': '12px'
                    }
                ),
                html.Button(
                    '載入目錄',
                    id='load-dir-button',
                    n_clicks=0,
                    style={
                        'marginTop': '5px',
                        'width': '100%',
                        'padding': '8px',
                        'backgroundColor': '#3498db',
                        'color': 'white',
                        'border': 'none',
                        'borderRadius': '4px',
                        'cursor': 'pointer',
                        'fontFamily': 'Microsoft YaHei, Arial'
                    }
                ),
            ], style={'marginBottom': '15px'}),

            # 狀態訊息
            html.Div(
                id='status-message',
                style={
                    'padding': '10px',
                    'marginBottom': '15px',
                    'borderRadius': '4px',
                    'fontFamily': 'Microsoft YaHei, Arial',
                    'fontSize': '12px'
                }
            ),

            # 股票選擇
            html.Label(
                '選擇股票:',
                style={'fontWeight': 'bold', 'marginTop': '10px', 'fontFamily': 'Microsoft YaHei, Arial'}
            ),
            dcc.Dropdown(
                id='stock-dropdown',
                options=[],
                placeholder='請先載入目錄...',
                style={'marginBottom': '15px', 'fontFamily': 'Microsoft YaHei, Arial'}
            ),

            # 顯示選項
            html.Label(
                '顯示選項:',
                style={'fontWeight': 'bold', 'marginTop': '10px', 'fontFamily': 'Microsoft YaHei, Arial'}
            ),
            dcc.Checklist(
                id='display-options',
                options=[
                    {'label': ' 中間價折線圖 (mids)', 'value': 'mids'},
                    {'label': ' LOB 特徵矩陣 (features)', 'value': 'features'},
                    {'label': ' 標籤陣列圖 (labels)', 'value': 'labels'},
                    {'label': ' 事件數量圖 (bucket_event_count)', 'value': 'bucket_event_count'},
                    {'label': ' 時間桶遮罩圖 (bucket_mask)', 'value': 'bucket_mask'},
                    {'label': ' 標籤預覽分布', 'value': 'label_preview'},
                    {'label': ' 元數據表格', 'value': 'metadata'}
                ],
                value=['mids', 'label_preview', 'metadata'],
                style={'fontFamily': 'Microsoft YaHei, Arial', 'fontSize': '12px'}
            ),

            # 快取資訊
            html.Hr(style={'margin': '20px 0'}),
            html.Div(
                id='cache-info',
                style={
                    'fontSize': '11px',
                    'color': '#7f8c8d',
                    'fontFamily': 'Consolas, monospace'
                }
            ),

        ], style={
            'width': '25%',
            'padding': '20px',
            'backgroundColor': '#f8f9fa',
            'borderRight': '1px solid #dee2e6',
            'minHeight': '80vh'
        }),

        # 右側圖表區域
        html.Div([
            html.Div(id='charts-container')
        ], style={
            'width': '75%',
            'padding': '20px'
        })

    ], style={'display': 'flex'}),

    # 隱藏的 Store 用於儲存目錄資訊
    dcc.Store(id='dir-info-store'),

], style={'fontFamily': 'Arial, sans-serif'})


# ============================================================================
# 回調函數
# ============================================================================

@app.callback(
    [Output('dir-info-store', 'data'),
     Output('status-message', 'children'),
     Output('status-message', 'style'),
     Output('stock-dropdown', 'options'),
     Output('stock-dropdown', 'value')],
    Input('load-dir-button', 'n_clicks'),
    State('daily-dir-input', 'value'),
    prevent_initial_call=False
)
def load_directory(n_clicks, daily_dir):
    """載入日期目錄"""
    if not daily_dir:
        return None, "請輸入日期目錄路徑", {
            'padding': '10px',
            'backgroundColor': '#fff3cd',
            'color': '#856404',
            'border': '1px solid #ffeeba',
            'borderRadius': '4px',
            'fontFamily': 'Microsoft YaHei, Arial',
            'fontSize': '12px'
        }, [], None

    # 掃描目錄
    dir_info, error = scan_daily_directory(daily_dir)

    if error:
        return None, f"❌ {error}", {
            'padding': '10px',
            'backgroundColor': '#f8d7da',
            'color': '#721c24',
            'border': '1px solid #f5c6cb',
            'borderRadius': '4px',
            'fontFamily': 'Microsoft YaHei, Arial',
            'fontSize': '12px'
        }, [], None

    # 準備股票選項
    stock_options = []

    # 第一個選項：全部股票
    stock_options.append({
        'label': f'📊 全部股票（整體統計）- {dir_info["total_stocks"]} 檔',
        'value': '__ALL_STOCKS__'
    })

    # 個別股票選項
    for symbol in sorted(dir_info['stocks'].keys()):
        stock_options.append({
            'label': symbol,
            'value': symbol
        })

    # 成功訊息
    success_msg = f"✅ 成功載入 {dir_info['date']} - 共 {dir_info['total_stocks']} 檔股票"

    return dir_info, success_msg, {
        'padding': '10px',
        'backgroundColor': '#d4edda',
        'color': '#155724',
        'border': '1px solid #c3e6cb',
        'borderRadius': '4px',
        'fontFamily': 'Microsoft YaHei, Arial',
        'fontSize': '12px'
    }, stock_options, '__ALL_STOCKS__'


@app.callback(
    Output('charts-container', 'children'),
    [Input('stock-dropdown', 'value'),
     Input('display-options', 'value')],
    State('dir-info-store', 'data'),
    prevent_initial_call=True
)
def update_charts(symbol, display_options, dir_info):
    """更新圖表"""
    if not dir_info or not symbol:
        return html.Div(
            "請先載入目錄並選擇股票",
            style={
                'textAlign': 'center',
                'color': '#7f8c8d',
                'padding': '50px',
                'fontFamily': 'Microsoft YaHei, Arial'
            }
        )

    # 檢查是否選擇「全部股票」
    if symbol == '__ALL_STOCKS__':
        return update_all_stocks_view(dir_info, display_options)

    # 單一股票檢視
    return update_single_stock_view(symbol, dir_info, display_options)


def update_single_stock_view(symbol, dir_info, display_options):
    """單一股票檢視"""
    npz_path = dir_info['stocks'].get(symbol)

    if not npz_path:
        return html.Div(
            f"找不到股票: {symbol}",
            style={'color': 'red', 'fontFamily': 'Microsoft YaHei, Arial'}
        )

    try:
        # 載入數據
        data = load_preprocessed_stock(npz_path)

        charts = []

        # 獲取基礎數據
        features = data['features']
        mids = data['mids']
        metadata = data.get('metadata', {})
        labels = data.get('labels')  # 可能為 None
        bucket_event_count = data.get('bucket_event_count')
        bucket_mask = data.get('bucket_mask')

        # 1. 中間價折線圖（帶標籤疊加）
        if 'mids' in display_options:
            # 優先使用已保存的標籤，否則實時計算
            labels_for_plot = labels
            if labels_for_plot is None:
                # 如果 NPZ 中沒有標籤，實時計算
                print(f"[INFO] NPZ 中無標籤數據，實時計算標籤...")
                labels_for_plot = compute_labels_from_mids(mids, metadata)
            else:
                print(f"[INFO] 使用 NPZ 中已保存的標籤數據")

            fig_mids = go.Figure()

            # 添加中間價折線
            fig_mids.add_trace(go.Scatter(
                y=mids,
                mode='lines',
                name='中間價',
                line=dict(color='#2c3e50', width=1.5),
                hovertemplate='時間步: %{x}<br>中間價: %{y:.2f}<extra></extra>'
            ))

            # 疊加標籤點（如果有標籤）
            if labels_for_plot is not None:
                # 過濾 NaN 值
                valid_mask = ~np.isnan(labels_for_plot)
                valid_labels = labels_for_plot[valid_mask]

                # Down (-1): 紅色
                down_indices = np.where(labels_for_plot == -1)[0]
                if len(down_indices) > 0:
                    fig_mids.add_trace(go.Scatter(
                        x=down_indices,
                        y=mids[down_indices],
                        mode='markers',
                        name='Down (-1)',
                        marker=dict(color='#e74c3c', size=4, opacity=0.6),
                        hovertemplate='時間步: %{x}<br>中間價: %{y:.2f}<br>標籤: Down<extra></extra>'
                    ))

                # Neutral (0): 灰色
                neutral_indices = np.where(labels_for_plot == 0)[0]
                if len(neutral_indices) > 0:
                    fig_mids.add_trace(go.Scatter(
                        x=neutral_indices,
                        y=mids[neutral_indices],
                        mode='markers',
                        name='Neutral (0)',
                        marker=dict(color='#95a5a6', size=4, opacity=0.6),
                        hovertemplate='時間步: %{x}<br>中間價: %{y:.2f}<br>標籤: Neutral<extra></extra>'
                    ))

                # Up (1): 綠色
                up_indices = np.where(labels_for_plot == 1)[0]
                if len(up_indices) > 0:
                    fig_mids.add_trace(go.Scatter(
                        x=up_indices,
                        y=mids[up_indices],
                        mode='markers',
                        name='Up (1)',
                        marker=dict(color='#27ae60', size=4, opacity=0.6),
                        hovertemplate='時間步: %{x}<br>中間價: %{y:.2f}<br>標籤: Up<extra></extra>'
                    ))

            fig_mids.update_layout(
                title=f'{symbol} - 中間價時序圖（含標籤）',
                xaxis_title='時間步',
                yaxis_title='中間價',
                height=500,
                hovermode='closest',
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                font=dict(family='Microsoft YaHei, Arial')
            )
            charts.append(dcc.Graph(figure=fig_mids))

        # 2. LOB 特徵矩陣熱圖
        if 'features' in display_options and features is not None:
            # 只顯示前 500 個時間點（避免過大）
            T_display = min(500, features.shape[0])
            features_display = features[:T_display, :]

            # 特徵名稱
            feature_names = (
                [f'ask_price_{i+1}' for i in range(5)] +
                [f'ask_vol_{i+1}' for i in range(5)] +
                [f'bid_price_{i+1}' for i in range(5)] +
                [f'bid_vol_{i+1}' for i in range(5)]
            )

            fig_features = go.Figure(data=go.Heatmap(
                z=features_display.T,
                x=list(range(T_display)),
                y=feature_names,
                colorscale='Viridis',
                hovertemplate='時間: %{x}<br>特徵: %{y}<br>值: %{z:.2f}<extra></extra>'
            ))

            fig_features.update_layout(
                title=f'{symbol} - LOB 特徵矩陣熱圖（前 {T_display} 時間步）',
                xaxis_title='時間步',
                yaxis_title='特徵',
                height=600,
                font=dict(family='Microsoft YaHei, Arial')
            )
            charts.append(dcc.Graph(figure=fig_features))

        # 3. 標籤陣列視覺化
        if 'labels' in display_options and labels is not None:
            # 過濾 NaN
            valid_mask = ~np.isnan(labels)
            valid_labels = labels[valid_mask]
            valid_indices = np.where(valid_mask)[0]

            # 創建標籤時間序列圖
            fig_labels = go.Figure()

            # 為每個標籤類別添加散點
            for label_val, label_name, color in [
                (-1, 'Down', '#e74c3c'),
                (0, 'Neutral', '#95a5a6'),
                (1, 'Up', '#27ae60')
            ]:
                label_mask = valid_labels == label_val
                label_indices = valid_indices[label_mask]
                label_values = valid_labels[label_mask]

                fig_labels.add_trace(go.Scatter(
                    x=label_indices,
                    y=label_values,
                    mode='markers',
                    name=label_name,
                    marker=dict(color=color, size=3, opacity=0.7),
                    hovertemplate=f'{label_name}<br>時間步: %{{x}}<extra></extra>'
                ))

            fig_labels.update_layout(
                title=f'{symbol} - 標籤時間序列（總計 {len(valid_labels)} 個有效標籤）',
                xaxis_title='時間步',
                yaxis_title='標籤值',
                height=400,
                yaxis=dict(
                    tickmode='array',
                    tickvals=[-1, 0, 1],
                    ticktext=['Down (-1)', 'Neutral (0)', 'Up (1)']
                ),
                font=dict(family='Microsoft YaHei, Arial')
            )
            charts.append(dcc.Graph(figure=fig_labels))

        # 4. 事件數量圖
        if 'bucket_event_count' in display_options and bucket_event_count is not None:
            fig_events = go.Figure()

            fig_events.add_trace(go.Scatter(
                y=bucket_event_count,
                mode='lines',
                name='事件數',
                line=dict(color='#3498db', width=1),
                fill='tozeroy',
                hovertemplate='時間步: %{x}<br>事件數: %{y}<extra></extra>'
            ))

            fig_events.update_layout(
                title=f'{symbol} - 每秒事件數量（平均: {bucket_event_count.mean():.1f}）',
                xaxis_title='時間步',
                yaxis_title='事件數',
                height=400,
                font=dict(family='Microsoft YaHei, Arial')
            )
            charts.append(dcc.Graph(figure=fig_events))

        # 5. 時間桶遮罩圖
        if 'bucket_mask' in display_options and bucket_mask is not None:
            valid_ratio = bucket_mask.mean()

            fig_mask = go.Figure()

            fig_mask.add_trace(go.Scatter(
                y=bucket_mask,
                mode='lines',
                name='遮罩',
                line=dict(color='#9b59b6', width=1),
                fill='tozeroy',
                hovertemplate='時間步: %{x}<br>狀態: %{y}<extra></extra>'
            ))

            fig_mask.update_layout(
                title=f'{symbol} - 時間桶遮罩（有效比例: {valid_ratio:.2%}）',
                xaxis_title='時間步',
                yaxis_title='遮罩值 (0=缺失, 1=有效)',
                height=400,
                yaxis=dict(
                    tickmode='array',
                    tickvals=[0, 1],
                    ticktext=['缺失', '有效']
                ),
                font=dict(family='Microsoft YaHei, Arial')
            )
            charts.append(dcc.Graph(figure=fig_mask))

        # 6. 標籤預覽
        if 'label_preview' in display_options:
            label_preview = metadata.get('label_preview')

            if label_preview:
                fig_label = create_label_preview_bar(label_preview)
                fig_label.update_layout(
                    title=f'{symbol} - 標籤預覽分布',
                    font=dict(family='Microsoft YaHei, Arial')
                )
                charts.append(dcc.Graph(figure=fig_label))
            else:
                charts.append(html.Div(
                    "此股票無標籤預覽資訊",
                    style={'color': '#7f8c8d', 'padding': '20px', 'fontFamily': 'Microsoft YaHei, Arial'}
                ))

        # 7. 元數據表格
        if 'metadata' in display_options:
            if metadata:
                fig_meta = create_metadata_table(metadata)
                fig_meta.update_layout(
                    title=f'{symbol} - 元數據資訊',
                    font=dict(family='Microsoft YaHei, Arial')
                )
                charts.append(dcc.Graph(figure=fig_meta))

        return html.Div(charts)

    except Exception as e:
        return html.Div(
            f"載入數據失敗: {str(e)}",
            style={'color': 'red', 'padding': '20px', 'fontFamily': 'Microsoft YaHei, Arial'}
        )


def update_all_stocks_view(dir_info, display_options):
    """全部股票整體統計檢視"""
    # 從 dir_info 取得 daily_dir 路徑
    # 重建路徑（從第一個股票的路徑推導）
    first_stock_path = list(dir_info['stocks'].values())[0]
    daily_dir = str(Path(first_stock_path).parent)

    try:
        # 統計整體標籤分布
        stats = get_overall_label_stats(daily_dir)

        charts = []

        # 1. 整體標籤分布柱狀圖
        if stats['overall_dist']:
            fig_overall = create_overall_label_stats_bar(
                stats['overall_dist'],
                stats['total_stocks']
            )
            fig_overall.update_layout(
                title=f"整體標籤分布 - {dir_info['date']} ({stats['stocks_with_labels']}/{stats['total_stocks']} 檔有標籤)",
                font=dict(family='Microsoft YaHei, Arial')
            )
            charts.append(dcc.Graph(figure=fig_overall))

        # 2. 前 10 檔股票堆疊柱狀圖
        if stats['stock_details']:
            top_10_stocks = sorted(
                stats['stock_details'],
                key=lambda x: sum(x['label_preview']['label_counts'].values()),
                reverse=True
            )[:10]

            fig_top10 = go.Figure()

            for stock in top_10_stocks:
                symbol = stock['symbol']
                counts = stock['label_preview']['label_counts']

                fig_top10.add_trace(go.Bar(
                    name=symbol,
                    x=['Down (-1)', 'Neutral (0)', 'Up (1)'],
                    y=[
                        counts.get('-1', 0),
                        counts.get('0', 0),
                        counts.get('1', 0)
                    ]
                ))

            fig_top10.update_layout(
                title='前 10 檔股票標籤分布',
                xaxis_title='標籤類別',
                yaxis_title='數量',
                barmode='group',
                height=500,
                font=dict(family='Microsoft YaHei, Arial')
            )
            charts.append(dcc.Graph(figure=fig_top10))

        # 3. Summary.json 資訊表格
        if dir_info.get('summary'):
            summary = dir_info['summary']

            table_data = []
            for key, value in summary.items():
                if key == 'stocks':
                    continue  # 跳過股票詳細列表
                table_data.append({'項目': key, '值': str(value)})

            if table_data:
                fig_summary = go.Figure(data=[go.Table(
                    header=dict(
                        values=['<b>項目</b>', '<b>值</b>'],
                        fill_color='#3498db',
                        font=dict(color='white', size=12, family='Microsoft YaHei, Arial'),
                        align='left'
                    ),
                    cells=dict(
                        values=[
                            [d['項目'] for d in table_data],
                            [d['值'] for d in table_data]
                        ],
                        fill_color='#ecf0f1',
                        font=dict(color='#2c3e50', size=11, family='Consolas, monospace'),
                        align='left',
                        height=25
                    )
                )])

                fig_summary.update_layout(
                    title='Summary.json 統計資訊',
                    height=400,
                    font=dict(family='Microsoft YaHei, Arial')
                )
                charts.append(dcc.Graph(figure=fig_summary))

        return html.Div(charts)

    except Exception as e:
        return html.Div(
            f"載入整體統計失敗: {str(e)}",
            style={'color': 'red', 'padding': '20px', 'fontFamily': 'Microsoft YaHei, Arial'}
        )


@app.callback(
    Output('cache-info', 'children'),
    Input('stock-dropdown', 'value'),
    prevent_initial_call=True
)
def update_cache_info(_):
    """更新快取資訊"""
    try:
        info = get_cache_info()
        if info and isinstance(info, dict):
            hits = info.get('hits', 0)
            misses = info.get('misses', 0)
            current = info.get('current_size', 0)
            max_size = info.get('max_size', 10)
            return f"快取: {hits}/{hits + misses} 命中 ({current}/{max_size})"
        return "快取: N/A"
    except Exception as e:
        return f"快取: 錯誤 ({str(e)})"


# ============================================================================
# 應用啟動
# ============================================================================

if __name__ == '__main__':
    print("="*70)
    print("Label Viewer - 預處理數據模式")
    print("="*70)
    print()
    print("應用已啟動，請在瀏覽器訪問:")
    print("  → http://localhost:8051")
    print()
    print("按 Ctrl+C 停止")
    print("="*70)
    print()

    app.run(debug=False, port=8051, host='0.0.0.0')
