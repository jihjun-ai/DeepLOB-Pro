"""
主圖表組件

功能：
1. 生成收盤價曲線圖
2. 添加標籤背景色（紅/灰/綠）
3. 支援互動（縮放、平移、懸停）
4. WebGL 加速渲染

作者：DeepLOB-Pro Team
最後更新：2025-10-20
"""

import numpy as np
import plotly.graph_objects as go
from typing import List, Tuple, Optional, Dict
import sys
from pathlib import Path

# 添加 utils 模組路徑
sys.path.append(str(Path(__file__).parent.parent))
from utils.config import LABEL_COLORS, LABEL_NAMES


def merge_consecutive_labels(labels: np.ndarray) -> List[Tuple[int, int, int]]:
    """
    合併連續相同標籤的區間（優化渲染性能）

    Args:
        labels: 標籤序列 (N,)

    Returns:
        列表，每個元素為 (start_idx, end_idx, label_value)

    Example:
        >>> labels = np.array([0, 0, 0, 1, 1, 2, 2, 2])
        >>> merge_consecutive_labels(labels)
        [(0, 3, 0), (3, 5, 1), (5, 8, 2)]

    Notes:
        - 合併後可以減少 shape 數量，提升渲染速度
        - 例如：1000 個標籤可能只需 50-100 個矩形
    """
    if len(labels) == 0:
        return []

    segments = []
    current_label = labels[0]
    start_idx = 0

    for i in range(1, len(labels)):
        if labels[i] != current_label:
            # 標籤變化，記錄前一個區間
            segments.append((start_idx, i, int(current_label)))
            start_idx = i
            current_label = labels[i]

    # 添加最後一個區間
    segments.append((start_idx, len(labels), int(current_label)))

    return segments


def create_label_background_shapes(
    labels: np.ndarray,
    y_min: float,
    y_max: float,
    time_range: Optional[Tuple[int, int]] = None
) -> List[dict]:
    """
    創建標籤背景色矩形（Plotly shapes）

    Args:
        labels: 標籤序列 (N,)
        y_min: Y 軸最小值（價格）
        y_max: Y 軸最大值（價格）
        time_range: 時間範圍 (start, end)，None=全部

    Returns:
        Plotly shape 列表

    Example:
        >>> labels = np.array([0, 1, 2, 1, 0])
        >>> shapes = create_label_background_shapes(labels, 100.0, 120.0)
        >>> len(shapes)  # 可能少於 5（因為合併了連續區間）
    """
    # 應用時間範圍篩選
    if time_range is not None:
        start, end = time_range
        labels = labels[start:end]
        x_offset = start
    else:
        x_offset = 0

    # 合併連續相同標籤
    segments = merge_consecutive_labels(labels)

    # 創建 shape 列表
    shapes = []

    for start_idx, end_idx, label_value in segments:
        # 獲取標籤對應的顏色
        color = LABEL_COLORS.get(label_value, LABEL_COLORS[1])  # 預設灰色

        # 創建矩形 shape
        shape = dict(
            type="rect",
            xref="x",
            yref="y",
            x0=x_offset + start_idx,
            x1=x_offset + end_idx,
            y0=y_min,
            y1=y_max,
            fillcolor=color,
            opacity=0.2,  # 半透明（不遮蔽價格曲線）
            line_width=0,  # 無邊框
            layer="below"  # 在曲線下方
        )

        shapes.append(shape)

    return shapes


def create_price_trace(
    close_prices: np.ndarray,
    labels: np.ndarray,
    time_range: Optional[Tuple[int, int]] = None,
    use_webgl: bool = True
) -> go.Scatter:
    """
    創建收盤價曲線 trace

    Args:
        close_prices: 收盤價序列 (N,)
        labels: 標籤序列 (N,) - 用於懸停資訊
        time_range: 時間範圍 (start, end)，None=全部
        use_webgl: 是否使用 WebGL 加速（推薦用於大數據量）

    Returns:
        Plotly Scatter trace

    Example:
        >>> close = np.array([100, 101, 102, 101, 100])
        >>> labels = np.array([0, 1, 2, 1, 0])
        >>> trace = create_price_trace(close, labels)
    """
    # 應用時間範圍篩選
    if time_range is not None:
        start, end = time_range
        close_prices = close_prices[start:end]
        labels = labels[start:end]
        x_values = np.arange(start, end)
    else:
        x_values = np.arange(len(close_prices))

    # 創建懸停文本（顯示：時間點、價格、標籤名稱）
    hover_texts = []
    for i, (price, label) in enumerate(zip(close_prices, labels)):
        label_name = LABEL_NAMES.get(int(label), "未知")
        text = (
            f"<b>時間點</b>: {x_values[i]}<br>"
            f"<b>收盤價</b>: {price:.4f}<br>"
            f"<b>標籤</b>: {label_name}"
        )
        hover_texts.append(text)

    # 選擇 Scatter 或 Scattergl（WebGL 加速）
    ScatterClass = go.Scattergl if use_webgl else go.Scatter

    # 創建 trace
    trace = ScatterClass(
        x=x_values,
        y=close_prices,
        mode='lines',
        name='收盤價',
        line=dict(
            color='#3498db',  # 藍色
            width=2
        ),
        hovertext=hover_texts,
        hoverinfo='text',
        hoverlabel=dict(
            bgcolor='white',
            font_size=12,
            font_family='Microsoft YaHei, Arial'
        )
    )

    return trace


def create_main_chart(
    stock_id: str,
    close_prices: np.ndarray,
    labels: np.ndarray,
    time_range: Optional[Tuple[int, int]] = None,
    chart_options: Optional[Dict] = None
) -> go.Figure:
    """
    創建主圖表（收盤價 + 標籤背景色）

    Args:
        stock_id: 股票 ID（例如：'2330'）
        close_prices: 收盤價序列 (N,)
        labels: 標籤序列 (N,)
        time_range: 時間範圍 (start, end)，None=全部
        chart_options: 圖表選項字典，包含：
            - width: 圖表寬度（預設 None=自適應）
            - height: 圖表高度（預設 500）
            - show_legend: 是否顯示圖例（預設 True）
            - use_webgl: 是否使用 WebGL（預設 True）

    Returns:
        Plotly Figure 對象

    Example:
        >>> close = np.random.randn(1000).cumsum() + 100
        >>> labels = np.random.randint(0, 3, 1000)
        >>> fig = create_main_chart('2330', close, labels)
        >>> fig.show()  # 在瀏覽器中顯示

    Notes:
        - 使用 WebGL 渲染，支援數萬點流暢顯示
        - 標籤背景色透明度 0.2，不遮蔽價格曲線
        - 支援縮放、平移、懸停等互動功能
    """
    # 預設選項
    default_options = {
        'width': None,  # 自適應
        'height': 500,
        'show_legend': True,
        'use_webgl': True
    }

    # 合併用戶選項
    if chart_options is None:
        chart_options = {}
    options = {**default_options, **chart_options}

    # 應用時間範圍篩選（用於計算 Y 軸範圍）
    if time_range is not None:
        start, end = time_range
        close_subset = close_prices[start:end]
        labels_subset = labels[start:end]
    else:
        close_subset = close_prices
        labels_subset = labels

    # 計算 Y 軸範圍（留 5% 邊距）
    y_min = close_subset.min()
    y_max = close_subset.max()
    y_margin = (y_max - y_min) * 0.05
    y_min -= y_margin
    y_max += y_margin

    # 創建標籤背景色 shapes
    shapes = create_label_background_shapes(
        labels=labels,
        y_min=y_min,
        y_max=y_max,
        time_range=time_range
    )

    # 創建價格曲線 trace
    price_trace = create_price_trace(
        close_prices=close_prices,
        labels=labels,
        time_range=time_range,
        use_webgl=options['use_webgl']
    )

    # 創建 Figure
    fig = go.Figure()

    # 添加 trace
    fig.add_trace(price_trace)

    # 更新布局
    fig.update_layout(
        title=dict(
            text=f'股票 {stock_id} - 收盤價與標籤視覺化',
            font=dict(size=18, family='Microsoft YaHei, Arial')
        ),
        xaxis=dict(
            title='時間點（樣本索引）',
            gridcolor='#ecf0f1',
            showgrid=True,
            zeroline=False
        ),
        yaxis=dict(
            title='收盤價',
            gridcolor='#ecf0f1',
            showgrid=True,
            zeroline=False
        ),
        shapes=shapes,  # 添加標籤背景色
        hovermode='x unified',  # 統一懸停模式
        template='plotly_white',  # 白色背景模板
        showlegend=options['show_legend'],
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor='#bdc3c7',
            borderwidth=1
        ),
        width=options['width'],
        height=options['height'],
        font=dict(family='Microsoft YaHei, Arial'),
        margin=dict(l=60, r=20, t=60, b=60)
    )

    # 配置工具列
    fig.update_layout(
        modebar=dict(
            bgcolor='rgba(255, 255, 255, 0.8)',
            color='#7f8c8d',
            activecolor='#3498db'
        )
    )

    return fig


def add_label_legend_annotation(fig: go.Figure) -> go.Figure:
    """
    添加標籤顏色圖例註解（輔助功能）

    Args:
        fig: Plotly Figure 對象

    Returns:
        更新後的 Figure 對象

    Example:
        >>> fig = create_main_chart('2330', close, labels)
        >>> fig = add_label_legend_annotation(fig)
    """
    # 創建圖例文本
    legend_text = (
        "<b>標籤顏色說明：</b><br>"
        f"<span style='color:{LABEL_COLORS[0]}'>█</span> {LABEL_NAMES[0]}<br>"
        f"<span style='color:{LABEL_COLORS[1]}'>█</span> {LABEL_NAMES[1]}<br>"
        f"<span style='color:{LABEL_COLORS[2]}'>█</span> {LABEL_NAMES[2]}"
    )

    # 添加註解
    fig.add_annotation(
        text=legend_text,
        xref="paper",
        yref="paper",
        x=0.98,
        y=0.02,
        showarrow=False,
        bgcolor='rgba(255, 255, 255, 0.9)',
        bordercolor='#bdc3c7',
        borderwidth=1,
        borderpad=8,
        font=dict(size=11, family='Microsoft YaHei, Arial'),
        align='left',
        xanchor='right',
        yanchor='bottom'
    )

    return fig


if __name__ == "__main__":
    """
    測試代碼
    """
    print("=" * 60)
    print("測試主圖表組件")
    print("=" * 60)

    # 生成測試數據
    np.random.seed(42)
    n_samples = 1000

    # 模擬收盤價（隨機遊走）
    close_prices = np.cumsum(np.random.randn(n_samples) * 0.5) + 100

    # 模擬標籤（根據價格變化生成）
    price_changes = np.diff(close_prices, prepend=close_prices[0])
    labels = np.where(price_changes > 0.5, 2, np.where(price_changes < -0.5, 0, 1))

    print(f"✅ 生成測試數據：{n_samples} 個樣本")
    print(f"   - 收盤價範圍: [{close_prices.min():.2f}, {close_prices.max():.2f}]")
    print(f"   - 標籤分布: {np.bincount(labels)}")

    # 測試 1: 合併連續標籤
    print("\n" + "=" * 60)
    print("測試 1: 合併連續標籤")
    print("=" * 60)
    segments = merge_consecutive_labels(labels)
    print(f"✅ 原始標籤數: {len(labels)}")
    print(f"✅ 合併後區間數: {len(segments)}")
    print(f"✅ 壓縮率: {len(segments) / len(labels):.1%}")
    print(f"   前 5 個區間: {segments[:5]}")

    # 測試 2: 創建標籤背景
    print("\n" + "=" * 60)
    print("測試 2: 創建標籤背景色 shapes")
    print("=" * 60)
    shapes = create_label_background_shapes(
        labels=labels,
        y_min=close_prices.min(),
        y_max=close_prices.max()
    )
    print(f"✅ 生成 {len(shapes)} 個背景矩形")
    print(f"   範例 shape: {shapes[0] if shapes else 'None'}")

    # 測試 3: 創建價格曲線
    print("\n" + "=" * 60)
    print("測試 3: 創建價格曲線 trace")
    print("=" * 60)
    trace = create_price_trace(close_prices, labels, use_webgl=True)
    print(f"✅ Trace 類型: {type(trace).__name__}")
    print(f"✅ 數據點數: {len(trace.x)}")

    # 測試 4: 創建完整圖表
    print("\n" + "=" * 60)
    print("測試 4: 創建完整主圖表")
    print("=" * 60)
    fig = create_main_chart(
        stock_id='TEST',
        close_prices=close_prices,
        labels=labels,
        chart_options={'height': 600}
    )
    print(f"✅ 圖表創建成功")
    print(f"   - Trace 數量: {len(fig.data)}")
    print(f"   - Shape 數量: {len(fig.layout.shapes)}")

    # 測試 5: 添加圖例註解
    print("\n" + "=" * 60)
    print("測試 5: 添加標籤圖例註解")
    print("=" * 60)
    fig = add_label_legend_annotation(fig)
    print(f"✅ 註解添加成功")
    print(f"   - Annotation 數量: {len(fig.layout.annotations)}")

    # 測試 6: 時間範圍篩選
    print("\n" + "=" * 60)
    print("測試 6: 時間範圍篩選")
    print("=" * 60)
    time_range = (100, 500)
    fig_subset = create_main_chart(
        stock_id='TEST',
        close_prices=close_prices,
        labels=labels,
        time_range=time_range
    )
    print(f"✅ 篩選範圍: {time_range}")
    print(f"   - 顯示數據點數: {len(fig_subset.data[0].x)}")

    print("\n" + "=" * 60)
    print("✅ 所有測試完成！")
    print("=" * 60)
    print("\n提示：在 Dash 應用中使用以下代碼顯示圖表：")
    print("  fig.show()  # Jupyter Notebook")
    print("  或將 fig 傳遞給 dcc.Graph(figure=fig)")
