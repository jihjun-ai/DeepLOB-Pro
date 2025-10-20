# -*- coding: utf-8 -*-
"""
label_timeline.py - 標籤時間軸組件
========================================================
顯示標籤序列的顏色條帶視覺化
"""

import plotly.graph_objects as go
import numpy as np
from typing import Dict, Any, Optional

# 導入配置
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from utils.config import LABEL_NAMES, LABEL_COLORS


def create_label_timeline(
    labels: np.ndarray,
    chart_options: Optional[Dict[str, Any]] = None
) -> go.Figure:
    """
    創建標籤時間軸圖（顏色條帶）

    Args:
        labels: (N,) 標籤陣列 (0=下跌, 1=持平, 2=上漲)
        chart_options: 圖表選項字典
                      {'height': 高度, 'title': 標題, ...}

    Returns:
        fig: Plotly Figure 物件

    Example:
        >>> labels = np.array([0, 0, 1, 1, 2, 2, 2])
        >>> fig = create_label_timeline(labels)
    """
    # 預設選項
    default_options = {
        'height': 120,
        'title': '標籤時間軸',
        'show_legend': False
    }

    if chart_options:
        default_options.update(chart_options)

    # 創建圖表
    fig = go.Figure()

    # 為每個標籤值添加一個 trace（條帶）
    # 使用 scatter 模式，透過 fill='tozeroy' 創建色塊

    # 準備數據：將標籤轉換為高度值（用於視覺化）
    # 0=下跌 -> y=0
    # 1=持平 -> y=0.5
    # 2=上漲 -> y=1

    x = np.arange(len(labels))

    # 為每個標籤類型創建 mask
    for label_val in [0, 1, 2]:
        mask = labels == label_val
        if not np.any(mask):
            continue  # 跳過沒有樣本的標籤

        # 找出連續區間
        segments = _find_label_segments(labels, label_val)

        # 為每個區間添加一個矩形 shape
        for start_idx, end_idx in segments:
            fig.add_shape(
                type="rect",
                x0=start_idx,
                x1=end_idx,
                y0=0,
                y1=1,
                fillcolor=LABEL_COLORS[label_val],
                opacity=0.7,
                line_width=0,
                layer="below"
            )

    # 添加一個虛擬 trace（用於設置坐標軸範圍）
    fig.add_trace(go.Scatter(
        x=[0, len(labels) - 1],
        y=[0, 1],
        mode='markers',
        marker=dict(size=0.1, opacity=0),  # 幾乎不可見
        showlegend=False,
        hoverinfo='skip'
    ))

    # 更新布局
    fig.update_layout(
        title={
            'text': default_options['title'],
            'font': {'size': 14, 'family': 'Microsoft YaHei, Arial'},
            'x': 0.5,
            'xanchor': 'center'
        },
        height=default_options['height'],
        margin=dict(l=50, r=20, t=40, b=40),
        template='plotly_white',
        font=dict(family='Microsoft YaHei, Arial'),
        xaxis=dict(
            title='樣本索引',
            showgrid=False,
            zeroline=False
        ),
        yaxis=dict(
            title='',
            showticklabels=False,
            showgrid=False,
            zeroline=False,
            fixedrange=True  # 禁用 Y 軸縮放
        ),
        hovermode='x unified',
        showlegend=default_options['show_legend']
    )

    return fig


def _find_label_segments(labels: np.ndarray, label_value: int) -> list:
    """
    找出指定標籤值的連續區間

    Args:
        labels: (N,) 標籤陣列
        label_value: 要查找的標籤值

    Returns:
        segments: 區間列表，每個元素為 (start_idx, end_idx)

    Example:
        >>> labels = np.array([0, 0, 1, 1, 0, 0])
        >>> segments = _find_label_segments(labels, 0)
        >>> segments
        [(0, 2), (4, 6)]
    """
    segments = []
    in_segment = False
    start_idx = None

    for i, label in enumerate(labels):
        if label == label_value:
            if not in_segment:
                # 開始新區間
                start_idx = i
                in_segment = True
        else:
            if in_segment:
                # 結束當前區間
                segments.append((start_idx, i))
                in_segment = False

    # 處理最後一個區間
    if in_segment:
        segments.append((start_idx, len(labels)))

    return segments


def create_label_timeline_with_legend(
    labels: np.ndarray,
    chart_options: Optional[Dict[str, Any]] = None
) -> go.Figure:
    """
    創建帶有圖例的標籤時間軸圖

    Args:
        labels: (N,) 標籤陣列
        chart_options: 圖表選項字典

    Returns:
        fig: Plotly Figure 物件

    Example:
        >>> labels = np.array([0, 0, 1, 1, 2, 2])
        >>> fig = create_label_timeline_with_legend(labels)
    """
    # 預設選項
    default_options = {
        'height': 150,
        'title': '標籤時間軸',
        'show_legend': True
    }

    if chart_options:
        default_options.update(chart_options)

    # 創建基礎圖表
    fig = create_label_timeline(labels, default_options)

    # 添加圖例（使用虛擬 traces）
    for label_val in [0, 1, 2]:
        # 檢查該標籤是否存在
        if not np.any(labels == label_val):
            continue

        fig.add_trace(go.Scatter(
            x=[None],
            y=[None],
            mode='markers',
            marker=dict(
                size=10,
                color=LABEL_COLORS[label_val],
                symbol='square'
            ),
            name=LABEL_NAMES[label_val],
            showlegend=True
        ))

    # 更新圖例位置
    fig.update_layout(
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=11)
        )
    )

    return fig


def create_label_heatmap(
    labels: np.ndarray,
    chart_options: Optional[Dict[str, Any]] = None
) -> go.Figure:
    """
    創建標籤熱力圖（另一種視覺化方式）

    Args:
        labels: (N,) 標籤陣列
        chart_options: 圖表選項字典

    Returns:
        fig: Plotly Figure 物件

    Example:
        >>> labels = np.array([0, 0, 1, 1, 2, 2])
        >>> fig = create_label_heatmap(labels)
    """
    # 預設選項
    default_options = {
        'height': 100,
        'title': '標籤熱力圖'
    }

    if chart_options:
        default_options.update(chart_options)

    # 創建熱力圖數據（單行）
    z = labels.reshape(1, -1)  # (1, N)

    # 創建自定義 colorscale
    colorscale = [
        [0.0, LABEL_COLORS[0]],  # 下跌 - 紅色
        [0.5, LABEL_COLORS[1]],  # 持平 - 灰色
        [1.0, LABEL_COLORS[2]]   # 上漲 - 綠色
    ]

    fig = go.Figure(data=go.Heatmap(
        z=z,
        colorscale=colorscale,
        showscale=False,
        hovertemplate='樣本 %{x}<br>標籤: %{z}<extra></extra>',
        zmin=0,
        zmax=2
    ))

    # 更新布局
    fig.update_layout(
        title={
            'text': default_options['title'],
            'font': {'size': 14, 'family': 'Microsoft YaHei, Arial'},
            'x': 0.5,
            'xanchor': 'center'
        },
        height=default_options['height'],
        margin=dict(l=50, r=20, t=40, b=40),
        template='plotly_white',
        font=dict(family='Microsoft YaHei, Arial'),
        xaxis=dict(
            title='樣本索引',
            showgrid=False
        ),
        yaxis=dict(
            title='',
            showticklabels=False,
            showgrid=False,
            fixedrange=True
        )
    )

    return fig
