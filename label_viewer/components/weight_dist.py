# -*- coding: utf-8 -*-
"""
weight_dist.py - 樣本權重分布組件
========================================================
顯示樣本權重的統計分布（直方圖、箱型圖等）
"""

import plotly.graph_objects as go
import numpy as np
from typing import Dict, Any, Optional

# 導入配置
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))


def create_weight_histogram(
    weights: np.ndarray,
    chart_options: Optional[Dict[str, Any]] = None
) -> go.Figure:
    """
    創建樣本權重直方圖

    Args:
        weights: (N,) 權重陣列
        chart_options: 圖表選項字典
                      {'height': 高度, 'title': 標題, 'nbins': 直方圖柱數, ...}

    Returns:
        fig: Plotly Figure 物件

    Example:
        >>> weights = np.random.uniform(0.5, 1.5, 1000)
        >>> fig = create_weight_histogram(weights)
    """
    # 預設選項
    default_options = {
        'height': 300,
        'title': '樣本權重分布',
        'nbins': 30,
        'show_statistics': True
    }

    if chart_options:
        default_options.update(chart_options)

    # 計算統計資訊
    stats = {
        'mean': float(np.mean(weights)),
        'median': float(np.median(weights)),
        'std': float(np.std(weights)),
        'min': float(np.min(weights)),
        'max': float(np.max(weights)),
        'q25': float(np.percentile(weights, 25)),
        'q75': float(np.percentile(weights, 75))
    }

    # 創建直方圖
    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=weights,
        nbinsx=default_options['nbins'],
        marker=dict(
            color='#3498db',
            line=dict(color='white', width=1)
        ),
        opacity=0.75,
        name='權重分布',
        hovertemplate='權重範圍: %{x}<br>樣本數: %{y}<extra></extra>'
    ))

    # 添加平均值線
    if default_options['show_statistics']:
        fig.add_vline(
            x=stats['mean'],
            line_dash="dash",
            line_color="red",
            annotation_text=f"平均值: {stats['mean']:.3f}",
            annotation_position="top",
            annotation=dict(font=dict(size=11, color='red'))
        )

        # 添加中位數線
        fig.add_vline(
            x=stats['median'],
            line_dash="dot",
            line_color="green",
            annotation_text=f"中位數: {stats['median']:.3f}",
            annotation_position="bottom",
            annotation=dict(font=dict(size=11, color='green'))
        )

    # 構建標題（包含統計資訊）
    if default_options['show_statistics']:
        title_text = f"{default_options['title']}<br><sub>範圍: [{stats['min']:.3f}, {stats['max']:.3f}] | 標準差: {stats['std']:.3f}</sub>"
    else:
        title_text = default_options['title']

    # 更新布局
    fig.update_layout(
        title={
            'text': title_text,
            'font': {'size': 14, 'family': 'Microsoft YaHei, Arial'},
            'x': 0.5,
            'xanchor': 'center'
        },
        height=default_options['height'],
        margin=dict(l=50, r=50, t=80, b=50),
        template='plotly_white',
        font=dict(family='Microsoft YaHei, Arial'),
        xaxis=dict(
            title='樣本權重',
            showgrid=True
        ),
        yaxis=dict(
            title='樣本數量',
            showgrid=True
        ),
        showlegend=False,
        bargap=0.1
    )

    return fig


def create_weight_boxplot(
    weights: np.ndarray,
    chart_options: Optional[Dict[str, Any]] = None
) -> go.Figure:
    """
    創建樣本權重箱型圖（Box Plot）

    Args:
        weights: (N,) 權重陣列
        chart_options: 圖表選項字典

    Returns:
        fig: Plotly Figure 物件

    Example:
        >>> weights = np.random.uniform(0.5, 1.5, 1000)
        >>> fig = create_weight_boxplot(weights)
    """
    # 預設選項
    default_options = {
        'height': 250,
        'title': '樣本權重箱型圖'
    }

    if chart_options:
        default_options.update(chart_options)

    # 創建箱型圖
    fig = go.Figure()

    fig.add_trace(go.Box(
        y=weights,
        name='權重',
        marker=dict(color='#3498db'),
        boxmean='sd',  # 顯示平均值和標準差
        hovertemplate='權重: %{y:.3f}<extra></extra>'
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
        margin=dict(l=50, r=50, t=50, b=50),
        template='plotly_white',
        font=dict(family='Microsoft YaHei, Arial'),
        yaxis=dict(
            title='樣本權重',
            showgrid=True
        ),
        showlegend=False
    )

    return fig


def create_weight_statistics_table(
    weights: np.ndarray
) -> Dict[str, float]:
    """
    計算權重統計摘要

    Args:
        weights: (N,) 權重陣列

    Returns:
        stats: 統計字典
               {
                   'mean': 平均值,
                   'median': 中位數,
                   'std': 標準差,
                   'min': 最小值,
                   'max': 最大值,
                   'q25': 第一四分位數,
                   'q75': 第三四分位數,
                   'iqr': 四分位距,
                   'cv': 變異係數
               }

    Example:
        >>> weights = np.random.uniform(0.5, 1.5, 1000)
        >>> stats = create_weight_statistics_table(weights)
        >>> stats['mean']
        1.0
    """
    mean_val = float(np.mean(weights))
    std_val = float(np.std(weights))

    stats = {
        'mean': mean_val,
        'median': float(np.median(weights)),
        'std': std_val,
        'min': float(np.min(weights)),
        'max': float(np.max(weights)),
        'q25': float(np.percentile(weights, 25)),
        'q75': float(np.percentile(weights, 75)),
        'iqr': float(np.percentile(weights, 75) - np.percentile(weights, 25)),
        'cv': std_val / mean_val if mean_val != 0 else 0.0  # 變異係數
    }

    return stats


def create_combined_weight_chart(
    weights: np.ndarray,
    chart_options: Optional[Dict[str, Any]] = None
) -> go.Figure:
    """
    創建組合圖表（直方圖 + 統計資訊）

    Args:
        weights: (N,) 權重陣列
        chart_options: 圖表選項字典

    Returns:
        fig: Plotly Figure 物件

    Example:
        >>> weights = np.random.uniform(0.5, 1.5, 1000)
        >>> fig = create_combined_weight_chart(weights)
    """
    # 預設選項
    default_options = {
        'height': 350,
        'title': '樣本權重分析',
        'nbins': 30
    }

    if chart_options:
        default_options.update(chart_options)

    # 計算統計資訊
    stats = create_weight_statistics_table(weights)

    # 創建子圖（1 行 2 列）
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.7, 0.3],
        subplot_titles=('權重分布直方圖', '箱型圖'),
        horizontal_spacing=0.1
    )

    # 左側：直方圖
    fig.add_trace(
        go.Histogram(
            x=weights,
            nbinsx=default_options['nbins'],
            marker=dict(
                color='#3498db',
                line=dict(color='white', width=1)
            ),
            opacity=0.75,
            name='權重分布',
            hovertemplate='權重: %{x:.3f}<br>樣本數: %{y}<extra></extra>'
        ),
        row=1,
        col=1
    )

    # 右側：箱型圖
    fig.add_trace(
        go.Box(
            y=weights,
            name='權重',
            marker=dict(color='#3498db'),
            boxmean='sd',
            hovertemplate='權重: %{y:.3f}<extra></extra>'
        ),
        row=1,
        col=2
    )

    # 更新布局
    fig.update_layout(
        title={
            'text': f"{default_options['title']}<br><sub>平均: {stats['mean']:.3f} | 中位數: {stats['median']:.3f} | 標準差: {stats['std']:.3f}</sub>",
            'font': {'size': 14, 'family': 'Microsoft YaHei, Arial'},
            'x': 0.5,
            'xanchor': 'center'
        },
        height=default_options['height'],
        margin=dict(l=50, r=50, t=100, b=50),
        template='plotly_white',
        font=dict(family='Microsoft YaHei, Arial'),
        showlegend=False
    )

    # 更新 X 軸（直方圖）
    fig.update_xaxes(title_text='樣本權重', row=1, col=1)
    fig.update_yaxes(title_text='樣本數量', row=1, col=1)

    # 更新 Y 軸（箱型圖）
    fig.update_yaxes(title_text='樣本權重', row=1, col=2)

    return fig
