"""
標籤分布圖組件

功能：
1. 生成標籤分布圓餅圖
2. 顯示百分比與絕對數量
3. 使用標籤對應的顏色
4. 互動懸停資訊

作者：DeepLOB-Pro Team
最後更新：2025-10-20
"""

import numpy as np
import plotly.graph_objects as go
from typing import Dict, Optional
import sys
from pathlib import Path

# 添加 utils 模組路徑
sys.path.append(str(Path(__file__).parent.parent))
from utils.config import LABEL_COLORS, LABEL_NAMES


def calculate_label_distribution(labels: np.ndarray) -> Dict[int, int]:
    """
    計算標籤分布統計

    Args:
        labels: 標籤序列 (N,)

    Returns:
        字典，key=標籤值, value=數量

    Example:
        >>> labels = np.array([0, 0, 1, 1, 1, 2, 2])
        >>> calculate_label_distribution(labels)
        {0: 2, 1: 3, 2: 2}
    """
    # 使用 numpy.bincount 快速計算
    counts = np.bincount(labels.astype(int))

    # 轉換為字典
    distribution = {label: int(count) for label, count in enumerate(counts) if count > 0}

    return distribution


def create_label_distribution_pie(
    labels: np.ndarray,
    chart_options: Optional[Dict] = None
) -> go.Figure:
    """
    創建標籤分布圓餅圖

    Args:
        labels: 標籤序列 (N,)
        chart_options: 圖表選項字典，包含：
            - width: 圖表寬度（預設 None=自適應）
            - height: 圖表高度（預設 300）
            - show_legend: 是否顯示圖例（預設 True）
            - hole: 環形圖中心空洞大小（0-1，0=圓餅圖，預設 0）

    Returns:
        Plotly Figure 對象

    Example:
        >>> labels = np.random.randint(0, 3, 1000)
        >>> fig = create_label_distribution_pie(labels)
        >>> fig.show()
    """
    # 預設選項
    default_options = {
        'width': None,  # 自適應
        'height': 300,
        'show_legend': True,
        'hole': 0  # 0=圓餅圖, 0.3=環形圖
    }

    # 合併用戶選項
    if chart_options is None:
        chart_options = {}
    options = {**default_options, **chart_options}

    # 計算標籤分布
    distribution = calculate_label_distribution(labels)

    # 準備數據
    label_values = sorted(distribution.keys())
    counts = [distribution[label] for label in label_values]
    label_names = [LABEL_NAMES.get(label, f"標籤 {label}") for label in label_values]
    colors = [LABEL_COLORS.get(label, '#95a5a6') for label in label_values]

    # 計算百分比
    total = sum(counts)
    percentages = [count / total * 100 for count in counts]

    # 創建懸停文本
    hover_texts = [
        f"<b>{name}</b><br>"
        f"數量: {count:,}<br>"
        f"百分比: {pct:.2f}%"
        for name, count, pct in zip(label_names, counts, percentages)
    ]

    # 創建圓餅圖
    fig = go.Figure(data=[go.Pie(
        labels=label_names,
        values=counts,
        marker=dict(colors=colors),
        hovertext=hover_texts,
        hoverinfo='text',
        textinfo='label+percent',  # 顯示標籤名稱和百分比
        textposition='inside',
        textfont=dict(size=12, color='white', family='Microsoft YaHei, Arial'),
        hole=options['hole']
    )])

    # 更新布局
    fig.update_layout(
        title=dict(
            text='標籤分布統計',
            font=dict(size=16, family='Microsoft YaHei, Arial')
        ),
        showlegend=options['show_legend'],
        legend=dict(
            orientation="h",  # 水平圖例
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5,
            font=dict(size=11, family='Microsoft YaHei, Arial')
        ),
        width=options['width'],
        height=options['height'],
        template='plotly_white',
        font=dict(family='Microsoft YaHei, Arial'),
        margin=dict(l=20, r=20, t=60, b=80),
        hoverlabel=dict(
            bgcolor='white',
            font_size=12,
            font_family='Microsoft YaHei, Arial'
        )
    )

    return fig


def create_label_distribution_bar(
    labels: np.ndarray,
    chart_options: Optional[Dict] = None
) -> go.Figure:
    """
    創建標籤分布柱狀圖（替代方案）

    Args:
        labels: 標籤序列 (N,)
        chart_options: 圖表選項字典，包含：
            - width: 圖表寬度（預設 None=自適應）
            - height: 圖表高度（預設 300）
            - show_legend: 是否顯示圖例（預設 False）
            - orientation: 方向（'v'=垂直, 'h'=水平，預設 'v'）

    Returns:
        Plotly Figure 對象

    Example:
        >>> labels = np.random.randint(0, 3, 1000)
        >>> fig = create_label_distribution_bar(labels)
        >>> fig.show()
    """
    # 預設選項
    default_options = {
        'width': None,  # 自適應
        'height': 300,
        'show_legend': False,
        'orientation': 'v'  # 垂直
    }

    # 合併用戶選項
    if chart_options is None:
        chart_options = {}
    options = {**default_options, **chart_options}

    # 計算標籤分布
    distribution = calculate_label_distribution(labels)

    # 準備數據
    label_values = sorted(distribution.keys())
    counts = [distribution[label] for label in label_values]
    label_names = [LABEL_NAMES.get(label, f"標籤 {label}") for label in label_values]
    colors = [LABEL_COLORS.get(label, '#95a5a6') for label in label_values]

    # 計算百分比
    total = sum(counts)
    percentages = [count / total * 100 for count in counts]

    # 創建懸停文本
    hover_texts = [
        f"<b>{name}</b><br>"
        f"數量: {count:,}<br>"
        f"百分比: {pct:.2f}%"
        for name, count, pct in zip(label_names, counts, percentages)
    ]

    # 創建柱狀圖
    if options['orientation'] == 'h':
        # 水平柱狀圖
        fig = go.Figure(data=[go.Bar(
            y=label_names,
            x=counts,
            orientation='h',
            marker=dict(color=colors),
            hovertext=hover_texts,
            hoverinfo='text',
            text=[f"{pct:.1f}%" for pct in percentages],
            textposition='inside',
            textfont=dict(size=12, color='white')
        )])
        xaxis_title = '樣本數'
        yaxis_title = '標籤類別'
    else:
        # 垂直柱狀圖
        fig = go.Figure(data=[go.Bar(
            x=label_names,
            y=counts,
            marker=dict(color=colors),
            hovertext=hover_texts,
            hoverinfo='text',
            text=[f"{pct:.1f}%" for pct in percentages],
            textposition='inside',
            textfont=dict(size=12, color='white')
        )])
        xaxis_title = '標籤類別'
        yaxis_title = '樣本數'

    # 更新布局
    fig.update_layout(
        title=dict(
            text='標籤分布統計（柱狀圖）',
            font=dict(size=16, family='Microsoft YaHei, Arial')
        ),
        xaxis=dict(
            title=xaxis_title,
            gridcolor='#ecf0f1'
        ),
        yaxis=dict(
            title=yaxis_title,
            gridcolor='#ecf0f1'
        ),
        showlegend=options['show_legend'],
        width=options['width'],
        height=options['height'],
        template='plotly_white',
        font=dict(family='Microsoft YaHei, Arial'),
        margin=dict(l=60, r=20, t=60, b=60),
        hoverlabel=dict(
            bgcolor='white',
            font_size=12,
            font_family='Microsoft YaHei, Arial'
        )
    )

    return fig


if __name__ == "__main__":
    """
    測試代碼
    """
    print("=" * 60)
    print("測試標籤分布圖組件")
    print("=" * 60)

    # 生成測試數據
    np.random.seed(42)
    n_samples = 1000

    # 模擬不平衡的標籤分布
    # 40% 持平, 30% 上漲, 30% 下跌
    labels = np.random.choice(
        [0, 1, 2],
        size=n_samples,
        p=[0.3, 0.4, 0.3]
    )

    print(f"✅ 生成測試數據：{n_samples} 個樣本")

    # 測試 1: 計算標籤分布
    print("\n" + "=" * 60)
    print("測試 1: 計算標籤分布")
    print("=" * 60)
    distribution = calculate_label_distribution(labels)
    print(f"✅ 標籤分布統計:")
    total = sum(distribution.values())
    for label, count in sorted(distribution.items()):
        label_name = LABEL_NAMES.get(label, f"標籤 {label}")
        percentage = count / total * 100
        print(f"   - {label_name} (標籤 {label}): {count:,} 樣本 ({percentage:.2f}%)")

    # 測試 2: 創建圓餅圖
    print("\n" + "=" * 60)
    print("測試 2: 創建標籤分布圓餅圖")
    print("=" * 60)
    fig_pie = create_label_distribution_pie(labels, chart_options={'height': 400})
    print(f"✅ 圓餅圖創建成功")
    print(f"   - Trace 數量: {len(fig_pie.data)}")
    print(f"   - 圖表類型: {type(fig_pie.data[0]).__name__}")

    # 測試 3: 創建環形圖
    print("\n" + "=" * 60)
    print("測試 3: 創建標籤分布環形圖（hole=0.3）")
    print("=" * 60)
    fig_donut = create_label_distribution_pie(
        labels,
        chart_options={'height': 400, 'hole': 0.3}
    )
    print(f"✅ 環形圖創建成功")
    print(f"   - Hole 大小: {fig_donut.data[0].hole}")

    # 測試 4: 創建垂直柱狀圖
    print("\n" + "=" * 60)
    print("測試 4: 創建標籤分布垂直柱狀圖")
    print("=" * 60)
    fig_bar_v = create_label_distribution_bar(labels, chart_options={'height': 400})
    print(f"✅ 垂直柱狀圖創建成功")
    print(f"   - Trace 數量: {len(fig_bar_v.data)}")
    print(f"   - 圖表類型: {type(fig_bar_v.data[0]).__name__}")

    # 測試 5: 創建水平柱狀圖
    print("\n" + "=" * 60)
    print("測試 5: 創建標籤分布水平柱狀圖")
    print("=" * 60)
    fig_bar_h = create_label_distribution_bar(
        labels,
        chart_options={'height': 400, 'orientation': 'h'}
    )
    print(f"✅ 水平柱狀圖創建成功")
    print(f"   - Orientation: {fig_bar_h.data[0].orientation}")

    # 測試 6: 邊界情況測試
    print("\n" + "=" * 60)
    print("測試 6: 邊界情況測試（單一標籤）")
    print("=" * 60)
    labels_single = np.ones(100, dtype=int)  # 全部為標籤 1
    distribution_single = calculate_label_distribution(labels_single)
    print(f"✅ 單一標籤分布: {distribution_single}")

    fig_single = create_label_distribution_pie(labels_single)
    print(f"✅ 單一標籤圓餅圖創建成功")

    print("\n" + "=" * 60)
    print("✅ 所有測試完成！")
    print("=" * 60)
    print("\n提示：在 Dash 應用中使用以下代碼顯示圖表：")
    print("  fig.show()  # Jupyter Notebook")
    print("  或將 fig 傳遞給 dcc.Graph(figure=fig)")
