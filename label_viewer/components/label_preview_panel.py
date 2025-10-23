"""
標籤預覽面板組件

功能：
1. 顯示標籤預覽統計（Down/Neutral/Up 比例）
2. 與 predicted_label_dist 對比
3. 視覺化標籤分布（圓餅圖）
4. 顯示元數據資訊

作者：DeepLOB-Pro Team
最後更新：2025-10-23
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, Optional


def create_label_preview_comparison(
    label_preview: Dict,
    predicted_dist: Optional[Dict] = None
) -> go.Figure:
    """
    創建標籤預覽對比圖

    Args:
        label_preview: label_preview 字典
            {
                'total_labels': 14278,
                'down_count': 4512,
                'neutral_count': 2145,
                'up_count': 7621,
                'down_pct': 0.316,
                'neutral_pct': 0.150,
                'up_pct': 0.534
            }
        predicted_dist: (可選) 預測標籤分布
            {
                'down': 0.30,
                'neutral': 0.40,
                'up': 0.30
            }

    Returns:
        Plotly Figure 對象
    """
    # 如果有 predicted_dist，創建雙圓餅圖對比
    if predicted_dist:
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('實際標籤分布（TB 計算）', '預測標籤分布（啟發式）'),
            specs=[[{'type': 'pie'}, {'type': 'pie'}]]
        )

        # 左側：實際標籤分布
        fig.add_trace(
            go.Pie(
                labels=['下跌', '持平', '上漲'],
                values=[
                    label_preview['down_count'],
                    label_preview['neutral_count'],
                    label_preview['up_count']
                ],
                marker=dict(colors=['#e74c3c', '#95a5a6', '#27ae60']),
                textinfo='label+percent',
                textposition='inside',
                hovertemplate='<b>%{label}</b><br>數量: %{value:,}<br>比例: %{percent}<extra></extra>'
            ),
            row=1, col=1
        )

        # 右側：預測標籤分布
        total = label_preview['total_labels']
        fig.add_trace(
            go.Pie(
                labels=['下跌', '持平', '上漲'],
                values=[
                    predicted_dist['down'] * total,
                    predicted_dist['neutral'] * total,
                    predicted_dist['up'] * total
                ],
                marker=dict(colors=['#e74c3c', '#95a5a6', '#27ae60']),
                textinfo='label+percent',
                textposition='inside',
                hovertemplate='<b>%{label}</b><br>預測數量: %{value:,.0f}<br>比例: %{percent}<extra></extra>'
            ),
            row=1, col=2
        )

        fig.update_layout(
            height=400,
            showlegend=False,
            title_text=f'標籤分布對比（總標籤數: {total:,}）',
            title_x=0.5,
            title_font_size=16
        )

    else:
        # 只有實際標籤分布，創建單個圓餅圖
        fig = go.Figure()

        fig.add_trace(
            go.Pie(
                labels=['下跌', '持平', '上漲'],
                values=[
                    label_preview['down_count'],
                    label_preview['neutral_count'],
                    label_preview['up_count']
                ],
                marker=dict(colors=['#e74c3c', '#95a5a6', '#27ae60']),
                textinfo='label+percent',
                textposition='inside',
                hovertemplate='<b>%{label}</b><br>數量: %{value:,}<br>比例: %{percent}<extra></extra>'
            )
        )

        fig.update_layout(
            height=400,
            title_text=f'標籤預覽分布（總標籤數: {label_preview["total_labels"]:,}）',
            title_x=0.5,
            title_font_size=16
        )

    return fig


def create_label_preview_bar(label_preview: Dict) -> go.Figure:
    """
    創建標籤預覽柱狀圖（顯示數量和比例）

    Args:
        label_preview: label_preview 字典

    Returns:
        Plotly Figure 對象
    """
    categories = ['下跌', '持平', '上漲']
    counts = [
        label_preview['down_count'],
        label_preview['neutral_count'],
        label_preview['up_count']
    ]
    percentages = [
        label_preview['down_pct'],
        label_preview['neutral_pct'],
        label_preview['up_pct']
    ]
    colors = ['#e74c3c', '#95a5a6', '#27ae60']

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=categories,
            y=counts,
            text=[f'{count:,}<br>({pct:.1%})' for count, pct in zip(counts, percentages)],
            textposition='outside',
            marker=dict(color=colors),
            hovertemplate='<b>%{x}</b><br>數量: %{y:,}<extra></extra>'
        )
    )

    fig.update_layout(
        title_text=f'標籤預覽統計（總標籤數: {label_preview["total_labels"]:,}）',
        title_x=0.5,
        xaxis_title='標籤類別',
        yaxis_title='標籤數量',
        height=400,
        showlegend=False
    )

    return fig


def create_metadata_table(metadata: Dict) -> go.Figure:
    """
    創建元數據表格

    Args:
        metadata: metadata 字典

    Returns:
        Plotly Figure 對象（表格）
    """
    # 提取關鍵資訊
    info_items = [
        ('股票代碼', metadata.get('symbol', 'N/A')),
        ('日期', metadata.get('date', 'N/A')),
        ('數據點數', f"{metadata.get('n_points', 0):,}"),
        ('波動範圍', f"{metadata.get('range_pct', 0):.2%}"),
        ('日內收益率', f"{metadata.get('return_pct', 0):.2%}"),
        ('開盤價', f"{metadata.get('open', 0):.2f}"),
        ('收盤價', f"{metadata.get('close', 0):.2f}"),
        ('最高價', f"{metadata.get('high', 0):.2f}"),
        ('最低價', f"{metadata.get('low', 0):.2f}"),
        ('通過過濾', '✅ 是' if metadata.get('pass_filter', False) else '❌ 否'),
        ('過濾閾值', f"{metadata.get('filter_threshold', 0):.4f}"),
        ('過濾方法', metadata.get('filter_method', 'N/A')),
    ]

    # 如果有 1Hz 聚合資訊
    if 'sampling_mode' in metadata and metadata['sampling_mode'] == 'time':
        info_items.extend([
            ('採樣模式', '時間聚合（1Hz）'),
            ('總秒數', f"{metadata.get('n_seconds', 0):,}"),
            ('前向填充比例', f"{metadata.get('ffill_ratio', 0):.1%}"),
            ('缺失比例', f"{metadata.get('missing_ratio', 0):.1%}"),
            ('多事件比例', f"{metadata.get('multi_event_ratio', 0):.1%}"),
            ('最大間隔（秒）', f"{metadata.get('max_gap_sec', 0)}"),
        ])

    # 創建表格
    keys = [item[0] for item in info_items]
    values = [item[1] for item in info_items]

    fig = go.Figure(data=[go.Table(
        header=dict(
            values=['<b>項目</b>', '<b>值</b>'],
            fill_color='#3498db',
            font=dict(color='white', size=12),
            align='left'
        ),
        cells=dict(
            values=[keys, values],
            fill_color=['#ecf0f1', 'white'],
            font=dict(size=11),
            align='left',
            height=25
        )
    )])

    fig.update_layout(
        title_text=f'元數據資訊 - {metadata.get("symbol", "N/A")}',
        title_x=0.5,
        height=max(400, len(info_items) * 30 + 100),
        margin=dict(l=10, r=10, t=50, b=10)
    )

    return fig


def create_overall_label_stats_bar(overall_dist: Dict, total_stocks: int) -> go.Figure:
    """
    創建整體標籤統計柱狀圖（用於顯示當天所有股票的標籤分布）

    Args:
        overall_dist: 整體標籤分布
            {
                'down_pct': 0.313,
                'neutral_pct': 0.150,
                'up_pct': 0.537,
                'total_labels': 3012456
            }
        total_stocks: 總股票數

    Returns:
        Plotly Figure 對象
    """
    categories = ['下跌', '持平', '上漲']
    percentages = [
        overall_dist['down_pct'],
        overall_dist['neutral_pct'],
        overall_dist['up_pct']
    ]
    counts = [
        overall_dist['down_pct'] * overall_dist['total_labels'],
        overall_dist['neutral_pct'] * overall_dist['total_labels'],
        overall_dist['up_pct'] * overall_dist['total_labels']
    ]
    colors = ['#e74c3c', '#95a5a6', '#27ae60']

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=categories,
            y=percentages,
            text=[f'{count:,.0f}<br>({pct:.1%})' for count, pct in zip(counts, percentages)],
            textposition='outside',
            marker=dict(color=colors),
            hovertemplate='<b>%{x}</b><br>比例: %{y:.2%}<br>數量: %{text}<extra></extra>'
        )
    )

    fig.update_layout(
        title_text=f'整體標籤分布（{total_stocks} 檔股票，總標籤數: {overall_dist["total_labels"]:,}）',
        title_x=0.5,
        xaxis_title='標籤類別',
        yaxis_title='標籤比例',
        yaxis_tickformat='.0%',
        height=400,
        showlegend=False
    )

    return fig


if __name__ == "__main__":
    # 測試代碼
    import sys

    # 測試數據
    label_preview = {
        'total_labels': 14278,
        'down_count': 4512,
        'neutral_count': 2145,
        'up_count': 7621,
        'down_pct': 0.316,
        'neutral_pct': 0.150,
        'up_pct': 0.534
    }

    predicted_dist = {
        'down': 0.30,
        'neutral': 0.40,
        'up': 0.30
    }

    metadata = {
        'symbol': '2330',
        'date': '20250901',
        'n_points': 14400,
        'range_pct': 0.0456,
        'return_pct': 0.0123,
        'open': 500.0,
        'close': 506.15,
        'high': 508.5,
        'low': 498.0,
        'pass_filter': True,
        'filter_threshold': 0.005,
        'filter_method': 'fixed_0.5%',
        'sampling_mode': 'time',
        'n_seconds': 14400,
        'ffill_ratio': 0.35,
        'missing_ratio': 0.02,
        'multi_event_ratio': 0.15,
        'max_gap_sec': 120
    }

    print("測試標籤預覽組件...\n")

    # 測試 1: 對比圖
    print("="*70)
    print("測試 1: 標籤預覽對比圖")
    print("="*70)
    fig1 = create_label_preview_comparison(label_preview, predicted_dist)
    print(f"圖表創建成功，高度: {fig1.layout.height}px")
    # fig1.show()  # 取消註釋以顯示

    # 測試 2: 柱狀圖
    print(f"\n{'='*70}")
    print("測試 2: 標籤預覽柱狀圖")
    print("="*70)
    fig2 = create_label_preview_bar(label_preview)
    print(f"圖表創建成功，高度: {fig2.layout.height}px")
    # fig2.show()

    # 測試 3: 元數據表格
    print(f"\n{'='*70}")
    print("測試 3: 元數據表格")
    print("="*70)
    fig3 = create_metadata_table(metadata)
    print(f"表格創建成功，高度: {fig3.layout.height}px")
    # fig3.show()

    # 測試 4: 整體標籤統計
    print(f"\n{'='*70}")
    print("測試 4: 整體標籤統計")
    print("="*70)
    overall_dist = {
        'down_pct': 0.313,
        'neutral_pct': 0.150,
        'up_pct': 0.537,
        'total_labels': 3012456
    }
    fig4 = create_overall_label_stats_bar(overall_dist, 211)
    print(f"圖表創建成功，高度: {fig4.layout.height}px")
    # fig4.show()

    print(f"\n{'='*70}")
    print("✅ 所有測試通過")
    print("="*70)
