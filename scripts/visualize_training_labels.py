# -*- coding: utf-8 -*-
"""
visualize_training_labels.py - 訓練數據標籤視覺化工具
=============================================================================
【功能】檢查 extract_tw_stock_data_v5.py 產生的訓練資料標籤是否正確

【輸出圖表】
1. 收盤價曲線與標籤顏色疊加（檢查標籤與趨勢是否對應）
2. 標籤分布統計（檢查類別平衡）
3. Triple-Barrier 觸發分析（檢查止盈止損合理性）
4. 樣本權重分布（檢查權重分配）

【使用方式】
python scripts/visualize_training_labels.py --data-dir ./data/processed_v5/npz --split train --n-stocks 5

【參數說明】
--data-dir: NPZ 數據目錄（包含 stock_embedding_*.npz）
--split: 數據集劃分 {train, val, test}
--n-stocks: 顯示前 N 檔股票的詳細圖表（預設 3）
--output-dir: 圖表輸出目錄（預設 ./results/label_visualization）

版本：v1.0
日期：2025-10-20
"""

import os
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# 設定日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# 設定中文字型（避免亂碼）
plt.rcParams['font.sans-serif'] = ['Microsoft YahHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 標籤定義
LABEL_NAMES = {0: '下跌', 1: '持平', 2: '上漲'}
LABEL_COLORS = {0: '#e74c3c', 1: '#95a5a6', 2: '#2ecc71'}  # 紅/灰/綠


def parse_args():
    """解析命令列參數"""
    parser = argparse.ArgumentParser(
        "visualize_training_labels",
        description="訓練數據標籤視覺化工具 - 檢查標籤趨勢是否正確"
    )
    parser.add_argument(
        "--data-dir",
        default="./data/processed_v5/npz",
        type=str,
        help="NPZ 數據目錄路徑"
    )
    parser.add_argument(
        "--split",
        default="train",
        choices=["train", "val", "test"],
        help="數據集劃分"
    )
    parser.add_argument(
        "--n-stocks",
        default=3,
        type=int,
        help="顯示前 N 檔股票的詳細圖表"
    )
    parser.add_argument(
        "--output-dir",
        default="./results/label_visualization",
        type=str,
        help="圖表輸出目錄"
    )
    parser.add_argument(
        "--max-points",
        default=500,
        type=int,
        help="單檔股票最多顯示的時間點數（避免圖表過於密集）"
    )
    return parser.parse_args()


def load_data(data_dir: str, split: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
    載入 NPZ 數據與 metadata

    Returns:
        X: (N, 100, 20) 時間序列特徵
        y: (N,) 標籤 {0, 1, 2}
        weights: (N,) 樣本權重
        stock_ids: (N,) 股票代碼
        metadata: 元數據字典
    """
    npz_path = os.path.join(data_dir, f"stock_embedding_{split}.npz")
    meta_path = os.path.join(data_dir, "normalization_meta.json")

    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"找不到數據文件: {npz_path}")

    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"找不到元數據文件: {meta_path}")

    # 載入 NPZ
    data = np.load(npz_path, allow_pickle=True)
    X = data['X']
    y = data['y']
    weights = data['weights']
    stock_ids = data['stock_ids']

    # 載入 metadata
    with open(meta_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    logging.info(f"✅ 已載入 {split} 數據: {X.shape[0]:,} 個樣本")
    logging.info(f"   形狀: X={X.shape}, y={y.shape}, weights={weights.shape}")

    return X, y, weights, stock_ids, metadata


def reconstruct_close_price(X: np.ndarray, metadata: Dict) -> np.ndarray:
    """
    從 Z-Score 正規化的特徵重建中間價（近似收盤價）

    Args:
        X: (N, 100, 20) 正規化後的 LOB 特徵
        metadata: 包含 Z-Score 參數的元數據

    Returns:
        close_prices: (N,) 每個窗口最後一個時間點的中間價
    """
    # 提取 Z-Score 參數
    mu = np.array(metadata['normalization']['feature_means'])
    sd = np.array(metadata['normalization']['feature_stds'])

    # 反向 Z-Score（只取最後一個時間點）
    X_last = X[:, -1, :]  # (N, 20)
    X_denorm = X_last * sd.reshape(1, -1) + mu.reshape(1, -1)

    # 計算中間價：(bid1 + ask1) / 2
    # 根據 extract_tw_stock_data_v5.py 的特徵定義：
    # feat = bids_p + asks_p + bids_q + asks_q
    # 索引 0-4: bid prices, 5-9: ask prices
    bid1 = X_denorm[:, 0]
    ask1 = X_denorm[:, 5]
    close = (bid1 + ask1) / 2.0

    return close


def plot_stock_labels(
    stock_id: str,
    close: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    output_path: str,
    max_points: int = 500
):
    """
    繪製單檔股票的收盤價與標籤趨勢圖

    Args:
        stock_id: 股票代碼
        close: (T,) 收盤價序列
        labels: (T,) 標籤序列
        weights: (T,) 樣本權重
        output_path: 圖表保存路徑
        max_points: 最多顯示的時間點數
    """
    # 如果時間點太多，進行降採樣
    T = len(close)
    if T > max_points:
        indices = np.linspace(0, T - 1, max_points, dtype=int)
        close = close[indices]
        labels = labels[indices]
        weights = weights[indices]
        T = max_points

    # 創建圖表（2 行佈局）
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(3, 2, figure=fig, height_ratios=[2, 1, 1])

    # 圖 1: 收盤價與標籤趨勢（主圖）
    ax1 = fig.add_subplot(gs[0, :])

    # 繪製收盤價曲線
    ax1.plot(range(T), close, color='#3498db', linewidth=1.5, label='收盤價', alpha=0.8)

    # 用標籤顏色標註背景
    for i in range(T):
        label = int(labels[i])
        ax1.axvspan(i - 0.5, i + 0.5, alpha=0.2, color=LABEL_COLORS[label])

    ax1.set_xlabel('時間點', fontsize=12)
    ax1.set_ylabel('價格', fontsize=12)
    ax1.set_title(f'{stock_id} - 收盤價與標籤趨勢', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    # 圖 2: 標籤序列（時間軸）
    ax2 = fig.add_subplot(gs[1, :])

    colors = [LABEL_COLORS[int(l)] for l in labels]
    ax2.bar(range(T), np.ones(T), color=colors, width=1.0, edgecolor='none')
    ax2.set_xlabel('時間點', fontsize=12)
    ax2.set_ylabel('標籤', fontsize=12)
    ax2.set_title('標籤序列（紅=下跌, 灰=持平, 綠=上漲）', fontsize=12)
    ax2.set_ylim(0, 1.2)
    ax2.set_yticks([])
    ax2.grid(True, alpha=0.3, axis='x')

    # 圖 3: 標籤分布統計
    ax3 = fig.add_subplot(gs[2, 0])

    label_counts = pd.Series(labels).value_counts().sort_index()
    label_pcts = label_counts / len(labels) * 100

    bars = ax3.bar(
        [LABEL_NAMES[i] for i in label_counts.index],
        label_counts.values,
        color=[LABEL_COLORS[i] for i in label_counts.index],
        edgecolor='black',
        linewidth=1.5
    )

    # 添加百分比標註
    for i, (bar, pct) in enumerate(zip(bars, label_pcts.values)):
        height = bar.get_height()
        ax3.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f'{pct:.1f}%',
            ha='center',
            va='bottom',
            fontsize=10,
            fontweight='bold'
        )

    ax3.set_ylabel('樣本數', fontsize=12)
    ax3.set_title('標籤分布', fontsize=12)
    ax3.grid(True, alpha=0.3, axis='y')

    # 圖 4: 樣本權重分布
    ax4 = fig.add_subplot(gs[2, 1])

    ax4.hist(weights, bins=50, color='#9b59b6', alpha=0.7, edgecolor='black')
    ax4.axvline(weights.mean(), color='red', linestyle='--', linewidth=2, label=f'均值={weights.mean():.2f}')
    ax4.set_xlabel('樣本權重', fontsize=12)
    ax4.set_ylabel('頻率', fontsize=12)
    ax4.set_title('樣本權重分布', fontsize=12)
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    logging.info(f"  ✅ 已保存: {output_path}")


def plot_overall_statistics(
    y: np.ndarray,
    weights: np.ndarray,
    stock_ids: np.ndarray,
    output_path: str
):
    """
    繪製整體數據集的統計圖表

    Args:
        y: (N,) 標籤
        weights: (N,) 樣本權重
        stock_ids: (N,) 股票代碼
        output_path: 圖表保存路徑
    """
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 3, figure=fig)

    # 圖 1: 標籤分布（整體）
    ax1 = fig.add_subplot(gs[0, 0])

    label_counts = pd.Series(y).value_counts().sort_index()
    label_pcts = label_counts / len(y) * 100

    bars = ax1.bar(
        [LABEL_NAMES[i] for i in label_counts.index],
        label_counts.values,
        color=[LABEL_COLORS[i] for i in label_counts.index],
        edgecolor='black',
        linewidth=1.5
    )

    for i, (bar, pct, count) in enumerate(zip(bars, label_pcts.values, label_counts.values)):
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f'{count:,}\n({pct:.1f}%)',
            ha='center',
            va='bottom',
            fontsize=11,
            fontweight='bold'
        )

    ax1.set_ylabel('樣本數', fontsize=12)
    ax1.set_title('標籤分布（整體）', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')

    # 圖 2: 每檔股票的樣本數分布
    ax2 = fig.add_subplot(gs[0, 1])

    stock_counts = pd.Series(stock_ids).value_counts().sort_values(ascending=False)
    ax2.hist(stock_counts.values, bins=30, color='#3498db', alpha=0.7, edgecolor='black')
    ax2.axvline(stock_counts.mean(), color='red', linestyle='--', linewidth=2, label=f'均值={stock_counts.mean():.0f}')
    ax2.set_xlabel('每檔股票的樣本數', fontsize=12)
    ax2.set_ylabel('股票數量', fontsize=12)
    ax2.set_title('股票樣本數分布', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 圖 3: 權重分布（整體）
    ax3 = fig.add_subplot(gs[0, 2])

    # 使用對數尺度處理極端權重
    weights_clipped = np.clip(weights, 0, np.percentile(weights, 99))
    ax3.hist(weights_clipped, bins=50, color='#9b59b6', alpha=0.7, edgecolor='black')
    ax3.axvline(weights.mean(), color='red', linestyle='--', linewidth=2, label=f'均值={weights.mean():.2f}')
    ax3.set_xlabel('樣本權重', fontsize=12)
    ax3.set_ylabel('頻率', fontsize=12)
    ax3.set_title('樣本權重分布（去除極值）', fontsize=12)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 圖 4: 各股票的標籤分布（堆疊圖）
    ax4 = fig.add_subplot(gs[1, :])

    # 選取樣本數前 20 的股票
    top_stocks = stock_counts.head(20).index
    stock_label_dist = []

    for stock in top_stocks:
        mask = stock_ids == stock
        labels_stock = y[mask]
        counts = pd.Series(labels_stock).value_counts().reindex([0, 1, 2], fill_value=0).values
        stock_label_dist.append(counts)

    stock_label_dist = np.array(stock_label_dist).T  # (3, 20)

    x = np.arange(len(top_stocks))
    width = 0.8

    bottom = np.zeros(len(top_stocks))
    for i, (label_name, color) in enumerate([(0, LABEL_COLORS[0]), (1, LABEL_COLORS[1]), (2, LABEL_COLORS[2])]):
        ax4.bar(x, stock_label_dist[i], width, label=LABEL_NAMES[i], color=color, bottom=bottom, edgecolor='black', linewidth=0.5)
        bottom += stock_label_dist[i]

    ax4.set_xlabel('股票代碼', fontsize=12)
    ax4.set_ylabel('樣本數', fontsize=12)
    ax4.set_title('前 20 檔股票的標籤分布（堆疊圖）', fontsize=14, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(top_stocks, rotation=45, ha='right')
    ax4.legend(loc='upper right')
    ax4.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    logging.info(f"✅ 已保存整體統計圖: {output_path}")


def main():
    """主程式"""
    args = parse_args()

    # 載入數據
    logging.info(f"載入 {args.split} 數據集...")
    X, y, weights, stock_ids, metadata = load_data(args.data_dir, args.split)

    # 重建收盤價
    logging.info("重建收盤價序列...")
    close_prices = reconstruct_close_price(X, metadata)

    # 創建輸出目錄
    output_dir = Path(args.output_dir) / args.split
    output_dir.mkdir(parents=True, exist_ok=True)

    # 生成整體統計圖
    logging.info("\n生成整體統計圖...")
    plot_overall_statistics(
        y, weights, stock_ids,
        str(output_dir / "overall_statistics.png")
    )

    # 選取前 N 檔股票繪製詳細圖表
    logging.info(f"\n生成前 {args.n_stocks} 檔股票的詳細圖表...")
    stock_counts = pd.Series(stock_ids).value_counts().sort_values(ascending=False)
    top_stocks = stock_counts.head(args.n_stocks).index

    for i, stock in enumerate(top_stocks, 1):
        mask = stock_ids == stock
        stock_close = close_prices[mask]
        stock_labels = y[mask]
        stock_weights = weights[mask]

        logging.info(f"  [{i}/{args.n_stocks}] 處理 {stock}（{len(stock_close):,} 個樣本）...")

        output_path = output_dir / f"stock_{stock}.png"
        plot_stock_labels(
            stock, stock_close, stock_labels, stock_weights,
            str(output_path),
            max_points=args.max_points
        )

    # 輸出摘要報告
    logging.info(f"\n{'='*60}")
    logging.info("📊 標籤檢查摘要報告")
    logging.info(f"{'='*60}")
    logging.info(f"數據集: {args.split}")
    logging.info(f"總樣本數: {len(y):,}")
    logging.info(f"股票數量: {len(np.unique(stock_ids))}")

    label_counts = pd.Series(y).value_counts().sort_index()
    label_pcts = label_counts / len(y) * 100
    logging.info(f"\n標籤分布:")
    for label, count in label_counts.items():
        pct = label_pcts[label]
        logging.info(f"  {LABEL_NAMES[label]}: {count:,} ({pct:.2f}%)")

    logging.info(f"\n樣本權重統計:")
    logging.info(f"  均值: {weights.mean():.3f}")
    logging.info(f"  標準差: {weights.std():.3f}")
    logging.info(f"  最大值: {weights.max():.3f}")
    logging.info(f"  最小值: {weights.min():.3f}")

    logging.info(f"\n圖表已保存至: {output_dir}")
    logging.info(f"{'='*60}\n")


if __name__ == "__main__":
    main()
