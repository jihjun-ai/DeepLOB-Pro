"""
預處理數據載入模組

功能：
1. 載入 preprocess_single_day.py 產生的 NPZ 數據
2. 解析 label_preview 資訊
3. 生成日期/股票列表
4. LRU Cache 快取機制

作者：DeepLOB-Pro Team
最後更新：2025-10-23
"""

import os
import json
import functools
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import numpy as np


@functools.lru_cache(maxsize=10)
def load_preprocessed_stock(npz_path: str) -> Dict:
    """
    載入單一股票的預處理數據

    Args:
        npz_path: NPZ 檔案路徑

    Returns:
        字典，包含：
        - features: (T, 20) LOB 特徵
        - mids: (T,) 中間價
        - metadata: 元數據字典（包含 label_preview）
        - bucket_event_count: 每秒事件數
        - bucket_mask: 時間桶遮罩

    Raises:
        FileNotFoundError: 檔案不存在
        ValueError: 數據格式錯誤
    """
    if not Path(npz_path).exists():
        raise FileNotFoundError(f"找不到檔案: {npz_path}")

    print(f"[INFO] 載入預處理數據: {npz_path}")

    # 載入 NPZ
    data = np.load(npz_path, allow_pickle=True)

    # 檢查必要鍵
    required_keys = ['features', 'mids', 'metadata']
    actual_keys = list(data.keys())
    missing_keys = set(required_keys) - set(actual_keys)

    if missing_keys:
        raise ValueError(f"缺少必要鍵: {missing_keys}，檔案鍵: {actual_keys}")

    # 解析 metadata
    metadata_str = str(data['metadata'])
    try:
        metadata = json.loads(metadata_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"無法解析 metadata: {e}")

    return {
        'features': data['features'],
        'mids': data['mids'],
        'metadata': metadata,
        'bucket_event_count': data.get('bucket_event_count', None),
        'bucket_mask': data.get('bucket_mask', None),
        'labels': data.get('labels', None)  # 🆕 讀取標籤數據（如果有）
    }


def scan_preprocessed_directory(preprocessed_dir: str) -> Dict[str, List[str]]:
    """
    掃描預處理數據目錄，生成日期和股票列表

    Args:
        preprocessed_dir: 預處理數據根目錄
            例如: data/preprocessed_v5 或 data/preprocessed_v5_1hz

    Returns:
        字典：
        {
            'dates': ['20250901', '20250902', ...],
            'stocks_by_date': {
                '20250901': ['2330', '2317', ...],
                '20250902': [...],
            }
        }

    目錄結構：
        preprocessed_dir/
        └── daily/
            ├── 20250901/
            │   ├── 2330.npz
            │   ├── 2317.npz
            │   └── summary.json
            ├── 20250902/
            │   ├── ...
    """
    daily_dir = Path(preprocessed_dir) / "daily"

    if not daily_dir.exists():
        raise FileNotFoundError(f"找不到 daily 目錄: {daily_dir}")

    dates = []
    stocks_by_date = {}

    # 遍歷日期目錄
    for date_dir in sorted(daily_dir.iterdir()):
        if not date_dir.is_dir():
            continue

        date = date_dir.name
        dates.append(date)

        # 掃描該日期的股票 NPZ
        stocks = []
        for npz_file in sorted(date_dir.glob("*.npz")):
            symbol = npz_file.stem  # 去掉 .npz 後綴
            stocks.append(symbol)

        stocks_by_date[date] = stocks

    print(f"[INFO] 掃描完成:")
    print(f"  找到 {len(dates)} 個交易日")
    total_stocks = sum(len(stocks) for stocks in stocks_by_date.values())
    print(f"  找到 {total_stocks} 個股票數據")

    return {
        'dates': dates,
        'stocks_by_date': stocks_by_date
    }


def get_label_preview_stats(preprocessed_dir: str, date: str) -> Dict:
    """
    獲取指定日期的標籤預覽統計

    Args:
        preprocessed_dir: 預處理數據根目錄
        date: 日期（例如 '20250901'）

    Returns:
        字典，包含：
        - total_stocks: 總股票數
        - stocks_with_labels: 有標籤預覽的股票數
        - overall_dist: 整體標籤分布 (down_pct, neutral_pct, up_pct)
        - stock_details: 每個股票的標籤預覽 [(symbol, label_preview), ...]
    """
    daily_dir = Path(preprocessed_dir) / "daily" / date

    if not daily_dir.exists():
        raise FileNotFoundError(f"找不到日期目錄: {daily_dir}")

    stock_details = []
    total_down = 0
    total_neutral = 0
    total_up = 0
    stocks_with_labels = 0

    # 遍歷該日期的所有 NPZ
    for npz_file in sorted(daily_dir.glob("*.npz")):
        symbol = npz_file.stem

        try:
            stock_data = load_preprocessed_stock(str(npz_file))
            metadata = stock_data['metadata']

            # 檢查是否有 label_preview
            if 'label_preview' in metadata and metadata['label_preview'] is not None:
                lp = metadata['label_preview']
                stock_details.append((symbol, lp))

                # 累加統計
                total_down += lp['down_count']
                total_neutral += lp['neutral_count']
                total_up += lp['up_count']
                stocks_with_labels += 1
            else:
                stock_details.append((symbol, None))

        except Exception as e:
            print(f"[WARN] 載入 {symbol} 失敗: {e}")
            stock_details.append((symbol, None))

    # 計算整體分布
    total_all = total_down + total_neutral + total_up
    if total_all > 0:
        overall_dist = {
            'down_pct': total_down / total_all,
            'neutral_pct': total_neutral / total_all,
            'up_pct': total_up / total_all,
            'total_labels': total_all
        }
    else:
        overall_dist = None

    return {
        'total_stocks': len(stock_details),
        'stocks_with_labels': stocks_with_labels,
        'overall_dist': overall_dist,
        'stock_details': stock_details
    }


def load_summary_json(preprocessed_dir: str, date: str) -> Optional[Dict]:
    """
    載入 summary.json

    Args:
        preprocessed_dir: 預處理數據根目錄
        date: 日期（例如 '20250901'）

    Returns:
        summary 字典，如果不存在則返回 None
    """
    summary_path = Path(preprocessed_dir) / "daily" / date / "summary.json"

    if not summary_path.exists():
        print(f"[WARN] 找不到 summary.json: {summary_path}")
        return None

    with open(summary_path, 'r', encoding='utf-8') as f:
        summary = json.load(f)

    return summary


# ============================================================================
# 快取管理
# ============================================================================

def get_cache_info() -> str:
    """獲取快取統計資訊"""
    info = load_preprocessed_stock.cache_info()
    return (
        f"快取統計：命中 {info.hits} 次，"
        f"未命中 {info.misses} 次，"
        f"當前大小 {info.currsize}/{info.maxsize}"
    )


def clear_cache():
    """清除快取"""
    load_preprocessed_stock.cache_clear()
    print("[INFO] 快取已清除")


# ============================================================================
# 便利函數
# ============================================================================

def get_stock_list_for_date(preprocessed_dir: str, date: str,
                            filter_by_label: bool = True) -> List[Tuple[str, Optional[Dict]]]:
    """
    獲取指定日期的股票列表

    Args:
        preprocessed_dir: 預處理數據根目錄
        date: 日期
        filter_by_label: 是否只返回有標籤預覽的股票

    Returns:
        [(symbol, label_preview), ...]
        按標籤總數排序（多的在前）
    """
    stats = get_label_preview_stats(preprocessed_dir, date)
    stock_details = stats['stock_details']

    if filter_by_label:
        # 過濾掉沒有標籤預覽的股票
        stock_details = [(sym, lp) for sym, lp in stock_details if lp is not None]

    # 按標籤總數排序（多的在前）
    stock_details_sorted = sorted(
        stock_details,
        key=lambda x: x[1]['total_labels'] if x[1] else 0,
        reverse=True
    )

    return stock_details_sorted


if __name__ == "__main__":
    # 測試代碼
    import sys

    if len(sys.argv) > 1:
        preprocessed_dir = sys.argv[1]
    else:
        preprocessed_dir = "data/preprocessed_v5_1hz"

    print(f"測試載入預處理數據: {preprocessed_dir}\n")

    # 測試 1: 掃描目錄
    print("="*70)
    print("測試 1: 掃描目錄")
    print("="*70)
    try:
        dir_info = scan_preprocessed_directory(preprocessed_dir)
        print(f"找到日期: {dir_info['dates'][:5]} ...")
        for date in dir_info['dates'][:2]:
            print(f"  {date}: {len(dir_info['stocks_by_date'][date])} 檔股票")
    except Exception as e:
        print(f"錯誤: {e}")

    # 測試 2: 載入單一股票
    if 'dates' in locals() and len(dir_info['dates']) > 0:
        test_date = dir_info['dates'][0]
        test_stocks = dir_info['stocks_by_date'][test_date][:2]

        print(f"\n{'='*70}")
        print(f"測試 2: 載入股票數據 ({test_date})")
        print("="*70)

        for symbol in test_stocks:
            npz_path = f"{preprocessed_dir}/daily/{test_date}/{symbol}.npz"
            try:
                stock_data = load_preprocessed_stock(npz_path)
                print(f"\n{symbol}:")
                print(f"  Features shape: {stock_data['features'].shape}")
                print(f"  Mids shape: {stock_data['mids'].shape}")
                print(f"  Pass filter: {stock_data['metadata'].get('pass_filter', 'N/A')}")

                if 'label_preview' in stock_data['metadata']:
                    lp = stock_data['metadata']['label_preview']
                    if lp:
                        print(f"  Label preview:")
                        print(f"    Total: {lp['total_labels']}")
                        print(f"    Down: {lp['down_pct']:.1%}")
                        print(f"    Neutral: {lp['neutral_pct']:.1%}")
                        print(f"    Up: {lp['up_pct']:.1%}")
            except Exception as e:
                print(f"  錯誤: {e}")

    # 測試 3: 獲取標籤統計
    if 'dates' in locals() and len(dir_info['dates']) > 0:
        test_date = dir_info['dates'][0]

        print(f"\n{'='*70}")
        print(f"測試 3: 標籤預覽統計 ({test_date})")
        print("="*70)

        try:
            stats = get_label_preview_stats(preprocessed_dir, test_date)
            print(f"總股票數: {stats['total_stocks']}")
            print(f"有標籤的股票: {stats['stocks_with_labels']}")

            if stats['overall_dist']:
                od = stats['overall_dist']
                print(f"\n整體標籤分布:")
                print(f"  Down: {od['down_pct']:.1%}")
                print(f"  Neutral: {od['neutral_pct']:.1%}")
                print(f"  Up: {od['up_pct']:.1%}")
                print(f"  Total: {od['total_labels']:,}")

            # 顯示前 5 檔股票
            print(f"\n前 5 檔股票（按標籤數排序）:")
            top_stocks = get_stock_list_for_date(preprocessed_dir, test_date)[:5]
            for symbol, lp in top_stocks:
                if lp:
                    print(f"  {symbol}: {lp['total_labels']:,} 標籤 "
                          f"(Down {lp['down_pct']:.1%} | "
                          f"Neutral {lp['neutral_pct']:.1%} | "
                          f"Up {lp['up_pct']:.1%})")

        except Exception as e:
            print(f"錯誤: {e}")

    print(f"\n{'='*70}")
    print(get_cache_info())
    print("="*70)
