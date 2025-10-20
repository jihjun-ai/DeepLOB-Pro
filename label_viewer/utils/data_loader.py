"""
數據載入模組

功能：
1. 載入 NPZ 格式的股票數據
2. LRU Cache 快取機制（提升性能）
3. 生成股票列表（按樣本數排序）
4. 錯誤處理與驗證

作者：DeepLOB-Pro Team
最後更新：2025-10-20
"""

import os
import functools
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np


# v5 格式不需要掃描檔案，因為所有數據都在 stock_embedding_{split}.npz 中


@functools.lru_cache(maxsize=3)
def load_split_data_v5(data_dir: str, split: str) -> Dict[str, np.ndarray]:
    """
    載入 extract_tw_stock_data_v5.py 產生的數據（v5 格式）

    Args:
        data_dir: 數據目錄路徑
        split: 數據集類型 ('train', 'val', 'test')

    Returns:
        字典，key=股票ID, value=數據字典（包含 features, labels, weights, metadata）

    Notes:
        v5 格式：
        - 檔案名：stock_embedding_train.npz, stock_embedding_val.npz, stock_embedding_test.npz
        - 鍵名：X (特徵), y (標籤), weights (權重), stock_ids (股票ID列表)
        - X shape: (N, 100, 20)
        - y shape: (N,)
        - weights shape: (N,)
        - stock_ids: list of stock IDs
    """
    # 驗證參數
    valid_splits = ['train', 'val', 'test']
    if split not in valid_splits:
        raise ValueError(f"無效的 split 參數: {split}，必須是 {valid_splits} 之一")

    # 載入合併的 NPZ 檔案
    file_path = Path(data_dir) / f"stock_embedding_{split}.npz"

    if not file_path.exists():
        raise FileNotFoundError(f"找不到檔案: {file_path}")

    print(f"[INFO] 載入檔案: {file_path}")
    npz_data = np.load(file_path, allow_pickle=True)

    # 載入 normalization metadata
    import json
    meta_path = Path(data_dir) / "normalization_meta.json"
    if meta_path.exists():
        with open(meta_path, 'r', encoding='utf-8') as f:
            norm_meta = json.load(f)
        print(f"[INFO] 載入 normalization metadata")
    else:
        norm_meta = None
        print(f"[WARN] 找不到 normalization_meta.json，將無法重建收盤價")

    # 檢查鍵名
    expected_keys = ['X', 'y', 'weights', 'stock_ids']
    actual_keys = list(npz_data.keys())
    print(f"[DEBUG] 檔案鍵: {actual_keys}")

    # 提取數據
    X = npz_data['X']           # (N, 100, 20)
    y = npz_data['y']           # (N,)
    weights = npz_data['weights']  # (N,)
    stock_ids = npz_data['stock_ids']  # 股票ID列表

    print(f"[INFO] 數據形狀: X={X.shape}, y={y.shape}, weights={weights.shape}")
    print(f"[INFO] 股票數量: {len(np.unique(stock_ids))}")

    # 按股票ID拆分數據
    all_data = {}
    unique_stocks = np.unique(stock_ids)

    for stock_id in unique_stocks:
        # 找出屬於這個股票的所有樣本
        mask = stock_ids == stock_id

        stock_data = {
            'features': X[mask],       # (N_stock, 100, 20)
            'labels': y[mask],         # (N_stock,)
            'weights': weights[mask],  # (N_stock,)
            'metadata': norm_meta if norm_meta else {},  # 使用共用的 normalization metadata
            'stock_id': str(stock_id),
            'n_samples': int(np.sum(mask))
        }

        all_data[str(stock_id)] = stock_data

    print(f"✅ 成功載入 {len(all_data)} 檔股票的 {split} 數據集")
    return all_data


def get_stock_list(data_dir: str, split: str, top_n: Optional[int] = None, sort_by: str = 'code') -> List[Tuple[str, int]]:
    """
    獲取股票列表

    Args:
        data_dir: 數據目錄路徑
        split: 數據集類型 ('train', 'val', 'test')
        top_n: 返回前 N 檔股票（None=全部）
        sort_by: 排序方式 ('code'=股票代碼, 'samples'=樣本數)

    Returns:
        列表，每個元素為 (股票ID, 樣本數)

    Example:
        >>> stocks = get_stock_list('data/processed_v5/npz', 'train', sort_by='code')
        >>> for stock_id, n_samples in stocks:
        ...     print(f"{stock_id}: {n_samples} 樣本")
    """
    # 載入數據（利用 LRU Cache）
    all_data = load_split_data_v5(data_dir, split)

    # 提取股票列表與樣本數
    stock_list = [(stock_id, data['n_samples']) for stock_id, data in all_data.items()]

    # 排序
    if sort_by == 'code':
        # 按股票代碼排序（數字優先，然後字母）
        stock_list.sort(key=lambda x: (int(x[0]) if x[0].isdigit() else float('inf'), x[0]))
    else:
        # 按樣本數降序排序
        stock_list.sort(key=lambda x: x[1], reverse=True)

    # 返回前 N 個
    if top_n is not None:
        stock_list = stock_list[:top_n]

    return stock_list


def load_stock_data(data_dir: str, stock_id: str, split: str) -> Dict[str, np.ndarray]:
    """
    載入單一股票的數據

    Args:
        data_dir: 數據目錄路徑
        stock_id: 股票 ID（例如：'2330'）
        split: 數據集類型 ('train', 'val', 'test')

    Returns:
        數據字典，包含：
        - features: (N, 100, 20) - LOB 特徵
        - labels: (N,) - 標籤 (0/1/2)
        - weights: (N,) - 樣本權重
        - metadata: 元數據字典
        - stock_id: 股票 ID
        - n_samples: 樣本數

    Raises:
        KeyError: 如果股票不存在

    Example:
        >>> data = load_stock_data('data/processed_v5/npz', '2330', 'train')
        >>> print(f"股票 {data['stock_id']} 有 {data['n_samples']} 個訓練樣本")
    """
    # 載入所有數據（利用 LRU Cache）
    all_data = load_split_data_v5(data_dir, split)

    # 提取指定股票
    if stock_id not in all_data:
        raise KeyError(f"股票 {stock_id} 不存在於 {split} 數據集中")

    return all_data[stock_id]


def clear_cache():
    """
    清除 LRU Cache（釋放記憶體）

    Example:
        >>> clear_cache()
        >>> print("快取已清除")
    """
    load_split_data_v5.cache_clear()


def get_cache_info() -> dict:
    """
    獲取 LRU Cache 資訊

    Returns:
        字典，包含：
        - hits: 快取命中次數
        - misses: 快取未命中次數
        - maxsize: 快取最大容量
        - currsize: 快取當前大小

    Example:
        >>> info = get_cache_info()
        >>> print(f"快取命中率: {info['hits'] / (info['hits'] + info['misses']):.1%}")
    """
    cache_info = load_split_data_v5.cache_info()
    return {
        'hits': cache_info.hits,
        'misses': cache_info.misses,
        'maxsize': cache_info.maxsize,
        'currsize': cache_info.currsize
    }


if __name__ == "__main__":
    """
    測試代碼（可選）
    """
    # 測試參數
    TEST_DATA_DIR = "../../data/processed_v5/npz"
    TEST_SPLIT = "train"

    # 測試 1: 掃描股票文件
    print("=" * 60)
    print("測試 1: 掃描股票文件")
    print("=" * 60)
    try:
        stock_ids = scan_stock_files(TEST_DATA_DIR)
        print(f"✅ 找到 {len(stock_ids)} 檔股票")
        print(f"前 10 檔: {stock_ids[:10]}")
    except Exception as e:
        print(f"❌ 錯誤: {e}")

    # 測試 2: 載入數據集
    print("\n" + "=" * 60)
    print("測試 2: 載入數據集（使用 LRU Cache）")
    print("=" * 60)
    try:
        data = load_split_data(TEST_DATA_DIR, TEST_SPLIT)
        print(f"✅ 成功載入 {len(data)} 檔股票的 {TEST_SPLIT} 數據")

        # 顯示第一檔股票的資訊
        first_stock_id = list(data.keys())[0]
        first_data = data[first_stock_id]
        print(f"\n範例：股票 {first_stock_id}")
        print(f"  - 特徵形狀: {first_data['features'].shape}")
        print(f"  - 標籤形狀: {first_data['labels'].shape}")
        print(f"  - 權重形狀: {first_data['weights'].shape}")
        print(f"  - 樣本數: {first_data['n_samples']}")
    except Exception as e:
        print(f"❌ 錯誤: {e}")

    # 測試 3: 獲取股票列表
    print("\n" + "=" * 60)
    print("測試 3: 獲取股票列表（按樣本數排序）")
    print("=" * 60)
    try:
        stock_list = get_stock_list(TEST_DATA_DIR, TEST_SPLIT, top_n=10)
        print(f"✅ 前 10 檔股票（按樣本數降序）：")
        for i, (stock_id, n_samples) in enumerate(stock_list, 1):
            print(f"  {i}. {stock_id}: {n_samples:,} 樣本")
    except Exception as e:
        print(f"❌ 錯誤: {e}")

    # 測試 4: 載入單一股票
    print("\n" + "=" * 60)
    print("測試 4: 載入單一股票數據")
    print("=" * 60)
    try:
        test_stock_id = stock_list[0][0]  # 使用樣本數最多的股票
        stock_data = load_stock_data(TEST_DATA_DIR, test_stock_id, TEST_SPLIT)
        print(f"✅ 成功載入股票 {test_stock_id}")
        print(f"  - 樣本數: {stock_data['n_samples']:,}")
        print(f"  - 特徵範圍: [{stock_data['features'].min():.2f}, {stock_data['features'].max():.2f}]")
        print(f"  - 標籤分布: {np.bincount(stock_data['labels'])}")
    except Exception as e:
        print(f"❌ 錯誤: {e}")

    # 測試 5: 快取資訊
    print("\n" + "=" * 60)
    print("測試 5: LRU Cache 資訊")
    print("=" * 60)
    cache_info = get_cache_info()
    print(f"✅ 快取統計：")
    print(f"  - 命中次數: {cache_info['hits']}")
    print(f"  - 未命中次數: {cache_info['misses']}")
    print(f"  - 當前大小: {cache_info['currsize']} / {cache_info['maxsize']}")
    if cache_info['hits'] + cache_info['misses'] > 0:
        hit_rate = cache_info['hits'] / (cache_info['hits'] + cache_info['misses'])
        print(f"  - 命中率: {hit_rate:.1%}")

    print("\n" + "=" * 60)
    print("✅ 所有測試完成！")
    print("=" * 60)
