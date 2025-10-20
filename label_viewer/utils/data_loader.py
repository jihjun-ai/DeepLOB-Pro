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


def scan_stock_files(data_dir: str) -> List[str]:
    """
    掃描數據目錄，找出所有 NPZ 文件

    Args:
        data_dir: 數據目錄路徑（例如：data/processed_v5/npz）

    Returns:
        股票 ID 列表（例如：['2330', '2317', ...]）

    Example:
        >>> stock_ids = scan_stock_files('data/processed_v5/npz')
        >>> print(f'找到 {len(stock_ids)} 檔股票')
    """
    data_path = Path(data_dir)

    if not data_path.exists():
        raise FileNotFoundError(f"數據目錄不存在: {data_dir}")

    # 尋找所有 stock_embedding_*.npz 文件
    npz_files = list(data_path.glob("stock_embedding_*.npz"))

    if not npz_files:
        raise ValueError(f"目錄中沒有找到 NPZ 文件: {data_dir}")

    # 提取股票 ID（從 stock_embedding_2330.npz 提取 2330）
    stock_ids = []
    for file_path in npz_files:
        file_name = file_path.stem  # stock_embedding_2330
        stock_id = file_name.replace("stock_embedding_", "")
        stock_ids.append(stock_id)

    return sorted(stock_ids)


@functools.lru_cache(maxsize=3)
def load_split_data(data_dir: str, split: str) -> Dict[str, np.ndarray]:
    """
    載入指定數據集的所有股票數據（使用 LRU Cache）

    Args:
        data_dir: 數據目錄路徑
        split: 數據集類型 ('train', 'val', 'test')

    Returns:
        字典，key=股票ID, value=數據字典（包含 features, labels, weights, metadata）

    Raises:
        ValueError: 如果 split 參數無效
        FileNotFoundError: 如果目錄不存在

    Example:
        >>> data = load_split_data('data/processed_v5/npz', 'train')
        >>> print(f"載入 {len(data)} 檔股票的訓練集數據")

    Notes:
        - 使用 LRU Cache (maxsize=3) 快取最近訪問的 3 個數據集
        - 典型快取：train, val, test 各一個
        - 快取命中時，載入時間從 5 秒降至 < 0.5 秒
    """
    # 驗證參數
    valid_splits = ['train', 'val', 'test']
    if split not in valid_splits:
        raise ValueError(f"無效的 split 參數: {split}，必須是 {valid_splits} 之一")

    # 掃描股票文件
    stock_ids = scan_stock_files(data_dir)

    # 載入所有股票的數據
    all_data = {}

    for stock_id in stock_ids:
        file_path = Path(data_dir) / f"stock_embedding_{stock_id}.npz"

        try:
            # 載入 NPZ 文件
            npz_data = np.load(file_path, allow_pickle=True)

            # 提取指定 split 的數據
            feat_key = f'feat_{split}'
            label_key = f'label_{split}'
            weight_key = f'weight_{split}'

            # 檢查必要的鍵是否存在
            if feat_key not in npz_data:
                print(f"警告：{stock_id} 缺少 {feat_key}，跳過")
                continue

            # 組織數據
            stock_data = {
                'features': npz_data[feat_key],      # (N, 100, 20)
                'labels': npz_data[label_key],       # (N,)
                'weights': npz_data[weight_key],     # (N,)
                'metadata': npz_data['metadata'].item() if 'metadata' in npz_data else {},
                'stock_id': stock_id,
                'n_samples': len(npz_data[label_key])
            }

            all_data[stock_id] = stock_data

        except Exception as e:
            print(f"警告：載入 {stock_id} 時發生錯誤: {e}，跳過")
            continue

    if not all_data:
        raise ValueError(f"未能載入任何股票數據（split={split}）")

    print(f"成功載入 {len(all_data)} 檔股票的 {split} 數據集")
    return all_data


def get_stock_list(data_dir: str, split: str, top_n: Optional[int] = None) -> List[Tuple[str, int]]:
    """
    獲取股票列表（按樣本數降序排序）

    Args:
        data_dir: 數據目錄路徑
        split: 數據集類型 ('train', 'val', 'test')
        top_n: 返回前 N 檔股票（None=全部）

    Returns:
        列表，每個元素為 (股票ID, 樣本數)，按樣本數降序排序

    Example:
        >>> stocks = get_stock_list('data/processed_v5/npz', 'train', top_n=10)
        >>> for stock_id, n_samples in stocks:
        ...     print(f"{stock_id}: {n_samples} 樣本")
    """
    # 載入數據（利用 LRU Cache）
    all_data = load_split_data(data_dir, split)

    # 提取股票列表與樣本數
    stock_list = [(stock_id, data['n_samples']) for stock_id, data in all_data.items()]

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
    all_data = load_split_data(data_dir, split)

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
    load_split_data.cache_clear()


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
    cache_info = load_split_data.cache_info()
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
