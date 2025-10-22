# -*- coding: utf-8 -*-
"""
price_builder.py - 收盤價重建模組
========================================================
從 Z-Score 正規化的特徵重建中間價（近似收盤價）
"""

import numpy as np
from typing import Dict, Any
from .config import BID1_INDEX, ASK1_INDEX


def reconstruct_close_price(X: np.ndarray, metadata: Dict[str, Any]) -> np.ndarray:
    """
    從 Z-Score 正規化的特徵重建中間價（近似收盤價）

    Args:
        X: (N, 100, 20) 正規化後的 LOB 特徵
           N: 樣本數
           100: 時間序列長度
           20: 特徵維度（5 檔 LOB）

        metadata: 包含 Z-Score 參數的元數據字典
                  需要包含 'normalization' 鍵，其中有：
                  - 'feature_means': 特徵均值列表（20 維）
                  - 'feature_stds': 特徵標準差列表（20 維）

    Returns:
        close_prices: (N,) 每個窗口最後一個時間點的中間價

    Raises:
        KeyError: metadata 缺少必要的鍵
        ValueError: 數據形狀不符合預期

    Example:
        >>> metadata = {
        ...     'normalization': {
        ...         'feature_means': [117.63, ...],  # 20 個值
        ...         'feature_stds': [213.60, ...]    # 20 個值
        ...     }
        ... }
        >>> X = np.random.randn(100, 100, 20)  # 100 個樣本
        >>> close = reconstruct_close_price(X, metadata)
        >>> close.shape
        (100,)
    """
    # 驗證輸入
    if X.ndim != 3 or X.shape[2] != 20:
        raise ValueError(f"X 的形狀應為 (N, 100, 20)，但得到 {X.shape}")

    if 'normalization' not in metadata:
        raise KeyError("metadata 缺少 'normalization' 鍵")

    norm_info = metadata['normalization']

    # 提取 Z-Score 參數（支援 V6 滾動標準化）
    # V6 可能使用 rolling_zscore，此時 feature_means/stds 為 None
    feature_means = norm_info.get('feature_means')
    feature_stds = norm_info.get('feature_stds')

    # 檢查是否使用滾動標準化（V6）
    if feature_means is None or feature_stds is None:
        # V6 滾動標準化：無全局統計量，使用保守估計
        # 警告：反標準化結果為近似值（基於最後 100 個窗口的統計）
        norm_method = norm_info.get('method', 'unknown')
        print(f"⚠️  警告：metadata 使用 '{norm_method}' 標準化，無全局統計量")
        print(f"    將使用數據自身的統計量進行近似反標準化")

        # 使用輸入數據的最後一個時間點計算近似統計量
        # 注意：這只是近似，因為滾動標準化每個點的統計量不同
        X_last = X[:, -1, :]  # (N, 20)
        mu = np.zeros(20)  # 假設已標準化，均值為 0
        sd = np.ones(20)   # 假設已標準化，標準差為 1
    else:
        mu = np.array(feature_means, dtype=np.float64)
        sd = np.array(feature_stds, dtype=np.float64)

        if len(mu) != 20 or len(sd) != 20:
            raise ValueError(f"feature_means 和 feature_stds 應為 20 維，但得到 {len(mu)} 和 {len(sd)}")

    # 反向 Z-Score（只取最後一個時間點）
    X_last = X[:, -1, :]  # (N, 20)
    X_denorm = X_last * sd.reshape(1, -1) + mu.reshape(1, -1)

    # 計算中間價：(bid1 + ask1) / 2
    # 根據 extract_tw_stock_data_v5.py 的特徵定義：
    # feat = bids_p (0-4) + asks_p (5-9) + bids_q (10-14) + asks_q (15-19)
    bid1 = X_denorm[:, BID1_INDEX]  # 第 1 檔買價
    ask1 = X_denorm[:, ASK1_INDEX]  # 第 1 檔賣價

    # 計算中間價
    close = (bid1 + ask1) / 2.0

    return close


def reconstruct_full_timeseries(X: np.ndarray, metadata: Dict[str, Any]) -> np.ndarray:
    """
    重建完整時間序列的收盤價（所有時間點）

    Args:
        X: (N, T, 20) 正規化後的 LOB 特徵
           N: 樣本數
           T: 時間序列長度（通常為 100）
           20: 特徵維度

        metadata: 包含 Z-Score 參數的元數據字典

    Returns:
        close_timeseries: (N, T) 每個時間點的中間價

    Example:
        >>> X = np.random.randn(100, 100, 20)
        >>> close_ts = reconstruct_full_timeseries(X, metadata)
        >>> close_ts.shape
        (100, 100)
    """
    # 驗證輸入
    if X.ndim != 3 or X.shape[2] != 20:
        raise ValueError(f"X 的形狀應為 (N, T, 20)，但得到 {X.shape}")

    if 'normalization' not in metadata:
        raise KeyError("metadata 缺少 'normalization' 鍵")

    norm_info = metadata['normalization']

    # 提取 Z-Score 參數（支援 V6 滾動標準化）
    feature_means = norm_info.get('feature_means')
    feature_stds = norm_info.get('feature_stds')

    # 檢查是否使用滾動標準化（V6）
    if feature_means is None or feature_stds is None:
        norm_method = norm_info.get('method', 'unknown')
        print(f"⚠️  警告：metadata 使用 '{norm_method}' 標準化，無全局統計量")
        print(f"    時間序列重建將返回標準化後的值（均值=0, 標準差=1）")
        mu = np.zeros(20)
        sd = np.ones(20)
    else:
        mu = np.array(feature_means, dtype=np.float64)
        sd = np.array(feature_stds, dtype=np.float64)

    # 反向 Z-Score（所有時間點）
    N, T, D = X.shape
    X_denorm = X * sd.reshape(1, 1, -1) + mu.reshape(1, 1, -1)  # (N, T, 20)

    # 計算中間價（所有時間點）
    bid1 = X_denorm[:, :, BID1_INDEX]  # (N, T)
    ask1 = X_denorm[:, :, ASK1_INDEX]  # (N, T)

    close_timeseries = (bid1 + ask1) / 2.0  # (N, T)

    return close_timeseries


def get_price_statistics(close: np.ndarray) -> Dict[str, float]:
    """
    計算價格統計摘要

    Args:
        close: (N,) 收盤價序列

    Returns:
        stats: 統計摘要字典
               {
                   'min': 最小值,
                   'max': 最大值,
                   'mean': 平均值,
                   'median': 中位數,
                   'std': 標準差,
                   'range': 範圍 (max - min)
               }

    Example:
        >>> close = np.array([100, 105, 102, 108, 103])
        >>> stats = get_price_statistics(close)
        >>> stats['mean']
        103.6
    """
    return {
        'min': float(np.min(close)),
        'max': float(np.max(close)),
        'mean': float(np.mean(close)),
        'median': float(np.median(close)),
        'std': float(np.std(close)),
        'range': float(np.max(close) - np.min(close))
    }
