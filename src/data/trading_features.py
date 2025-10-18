"""交易特征工程模块

将原始 LOB 数据转换为交易员能理解的特征
帮助模型"看懂"上涨/下跌/持平信号

核心思想:
    不只给模型原始价格/数量，而是给它"交易信号"
    例如: 买卖压力、订单失衡、价格动能等
"""

import numpy as np
from typing import Tuple


class TradingFeatureEngineer:
    """交易特征工程器

    将 5 档 LOB 的 20 维原始特征扩展为包含交易含义的特征
    """

    def __init__(self, use_advanced_features: bool = True):
        """初始化

        Args:
            use_advanced_features: 是否使用高级特征
        """
        self.use_advanced_features = use_advanced_features

    def transform(self, lob_data: np.ndarray) -> np.ndarray:
        """转换 LOB 数据为交易特征

        Args:
            lob_data: (N, T, 20) 原始 LOB 数据
                前 5 列: Ask Price [1-5]
                6-10 列: Ask Volume [1-5]
                11-15 列: Bid Price [1-5]
                16-20 列: Bid Volume [1-5]

        Returns:
            features: (N, T, F) 交易特征
                F = 20 (原始) + 额外特征
        """
        N, T, _ = lob_data.shape

        # 分离价格和数量
        ask_prices = lob_data[:, :, 0:5]      # (N, T, 5)
        ask_volumes = lob_data[:, :, 5:10]    # (N, T, 5)
        bid_prices = lob_data[:, :, 10:15]    # (N, T, 5)
        bid_volumes = lob_data[:, :, 15:20]   # (N, T, 5)

        features = [lob_data]  # 保留原始特征

        # ========== 核心交易特征 (必须) ==========

        # 1. 买卖压力 (Order Imbalance) - 最重要！
        # 理念: 买盘强 → 上涨，卖盘强 → 下跌
        total_ask_vol = ask_volumes.sum(axis=2, keepdims=True)  # (N, T, 1)
        total_bid_vol = bid_volumes.sum(axis=2, keepdims=True)

        imbalance = (total_bid_vol - total_ask_vol) / (total_bid_vol + total_ask_vol + 1e-10)
        features.append(imbalance)  # (N, T, 1)

        # 2. 价差 (Spread)
        # 理念: 价差缩小 → 流动性好，价差扩大 → 不确定性
        mid_price = (ask_prices[:, :, 0:1] + bid_prices[:, :, 0:1]) / 2
        spread = ask_prices[:, :, 0:1] - bid_prices[:, :, 0:1]
        spread_ratio = spread / (mid_price + 1e-10)
        features.append(spread_ratio)  # (N, T, 1)

        # 3. 价格动能 (Price Momentum)
        # 理念: 价格持续上涨/下跌的趋势
        price_change = np.diff(mid_price, axis=1, prepend=mid_price[:, 0:1, :])
        price_momentum = price_change / (mid_price + 1e-10)
        features.append(price_momentum)  # (N, T, 1)

        # 4. 成交量动能 (Volume Momentum)
        # 理念: 量能放大配合价格变动 → 强信号
        total_vol = total_ask_vol + total_bid_vol
        vol_change = np.diff(total_vol, axis=1, prepend=total_vol[:, 0:1, :])
        vol_momentum = vol_change / (total_vol + 1e-10)
        features.append(vol_momentum)  # (N, T, 1)

        # 5. 第一档买卖比 (First Level Imbalance)
        # 理念: 第一档最接近成交，最重要
        first_imbalance = (bid_volumes[:, :, 0:1] - ask_volumes[:, :, 0:1]) / \
                         (bid_volumes[:, :, 0:1] + ask_volumes[:, :, 0:1] + 1e-10)
        features.append(first_imbalance)  # (N, T, 1)

        # ========== 高级交易特征 (可选) ==========
        if self.use_advanced_features:

            # 6. 深度加权价格 (Depth Weighted Price)
            # 理念: 考虑深度的"真实"中间价
            ask_depth = (ask_volumes[:, :, 0:3] * ask_prices[:, :, 0:3]).sum(axis=2, keepdims=True)
            bid_depth = (bid_volumes[:, :, 0:3] * bid_prices[:, :, 0:3]).sum(axis=2, keepdims=True)
            total_depth_vol = ask_volumes[:, :, 0:3].sum(axis=2, keepdims=True) + \
                             bid_volumes[:, :, 0:3].sum(axis=2, keepdims=True)
            dwmp = (ask_depth + bid_depth) / (total_depth_vol + 1e-10)
            dwmp_diff = (dwmp - mid_price) / (mid_price + 1e-10)
            features.append(dwmp_diff)  # (N, T, 1)

            # 7. 买卖量分布 (Volume Distribution)
            # 理念: 量能集中在哪几档
            ask_vol_ratio_1 = ask_volumes[:, :, 0:1] / (total_ask_vol + 1e-10)
            bid_vol_ratio_1 = bid_volumes[:, :, 0:1] / (total_bid_vol + 1e-10)
            features.extend([ask_vol_ratio_1, bid_vol_ratio_1])  # (N, T, 2)

            # 8. 价格压力 (Price Pressure)
            # 理念: 各档价格距离中间价的程度
            ask_pressure = (ask_prices[:, :, 0:1] - mid_price) / (mid_price + 1e-10)
            bid_pressure = (mid_price - bid_prices[:, :, 0:1]) / (mid_price + 1e-10)
            features.extend([ask_pressure, bid_pressure])  # (N, T, 2)

            # 9. 流动性指标 (Liquidity Indicator)
            # 理念: 总量越大 → 流动性越好
            liquidity = np.log1p(total_vol)  # log(1+x) 避免极端值
            features.append(liquidity)  # (N, T, 1)

            # 10. 买卖方向性 (Directional Strength)
            # 理念: 综合多档的买卖力量
            weighted_bid = (bid_volumes * np.arange(5, 0, -1)).sum(axis=2, keepdims=True)
            weighted_ask = (ask_volumes * np.arange(5, 0, -1)).sum(axis=2, keepdims=True)
            directional = (weighted_bid - weighted_ask) / (weighted_bid + weighted_ask + 1e-10)
            features.append(directional)  # (N, T, 1)

        # 合并所有特征
        all_features = np.concatenate(features, axis=2)

        return all_features

    def get_feature_dim(self) -> int:
        """获取输出特征维度

        Returns:
            特征维度数
        """
        if self.use_advanced_features:
            # 20 (原始) + 1 (imbalance) + 1 (spread) + 1 (price_mom) +
            # 1 (vol_mom) + 1 (first_imb) + 1 (dwmp) + 2 (vol_ratio) +
            # 2 (pressure) + 1 (liquidity) + 1 (directional) = 32
            return 32
        else:
            # 20 (原始) + 5 (核心特征) = 25
            return 25


def add_trading_features_to_dataset(
    X: np.ndarray,
    y: np.ndarray,
    use_advanced: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """为数据集添加交易特征

    Args:
        X: (N, T, 20) 原始 LOB 数据
        y: (N,) 标签
        use_advanced: 是否使用高级特征

    Returns:
        X_enhanced: (N, T, F) 增强后的特征
        y: (N,) 标签 (不变)
    """
    engineer = TradingFeatureEngineer(use_advanced_features=use_advanced)
    X_enhanced = engineer.transform(X)

    print(f"特征维度: {X.shape[-1]} → {X_enhanced.shape[-1]}")
    print(f"  原始特征: 20 维")
    print(f"  新增特征: {X_enhanced.shape[-1] - 20} 维")

    return X_enhanced, y


if __name__ == "__main__":
    # 测试
    import sys
    from pathlib import Path

    # 加载数据
    data_path = Path(__file__).parent.parent.parent / "data" / "processed" / "stock_embedding_train.npz"
    data = np.load(data_path)
    X_train = data['X']
    y_train = data['y']

    print(f"原始数据: {X_train.shape}")

    # 转换特征
    X_enhanced, y_enhanced = add_trading_features_to_dataset(X_train, y_train, use_advanced=True)

    print(f"增强数据: {X_enhanced.shape}")
    print(f"\n特征统计:")
    print(f"  均值: {X_enhanced.mean():.4f}")
    print(f"  标准差: {X_enhanced.std():.4f}")
    print(f"  最小值: {X_enhanced.min():.4f}")
    print(f"  最大值: {X_enhanced.max():.4f}")

    # 保存增强数据
    output_path = data_path.parent / "stock_embedding_train_enhanced.npz"
    np.savez_compressed(
        output_path,
        X=X_enhanced,
        y=y_enhanced,
        stock_ids=data.get('stock_ids', None)
    )
    print(f"\n已保存增强数据至: {output_path}")
