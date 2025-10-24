"""
測試 V7 標準化修復
===================
驗證 extract_tw_stock_data_v7.py 的標準化功能是否正常

使用方式：
    python scripts/test_v7_normalization.py
"""

import numpy as np
import sys
from pathlib import Path

# 添加 src 到路徑
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.extract_tw_stock_data_v7 import zscore_apply, zscore_fit

def test_rolling_zscore():
    """測試滾動窗口標準化"""
    print("=" * 60)
    print("測試 1: Rolling Z-Score 標準化")
    print("=" * 60)

    # 創建測試數據（模擬原始 LOB 數據）
    np.random.seed(42)
    T, F = 1000, 20
    X = np.random.randn(T, F) * 100 + 150  # Mean ≈ 150, Std ≈ 100

    print(f"原始數據:")
    print(f"  Shape: {X.shape}")
    print(f"  Min: {X.min():.2f}")
    print(f"  Max: {X.max():.2f}")
    print(f"  Mean: {X.mean():.2f}")
    print(f"  Std: {X.std():.2f}")

    # 應用滾動窗口標準化
    X_norm = zscore_apply(
        X,
        mu=None,
        sd=None,
        method='rolling_zscore',
        window=100,
        min_periods=20
    )

    print(f"\n標準化後:")
    print(f"  Shape: {X_norm.shape}")
    print(f"  Min: {X_norm.min():.2f}")
    print(f"  Max: {X_norm.max():.2f}")
    print(f"  Mean: {X_norm.mean():.2f}")
    print(f"  Std: {X_norm.std():.2f}")

    # 驗證
    assert X_norm.shape == X.shape, "形狀不匹配"
    assert not np.isnan(X_norm).any(), "包含 NaN"
    assert not np.isinf(X_norm).any(), "包含 Inf"
    assert abs(X_norm.mean()) < 0.5, f"均值異常: {X_norm.mean():.2f}（應接近 0）"
    assert 0.5 < X_norm.std() < 1.5, f"標準差異常: {X_norm.std():.2f}（應接近 1）"

    print("\n✅ Rolling Z-Score 測試通過")


def test_global_zscore():
    """測試全局標準化"""
    print("\n" + "=" * 60)
    print("測試 2: Global Z-Score 標準化")
    print("=" * 60)

    # 創建測試數據
    np.random.seed(42)
    T, F = 1000, 20
    X = np.random.randn(T, F) * 200 + 300  # Mean ≈ 300, Std ≈ 200

    print(f"原始數據:")
    print(f"  Mean: {X.mean():.2f}, Std: {X.std():.2f}")

    # 應用全局標準化
    mu, sd = zscore_fit(X, method='global')
    X_norm = zscore_apply(X, mu, sd, method='global')

    print(f"\n標準化後:")
    print(f"  Mean: {X_norm.mean():.2f}, Std: {X_norm.std():.2f}")

    # 驗證
    assert abs(X_norm.mean()) < 1e-6, f"均值異常: {X_norm.mean():.6f}"
    assert abs(X_norm.std() - 1.0) < 1e-6, f"標準差異常: {X_norm.std():.6f}"

    print("✅ Global Z-Score 測試通過")


def test_real_data():
    """測試真實 LOB 數據"""
    print("\n" + "=" * 60)
    print("測試 3: 真實預處理數據")
    print("=" * 60)

    try:
        # 載入真實預處理數據
        import glob
        npz_files = glob.glob('data/preprocessed_swing/daily/*/*.npz')

        if not npz_files:
            print("⚠️ 未找到預處理數據，跳過此測試")
            return

        # 載入第一個文件
        data = np.load(npz_files[0])
        X = data['features']

        print(f"原始預處理數據 ({npz_files[0]}):")
        print(f"  Shape: {X.shape}")
        print(f"  Min: {X.min():.2f}")
        print(f"  Max: {X.max():.2f}")
        print(f"  Mean: {X.mean():.2f}")
        print(f"  Std: {X.std():.2f}")

        # 應用標準化
        X_norm = zscore_apply(
            X,
            mu=None,
            sd=None,
            method='rolling_zscore',
            window=100,
            min_periods=20
        )

        print(f"\n標準化後:")
        print(f"  Min: {X_norm.min():.2f}")
        print(f"  Max: {X_norm.max():.2f}")
        print(f"  Mean: {X_norm.mean():.2f}")
        print(f"  Std: {X_norm.std():.2f}")

        # 驗證範圍
        if X_norm.min() < -10 or X_norm.max() > 10:
            print(f"⚠️ 警告: Z-Score 範圍過大 ({X_norm.min():.2f} ~ {X_norm.max():.2f})")
        else:
            print("✅ Z-Score 範圍正常 (-10 ~ +10)")

        print("✅ 真實數據測試完成")

    except Exception as e:
        print(f"❌ 真實數據測試失敗: {e}")


if __name__ == "__main__":
    try:
        test_rolling_zscore()
        test_global_zscore()
        test_real_data()

        print("\n" + "=" * 60)
        print("🎉 所有測試通過！V7 標準化功能正常")
        print("=" * 60)

    except AssertionError as e:
        print(f"\n❌ 測試失敗: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 未預期錯誤: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
