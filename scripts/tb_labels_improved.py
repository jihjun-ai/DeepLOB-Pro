"""
改進版 Triple-Barrier 標籤生成
==================================================
針對「日沖交易」優化的標籤邏輯

主要改進：
1. PT/SL 觸發也檢查 min_return（避免微小波動被標為 ±1）
2. 可選的「終點導向」模式（固定視窗）
3. 更清晰的同時觸發處理
4. 詳細的觸發統計

版本：v1.0
日期：2025-10-21
"""

import numpy as np
import pandas as pd
from typing import Optional, Literal


def tb_labels_v2(
    close: pd.Series,
    vol: pd.Series,
    pt_mult: float = 2.0,
    sl_mult: float = 2.0,
    max_holding: int = 200,
    min_return: float = 0.002,  # 提高到 0.2%
    day_end_idx: Optional[int] = None,
    mode: Literal['path', 'endpoint'] = 'path',
    pt_sl_check_min_return: bool = True,  # 新增：PT/SL 也檢查 min_return
    tie_prefer: Literal['zero', 'up', 'down'] = 'zero'  # 同時觸發的處理
) -> pd.DataFrame:
    """
    改進版 Triple-Barrier 標籤生成

    Args:
        close: 收盤價序列
        vol: 波動率序列
        pt_mult: 止盈倍數
        sl_mult: 止損倍數
        max_holding: 最大持有期（bars）
        min_return: 最小報酬閾值（建議 0.001-0.005，即 0.1%-0.5%）
        day_end_idx: 當日最後索引（防止越日）
        mode: 'path' (路徑導向) 或 'endpoint' (終點導向)
        pt_sl_check_min_return: PT/SL 觸發時是否也檢查 min_return
        tie_prefer: 同時觸及上下界時的處理方式

    Returns:
        DataFrame 包含:
            - y: {-1, 0, 1} 標籤
            - ret: 實際收益
            - tt: 觸發時間步數
            - why: 觸發原因 {'up', 'down', 'time', 'both'}
            - up_p: 止盈價格
            - dn_p: 止損價格

    標籤邏輯（mode='path'，pt_sl_check_min_return=True）：
        1. 若先觸及 PT，且 |ret| >= min_return → y=+1
        2. 若先觸及 SL，且 |ret| >= min_return → y=-1
        3. 若 PT/SL 觸及但 |ret| < min_return → y=0（視為雜訊）
        4. 若時間到期：
           - |ret| < min_return → y=0
           - |ret| >= min_return → y=sign(ret)
        5. 若同時觸及 PT 和 SL → 依 tie_prefer 處理
    """

    n = len(close)
    results = []

    for i in range(n - 1):
        entry_price = close.iloc[i]
        entry_vol = vol.iloc[i]

        # 計算止盈止損價格
        up_barrier = entry_price * (1 + pt_mult * entry_vol)
        dn_barrier = entry_price * (1 - sl_mult * entry_vol)

        # 決定觀察窗口結束位置
        if day_end_idx is not None:
            end_idx = min(i + max_holding, day_end_idx + 1, n)
        else:
            end_idx = min(i + max_holding, n)

        if mode == 'endpoint':
            # 終點導向：直接看固定視窗後的結果
            trigger_idx = end_idx - 1
            trigger_why = 'time'
        else:
            # 路徑導向：尋找先觸及的 barrier
            triggered = False
            trigger_idx = end_idx - 1
            trigger_why = 'time'

            for j in range(i + 1, end_idx):
                future_price = close.iloc[j]

                hit_up = future_price >= up_barrier
                hit_dn = future_price <= dn_barrier

                # 同時觸及處理
                if hit_up and hit_dn:
                    trigger_idx = j
                    trigger_why = 'both'
                    triggered = True
                    break

                # 單獨觸及
                if hit_up:
                    trigger_idx = j
                    trigger_why = 'up'
                    triggered = True
                    break

                if hit_dn:
                    trigger_idx = j
                    trigger_why = 'down'
                    triggered = True
                    break

        # 計算實際收益
        exit_price = close.iloc[trigger_idx]
        ret = (exit_price - entry_price) / entry_price

        # 決定標籤（核心邏輯）
        if trigger_why == 'time':
            # 時間到期：檢查收益大小
            if np.abs(ret) < min_return:
                label = 0
            else:
                label = int(np.sign(ret))

        elif trigger_why == 'both':
            # 同時觸及：依偏好處理
            if tie_prefer == 'zero':
                label = 0
            elif tie_prefer == 'up':
                label = 1 if np.abs(ret) >= min_return else 0
            else:  # 'down'
                label = -1 if np.abs(ret) >= min_return else 0

        else:  # 'up' or 'down'
            # PT/SL 觸發
            if pt_sl_check_min_return:
                # 新邏輯：也檢查 min_return（避免微小波動）
                if np.abs(ret) < min_return:
                    label = 0  # 雖然觸及 barrier，但收益太小視為雜訊
                else:
                    label = int(np.sign(ret))
            else:
                # 舊邏輯：PT/SL 觸發一律 ±1
                label = int(np.sign(ret))

        results.append({
            'ret': ret,
            'y': label,
            'tt': trigger_idx - i,
            'why': trigger_why,
            'up_p': up_barrier,
            'dn_p': dn_barrier
        })

    # 最後一個點
    if n > 0:
        results.append({
            'ret': 0.0,
            'y': 0,
            'tt': 0,
            'why': 'time',
            'up_p': close.iloc[-1],
            'dn_p': close.iloc[-1]
        })

    return pd.DataFrame(results, index=close.index)


def label_horizon(
    close: pd.Series,
    horizon: int = 100,
    min_return: float = 0.002,
    day_end_idx: Optional[int] = None
) -> pd.DataFrame:
    """
    終點導向標籤（極簡版本 B）

    只看固定視窗後的漲跌，不考慮路徑

    Args:
        close: 收盤價序列
        horizon: 固定視窗大小
        min_return: 最小報酬閾值
        day_end_idx: 當日最後索引（防止越日）

    Returns:
        DataFrame 包含 y, ret, tt
    """
    n = len(close)
    results = []

    for i in range(n - 1):
        entry_price = close.iloc[i]

        # 決定退出位置
        if day_end_idx is not None:
            exit_idx = min(i + horizon, day_end_idx + 1, n)
        else:
            exit_idx = min(i + horizon, n)

        exit_idx = exit_idx - 1
        exit_price = close.iloc[exit_idx]

        ret = (exit_price - entry_price) / entry_price

        # 簡單邏輯
        if np.abs(ret) < min_return:
            label = 0
        else:
            label = int(np.sign(ret))

        results.append({
            'ret': ret,
            'y': label,
            'tt': exit_idx - i
        })

    # 最後一個點
    if n > 0:
        results.append({'ret': 0.0, 'y': 0, 'tt': 0})

    return pd.DataFrame(results, index=close.index)


def compare_labeling_methods(
    close: pd.Series,
    vol: pd.Series,
    config: dict
) -> pd.DataFrame:
    """
    比較不同標籤方法的結果

    用於實驗和選擇最佳方法
    """

    # 方法 1：原始 TB（PT/SL 不檢查 min_return）
    tb1 = tb_labels_v2(
        close, vol,
        pt_mult=config['pt_multiplier'],
        sl_mult=config['sl_multiplier'],
        max_holding=config['max_holding'],
        min_return=config['min_return'],
        pt_sl_check_min_return=False,
        mode='path'
    )

    # 方法 2：改進 TB（PT/SL 也檢查 min_return）
    tb2 = tb_labels_v2(
        close, vol,
        pt_mult=config['pt_multiplier'],
        sl_mult=config['sl_multiplier'],
        max_holding=config['max_holding'],
        min_return=config['min_return'],
        pt_sl_check_min_return=True,
        mode='path'
    )

    # 方法 3：終點導向
    horizon_labels = label_horizon(
        close,
        horizon=config['max_holding'],
        min_return=config['min_return']
    )

    # 統計比較
    comparison = pd.DataFrame({
        'method': ['Original TB', 'Improved TB', 'Horizon'],
        'label_-1': [
            (tb1['y'] == -1).sum(),
            (tb2['y'] == -1).sum(),
            (horizon_labels['y'] == -1).sum()
        ],
        'label_0': [
            (tb1['y'] == 0).sum(),
            (tb2['y'] == 0).sum(),
            (horizon_labels['y'] == 0).sum()
        ],
        'label_+1': [
            (tb1['y'] == 1).sum(),
            (tb2['y'] == 1).sum(),
            (horizon_labels['y'] == 1).sum()
        ]
    })

    # 計算百分比
    comparison['pct_-1'] = comparison['label_-1'] / len(close) * 100
    comparison['pct_0'] = comparison['label_0'] / len(close) * 100
    comparison['pct_+1'] = comparison['label_+1'] / len(close) * 100

    return comparison


if __name__ == '__main__':
    # 示範使用
    print("改進版 Triple-Barrier 標籤函數")
    print("=" * 60)
    print("\n使用範例：")
    print("""
    from scripts.tb_labels_improved import tb_labels_v2
    
    # 計算標籤（改進版）
    tb_df = tb_labels_v2(
        close=close,
        vol=vol,
        pt_mult=2.0,
        sl_mult=2.0,
        max_holding=200,
        min_return=0.002,  # 0.2%
        day_end_idx=len(close)-1,
        pt_sl_check_min_return=True,  # 關鍵改進
        tie_prefer='zero'
    )
    
    # 檢查標籤分布
    print(tb_df['y'].value_counts())
    print(tb_df['why'].value_counts())
    """)

