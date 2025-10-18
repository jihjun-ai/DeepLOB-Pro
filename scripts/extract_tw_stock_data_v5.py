# -*- coding: utf-8 -*-
"""
extract_tw_stock_data_v5.py - V5 Pro 专业化资料流水线
=============================================================================
【更新日期】2025-10-18
【版本说明】v5.0 - 专业级标签与波动率（triple-barrier + arch）

基于《V5_Pro_NoMLFinLab_Guide.md》实现，将 V4 的固定 k 步标签全面升级为：
  1) 专业波动率估计（EWMA / Yang-Zhang / GARCH）
  2) Triple-Barrier 事件打标（止盈/止损/到期）
  3) 样本权重（收益 × 时间衰减 × 类别平衡）

=============================================================================
核心改进（V4 → V5）
=============================================================================
✅ 波动率估计：
  - V4: 简单相对波动率 std(mid) / mean(mid)
  - V5: arch 库专业估计（EWMA / Yang-Zhang / GARCH）

✅ 标签生成：
  - V4: 固定 k 步价格变动（alpha=0.002 阈值）
  - V5: Triple-Barrier 自适应标签（基于波动率倍数）

✅ 样本权重：
  - V4: 无权重（或简单过滤）
  - V5: 收益加权 + 时间衰减 + 类别平衡

✅ 输出格式：
  - V4: X (N,100,20), y {0,1,2}
  - V5: X (N,100,20), y {0,1,2}, w (N,) + 详细 metadata

=============================================================================
使用方式
=============================================================================

【基本使用】（使用预设配置）：
  python scripts/extract_tw_stock_data_v5.py \
      --input-dir ./data/temp \
      --output-dir ./data/processed_v5 \
      --config configs/config_pro_v5.yaml

【输出结果】：
  1. NPZ 文件（新增样本权重）：
     - ./data/processed_v5/npz/stock_embedding_train.npz
       内含：X, y, stock_ids, weights (新增)
     - ./data/processed_v5/npz/stock_embedding_val.npz
     - ./data/processed_v5/npz/stock_embedding_test.npz

  2. Metadata（包含完整 V5 配置）：
     - ./data/processed_v5/npz/normalization_meta.json
       {
         "version": "5.0.0",
         "volatility_method": "ewma",
         "triple_barrier": {...},
         "sample_weights": {...},
         "label_distribution": {...}
       }

=============================================================================
固定设定（继承自 V4）
=============================================================================
- NoAuction：移除 IsTrialMatch=='1'，并限制 09:00:00–13:30:00
- 10 事件聚合（不重叠）：价格/数量皆取视窗末端
- 标准化：Z-Score（基于训练集）
- 时间序列窗口：100 timesteps
- 输出维度：20 维 LOB 特征（5 档 LOB）

=============================================================================
依赖套件（新增）
=============================================================================
pip install triple-barrier arch ruamel.yaml pandas scikit-learn

【可选】：
pip install skfolio vectorbt nautilus-trader  # 用于后续 CV/回测/条形聚合

版本：v5.0
更新：2025-10-18
"""

import os
import re
import json
import argparse
import glob
import logging
import traceback
import warnings
from collections import defaultdict
from datetime import datetime
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import pandas as pd
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils.yaml_manager import YAMLManager

# 设定版本号
VERSION = "5.0.0"

# 设定日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# 忽略部分第三方库警告
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

# 固定常数（继承自 V4）
AGG_FACTOR = 10
SEQ_LEN = 100
TRADING_START = 90000   # 09:00:00
TRADING_END   = 133000  # 13:30:00

# 欄位索引（輸入）
IDX_REF = 3
IDX_UPPER = 4
IDX_LOWER = 5
IDX_LASTPRICE = 9
IDX_LASTVOL = 10
IDX_TV = 11
IDX_TIME = 32
IDX_TRIAL = 33

# 五檔價量索引對應
BID_P_IDX = [12, 14, 16, 18, 20]
BID_Q_IDX = [13, 15, 17, 19, 21]
ASK_P_IDX = [22, 24, 26, 28, 30]
ASK_Q_IDX = [23, 25, 27, 29, 31]

# 全域統計變數
global_stats = {
    "total_raw_events": 0,
    "cleaned_events": 0,
    "aggregated_points": 0,
    "valid_windows": 0,
    "tb_success": 0
}


# ============================================================
# V5 核心模块：专业波动率估计（arch）
# ============================================================

def ewma_vol(close: pd.Series, halflife: int = 60) -> pd.Series:
    """
    EWMA 波动率估计

    Args:
        close: 收盘价序列
        halflife: EWMA 半衰期（bars）

    Returns:
        vol: 波动率序列
    """
    ret = np.log(close).diff()
    var = ret.ewm(halflife=halflife, adjust=False).var(ddof=1)
    vol = np.sqrt(var).fillna(method="bfill")
    return vol


def yang_zhang_vol(ohlc: pd.DataFrame, window: int = 60) -> pd.Series:
    """
    Yang-Zhang 波动率估计（利用 OHLC）

    Args:
        ohlc: 包含 open, high, low, close 的 DataFrame
        window: 滚动窗口大小

    Returns:
        vol: Yang-Zhang 波动率序列
    """
    o = ohlc['open'].astype(float)
    h = ohlc['high'].astype(float)
    l = ohlc['low'].astype(float)
    c = ohlc['close'].astype(float)

    k = 0.34 / (1.34 + (window + 1) / (window - 1))

    log_oc = np.log(o / c.shift(1))
    log_co = np.log(c / o)
    rs = (np.log(h / l)) ** 2

    sigma_o = log_oc.rolling(window).var()
    sigma_c = log_co.rolling(window).var()
    sigma_rs = rs.rolling(window).mean()

    yz = sigma_o + k * sigma_c + (1 - k) * sigma_rs
    vol = np.sqrt(yz).replace([np.inf, -np.inf], np.nan).fillna(method="bfill")

    return vol


def garch11_vol(close: pd.Series) -> pd.Series:
    """
    GARCH(1,1) 波动率估计

    Args:
        close: 收盘价序列

    Returns:
        vol: GARCH 波动率序列
    """
    try:
        from arch import arch_model

        ret = 100 * np.log(close).diff().dropna()

        if len(ret) < 50:
            logging.warning("GARCH: 资料点不足，回退到 EWMA")
            return ewma_vol(close, halflife=60)

        am = arch_model(ret, vol='GARCH', p=1, q=1, dist='normal')
        res = am.fit(disp='off', show_warning=False)
        fcast = res.forecast(horizon=1, reindex=True).variance
        vol = np.sqrt(fcast.squeeze()) / 100.0

        return vol.reindex(close.index).fillna(method="bfill")

    except Exception as e:
        logging.warning(f"GARCH 失败: {e}，回退到 EWMA")
        return ewma_vol(close, halflife=60)


# ============================================================
# V5 核心模块：Triple-Barrier 标签生成
# ============================================================

def tb_labels(close: pd.Series,
              vol: pd.Series,
              pt_mult: float = 2.0,
              sl_mult: float = 2.0,
              max_holding: int = 200,
              min_return: float = 0.0001) -> pd.DataFrame:
    """
    Triple-Barrier 标签生成（使用 triple-barrier PyPI 包）

    Args:
        close: 收盘价序列
        vol: 波动率序列
        pt_mult: 止盈倍数
        sl_mult: 止损倍数
        max_holding: 最大持有期（bars）
        min_return: 最小报酬阈值

    Returns:
        DataFrame 包含:
            - y: {-1, 0, 1} 标签
            - ret: 实际收益
            - tt: 触发时间步数
            - why: 触发原因 {'up', 'down', 'time'}
            - up_p: 止盈价格
            - dn_p: 止损价格
    """
    try:
        from triple_barrier import triple_barrier as tb

        events = tb.get_events(
            close=close.values,
            daily_vol=vol.values,
            tp_multiplier=pt_mult,
            sl_multiplier=sl_mult,
            max_holding=max_holding
        )

        bins = tb.get_bins(close.values, events)

        out = pd.DataFrame(index=close.index[:len(bins)])
        out["ret"] = bins["ret"].astype(float)

        # 应用最小报酬阈值
        out["y"] = np.where(
            np.abs(out["ret"]) < min_return,
            0,  # 持平
            np.sign(out["ret"])  # -1 或 1
        ).astype(int)

        out["tt"] = events["t1"]

        # 映射触发原因
        m = {"tp": "up", "sl": "down", "t1": "time"}
        out["why"] = pd.Series(events["label"]).map(m).values

        # 计算止盈止损价格
        out["up_p"] = close.iloc[:len(bins)] * (1 + pt_mult * vol.iloc[:len(bins)])
        out["dn_p"] = close.iloc[:len(bins)] * (1 - sl_mult * vol.iloc[:len(bins)])

        global_stats["tb_success"] += 1

        return out.dropna()

    except Exception as e:
        logging.error(f"Triple-Barrier 失败: {e}")
        raise


# ============================================================
# V5 核心模块：样本权重计算
# ============================================================

def make_sample_weight(ret: pd.Series,
                      tt: pd.Series,
                      y: pd.Series,
                      tau: float = 100.0,
                      scale: float = 10.0,
                      balance: bool = True) -> pd.Series:
    """
    样本权重计算（收益 × 时间衰减 × 类别平衡）

    Args:
        ret: 实际收益序列
        tt: 触发时间步数
        y: 标签序列
        tau: 时间衰减参数
        scale: 收益缩放系数
        balance: 是否启用类别平衡

    Returns:
        w: 样本权重序列（归一化到均值为 1）
    """
    from sklearn.utils.class_weight import compute_class_weight

    # 基础权重：|收益| × 时间衰减
    base = np.abs(ret.values) * scale * np.exp(-tt.values / float(tau))
    base = np.clip(base, 1e-3, None)

    # 类别平衡
    if balance:
        classes = np.array(sorted(y.unique()))
        cls_w = compute_class_weight('balanced', classes=classes, y=y.values)
        w_map = dict(zip(classes, cls_w))
        cw = y.map(w_map).values
        w = base * cw
    else:
        w = base

    # 归一化（均值为 1）
    w = w / np.mean(w)

    return pd.Series(w, index=y.index)


# ============================================================
# V4 兼容函数（继承）
# ============================================================

def parse_args():
    """解析命令列参数"""
    p = argparse.ArgumentParser(
        "extract_tw_stock_data_v5",
        description="台股 DeepLOB 资料前处理工具 (V5 Pro 版本)"
    )
    p.add_argument(
        "--input-dir",
        default="./data/temp",
        type=str,
        help="含每日原始 .txt 的资料夹"
    )
    p.add_argument(
        "--output-dir",
        default="./data/processed_v5",
        type=str,
        help="输出资料夹"
    )
    p.add_argument(
        "--config",
        default="./configs/config_pro_v5.yaml",
        type=str,
        help="V5 配置文件路径"
    )
    p.add_argument(
        "--make-npz",
        action="store_true",
        default=True,
        help="输出 70/15/15 的 .npz 文件"
    )
    return p.parse_args()


def load_config(config_path: str) -> Dict[str, Any]:
    """载入 V5 配置文件（使用 YAMLManager）

    Args:
        config_path: 配置文件路径

    Returns:
        配置字典

    Raises:
        FileNotFoundError: 配置文件不存在时抛出异常
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"配置文件不存在: {config_path}\n"
            f"请确保配置文件存在，或使用 --config 参数指定正确的配置文件路径。\n"
            f"示例配置文件位于: configs/config_pro_v5.yaml"
        )

    # 使用 YAMLManager 加载配置文件（会自动检查文件存在性）
    yaml_manager = YAMLManager(config_path)
    cfg = yaml_manager.as_dict()

    logging.info(f"已载入配置文件: {config_path}")
    return cfg


def hhmmss_to_int(s: str) -> int:
    s = s.strip()
    if not s.isdigit():
        return -1
    return int(s)


def to_float(x: str, default=0.0) -> float:
    try:
        return float(x)
    except:
        return default


def is_in_trading_window(t: int) -> bool:
    return TRADING_START <= t <= TRADING_END


def spread_ok(bid1: float, ask1: float) -> bool:
    if bid1 <= 0 or ask1 <= 0:
        return False
    if bid1 >= ask1:
        return False
    mid = 0.5 * (bid1 + ask1)
    return (ask1 - bid1) / max(mid, 1e-12) <= 0.05


def within_limits(px: float, lo: float, hi: float) -> bool:
    if lo > 0 and px < lo - 1e-12:
        return False
    if hi > 0 and px > hi + 1e-12:
        return False
    return True


def parse_line(raw: str) -> Tuple[str, int, Optional[Dict[str, Any]]]:
    """解析单行数据（继承自 V4）"""
    global global_stats
    global_stats["total_raw_events"] += 1

    parts = raw.strip().split("||")
    if len(parts) < 34:
        return ("", -1, None)

    sym = parts[1].strip()

    try:
        t = hhmmss_to_int(parts[IDX_TIME])
    except:
        t = -1

    # 试撮移除／时间窗检查
    if parts[IDX_TRIAL].strip() == "1":
        return (sym, t, None)
    if not is_in_trading_window(t):
        return (sym, t, None)

    # 取五档价量
    bids_p = [to_float(parts[i], 0.0) for i in BID_P_IDX]
    bids_q = [to_float(parts[i], 0.0) for i in BID_Q_IDX]
    asks_p = [to_float(parts[i], 0.0) for i in ASK_P_IDX]
    asks_q = [to_float(parts[i], 0.0) for i in ASK_Q_IDX]

    bid1, ask1 = bids_p[0], asks_p[0]
    if not spread_ok(bid1, ask1):
        return (sym, t, None)

    # 零值处理
    for p, q in zip(bids_p + asks_p, bids_q + asks_q):
        if p == 0.0 and q != 0.0:
            return (sym, t, None)

    ref = to_float(parts[IDX_REF], 0.0)
    upper = to_float(parts[IDX_UPPER], 0.0)
    lower = to_float(parts[IDX_LOWER], 0.0)
    last_px = to_float(parts[IDX_LASTPRICE], 0.0)
    tv = max(0, int(to_float(parts[IDX_TV], 0.0)))

    # 价格限制检查
    prices_to_check = [p for p in bids_p + asks_p if p > 0]
    if not all(within_limits(p, lower, upper) for p in prices_to_check):
        return (sym, t, None)

    # 组 20 维特征
    feat = np.array(bids_p + asks_p + bids_q + asks_q, dtype=np.float64)
    mid = 0.5 * (bid1 + ask1)

    rec = {
        "feat": feat,
        "mid": mid,
        "ref": ref,
        "upper": upper,
        "lower": lower,
        "last_px": last_px,
        "tv": tv,
        "raw": raw.strip()
    }

    global_stats["cleaned_events"] += 1
    return (sym, t, rec)


def dedup_by_timestamp_keep_last(rows: List[Tuple[int, Dict[str,Any]]]) -> List[Tuple[int, Dict[str,Any]]]:
    """时间戳去重，保留最后一笔（继承自 V4）"""
    if not rows:
        return rows

    dedup_dict = {}
    for idx, (t, r) in enumerate(rows):
        tv = r.get("tv", 0)
        key = (t, tv)
        dedup_dict[key] = (idx, r)

    result_with_idx = sorted(dedup_dict.values(), key=lambda x: x[0])
    result = [(rows[idx][0], r) for idx, r in result_with_idx]

    return result


def aggregate_chunks_of_10(seq: List[Tuple[int, Dict[str,Any]]]) -> Tuple[np.ndarray, np.ndarray]:
    """每 10 笔快照 → 1 时间点（继承自 V4）"""
    global global_stats

    if len(seq) < AGG_FACTOR:
        return np.zeros((0, 20), dtype=np.float64), np.zeros((0,), dtype=np.float64)

    feats, mids = [], []
    for i in range(AGG_FACTOR-1, len(seq), AGG_FACTOR):
        feat = seq[i][1]["feat"]
        mid = seq[i][1]["mid"]
        feats.append(feat)
        mids.append(mid)

    global_stats["aggregated_points"] += len(feats)
    return np.stack(feats, axis=0), np.array(mids, dtype=np.float64)


def zscore_fit(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """计算 Z-Score 参数（继承自 V4）"""
    mu = X.mean(axis=0)
    sd = X.std(axis=0, ddof=0)
    sd = np.where(sd < 1e-8, 1.0, sd)

    if np.any(np.abs(mu) > 1e6):
        logging.warning(f"侦测到异常大的均值: max|μ|={np.max(np.abs(mu)):.2f}")

    return mu, sd


def zscore_apply(X: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    """应用 Z-Score 正规化（继承自 V4）"""
    return (X - mu.reshape(1, -1)) / sd.reshape(1, -1)


def extract_date_from_filename(fp: str) -> str:
    """从档名抓取日期（继承自 V4）"""
    name = os.path.basename(fp)
    m = re.search(r"(20\d{6})", name)
    if m:
        return m.group(1)
    return name


# ============================================================
# V5 核心：滑窗 + Triple-Barrier 标签 + 样本权重
# ============================================================

def sliding_windows_v5(
    days_points: List[Tuple[str, str, np.ndarray, np.ndarray]],
    out_dir: str,
    config: Dict[str, Any]
):
    """
    V5 滑窗流程：
    1. 以股票为单位串接资料
    2. 计算波动率（EWMA/YZ/GARCH）
    3. 生成 Triple-Barrier 标签
    4. 计算样本权重
    5. 产生滑窗样本
    6. 70/15/15 切分
    """
    global global_stats

    if not days_points:
        logging.warning("没有资料可供产生 .npz 档案")
        return

    logging.info(f"\n{'='*60}")
    logging.info(f"V5 滑窗流程开始，共 {len(days_points)} 个 symbol-day 组合")
    logging.info(f"{'='*60}")

    # 步骤 1: 以股票为单位重组资料
    stock_data = defaultdict(lambda: {'dates': [], 'X': [], 'mids': []})

    for date, sym, Xd, mids in days_points:
        stock_data[sym]['dates'].append(date)
        stock_data[sym]['X'].append(Xd)
        stock_data[sym]['mids'].append(mids)

    logging.info(f"共 {len(stock_data)} 个股票有资料")

    # 对每个股票，按日期排序并串接
    stock_sequences = []

    for sym, data in stock_data.items():
        sorted_indices = np.argsort(data['dates'])

        X_concat = np.concatenate([data['X'][i] for i in sorted_indices], axis=0)
        mids_concat = np.concatenate([data['mids'][i] for i in sorted_indices], axis=0)

        stock_sequences.append((sym, X_concat.shape[0], X_concat, mids_concat))

    # 按时间点数量排序
    stock_sequences.sort(key=lambda x: x[1], reverse=True)

    logging.info(f"\n股票时间点统计:")
    logging.info(f"  最多: {stock_sequences[0][1]} 个点 ({stock_sequences[0][0]})")
    logging.info(f"  最少: {stock_sequences[-1][1]} 个点 ({stock_sequences[-1][0]})")

    # 步骤 2: 过滤太短的股票序列
    MIN_POINTS = SEQ_LEN + 50  # 100 + 50 = 150

    valid_stocks = [s for s in stock_sequences if s[1] >= MIN_POINTS]
    filtered_stocks = len(stock_sequences) - len(valid_stocks)

    if filtered_stocks > 0:
        logging.warning(f"过滤 {filtered_stocks} 档序列太短的股票（< {MIN_POINTS} 个点）")

    if not valid_stocks:
        logging.error("没有股票有足够的时间点产生滑窗样本！")
        return

    logging.info(f"有效股票: {len(valid_stocks)} 档")

    # 步骤 3: 按股票数量切分 70/15/15
    n_stocks = len(valid_stocks)
    n_train = max(1, int(n_stocks * config['split']['train_ratio']))
    n_val = max(1, int(n_stocks * config['split']['val_ratio']))

    train_stocks = valid_stocks[:n_train]
    val_stocks = valid_stocks[n_train:n_train + n_val]
    test_stocks = valid_stocks[n_train + n_val:]

    logging.info(f"\n资料切分（按股票数）:")
    logging.info(f"  Train: {len(train_stocks)} 档股票")
    logging.info(f"  Val:   {len(val_stocks)} 档股票")
    logging.info(f"  Test:  {len(test_stocks)} 档股票")

    splits = {
        'train': train_stocks,
        'val': val_stocks,
        'test': test_stocks
    }

    # 步骤 4: 计算训练集的 Z-Score 参数
    logging.info(f"\n{'='*60}")
    logging.info("计算 Z-Score 参数（基于训练集）")
    logging.info(f"{'='*60}")

    train_X_list = [stock[2] for stock in train_stocks]
    Xtr = np.concatenate(train_X_list, axis=0) if train_X_list else np.zeros((0, 20))

    if Xtr.size == 0:
        mu = np.zeros((20,), dtype=np.float64)
        sd = np.ones((20,), dtype=np.float64)
        logging.warning("训练集为空，使用预设 Z-Score 参数")
    else:
        mu, sd = zscore_fit(Xtr)
        logging.info(f"训练集大小: {Xtr.shape[0]:,} 个时间点")

    # 步骤 5: 产生滑窗样本（V5 核心）
    def build_split_v5(split_name):
        """对一个 split 产生 V5 滑窗样本"""
        stock_list = splits[split_name]

        logging.info(f"\n{'='*60}")
        logging.info(f"处理 {split_name.upper()} 集，共 {len(stock_list)} 档股票")
        logging.info(f"{'='*60}")

        X_windows = []
        y_labels = []
        weights = []
        stock_ids = []

        total_windows = 0
        tb_stats = {"up": 0, "down": 0, "time": 0}

        for sym, n_points, Xd, mids in stock_list:
            # Z-score 正规化
            Xn = zscore_apply(Xd, mu, sd)

            # 构建 DataFrame 用于 V5 标签生成
            close = pd.Series(mids, name='close')

            # 计算波动率
            vol_method = config['volatility']['method']

            # 计算波动率（失败则抛出异常）
            if vol_method == 'ewma':
                vol = ewma_vol(close, halflife=config['volatility']['halflife'])
            elif vol_method == 'garch':
                vol = garch11_vol(close)
            else:
                raise ValueError(f"不支援的波动率方法: {vol_method}，请使用 'ewma' 或 'garch'")

            # 生成 Triple-Barrier 标签（失败则停止流程）
            tb_cfg = config['triple_barrier']
            tb_df = tb_labels(
                close=close,
                vol=vol,
                pt_mult=tb_cfg['pt_multiplier'],
                sl_mult=tb_cfg['sl_multiplier'],
                max_holding=tb_cfg['max_holding'],
                min_return=tb_cfg['min_return']
            )

            # 转换标签 {-1,0,1} → {0,1,2}
            y_tb = tb_df["y"].map({-1: 0, 0: 1, 1: 2})

            # 计算样本权重
            if config['sample_weights']['enabled']:
                w = make_sample_weight(
                    ret=tb_df["ret"],
                    tt=tb_df["tt"],
                    y=y_tb,
                    tau=config['sample_weights']['tau'],
                    scale=config['sample_weights']['return_scaling'],
                    balance=config['sample_weights']['balance_classes']
                )
            else:
                w = pd.Series(np.ones(len(y_tb)), index=y_tb.index)

            # 统计触发原因
            for reason in tb_df["why"].value_counts().items():
                if reason[0] in tb_stats:
                    tb_stats[reason[0]] += reason[1]

            # 产生滑窗样本
            T = Xn.shape[0]
            max_t = min(T, len(y_tb))

            if max_t < SEQ_LEN:
                logging.warning(f"  {sym}: 跳过（只有 {max_t} 个点）")
                continue

            windows_count = 0

            for t in range(SEQ_LEN - 1, max_t):
                window = Xn[t - SEQ_LEN + 1:t + 1, :]

                if window.shape[0] != SEQ_LEN:
                    continue

                label = int(y_tb.iloc[t])
                weight = float(w.iloc[t])

                if label not in [0, 1, 2]:
                    continue

                X_windows.append(window.astype(np.float32))
                y_labels.append(label)
                weights.append(weight)
                stock_ids.append(sym)
                windows_count += 1

            if windows_count > 0:
                logging.info(f"  {sym}: {windows_count:,} 个样本 (共 {T} 个点)")

            total_windows += windows_count

        logging.info(f"\n{split_name.upper()} 总计: {total_windows:,} 个样本")
        logging.info(f"触发原因分布: {tb_stats}")

        global_stats["valid_windows"] += total_windows

        if X_windows:
            X_array = np.stack(X_windows, axis=0)
            y_array = np.array(y_labels, dtype=np.int64)
            w_array = np.array(weights, dtype=np.float32)
            sid_array = np.array(stock_ids, dtype=object)

            # 统计资讯
            unique_stocks = len(np.unique(sid_array))
            label_dist = np.bincount(y_array, minlength=3)

            logging.info(f"  形状: X={X_array.shape}, y={y_array.shape}, w={w_array.shape}")
            logging.info(f"  涵盖股票: {unique_stocks} 档")
            logging.info(f"  标签分布: 上涨={label_dist[0]}, 持平={label_dist[1]}, 下跌={label_dist[2]}")
            logging.info(f"  权重统计: mean={w_array.mean():.3f}, std={w_array.std():.3f}, max={w_array.max():.3f}")

            return X_array, y_array, w_array, sid_array
        else:
            logging.warning(f"{split_name} 集没有产生任何样本！")
            return (
                np.zeros((0, SEQ_LEN, 20), dtype=np.float32),
                np.zeros((0,), dtype=np.int64),
                np.zeros((0,), dtype=np.float32),
                np.array([], dtype=object)
            )

    # 步骤 6: 产生并保存 npz 档案
    os.makedirs(out_dir, exist_ok=True)

    results = {}

    for split in ["train", "val", "test"]:
        X, y, w, sid = build_split_v5(split)

        npz_path = os.path.join(out_dir, f"stock_embedding_{split}.npz")
        np.savez_compressed(npz_path, X=X, y=y, weights=w, stock_ids=sid)

        logging.info(f"\n✅ 已保存: {npz_path}")

        results[split] = {
            "samples": len(y),
            "label_dist": np.bincount(y, minlength=3).tolist() if len(y) > 0 else [0, 0, 0],
            "weight_stats": {
                "mean": float(w.mean()) if len(w) > 0 else 0.0,
                "std": float(w.std()) if len(w) > 0 else 0.0,
                "max": float(w.max()) if len(w) > 0 else 0.0
            }
        }

    # 步骤 7: 写入 metadata
    meta = {
        "format": "deeplob_v5_pro",
        "version": VERSION,
        "creation_date": datetime.now().isoformat(),
        "seq_len": SEQ_LEN,
        "feature_dim": 20,

        "volatility": {
            "method": config['volatility']['method'],
            "halflife": config['volatility'].get('halflife', 60)
        },

        "triple_barrier": config['triple_barrier'],

        "sample_weights": {
            "enabled": config['sample_weights']['enabled'],
            "tau": config['sample_weights']['tau'],
            "return_scaling": config['sample_weights']['return_scaling'],
            "balance_classes": config['sample_weights']['balance_classes']
        },

        "normalization": {
            "method": "zscore",
            "computed_on": "train_set",
            "feature_means": mu.tolist(),
            "feature_stds": sd.tolist()
        },

        "data_quality": global_stats,

        "data_split": {
            "method": "by_stock_count",
            "train_stocks": len(train_stocks),
            "val_stocks": len(val_stocks),
            "test_stocks": len(test_stocks),
            "total_stocks": len(valid_stocks),
            "filtered_stocks": filtered_stocks,
            "results": results
        },

        "note": "V5 Pro: Triple-Barrier labels + Sample weights. Labels: {0:下跌, 1:持平, 2:上涨}",
    }

    meta_path = os.path.join(out_dir, "normalization_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    logging.info(f"\n✅ Metadata 已保存: {meta_path}")
    logging.info(f"\n{'='*60}")
    logging.info("NPZ 档案产生完成！")
    logging.info(f"{'='*60}")


# ============================================================
# 主程式
# ============================================================

def main():
    """主程式"""
    try:
        args = parse_args()
        in_dir = args.input_dir
        out_dir = args.output_dir

        # 载入配置
        config = load_config(args.config)

        # 验证输入目录存在
        if not os.path.exists(in_dir):
            logging.error(f"输入目录不存在: {in_dir}")
            return 1

        # 建立输出目录
        try:
            os.makedirs(out_dir, exist_ok=True)
        except Exception as e:
            logging.error(f"无法建立输出目录 {out_dir}: {e}")
            return 1

        logging.info(f"{'='*60}")
        logging.info(f"V5 Pro 资料流水线启动")
        logging.info(f"{'='*60}")
        logging.info(f"输入目录: {in_dir}")
        logging.info(f"输出目录: {out_dir}")
        logging.info(f"配置版本: {config['version']}")
        logging.info(f"波动率方法: {config['volatility']['method']}")
        logging.info(f"Triple-Barrier: PT={config['triple_barrier']['pt_multiplier']}σ, "
                    f"SL={config['triple_barrier']['sl_multiplier']}σ, "
                    f"MaxHold={config['triple_barrier']['max_holding']}")
        logging.info(f"{'='*60}\n")

        # 读取所有 .txt 档
        files = sorted(glob.glob(os.path.join(in_dir, "*.txt")))
        if not files:
            logging.error(f"在 {in_dir} 找不到 .txt 档案")
            return 1

        logging.info(f"找到 {len(files)} 个档案待处理")

        # 先按档名日期分组
        day_map: Dict[str, List[str]] = defaultdict(list)
        for fp in files:
            day = extract_date_from_filename(fp)
            day_map[day].append(fp)

        # 逐日逐档案读取 → 逐 symbol 汇整
        per_day_symbol_points = []  # [(date, symbol, X_points, mids)]
        day_keys = sorted(day_map.keys())

        logging.info(f"开始处理 {len(day_keys)} 个交易日的资料")

        for day in day_keys:
            fps = sorted(day_map[day])
            logging.info(f"处理日期 {day}，共 {len(fps)} 个档案")

            # 读本日所有行
            per_symbol_raw: Dict[str, List[Tuple[int, Dict[str,Any]]]] = defaultdict(list)

            for fp in fps:
                try:
                    with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                        for raw in f:
                            sym, t, rec = parse_line(raw)
                            if rec is None or sym == "" or t < 0:
                                continue
                            per_symbol_raw[sym].append((t, rec))
                except Exception as e:
                    logging.warning(f"读取档案 {fp} 时发生错误: {e}")
                    continue

            # 对每个 symbol：去重、聚合、保存中间价
            for sym, rows in per_symbol_raw.items():
                if not rows:
                    continue

                # 时间排序
                rows.sort(key=lambda x: x[0])

                # 去重
                rows = dedup_by_timestamp_keep_last(rows)
                if not rows:
                    continue

                # 10 事件聚合
                Xp, mids = aggregate_chunks_of_10(rows)
                if Xp.shape[0] == 0:
                    continue

                # V5: 保存 mids（用于后续标签生成）
                per_day_symbol_points.append((day, sym, Xp, mids))

        # 若没有可用资料
        if not per_day_symbol_points:
            logging.error("清洗或聚合后没有可用资料")
            return 1

        logging.info(f"共处理 {len(per_day_symbol_points)} 个 symbol-day 组合")

        # 产出 70/15/15 的 .npz（V5 滑窗流程）
        if args.make_npz:
            logging.info("开始产生 V5 .npz 档案")
            sliding_windows_v5(
                per_day_symbol_points,
                os.path.join(out_dir, "npz"),
                config
            )

        logging.info(f"\n{'='*60}")
        logging.info(f"[完成] V5 转换成功，输出资料夹: {out_dir}")
        logging.info(f"{'='*60}")
        logging.info(f"统计资料:")
        logging.info(f"  原始事件数: {global_stats['total_raw_events']:,}")
        logging.info(f"  清洗后: {global_stats['cleaned_events']:,}")
        logging.info(f"  聚合后时间点: {global_stats['aggregated_points']:,}")
        logging.info(f"  有效窗口: {global_stats['valid_windows']:,}")
        logging.info(f"  Triple-Barrier 成功: {global_stats['tb_success']:,}")
        logging.info(f"{'='*60}\n")

        return 0

    except Exception as e:
        logging.error(f"程式执行失败: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
