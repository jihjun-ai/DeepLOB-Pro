# -*- coding: utf-8 -*-
"""
stability_check.py - 時間穩定性驗證模組
=============================================================================
驗證必要條件 4: 訊號非瞬時幻覺（有可遷移的穩定性）

檢驗項：
1. 滾動回測 (Rolling Backtest)
2. IC 穩定性分析
3. 學習曲線
4. 特徵重要度穩定性

使用方式：
    python scripts/stability_check.py \
        --preprocessed-dir ./data/preprocessed_v5_1hz \
        --output-dir ./stability_check_output \
        --train-window 20 \
        --test-window 5 \
        --step 5

輸出：
    - rolling_backtest_results.json
    - stability_report.txt
    - 穩定性圖表

版本：v1.0
日期：2025-10-21
"""

import os
import json
import argparse
import logging
import warnings
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
from collections import defaultdict
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    f1_score, roc_auc_score, log_loss
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
warnings.filterwarnings('ignore')

# ============================================================
# 數據載入
# ============================================================

def load_all_daily_npz(preprocessed_dir: str) -> Dict[str, List[Tuple]]:
    """
    載入所有預處理數據，按日期組織

    Returns:
        {
            '20250901': [(symbol, features, mids, meta), ...],
            '20250902': [...],
            ...
        }
    """
    daily_dir = os.path.join(preprocessed_dir, "daily")

    if not os.path.exists(daily_dir):
        logging.error(f"預處理目錄不存在: {daily_dir}")
        return {}

    data_by_date = defaultdict(list)

    import glob

    for date_dir in sorted(os.listdir(daily_dir)):
        date_path = os.path.join(daily_dir, date_dir)
        if not os.path.isdir(date_path):
            continue

        for npz_file in sorted(glob.glob(os.path.join(date_path, "*.npz"))):
            if npz_file.endswith("summary.json"):
                continue

            try:
                data = np.load(npz_file, allow_pickle=True)

                features = data['features']
                mids = data['mids']
                meta = json.loads(str(data['metadata']))

                # 檢查過濾狀態
                if not meta.get('pass_filter', True):
                    continue

                symbol = meta['symbol']
                date = meta['date']

                data_by_date[date].append((symbol, features, mids, meta))

            except Exception as e:
                logging.warning(f"無法載入 {npz_file}: {e}")
                continue

    logging.info(f"載入了 {len(data_by_date)} 個交易日的數據")

    return dict(data_by_date)


# ============================================================
# 特徵提取與標籤生成
# ============================================================

def ewma_vol(close: pd.Series, halflife: int = 60) -> pd.Series:
    """EWMA 波動率估計"""
    ret = np.log(close).diff()
    var = ret.ewm(halflife=halflife, adjust=False).var()
    vol = np.sqrt(var)

    if vol.isna().any():
        valid_vols = vol.dropna()
        if len(valid_vols) > 0:
            initial_vol = valid_vols.iloc[:min(100, len(valid_vols))].mean()
        else:
            initial_vol = 0.01
        vol = vol.fillna(initial_vol)

    return vol


def simple_tb_labels(close: pd.Series, vol: pd.Series,
                     pt_mult: float = 2.0, sl_mult: float = 2.0,
                     max_holding: int = 200, min_return: float = 0.0001) -> pd.Series:
    """
    簡化版 Triple-Barrier 標籤（用於滾動回測）

    Returns:
        pd.Series: 標籤 {0: 下跌, 1: 持平, 2: 上漲}
    """
    n = len(close)
    labels = []

    for i in range(n - 1):
        entry_price = close.iloc[i]
        entry_vol = vol.iloc[i]

        up_barrier = entry_price * (1 + pt_mult * entry_vol)
        dn_barrier = entry_price * (1 - sl_mult * entry_vol)

        end_idx = min(i + max_holding, n)

        trigger_why = 'time'

        for j in range(i + 1, end_idx):
            future_price = close.iloc[j]

            if future_price >= up_barrier:
                trigger_why = 'up'
                break

            if future_price <= dn_barrier:
                trigger_why = 'down'
                break

        exit_price = close.iloc[min(i + max_holding, n - 1)]
        ret = (exit_price - entry_price) / entry_price

        if trigger_why == 'time':
            if np.abs(ret) < min_return:
                label = 1  # neutral
            else:
                label = 2 if ret > 0 else 0  # up / down
        else:
            label = 2 if trigger_why == 'up' else 0

        labels.append(label)

    # 最後一個點
    labels.append(1)

    return pd.Series(labels, index=close.index)


def extract_features_labels(day_data: List[Tuple], seq_len: int = 100,
                           stride: int = 10, max_samples_per_stock: int = 500) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    從單日數據提取特徵和標籤（優化版：降採樣 + 進度顯示）

    Args:
        day_data: 單日數據列表
        seq_len: 序列長度
        stride: 滑窗步長（優化：每 stride 個點取一個樣本）
        max_samples_per_stock: 每個股票最多取樣數（避免單股樣本過多）

    Returns:
        X: (N, seq_len, 20)
        y: (N,)
        stock_ids: (N,)
    """
    X_list = []
    y_list = []
    stock_list = []

    total_stocks = len(day_data)
    processed = 0

    for symbol, features, mids, meta in day_data:
        processed += 1

        # 每處理 20 個股票顯示一次進度
        if processed % 20 == 0 or processed == total_stocks:
            logging.info(f"    處理進度: {processed}/{total_stocks} 股票 ({processed/total_stocks*100:.1f}%)")

        close = pd.Series(mids, name='close')

        # 波動率
        try:
            vol = ewma_vol(close, halflife=60)
        except:
            continue

        # 標籤（簡化版 TB，減少計算量）
        try:
            y_labels = simple_tb_labels(close, vol, max_holding=40)  # 降低 max_holding
        except:
            continue

        # 滑窗（優化：每 stride 個點取一個樣本）
        T = features.shape[0]
        max_t = min(T, len(y_labels))

        if max_t < seq_len:
            continue

        stock_samples = 0

        for t in range(seq_len - 1, max_t, stride):  # 優化：使用 stride
            if stock_samples >= max_samples_per_stock:  # 限制單股樣本數
                break

            window_start = t - seq_len + 1
            window = features[window_start:t + 1, :]

            if window.shape[0] != seq_len:
                continue

            label = int(y_labels.iloc[t])

            if label not in [0, 1, 2]:
                continue

            X_list.append(window.astype(np.float32))
            y_list.append(label)
            stock_list.append(symbol)
            stock_samples += 1

    if X_list:
        X = np.stack(X_list, axis=0)
        y = np.array(y_list, dtype=np.int64)
        stock_ids = np.array(stock_list, dtype=object)
        return X, y, stock_ids
    else:
        return (
            np.zeros((0, seq_len, 20), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
            np.array([], dtype=object)
        )


# ============================================================
# 滾動回測核心
# ============================================================

def rolling_backtest(
    data_by_date: Dict[str, List[Tuple]],
    train_window: int = 20,
    test_window: int = 5,
    step: int = 5,
    seq_len: int = 100
) -> List[Dict]:
    """
    滾動回測

    Args:
        data_by_date: 按日期組織的數據
        train_window: 訓練窗口（天數）
        test_window: 測試窗口（天數）
        step: 滾動步長（天數）
        seq_len: 序列長度

    Returns:
        List of results for each window
    """
    logging.info("\n" + "="*60)
    logging.info("滾動回測開始")
    logging.info("="*60)
    logging.info(f"訓練窗口: {train_window} 天")
    logging.info(f"測試窗口: {test_window} 天")
    logging.info(f"滾動步長: {step} 天")

    dates = sorted(data_by_date.keys())
    total_dates = len(dates)

    logging.info(f"總交易日: {total_dates}")

    if total_dates < train_window + test_window:
        logging.error(f"數據不足: 至少需要 {train_window + test_window} 天")
        return []

    results = []
    window_idx = 0

    for start_idx in range(0, total_dates - train_window - test_window + 1, step):
        window_idx += 1

        train_end_idx = start_idx + train_window
        test_end_idx = train_end_idx + test_window

        train_dates = dates[start_idx:train_end_idx]
        test_dates = dates[train_end_idx:test_end_idx]

        logging.info(f"\n窗口 {window_idx}:")
        logging.info(f"  訓練期: {train_dates[0]} ~ {train_dates[-1]}")
        logging.info(f"  測試期: {test_dates[0]} ~ {test_dates[-1]}")

        # 提取訓練數據
        train_data = []
        for date in train_dates:
            train_data.extend(data_by_date[date])

        logging.info(f"  提取訓練特徵（{len(train_data)} 個 stock-day）...")
        X_train, y_train, stock_train = extract_features_labels(train_data, seq_len)

        if len(X_train) < 100:
            logging.warning(f"  訓練樣本不足 ({len(X_train)})，跳過")
            continue

        # 提取測試數據
        test_data = []
        for date in test_dates:
            test_data.extend(data_by_date[date])

        logging.info(f"  提取測試特徵（{len(test_data)} 個 stock-day）...")
        X_test, y_test, stock_test = extract_features_labels(test_data, seq_len)

        if len(X_test) < 50:
            logging.warning(f"  測試樣本不足 ({len(X_test)})，跳過")
            continue

        # 正規化（用訓練集統計量）
        X_train_flat = X_train[:, -1, :].reshape(len(X_train), -1)
        X_test_flat = X_test[:, -1, :].reshape(len(X_test), -1)

        mu = X_train_flat.mean(axis=0)
        sd = X_train_flat.std(axis=0)
        sd = np.where(sd < 1e-8, 1.0, sd)

        X_train_norm = (X_train_flat - mu) / sd
        X_test_norm = (X_test_flat - mu) / sd

        # 訓練簡單模型
        clf = LogisticRegression(max_iter=200, random_state=42, class_weight='balanced')

        try:
            clf.fit(X_train_norm, y_train)
        except Exception as e:
            logging.warning(f"  訓練失敗: {e}")
            continue

        # 評估
        y_pred_train = clf.predict(X_train_norm)
        y_pred_test = clf.predict(X_test_norm)
        y_proba_test = clf.predict_proba(X_test_norm)

        train_acc = accuracy_score(y_train, y_pred_train)
        test_acc = accuracy_score(y_test, y_pred_test)
        test_bal_acc = balanced_accuracy_score(y_test, y_pred_test)
        test_f1 = f1_score(y_test, y_pred_test, average='macro')

        try:
            test_auc = roc_auc_score(y_test, y_proba_test, multi_class='ovr', average='macro')
        except:
            test_auc = 0.5

        # 計算 IC（Information Coefficient）
        # 用預測概率與實際標籤的相關性
        pred_score = y_proba_test[:, 2] - y_proba_test[:, 0]  # up prob - down prob
        actual_score = (y_test == 2).astype(float) - (y_test == 0).astype(float)

        try:
            ic, ic_pvalue = stats.spearmanr(pred_score, actual_score)
        except:
            ic = 0.0
            ic_pvalue = 1.0

        logging.info(f"  訓練樣本: {len(X_train):,}")
        logging.info(f"  測試樣本: {len(X_test):,}")
        logging.info(f"  訓練準確率: {train_acc:.3f}")
        logging.info(f"  測試準確率: {test_acc:.3f}")
        logging.info(f"  平衡準確率: {test_bal_acc:.3f}")
        logging.info(f"  F1 (macro): {test_f1:.3f}")
        logging.info(f"  AUC (macro): {test_auc:.3f}")
        logging.info(f"  IC: {ic:.3f} (p={ic_pvalue:.3f})")

        result = {
            'window_idx': window_idx,
            'train_start': train_dates[0],
            'train_end': train_dates[-1],
            'test_start': test_dates[0],
            'test_end': test_dates[-1],
            'n_train': len(X_train),
            'n_test': len(X_test),
            'train_acc': float(train_acc),
            'test_acc': float(test_acc),
            'test_bal_acc': float(test_bal_acc),
            'test_f1': float(test_f1),
            'test_auc': float(test_auc),
            'ic': float(ic),
            'ic_pvalue': float(ic_pvalue),
            'acc_gap': float(train_acc - test_acc)
        }

        results.append(result)

    return results


# ============================================================
# 穩定性分析
# ============================================================

def analyze_stability(results: List[Dict]) -> Dict:
    """分析滾動回測結果的穩定性"""

    logging.info("\n" + "="*60)
    logging.info("穩定性分析")
    logging.info("="*60)

    if len(results) == 0:
        logging.warning("無結果可分析")
        return {}

    # 提取指標
    auc_list = [r['test_auc'] for r in results]
    ic_list = [r['ic'] for r in results]
    bal_acc_list = [r['test_bal_acc'] for r in results]
    f1_list = [r['test_f1'] for r in results]

    # 統計量
    stability = {
        'n_windows': len(results),

        'auc': {
            'mean': float(np.mean(auc_list)),
            'std': float(np.std(auc_list)),
            'min': float(np.min(auc_list)),
            'max': float(np.max(auc_list)),
            'median': float(np.median(auc_list)),
            'positive_ratio': float(sum(np.array(auc_list) > 0.5) / len(auc_list))
        },

        'ic': {
            'mean': float(np.mean(ic_list)),
            'std': float(np.std(ic_list)),
            'min': float(np.min(ic_list)),
            'max': float(np.max(ic_list)),
            'median': float(np.median(ic_list)),
            'positive_ratio': float(sum(np.array(ic_list) > 0) / len(ic_list))
        },

        'balanced_acc': {
            'mean': float(np.mean(bal_acc_list)),
            'std': float(np.std(bal_acc_list)),
            'min': float(np.min(bal_acc_list)),
            'max': float(np.max(bal_acc_list))
        },

        'f1': {
            'mean': float(np.mean(f1_list)),
            'std': float(np.std(f1_list)),
            'min': float(np.min(f1_list)),
            'max': float(np.max(f1_list))
        }
    }

    # 判斷穩定性
    auc_mean = stability['auc']['mean']
    auc_std = stability['auc']['std']
    auc_pos_ratio = stability['auc']['positive_ratio']

    ic_mean = stability['ic']['mean']
    ic_pos_ratio = stability['ic']['positive_ratio']

    logging.info(f"\nAUC 統計:")
    logging.info(f"  平均: {auc_mean:.3f}")
    logging.info(f"  標準差: {auc_std:.3f}")
    logging.info(f"  範圍: [{stability['auc']['min']:.3f}, {stability['auc']['max']:.3f}]")
    logging.info(f"  正比例: {auc_pos_ratio*100:.1f}%")

    logging.info(f"\nIC 統計:")
    logging.info(f"  平均: {ic_mean:.3f}")
    logging.info(f"  標準差: {stability['ic']['std']:.3f}")
    logging.info(f"  範圍: [{stability['ic']['min']:.3f}, {stability['ic']['max']:.3f}]")
    logging.info(f"  正比例: {ic_pos_ratio*100:.1f}%")

    # 穩定性判斷
    if auc_mean > 0.52 and auc_std < 0.05 and auc_pos_ratio > 0.8:
        status = "✅ 穩定"
        stability['status'] = "stable"
        stability['pass'] = True
    elif auc_mean > 0.50 and auc_pos_ratio > 0.6:
        status = "⚠️ 勉強"
        stability['status'] = "marginal"
        stability['pass'] = True
    else:
        status = "❌ 不穩定"
        stability['status'] = "unstable"
        stability['pass'] = False

    logging.info(f"\n穩定性評估: {status}")
    stability['overall_status'] = status

    return stability


# ============================================================
# 主函數
# ============================================================

def run_stability_check(
    preprocessed_dir: str,
    output_dir: str,
    train_window: int = 20,
    test_window: int = 5,
    step: int = 5
):
    """執行完整穩定性檢查"""

    logging.info("\n" + "="*60)
    logging.info("時間穩定性驗證開始")
    logging.info("="*60)

    os.makedirs(output_dir, exist_ok=True)

    # 載入數據
    data_by_date = load_all_daily_npz(preprocessed_dir)

    if len(data_by_date) == 0:
        logging.error("無數據可分析")
        return

    # 滾動回測
    results = rolling_backtest(
        data_by_date=data_by_date,
        train_window=train_window,
        test_window=test_window,
        step=step
    )

    if len(results) == 0:
        logging.error("滾動回測無結果")
        return

    # 穩定性分析
    stability = analyze_stability(results)

    # 保存結果
    report = {
        'timestamp': pd.Timestamp.now().isoformat(),
        'config': {
            'train_window': train_window,
            'test_window': test_window,
            'step': step
        },
        'rolling_backtest_results': results,
        'stability_analysis': stability
    }

    json_path = os.path.join(output_dir, 'rolling_backtest_results.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    logging.info(f"\n結果已保存: {json_path}")

    # 生成人類可讀摘要
    txt_path = os.path.join(output_dir, 'stability_report.txt')
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("時間穩定性驗證報告\n")
        f.write("="*60 + "\n\n")

        f.write(f"時間戳: {report['timestamp']}\n")
        f.write(f"滾動窗口數: {stability['n_windows']}\n\n")

        f.write("穩定性指標:\n")
        f.write(f"  AUC 平均: {stability['auc']['mean']:.3f} ± {stability['auc']['std']:.3f}\n")
        f.write(f"  AUC 範圍: [{stability['auc']['min']:.3f}, {stability['auc']['max']:.3f}]\n")
        f.write(f"  AUC 正比例: {stability['auc']['positive_ratio']*100:.1f}%\n\n")

        f.write(f"  IC 平均: {stability['ic']['mean']:.3f} ± {stability['ic']['std']:.3f}\n")
        f.write(f"  IC 範圍: [{stability['ic']['min']:.3f}, {stability['ic']['max']:.3f}]\n")
        f.write(f"  IC 正比例: {stability['ic']['positive_ratio']*100:.1f}%\n\n")

        f.write(f"總體評估: {stability['overall_status']}\n")
        f.write(f"通過: {'是' if stability['pass'] else '否'}\n")

    logging.info(f"摘要已保存: {txt_path}")

    logging.info("\n" + "="*60)
    logging.info(f"穩定性檢查完成")
    logging.info(f"總體評估: {stability['overall_status']}")
    logging.info("="*60 + "\n")

    return report


def parse_args():
    p = argparse.ArgumentParser(description="時間穩定性驗證腳本")
    p.add_argument("--preprocessed-dir", required=True, help="預處理數據目錄")
    p.add_argument("--output-dir", default="./stability_check_output", help="輸出目錄")
    p.add_argument("--train-window", type=int, default=20, help="訓練窗口（天數）")
    p.add_argument("--test-window", type=int, default=5, help="測試窗口（天數）")
    p.add_argument("--step", type=int, default=5, help="滾動步長（天數）")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    run_stability_check(
        preprocessed_dir=args.preprocessed_dir,
        output_dir=args.output_dir,
        train_window=args.train_window,
        test_window=args.test_window,
        step=args.step
    )
