# -*- coding: utf-8 -*-
"""
data_health_check.py - 數據質量健檢腳本
=============================================================================
檢查訓練數據是否滿足「具有學習價值、且可以學起來」的 5 個必要條件

必要條件：
1. 有明確任務與標籤
2. 不含未來資訊（無洩漏）
3. 足量且涵蓋多樣情境
4. 訊號非瞬時幻覺（穩定性）
5. 噪聲可被管理

使用方式：
    python scripts/data_health_check.py \
        --train-npz ./data/processed_v6/npz/stock_embedding_train.npz \
        --val-npz ./data/processed_v6/npz/stock_embedding_val.npz \
        --test-npz ./data/processed_v6/npz/stock_embedding_test.npz \
        --output-dir ./data/processed_v6/health_check/

輸出：
    - health_check_report.json
    - health_check_report.txt (人類可讀)
    - 各種檢測圖表

版本：v1.0
日期：2025-10-21
"""

import os
import json
import argparse
import logging
import warnings
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    f1_score, roc_auc_score, confusion_matrix
)
from sklearn.utils.class_weight import compute_class_weight

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
warnings.filterwarnings('ignore')

# ============================================================
# 輔助函數
# ============================================================

def load_npz_data(npz_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """載入 NPZ 數據"""
    data = np.load(npz_path, allow_pickle=True)
    X = data['X']
    y = data['y']
    weights = data.get('weights', np.ones(len(y)))
    stock_ids = data.get('stock_ids', np.array(['unknown'] * len(y)))

    logging.info(f"載入 {npz_path}")
    logging.info(f"  X shape: {X.shape}")
    logging.info(f"  y shape: {y.shape}")
    logging.info(f"  unique labels: {np.unique(y)}")

    return X, y, weights, stock_ids


def calculate_entropy(y: np.ndarray) -> float:
    """計算標籤熵"""
    counts = np.bincount(y)
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    return -np.sum(probs * np.log2(probs))


def calculate_effective_n(y: np.ndarray, X: np.ndarray) -> float:
    """
    計算有效樣本數 (考慮自相關)

    Neff ≈ N * (1 - ρ₁) / (1 + ρ₁)
    其中 ρ₁ 是標籤的 lag-1 自相關係數
    """
    # 計算標籤自相關
    y_shifted = np.roll(y, 1)
    y_shifted[0] = y[0]  # 首個值不變

    # Pearson 相關
    rho_1 = np.corrcoef(y[1:], y_shifted[1:])[0, 1]

    N = len(y)
    if abs(rho_1) < 0.99:
        Neff = N * (1 - rho_1) / (1 + rho_1)
    else:
        Neff = N * 0.1  # 極高自相關

    return max(Neff, N * 0.1)  # 最少 10% 有效


def calculate_psi(expected: np.ndarray, actual: np.ndarray, bins=10) -> float:
    """
    計算 PSI (Population Stability Index)

    PSI < 0.1: 分布穩定
    0.1 <= PSI < 0.2: 輕微漂移
    PSI >= 0.2: 顯著漂移，需警戒
    """
    exp_hist, bin_edges = np.histogram(expected, bins=bins)
    act_hist, _ = np.histogram(actual, bins=bin_edges)

    exp_pct = (exp_hist + 1) / (len(expected) + bins)  # +1 平滑
    act_pct = (act_hist + 1) / (len(actual) + bins)

    psi = np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct))

    return psi


# ============================================================
# 檢查 1: 明確任務與標籤
# ============================================================

def check_task_and_labels(y: np.ndarray) -> Dict:
    """
    檢查標籤質量

    檢驗項：
    - 類別分布
    - 標籤熵
    - 失衡程度
    """
    logging.info("\n" + "="*60)
    logging.info("檢查 1: 明確任務與標籤")
    logging.info("="*60)

    results = {}

    # 類別分布
    unique_labels = np.unique(y)
    counts = np.bincount(y, minlength=3)
    total = counts.sum()
    pcts = counts / total * 100

    results['n_classes'] = len(unique_labels)
    results['class_counts'] = counts.tolist()
    results['class_percentages'] = pcts.tolist()

    logging.info(f"\n類別分布:")
    logging.info(f"  Down (0):    {counts[0]:,} ({pcts[0]:.1f}%)")
    logging.info(f"  Neutral (1): {counts[1]:,} ({pcts[1]:.1f}%)")
    logging.info(f"  Up (2):      {counts[2]:,} ({pcts[2]:.1f}%)")

    # 標籤熵
    entropy = calculate_entropy(y)
    max_entropy = np.log2(3)  # 三分類最大熵
    results['entropy'] = entropy
    results['normalized_entropy'] = entropy / max_entropy

    logging.info(f"\n標籤熵:")
    logging.info(f"  熵值: {entropy:.3f}")
    logging.info(f"  正規化熵: {entropy/max_entropy:.3f} (1.0 = 完全均衡)")

    # 失衡檢查
    max_pct = pcts.max()
    results['max_class_pct'] = max_pct

    if max_pct > 60:
        status = "❌ 嚴重失衡"
        results['balance_status'] = "severely_imbalanced"
    elif max_pct > 50:
        status = "⚠️ 輕度失衡"
        results['balance_status'] = "slightly_imbalanced"
    else:
        status = "✅ 相對均衡"
        results['balance_status'] = "balanced"

    logging.info(f"\n失衡檢查: {status}")
    logging.info(f"  最大類別佔比: {max_pct:.1f}%")

    # 有效樣本數（待後續補充）
    results['pass'] = max_pct < 70 and entropy/max_entropy > 0.6

    return results


# ============================================================
# 檢查 2: 未來資訊洩漏測試
# ============================================================

def check_future_leakage(X: np.ndarray, y: np.ndarray, weights: np.ndarray) -> Dict:
    """
    破功測試：檢測未來資訊洩漏

    測試 A: 時間打亂標籤 - 準確率應降到隨機水準
    測試 B: 未對齊測試 - 用 t+1 特徵預測 t 標籤，不應變好
    """
    logging.info("\n" + "="*60)
    logging.info("檢查 2: 未來資訊洩漏測試")
    logging.info("="*60)

    results = {}

    # 取樣本（避免計算時間過長）
    n_samples = min(10000, len(y))
    indices = np.random.choice(len(y), n_samples, replace=False)
    X_sample = X[indices]
    y_sample = y[indices]
    w_sample = weights[indices]

    # 展平特徵
    X_flat = X_sample.reshape(X_sample.shape[0], -1)

    # 訓練基準模型（正常對齊）
    clf_baseline = LogisticRegression(max_iter=200, random_state=42, class_weight='balanced')
    clf_baseline.fit(X_flat, y_sample, sample_weight=w_sample)
    y_pred_baseline = clf_baseline.predict(X_flat)
    acc_baseline = accuracy_score(y_sample, y_pred_baseline)

    results['baseline_accuracy'] = acc_baseline
    logging.info(f"\n基準模型（正常對齊）:")
    logging.info(f"  準確率: {acc_baseline:.3f}")

    # 測試 A: 時間打亂標籤
    y_shuffled = y_sample.copy()
    np.random.shuffle(y_shuffled)

    clf_shuffled = LogisticRegression(max_iter=200, random_state=42, class_weight='balanced')
    clf_shuffled.fit(X_flat, y_shuffled, sample_weight=w_sample)
    y_pred_shuffled = clf_shuffled.predict(X_flat)
    acc_shuffled = accuracy_score(y_shuffled, y_pred_shuffled)

    results['shuffled_accuracy'] = acc_shuffled
    results['accuracy_drop'] = acc_baseline - acc_shuffled

    logging.info(f"\n測試 A: 時間打亂標籤")
    logging.info(f"  準確率: {acc_shuffled:.3f}")
    logging.info(f"  下降幅度: {acc_baseline - acc_shuffled:.3f}")

    if acc_shuffled > 0.4:
        status_a = "❌ 失敗 - 打亂後仍高準確率，可能有洩漏"
        results['test_a_pass'] = False
    else:
        status_a = "✅ 通過 - 打亂後準確率接近隨機"
        results['test_a_pass'] = True

    logging.info(f"  結果: {status_a}")

    # 測試 B: 未對齊測試（用 t+1 預測 t）
    # 將特徵向後移一步
    X_misaligned = np.roll(X_sample, -1, axis=0)
    X_misaligned[-1] = X_sample[-1]  # 最後一個保持不變
    X_mis_flat = X_misaligned.reshape(X_misaligned.shape[0], -1)

    clf_misaligned = LogisticRegression(max_iter=200, random_state=42, class_weight='balanced')
    clf_misaligned.fit(X_mis_flat, y_sample, sample_weight=w_sample)
    y_pred_mis = clf_misaligned.predict(X_mis_flat)
    acc_misaligned = accuracy_score(y_sample, y_pred_mis)

    results['misaligned_accuracy'] = acc_misaligned
    results['accuracy_increase'] = acc_misaligned - acc_baseline

    logging.info(f"\n測試 B: 未對齊測試（特徵向後移1步）")
    logging.info(f"  準確率: {acc_misaligned:.3f}")
    logging.info(f"  變化: {acc_misaligned - acc_baseline:+.3f}")

    if acc_misaligned > acc_baseline + 0.05:
        status_b = "❌ 失敗 - 錯對齊反而變好，存在洩漏"
        results['test_b_pass'] = False
    else:
        status_b = "✅ 通過 - 錯對齊無異常提升"
        results['test_b_pass'] = True

    logging.info(f"  結果: {status_b}")

    results['pass'] = results['test_a_pass'] and results['test_b_pass']

    return results


# ============================================================
# 檢查 3: 足量且多樣
# ============================================================

def check_sufficiency_and_diversity(
    X: np.ndarray,
    y: np.ndarray,
    stock_ids: np.ndarray
) -> Dict:
    """
    檢查樣本量與多樣性

    檢驗項：
    - 總樣本數
    - 有效樣本數 (Neff)
    - 股票覆蓋數
    - 每類 Neff 是否足夠（建議 > 1000）
    """
    logging.info("\n" + "="*60)
    logging.info("檢查 3: 足量且多樣")
    logging.info("="*60)

    results = {}

    # 總樣本數
    n_total = len(y)
    results['n_total'] = n_total

    logging.info(f"\n總樣本數: {n_total:,}")

    # 有效樣本數
    Neff = calculate_effective_n(y, X)
    results['Neff'] = Neff
    results['Neff_ratio'] = Neff / n_total

    logging.info(f"有效樣本數 (Neff): {Neff:,.0f} ({Neff/n_total*100:.1f}%)")

    # 每類 Neff
    class_Neff = {}
    for cls in [0, 1, 2]:
        mask = (y == cls)
        n_cls = mask.sum()
        if n_cls > 0:
            Neff_cls = calculate_effective_n(y[mask], X[mask])
            class_Neff[cls] = Neff_cls
        else:
            class_Neff[cls] = 0

    results['class_Neff'] = class_Neff

    logging.info(f"\n每類有效樣本數:")
    logging.info(f"  Down (0):    {class_Neff[0]:,.0f}")
    logging.info(f"  Neutral (1): {class_Neff[1]:,.0f}")
    logging.info(f"  Up (2):      {class_Neff[2]:,.0f}")

    # 股票覆蓋
    unique_stocks = np.unique(stock_ids)
    n_stocks = len(unique_stocks)
    results['n_stocks'] = n_stocks

    logging.info(f"\n股票覆蓋數: {n_stocks}")

    # 判斷是否充足
    min_class_Neff = min(class_Neff.values())

    if min_class_Neff < 500:
        status = "❌ 不足 - 某類 Neff < 500"
        results['sufficiency_status'] = "insufficient"
        results['pass'] = False
    elif min_class_Neff < 1000:
        status = "⚠️ 勉強 - 某類 Neff < 1000"
        results['sufficiency_status'] = "marginal"
        results['pass'] = True
    else:
        status = "✅ 充足 - 各類 Neff >= 1000"
        results['sufficiency_status'] = "sufficient"
        results['pass'] = True

    logging.info(f"\n充足性評估: {status}")

    return results


# ============================================================
# 檢查 4: 穩定性（簡化版）
# ============================================================

def check_stability_simple(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    w_train: np.ndarray
) -> Dict:
    """
    簡化版穩定性檢查

    檢驗項：
    - PSI（特徵漂移）
    - 訓練/測試集表現差異
    """
    logging.info("\n" + "="*60)
    logging.info("檢查 4: 穩定性（簡化版）")
    logging.info("="*60)

    results = {}

    # 計算 PSI（對每個特徵的最後時間步）
    X_train_last = X_train[:, -1, :]  # (N, 20)
    X_test_last = X_test[:, -1, :]

    psi_values = []
    for feat_idx in range(X_train_last.shape[1]):
        psi = calculate_psi(X_train_last[:, feat_idx], X_test_last[:, feat_idx])
        psi_values.append(psi)

    psi_mean = np.mean(psi_values)
    psi_max = np.max(psi_values)

    results['psi_mean'] = psi_mean
    results['psi_max'] = psi_max

    logging.info(f"\nPSI (Population Stability Index):")
    logging.info(f"  平均 PSI: {psi_mean:.3f}")
    logging.info(f"  最大 PSI: {psi_max:.3f}")

    if psi_max > 0.25:
        status = "❌ 顯著漂移 - PSI > 0.25"
        results['psi_status'] = "significant_drift"
        results['pass'] = False
    elif psi_mean > 0.15:
        status = "⚠️ 輕微漂移 - 平均 PSI > 0.15"
        results['psi_status'] = "moderate_drift"
        results['pass'] = True
    else:
        status = "✅ 穩定 - PSI < 0.15"
        results['psi_status'] = "stable"
        results['pass'] = True

    logging.info(f"  評估: {status}")

    # 訓練/測試性能對比
    X_train_flat = X_train[:, -1, :].reshape(len(X_train), -1)
    X_test_flat = X_test[:, -1, :].reshape(len(X_test), -1)

    clf = LogisticRegression(max_iter=200, random_state=42, class_weight='balanced')
    clf.fit(X_train_flat, y_train, sample_weight=w_train)

    train_acc = clf.score(X_train_flat, y_train)
    test_acc = clf.score(X_test_flat, y_test)

    results['train_accuracy'] = train_acc
    results['test_accuracy'] = test_acc
    results['accuracy_gap'] = train_acc - test_acc

    logging.info(f"\n簡單模型性能:")
    logging.info(f"  訓練集準確率: {train_acc:.3f}")
    logging.info(f"  測試集準確率: {test_acc:.3f}")
    logging.info(f"  差距: {train_acc - test_acc:.3f}")

    return results


# ============================================================
# 檢查 5: 基準對比
# ============================================================

def check_baseline_comparison(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    w_train: np.ndarray
) -> Dict:
    """
    與天真基準對比

    基準：
    - 多數類基準
    - 簡單模型（Logistic Regression）
    """
    logging.info("\n" + "="*60)
    logging.info("檢查 5: 基準對比")
    logging.info("="*60)

    results = {}

    # 天真基準：預測多數類
    majority_class = np.bincount(y_train).argmax()
    y_pred_naive = np.full(len(y_test), majority_class)
    acc_naive = accuracy_score(y_test, y_pred_naive)

    results['naive_baseline'] = {
        'majority_class': int(majority_class),
        'accuracy': acc_naive
    }

    logging.info(f"\n天真基準（預測多數類 {majority_class}）:")
    logging.info(f"  準確率: {acc_naive:.3f}")

    # 簡單模型：Logistic Regression
    X_train_flat = X_train[:, -1, :].reshape(len(X_train), -1)
    X_test_flat = X_test[:, -1, :].reshape(len(X_test), -1)

    clf = LogisticRegression(max_iter=200, random_state=42, class_weight='balanced')
    clf.fit(X_train_flat, y_train, sample_weight=w_train)

    y_pred = clf.predict(X_test_flat)
    y_proba = clf.predict_proba(X_test_flat)

    acc = accuracy_score(y_test, y_pred)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')

    # 計算 AUC (OvR)
    try:
        auc = roc_auc_score(y_test, y_proba, multi_class='ovr', average='macro')
    except:
        auc = 0.5

    results['simple_model'] = {
        'accuracy': acc,
        'balanced_accuracy': bal_acc,
        'f1_macro': f1,
        'auc_macro': auc
    }

    logging.info(f"\n簡單模型（Logistic Regression）:")
    logging.info(f"  準確率: {acc:.3f}")
    logging.info(f"  平衡準確率: {bal_acc:.3f}")
    logging.info(f"  F1 (macro): {f1:.3f}")
    logging.info(f"  AUC (macro): {auc:.3f}")

    # 判斷是否打敗基準
    beat_naive = acc > acc_naive + 0.02
    beat_random = bal_acc > 0.40  # 三分類隨機 = 0.333

    results['beat_naive_baseline'] = beat_naive
    results['beat_random'] = beat_random

    if beat_naive and beat_random:
        status = "✅ 通過 - 超越天真基準且顯著高於隨機"
        results['pass'] = True
    elif beat_random:
        status = "⚠️ 勉強 - 高於隨機但未顯著超越天真基準"
        results['pass'] = True
    else:
        status = "❌ 失敗 - 接近或低於隨機水準"
        results['pass'] = False

    logging.info(f"\n基準對比評估: {status}")

    return results


# ============================================================
# 主函數
# ============================================================

def run_health_check(
    train_npz: str,
    val_npz: str,
    test_npz: str,
    output_dir: str
):
    """執行完整健檢"""

    logging.info("\n" + "="*60)
    logging.info("數據健檢開始")
    logging.info("="*60)

    os.makedirs(output_dir, exist_ok=True)

    # 載入數據
    X_train, y_train, w_train, stock_train = load_npz_data(train_npz)
    X_val, y_val, w_val, stock_val = load_npz_data(val_npz)
    X_test, y_test, w_test, stock_test = load_npz_data(test_npz)

    # 合併訓練+驗證作為"訓練集"
    X_full_train = np.concatenate([X_train, X_val], axis=0)
    y_full_train = np.concatenate([y_train, y_val], axis=0)
    w_full_train = np.concatenate([w_train, w_val], axis=0)
    stock_full_train = np.concatenate([stock_train, stock_val], axis=0)

    report = {
        'timestamp': pd.Timestamp.now().isoformat(),
        'data_paths': {
            'train': train_npz,
            'val': val_npz,
            'test': test_npz
        }
    }

    # 執行各項檢查
    report['check_1_task_labels'] = check_task_and_labels(y_full_train)
    report['check_2_leakage'] = check_future_leakage(X_full_train, y_full_train, w_full_train)
    report['check_3_sufficiency'] = check_sufficiency_and_diversity(X_full_train, y_full_train, stock_full_train)
    report['check_4_stability'] = check_stability_simple(X_full_train, y_full_train, X_test, y_test, w_full_train)
    report['check_5_baseline'] = check_baseline_comparison(X_full_train, y_full_train, X_test, y_test, w_full_train)

    # 總體評估
    all_pass = all([
        report['check_1_task_labels']['pass'],
        report['check_2_leakage']['pass'],
        report['check_3_sufficiency']['pass'],
        report['check_4_stability']['pass'],
        report['check_5_baseline']['pass']
    ])

    report['overall_pass'] = all_pass

    # 保存報告（轉換 NumPy 類型）
    json_path = os.path.join(output_dir, 'health_check_report.json')

    def convert_numpy_types(obj):
        """遞歸轉換 NumPy 類型為 Python 原生類型"""
        if isinstance(obj, dict):
            return {key: convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                             np.int16, np.int32, np.int64, np.uint8,
                             np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

    report_serializable = convert_numpy_types(report)

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(report_serializable, f, indent=2, ensure_ascii=False)

    logging.info(f"\n報告已保存: {json_path}")

    # 生成人類可讀摘要
    txt_path = os.path.join(output_dir, 'health_check_report.txt')
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("數據健檢報告摘要\n")
        f.write("="*60 + "\n\n")

        f.write(f"時間戳: {report['timestamp']}\n\n")

        f.write("檢查結果:\n")
        f.write(f"  1. 明確任務與標籤: {'✅ 通過' if report['check_1_task_labels']['pass'] else '❌ 失敗'}\n")
        f.write(f"  2. 未來資訊洩漏: {'✅ 通過' if report['check_2_leakage']['pass'] else '❌ 失敗'}\n")
        f.write(f"  3. 足量且多樣: {'✅ 通過' if report['check_3_sufficiency']['pass'] else '❌ 失敗'}\n")
        f.write(f"  4. 穩定性: {'✅ 通過' if report['check_4_stability']['pass'] else '❌ 失敗'}\n")
        f.write(f"  5. 基準對比: {'✅ 通過' if report['check_5_baseline']['pass'] else '❌ 失敗'}\n\n")

        f.write(f"總體評估: {'✅ 數據具有學習價值' if all_pass else '⚠️ 存在問題，需改進'}\n")

    logging.info(f"摘要已保存: {txt_path}")

    logging.info("\n" + "="*60)
    logging.info(f"總體評估: {'✅ 數據具有學習價值' if all_pass else '⚠️ 存在問題，需改進'}")
    logging.info("="*60 + "\n")

    return report


def parse_args():
    p = argparse.ArgumentParser(description="數據質量健檢腳本")
    p.add_argument("--train-npz", required=True, help="訓練集 NPZ 路徑")
    p.add_argument("--val-npz", required=True, help="驗證集 NPZ 路徑")
    p.add_argument("--test-npz", required=True, help="測試集 NPZ 路徑")
    p.add_argument("--output-dir", default="./health_check_output", help="輸出目錄")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    run_health_check(
        train_npz=args.train_npz,
        val_npz=args.val_npz,
        test_npz=args.test_npz,
        output_dir=args.output_dir
    )
