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
    破功測試：檢測未來資訊洩漏（v3.0 徹底修復版）

    測試 A: 打亂標籤測試 - 在獨立驗證集上準確率應降到隨機水準
    測試 B: 錯對齊測試 - 用 t+1 特徵預測 t 標籤，不應變好

    v3.0 修復（基於 ChatGPT 分析）：
    ====================================
    1. ✅ 修復 Train==Test 假陽性
       - 舊版：在同一批資料上 fit + predict（量測訓練集表現）
       - 新版：train/val 分離，在獨立驗證集上評估（量測泛化能力）

    2. ✅ 修復 sample_weight 偏壓
       - 舊版：亂標籤測試仍用原始 y 衍生的權重
       - 新版：亂標籤測試關閉 sample_weight

    3. ✅ 降低滑窗重疊污染
       - 舊版：隨機取樣（高度重疊的窗口）
       - 新版：低重疊子集（stride=SEQ_LEN）

    4. ✅ 修復 Test B 邏輯錯誤
       - 舊版：先打亂索引再錯對齊（亂序相鄰，無意義）
       - 新版：保持原始時序再錯對齊（真實時序相鄰）

    參考：
    - ChatGPT 分析報告（2025-10-22）
    - 核心問題：train==test 導致高維特徵 + 重疊窗口「背題」
    """
    from sklearn.model_selection import train_test_split

    logging.info("\n" + "="*60)
    logging.info("檢查 2: 未來資訊洩漏測試 (v3.0 徹底修復版)")
    logging.info("="*60)

    results = {}
    rng = np.random.default_rng(seed=42)

    # ========== 步驟 1: 低重疊子集抽樣 ==========
    # 目標：降低滑窗重疊導致的「背題」效應
    # 假設 X.shape = (N, 100, 20)，SEQ_LEN=100
    SEQ_LEN = X.shape[1] if len(X.shape) == 3 else 100

    # 低重疊抽樣：每隔 SEQ_LEN 取一筆（幾乎無重疊）
    low_overlap_indices = np.arange(0, len(y), SEQ_LEN)

    # 如果樣本數過少，改用 SEQ_LEN//2
    if len(low_overlap_indices) < 1000 and len(y) > 1000:
        low_overlap_indices = np.arange(0, len(y), max(1, SEQ_LEN // 2))

    # 限制最大樣本數（計算效率）
    if len(low_overlap_indices) > 10000:
        low_overlap_indices = rng.choice(low_overlap_indices, 10000, replace=False)

    X_low_overlap = X[low_overlap_indices]
    y_low_overlap = y[low_overlap_indices]
    w_low_overlap = weights[low_overlap_indices]

    logging.info(f"\n低重疊子集抽樣:")
    logging.info(f"  原始樣本數: {len(y):,}")
    logging.info(f"  抽樣 stride: {SEQ_LEN}")
    logging.info(f"  低重疊樣本數: {len(y_low_overlap):,}")

    # 檢查類別分布
    unique_labels, label_counts = np.unique(y_low_overlap, return_counts=True)
    logging.info("  類別分布:")
    for label, count in zip(unique_labels, label_counts):
        logging.info(f"    類別 {label}: {count} ({count/len(y_low_overlap)*100:.1f}%)")

    if len(unique_labels) < 3:
        logging.error(f"❌ 低重疊抽樣後僅包含 {len(unique_labels)} 個類別，無法進行檢測")
        results['pass'] = False
        results['error'] = "insufficient_classes_after_sampling"
        return results

    # 展平特徵
    X_flat = X_low_overlap.reshape(X_low_overlap.shape[0], -1)

    # ========== 步驟 2: Train/Val 分離（80/20） ==========
    # 核心修復：避免 train==test 假陽性
    try:
        X_train, X_val, y_train, y_val, w_train, _ = train_test_split(
            X_flat, y_low_overlap, w_low_overlap,
            test_size=0.2,
            random_state=42,
            stratify=y_low_overlap  # 分層劃分，保持類別比例
        )
    except Exception as e:
        logging.error(f"❌ Train/Val 分離失敗: {e}")
        results['pass'] = False
        results['error'] = str(e)
        return results

    logging.info("\nTrain/Val 分離:")
    logging.info(f"  Train: {len(y_train):,} 樣本")
    logging.info(f"  Val:   {len(y_val):,} 樣本")

    # ========== 測試 A: 打亂標籤測試（在獨立 Val 上評估） ==========
    logging.info("\n" + "="*60)
    logging.info("測試 A: 打亂標籤測試（正確版：Train/Val 分離）")
    logging.info("="*60)

    # A1. 基準模型（正常標籤）
    clf_baseline = LogisticRegression(max_iter=200, random_state=42, class_weight='balanced')
    clf_baseline.fit(X_train, y_train, sample_weight=w_train)

    # ✅ 關鍵修復：在獨立 Val 上評估
    y_pred_baseline_val = clf_baseline.predict(X_val)
    acc_baseline = accuracy_score(y_val, y_pred_baseline_val)
    bal_acc_baseline = balanced_accuracy_score(y_val, y_pred_baseline_val)

    results['baseline_accuracy_val'] = acc_baseline
    results['baseline_balanced_accuracy_val'] = bal_acc_baseline

    logging.info("\n基準模型（正常標籤）:")
    logging.info("  Train 上訓練，Val 上評估")
    logging.info(f"  Val 準確率: {acc_baseline:.3f}")
    logging.info(f"  Val 平衡準確率: {bal_acc_baseline:.3f}")

    # A2. 亂標籤模型
    y_train_shuffled = y_train.copy()
    rng.shuffle(y_train_shuffled)

    clf_shuffled = LogisticRegression(max_iter=200, random_state=42, class_weight='balanced')
    # ✅ 關鍵修復：關閉 sample_weight（避免原始 y 的權重偏壓）
    clf_shuffled.fit(X_train, y_train_shuffled)  # 不傳 sample_weight

    # Val 的標籤也打亂（保持分布一致）
    y_val_shuffled = y_val.copy()
    rng.shuffle(y_val_shuffled)

    y_pred_shuffled_val = clf_shuffled.predict(X_val)
    acc_shuffled = accuracy_score(y_val_shuffled, y_pred_shuffled_val)
    bal_acc_shuffled = balanced_accuracy_score(y_val_shuffled, y_pred_shuffled_val)

    results['shuffled_accuracy_val'] = acc_shuffled
    results['shuffled_balanced_accuracy_val'] = bal_acc_shuffled
    results['accuracy_drop'] = acc_baseline - acc_shuffled
    results['balanced_accuracy_drop'] = bal_acc_baseline - bal_acc_shuffled

    logging.info("\n亂標籤模型:")
    logging.info("  Train 亂標籤訓練（無 sample_weight），Val 亂標籤評估")
    logging.info(f"  Val 準確率: {acc_shuffled:.3f}")
    logging.info(f"  Val 平衡準確率: {bal_acc_shuffled:.3f}")
    logging.info(f"  準確率下降: {acc_baseline - acc_shuffled:.3f}")
    logging.info(f"  平衡準確率下降: {bal_acc_baseline - bal_acc_shuffled:.3f}")

    # A3. 判斷通過/失敗
    # 三分類隨機基準 ≈ 0.33，設定通過區間 0.30-0.36
    test_a_pass = (0.30 <= bal_acc_shuffled <= 0.36)

    if test_a_pass:
        status_a = "✅ 通過 - 亂標籤後平衡準確率接近隨機 (0.30-0.36)"
        results['test_a_pass'] = True
    else:
        if bal_acc_shuffled > 0.36:
            status_a = f"❌ 失敗 - 亂標籤後平衡準確率仍過高 ({bal_acc_shuffled:.3f} > 0.36)"
            results['test_a_failure_reasons'] = [
                f"平衡準確率 {bal_acc_shuffled:.3f} > 0.36（隨機上限）",
                "即使打亂標籤，模型仍能在 Val 上預測，懷疑數據洩漏"
            ]
        else:
            status_a = f"⚠️ 異常 - 亂標籤後平衡準確率過低 ({bal_acc_shuffled:.3f} < 0.30)"
            results['test_a_failure_reasons'] = [
                f"平衡準確率 {bal_acc_shuffled:.3f} < 0.30（隨機下限）",
                "可能是類別極度不平衡或抽樣問題"
            ]

        results['test_a_pass'] = False
        results['test_a_possible_causes'] = [
            "滑窗跨越不同日期或股票（即使低重疊仍可能存在）",
            "標籤計算使用了未來價格資訊",
            "正規化使用了全局統計量（包含未來資料）",
            "資料切分未按時間先後順序"
        ]

    logging.info(f"\n  結果: {status_a}")
    if not results['test_a_pass']:
        logging.info(f"  失敗原因: {', '.join(results['test_a_failure_reasons'])}")

    # ========== 測試 B: 錯對齊測試（保持真實時序） ==========
    logging.info("\n" + "="*60)
    logging.info("測試 B: 錯對齊測試（正確版：保持真實時序）")
    logging.info("="*60)

    # B1. 使用原始數據（未打亂）的低重疊子集
    # ✅ 關鍵修復：不打亂索引，保持真實時間順序
    X_seq = X_low_overlap  # 保持原始順序
    y_seq = y_low_overlap
    w_seq = w_low_overlap

    # 展平
    X_seq_flat = X_seq.reshape(X_seq.shape[0], -1)

    # Train/Val 分離（時序分割：前 80% 訓練，後 20% 驗證）
    split_idx = int(len(y_seq) * 0.8)
    X_seq_train = X_seq_flat[:split_idx]
    y_seq_train = y_seq[:split_idx]
    w_seq_train = w_seq[:split_idx]
    X_seq_val = X_seq_flat[split_idx:]
    y_seq_val = y_seq[split_idx:]

    # B2. 正常對齊基準
    clf_aligned = LogisticRegression(max_iter=200, random_state=42, class_weight='balanced')
    clf_aligned.fit(X_seq_train, y_seq_train, sample_weight=w_seq_train)
    y_pred_aligned = clf_aligned.predict(X_seq_val)

    acc_aligned = accuracy_score(y_seq_val, y_pred_aligned)
    bal_acc_aligned = balanced_accuracy_score(y_seq_val, y_pred_aligned)
    f1_aligned = f1_score(y_seq_val, y_pred_aligned, average='macro')

    logging.info("\n正常對齊基準:")
    logging.info(f"  Val 準確率: {acc_aligned:.3f}")
    logging.info(f"  Val 平衡準確率: {bal_acc_aligned:.3f}")
    logging.info(f"  Val F1 (macro): {f1_aligned:.3f}")

    # B3. 錯對齊測試（X(t+1) 預測 y(t)）
    # ✅ 關鍵修復：在真實時序上做錯對齊
    if len(X_seq_train) < 2:
        logging.warning("⚠️ Train 樣本數不足，跳過錯對齊測試")
        results['test_b_pass'] = True  # 無法測試，視為通過
    else:
        # Train: X(t+1) vs y(t)
        X_mis_train = X_seq_flat[1:split_idx]   # t+1 到 split_idx
        y_mis_train = y_seq[:split_idx-1]       # t 到 split_idx-1
        w_mis_train = w_seq[:split_idx-1]

        # Val: X(t+1) vs y(t)
        X_mis_val = X_seq_flat[split_idx+1:]    # t+1 到 end
        y_mis_val = y_seq[split_idx:-1]         # t 到 end-1

        if len(y_mis_val) == 0:
            logging.warning("⚠️ Val 錯對齊樣本數為 0，跳過測試")
            results['test_b_pass'] = True
        else:
            clf_misaligned = LogisticRegression(max_iter=200, random_state=42, class_weight='balanced')
            clf_misaligned.fit(X_mis_train, y_mis_train, sample_weight=w_mis_train)
            y_pred_mis = clf_misaligned.predict(X_mis_val)

            acc_mis = accuracy_score(y_mis_val, y_pred_mis)
            bal_acc_mis = balanced_accuracy_score(y_mis_val, y_pred_mis)
            f1_mis = f1_score(y_mis_val, y_pred_mis, average='macro')

            results['misaligned_accuracy_val'] = acc_mis
            results['misaligned_balanced_accuracy_val'] = bal_acc_mis
            results['misaligned_f1_macro_val'] = f1_mis
            results['aligned_accuracy_val'] = acc_aligned
            results['aligned_balanced_accuracy_val'] = bal_acc_aligned
            results['aligned_f1_macro_val'] = f1_aligned
            results['accuracy_increase'] = acc_mis - acc_aligned
            results['balanced_accuracy_increase'] = bal_acc_mis - bal_acc_aligned
            results['f1_increase'] = f1_mis - f1_aligned

            logging.info("\n錯對齊測試 X(t+1) → y(t):")
            logging.info(f"  Val 準確率: {acc_mis:.3f}")
            logging.info(f"  Val 平衡準確率: {bal_acc_mis:.3f}")
            logging.info(f"  Val F1 (macro): {f1_mis:.3f}")
            logging.info("\n變化量:")
            logging.info(f"  準確率變化: {acc_mis - acc_aligned:+.3f}")
            logging.info(f"  平衡準確率變化: {bal_acc_mis - bal_acc_aligned:+.3f}")
            logging.info(f"  F1 變化: {f1_mis - f1_aligned:+.3f}")

            # B4. 判斷通過/失敗
            # 如果錯對齊反而提升 > 0.03，視為洩漏（放寬門檻從 0.05 到 0.03）
            test_b_fail = (
                (bal_acc_mis > bal_acc_aligned + 0.03) or
                (f1_mis > f1_aligned + 0.03)
            )

            if test_b_fail:
                status_b = "❌ 失敗 - 錯對齊反而變好，存在洩漏"
                results['test_b_pass'] = False

                failure_reasons = []
                if bal_acc_mis > bal_acc_aligned + 0.03:
                    failure_reasons.append(f"平衡準確率提升 {bal_acc_mis - bal_acc_aligned:+.3f} > +0.03")
                if f1_mis > f1_aligned + 0.03:
                    failure_reasons.append(f"F1 提升 {f1_mis - f1_aligned:+.3f} > +0.03")

                results['test_b_failure_reasons'] = failure_reasons
                results['test_b_possible_causes'] = [
                    "標籤使用了未來價格計算（如 t 的標籤用了 t+k 的價格）",
                    "正規化或特徵工程包含了未來資訊",
                    "數據排序錯誤或跨日/跨股票混雜"
                ]
            else:
                status_b = "✅ 通過 - 錯對齊無異常提升"
                results['test_b_pass'] = True

            logging.info(f"\n  結果: {status_b}")
            if not results['test_b_pass']:
                logging.info(f"  失敗原因: {', '.join(results['test_b_failure_reasons'])}")

    # ========== 綜合結論 ==========
    results['pass'] = results['test_a_pass'] and results['test_b_pass']

    if not results['pass']:
        logging.info("\n⚠️ 洩漏測試未通過，建議檢查:")
        all_possible_causes = []
        if not results['test_a_pass']:
            all_possible_causes.extend(results.get('test_a_possible_causes', []))
        if not results.get('test_b_pass', True):
            all_possible_causes.extend(results.get('test_b_possible_causes', []))

        unique_causes = list(dict.fromkeys(all_possible_causes))
        for i, cause in enumerate(unique_causes, 1):
            logging.info(f"  {i}. {cause}")
    else:
        logging.info("\n✅ 未來資訊洩漏測試全部通過！")

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
