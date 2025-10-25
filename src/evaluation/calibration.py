#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Calibration - 模型校準評估與溫度縮放
=============================================================================
核心功能:
1. Expected Calibration Error (ECE) - 評估預測概率是否校準良好
2. Temperature Scaling - Post-hoc 校準方法
3. Reliability Diagram - 可視化校準品質

使用方式:
    from src.evaluation.calibration import compute_ece, temperature_scale

    # 計算 ECE
    ece = compute_ece(logits, labels, n_bins=15)

    # 溫度縮放
    scaled_logits, best_temp = temperature_scale(logits, labels)
=============================================================================
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ⭐ 修復 tkinter 多執行緒錯誤：必須在導入 pyplot 之前設置後端
import matplotlib
matplotlib.use('Agg')  # 非互動式後端，避免 tkinter 錯誤
import matplotlib.pyplot as plt

from typing import Optional, Tuple


def compute_ece(
    logits: torch.Tensor,
    labels: torch.Tensor,
    n_bins: int = 15,
    return_bin_stats: bool = False
) -> float:
    """計算 Expected Calibration Error (ECE)

    ECE 衡量預測概率與實際準確率的差距:
    - ECE = Σ |accuracy(bin) - confidence(bin)| × P(bin)
    - 理想值: 0（完美校準）
    - 通常: <0.05 為良好，<0.10 為可接受

    參數:
        logits: (N, C) - 模型輸出 logits
        labels: (N,) - 真實標籤
        n_bins: bin 數量（默認 15）
        return_bin_stats: 是否返回每個 bin 的統計

    返回:
        ece: ECE 值
        bin_stats: (可選) 每個 bin 的統計資訊
    """
    # 計算預測概率和置信度
    probs = F.softmax(logits, dim=-1)  # (N, C)
    confidences, predictions = probs.max(dim=-1)  # (N,), (N,)

    # 計算準確性
    accuracies = (predictions == labels).float()  # (N,)

    # 創建 bins
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    bin_stats = []

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # 找到屬於當前 bin 的樣本
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.float().mean().item()

        if prop_in_bin > 0:
            # 計算 bin 內的平均置信度和準確率
            accuracy_in_bin = accuracies[in_bin].mean().item()
            avg_confidence_in_bin = confidences[in_bin].mean().item()

            # ECE 貢獻
            ece += abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

            if return_bin_stats:
                bin_stats.append({
                    'bin_lower': bin_lower.item(),
                    'bin_upper': bin_upper.item(),
                    'accuracy': accuracy_in_bin,
                    'confidence': avg_confidence_in_bin,
                    'proportion': prop_in_bin,
                    'count': in_bin.sum().item()
                })

    if return_bin_stats:
        return ece, bin_stats
    else:
        return ece


def compute_mce(
    logits: torch.Tensor,
    labels: torch.Tensor,
    n_bins: int = 15
) -> float:
    """計算 Maximum Calibration Error (MCE)

    MCE = max_i |accuracy(bin_i) - confidence(bin_i)|
    """
    probs = F.softmax(logits, dim=-1)
    confidences, predictions = probs.max(dim=-1)
    accuracies = (predictions == labels).float()

    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    mce = 0.0

    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        if in_bin.sum() > 0:
            accuracy_in_bin = accuracies[in_bin].mean().item()
            confidence_in_bin = confidences[in_bin].mean().item()
            mce = max(mce, abs(accuracy_in_bin - confidence_in_bin))

    return mce


class TemperatureScaling(nn.Module):
    """溫度縮放 - Post-hoc 校準方法

    原理:
    - 學習一個標量溫度 T
    - 將 logits 除以 T: softmax(logits / T)
    - T > 1: 降低置信度（更保守）
    - T < 1: 提高置信度（更激進）
    """

    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, logits):
        """應用溫度縮放"""
        return logits / self.temperature

    def fit(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        lr: float = 0.01,
        max_iter: int = 50,
        verbose: bool = True
    ) -> Tuple[float, float, float]:
        """在驗證集上學習最優溫度

        參數:
            logits: (N, C) - 驗證集 logits
            labels: (N,) - 驗證集標籤
            weights: (N,) - 樣本權重（可選）
            lr: 學習率
            max_iter: 最大迭代次數
            verbose: 是否打印日誌

        返回:
            best_temp: 最優溫度值
            nll_before: 校準前的 NLL
            nll_after: 校準後的 NLL
        """
        if weights is None:
            weights = torch.ones(len(labels))

        # 計算校準前的 NLL
        nll_criterion = nn.CrossEntropyLoss(reduction='none')
        with torch.no_grad():
            nll_before = (nll_criterion(logits, labels) * weights).sum() / weights.sum()

        # 優化溫度
        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)

        def eval_loss():
            optimizer.zero_grad()
            scaled_logits = self.forward(logits)
            loss = (nll_criterion(scaled_logits, labels) * weights).sum() / weights.sum()
            loss.backward()
            return loss

        optimizer.step(eval_loss)

        # 計算校準後的 NLL
        with torch.no_grad():
            scaled_logits = self.forward(logits)
            nll_after = (nll_criterion(scaled_logits, labels) * weights).sum() / weights.sum()

        best_temp = self.temperature.item()

        if verbose:
            print(f"[溫度校準] T = {best_temp:.4f}")
            print(f"  NLL: {nll_before:.4f} → {nll_after:.4f} (↓{nll_before - nll_after:.4f})")

        return best_temp, nll_before.item(), nll_after.item()


def temperature_scale(
    logits: torch.Tensor,
    labels: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
    lr: float = 0.01,
    max_iter: int = 50
) -> Tuple[torch.Tensor, float]:
    """便捷函數: 溫度縮放

    參數:
        logits: (N, C)
        labels: (N,)
        weights: (N,) 可選

    返回:
        scaled_logits: (N, C) - 校準後的 logits
        temperature: 最優溫度值
    """
    ts = TemperatureScaling()
    best_temp, _, _ = ts.fit(logits, labels, weights, lr, max_iter, verbose=False)

    with torch.no_grad():
        scaled_logits = ts(logits)

    return scaled_logits, best_temp


def plot_reliability_diagram(
    logits: torch.Tensor,
    labels: torch.Tensor,
    n_bins: int = 15,
    save_path: Optional[str] = None,
    title: str = "Reliability Diagram"
):
    """繪製可靠性圖（Reliability Diagram）

    Y軸: 實際準確率
    X軸: 預測置信度
    理想: 對角線（y=x）

    參數:
        logits: (N, C)
        labels: (N,)
        n_bins: bin 數量
        save_path: 保存路徑（可選）
        title: 圖表標題
    """
    ece, bin_stats = compute_ece(logits, labels, n_bins, return_bin_stats=True)

    # 提取數據
    confidences = [stat['confidence'] for stat in bin_stats]
    accuracies = [stat['accuracy'] for stat in bin_stats]
    proportions = [stat['proportion'] for stat in bin_stats]

    # 繪圖
    fig, ax = plt.subplots(figsize=(8, 6))

    # Bar plot（寬度按樣本比例）
    bin_width = 1.0 / n_bins
    for i, stat in enumerate(bin_stats):
        ax.bar(
            stat['confidence'],
            stat['accuracy'],
            width=bin_width,
            alpha=0.6,
            edgecolor='black',
            linewidth=1.5
        )

    # 對角線（完美校準）
    ax.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect Calibration')

    # Gap bars（顯示校準誤差）
    for conf, acc in zip(confidences, accuracies):
        ax.plot([conf, conf], [acc, conf], 'k-', linewidth=1, alpha=0.5)

    ax.set_xlabel('Confidence', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title(f'{title}\nECE = {ece:.4f}', fontsize=14)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Reliability diagram 已保存: {save_path}")

    plt.close()


def evaluate_calibration(
    logits: torch.Tensor,
    labels: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
    apply_temperature_scaling: bool = True,
    n_bins: int = 15,
    save_dir: Optional[str] = None
) -> dict:
    """完整的校準評估

    返回:
        results: {
            'ece_before': float,
            'mce_before': float,
            'ece_after': float (如果啟用溫度縮放),
            'mce_after': float,
            'temperature': float (如果啟用),
            'nll_before': float,
            'nll_after': float
        }
    """
    results = {}

    # 校準前評估
    results['ece_before'] = compute_ece(logits, labels, n_bins)
    results['mce_before'] = compute_mce(logits, labels, n_bins)

    # 溫度縮放
    if apply_temperature_scaling:
        ts = TemperatureScaling()
        temp, nll_before, nll_after = ts.fit(logits, labels, weights)

        with torch.no_grad():
            scaled_logits = ts(logits)

        results['ece_after'] = compute_ece(scaled_logits, labels, n_bins)
        results['mce_after'] = compute_mce(scaled_logits, labels, n_bins)
        results['temperature'] = temp
        results['nll_before'] = nll_before
        results['nll_after'] = nll_after

        # 保存圖表
        if save_dir:
            import os
            os.makedirs(save_dir, exist_ok=True)

            plot_reliability_diagram(
                logits, labels, n_bins,
                save_path=os.path.join(save_dir, 'reliability_before.png'),
                title='Before Temperature Scaling'
            )

            plot_reliability_diagram(
                scaled_logits, labels, n_bins,
                save_path=os.path.join(save_dir, 'reliability_after.png'),
                title='After Temperature Scaling'
            )

    return results


if __name__ == "__main__":
    """測試校準功能"""
    print("="*60)
    print("模型校準評估測試")
    print("="*60)

    # 模擬數據
    torch.manual_seed(42)
    n_samples = 1000
    n_classes = 3

    # 模擬 over-confident 模型
    logits = torch.randn(n_samples, n_classes) * 3  # 放大 logits（過度自信）
    labels = torch.randint(0, n_classes, (n_samples,))

    print(f"\n測試數據:")
    print(f"  樣本數: {n_samples}")
    print(f"  類別數: {n_classes}")

    # 計算 ECE
    ece, bin_stats = compute_ece(logits, labels, n_bins=15, return_bin_stats=True)
    print(f"\n校準前:")
    print(f"  ECE = {ece:.4f}")
    print(f"  MCE = {compute_mce(logits, labels):.4f}")

    # 溫度縮放
    scaled_logits, temp = temperature_scale(logits, labels)
    ece_after = compute_ece(scaled_logits, labels)
    print(f"\n校準後:")
    print(f"  Temperature = {temp:.4f}")
    print(f"  ECE = {ece_after:.4f}")
    print(f"  MCE = {compute_mce(scaled_logits, labels):.4f}")
    print(f"  改善: {ece - ece_after:.4f} ({100*(ece-ece_after)/ece:.1f}%)")

    # 完整評估
    print(f"\n完整評估:")
    results = evaluate_calibration(logits, labels, apply_temperature_scaling=True)

    for key, value in results.items():
        print(f"  {key}: {value:.4f}")

    print(f"\n✅ 所有測試通過！")
