# -*- coding: utf-8 -*-
"""
check_data_health_v5.py - V5 資料健康檢查工具
=============================================================================
【功能說明】
檢查 extract_tw_stock_data_v5.py 產生的資料品質，評估是否適合用於訓練模型

【檢查項目】
1. 基礎檢查：檔案存在性、資料格式、維度正確性
2. 資料品質：NaN/Inf、數值範圍、標準化狀態
3. 標籤分布：類別平衡、標籤有效性
4. 樣本權重：分布合理性、極端值
5. 統計分析：特徵統計、相關性分析
6. 訓練適用性：資料量、股票覆蓋、時序完整性

【使用方式】
  python scripts/check_data_health_v5.py --data-dir ./data/processed_v5/npz

【輸出】
  - 控制台：彩色健康報告（✅ 健康 / ⚠️ 警告 / ❌ 錯誤）
  - JSON：詳細檢查結果（health_report.json）
  - 視覺化：標籤分布、權重分布、特徵統計圖（可選）

版本：v1.0
更新：2025-10-20
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd

# 設定日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# ANSI 顏色碼
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def parse_args():
    """解析命令列參數"""
    p = argparse.ArgumentParser(
        "check_data_health_v5",
        description="檢查 V5 資料健康程度與訓練適用性"
    )
    p.add_argument(
        "--data-dir",
        default="./data/processed_v5/npz",
        type=str,
        help="NPZ 資料目錄（包含 train/val/test.npz）"
    )
    p.add_argument(
        "--save-report",
        action="store_true",
        default=True,
        help="保存 JSON 報告"
    )
    p.add_argument(
        "--plot",
        action="store_true",
        default=False,
        help="生成視覺化圖表（需要 matplotlib）"
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="顯示詳細檢查資訊"
    )
    return p.parse_args()


def print_section(title: str):
    """輸出區塊標題"""
    print(f"\n{Colors.BOLD}{Colors.HEADER}{'='*70}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.HEADER}{title}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.HEADER}{'='*70}{Colors.ENDC}\n")


def print_status(name: str, status: str, message: str = ""):
    """輸出狀態訊息"""
    if status == "pass":
        symbol = f"{Colors.OKGREEN}✅{Colors.ENDC}"
    elif status == "warning":
        symbol = f"{Colors.WARNING}⚠️{Colors.ENDC}"
    else:  # fail
        symbol = f"{Colors.FAIL}❌{Colors.ENDC}"

    msg = f" - {message}" if message else ""
    print(f"{symbol} {name}{msg}")


def load_npz_data(data_dir: str) -> Dict[str, Dict[str, np.ndarray]]:
    """載入所有 NPZ 檔案"""
    splits = {}

    for split in ["train", "val", "test"]:
        npz_path = os.path.join(data_dir, f"stock_embedding_{split}.npz")

        if not os.path.exists(npz_path):
            logging.warning(f"找不到檔案: {npz_path}")
            splits[split] = None
            continue

        try:
            data = np.load(npz_path, allow_pickle=True)
            splits[split] = {
                "X": data["X"],
                "y": data["y"],
                "weights": data.get("weights", np.ones(len(data["y"]))),
                "stock_ids": data.get("stock_ids", np.array([]))
            }
            logging.info(f"✅ 已載入 {split} 集: {npz_path}")
        except Exception as e:
            logging.error(f"❌ 載入失敗 {npz_path}: {e}")
            splits[split] = None

    return splits


def load_metadata(data_dir: str) -> Optional[Dict[str, Any]]:
    """載入 metadata"""
    meta_path = os.path.join(data_dir, "normalization_meta.json")

    if not os.path.exists(meta_path):
        logging.warning(f"找不到 metadata: {meta_path}")
        return None

    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        logging.info(f"✅ 已載入 metadata: {meta_path}")
        return meta
    except Exception as e:
        logging.error(f"❌ 載入 metadata 失敗: {e}")
        return None


# ============================================================
# 檢查函數
# ============================================================

def check_basic(splits: Dict, metadata: Dict) -> Dict[str, Any]:
    """基礎檢查：檔案存在、格式、維度"""
    results = {
        "status": "pass",
        "checks": []
    }

    print_section("1. 基礎檢查 (Basic Validation)")

    # 檢查所有 split 是否存在
    for split in ["train", "val", "test"]:
        if splits[split] is None:
            results["checks"].append({
                "name": f"{split} 集檔案存在性",
                "status": "fail",
                "message": "檔案不存在或無法載入"
            })
            print_status(f"{split} 集檔案", "fail", "不存在")
            results["status"] = "fail"
        else:
            print_status(f"{split} 集檔案", "pass", "存在")
            results["checks"].append({
                "name": f"{split} 集檔案存在性",
                "status": "pass"
            })

    # 檢查 metadata
    if metadata is None:
        results["checks"].append({
            "name": "Metadata 檔案",
            "status": "warning",
            "message": "找不到 normalization_meta.json"
        })
        print_status("Metadata 檔案", "warning", "不存在")
        if results["status"] == "pass":
            results["status"] = "warning"
    else:
        print_status("Metadata 檔案", "pass", f"版本: {metadata.get('version', 'unknown')}")
        results["checks"].append({
            "name": "Metadata 檔案",
            "status": "pass",
            "version": metadata.get('version', 'unknown')
        })

    # 檢查維度
    for split in ["train", "val", "test"]:
        if splits[split] is None:
            continue

        X = splits[split]["X"]
        y = splits[split]["y"]
        w = splits[split]["weights"]

        # 檢查形狀
        expected_shape = (len(y), 100, 20)
        if X.shape != expected_shape:
            results["checks"].append({
                "name": f"{split} 集形狀",
                "status": "fail",
                "message": f"預期 {expected_shape}，實際 {X.shape}"
            })
            print_status(f"{split} 集形狀", "fail", f"預期 {expected_shape}，實際 {X.shape}")
            results["status"] = "fail"
        else:
            print_status(f"{split} 集形狀", "pass", f"{X.shape}")
            results["checks"].append({
                "name": f"{split} 集形狀",
                "status": "pass",
                "shape": str(X.shape)
            })

        # 檢查樣本數量一致性
        if len(X) != len(y) or len(y) != len(w):
            results["checks"].append({
                "name": f"{split} 集樣本數一致性",
                "status": "fail",
                "message": f"X={len(X)}, y={len(y)}, w={len(w)}"
            })
            print_status(f"{split} 集樣本數一致", "fail", f"X={len(X)}, y={len(y)}, w={len(w)}")
            results["status"] = "fail"
        else:
            print_status(f"{split} 集樣本數一致", "pass", f"{len(y):,} 樣本")
            results["checks"].append({
                "name": f"{split} 集樣本數一致性",
                "status": "pass",
                "samples": len(y)
            })

    return results


def check_data_quality(splits: Dict) -> Dict[str, Any]:
    """資料品質檢查：NaN、Inf、數值範圍"""
    results = {
        "status": "pass",
        "checks": []
    }

    print_section("2. 資料品質檢查 (Data Quality)")

    for split in ["train", "val", "test"]:
        if splits[split] is None:
            continue

        X = splits[split]["X"]
        y = splits[split]["y"]
        w = splits[split]["weights"]

        # 檢查 NaN
        nan_count = np.isnan(X).sum()
        if nan_count > 0:
            pct = nan_count / X.size * 100
            results["checks"].append({
                "name": f"{split} 集 NaN 檢查",
                "status": "fail",
                "message": f"發現 {nan_count:,} 個 NaN ({pct:.2f}%)"
            })
            print_status(f"{split} 集 NaN", "fail", f"{nan_count:,} 個 ({pct:.2f}%)")
            results["status"] = "fail"
        else:
            print_status(f"{split} 集 NaN", "pass", "無 NaN")
            results["checks"].append({
                "name": f"{split} 集 NaN 檢查",
                "status": "pass"
            })

        # 檢查 Inf
        inf_count = np.isinf(X).sum()
        if inf_count > 0:
            pct = inf_count / X.size * 100
            results["checks"].append({
                "name": f"{split} 集 Inf 檢查",
                "status": "fail",
                "message": f"發現 {inf_count:,} 個 Inf ({pct:.2f}%)"
            })
            print_status(f"{split} 集 Inf", "fail", f"{inf_count:,} 個 ({pct:.2f}%)")
            results["status"] = "fail"
        else:
            print_status(f"{split} 集 Inf", "pass", "無 Inf")
            results["checks"].append({
                "name": f"{split} 集 Inf 檢查",
                "status": "pass"
            })

        # 檢查數值範圍（Z-score 應該大致在 [-5, 5]）
        X_flat = X.reshape(-1)
        min_val, max_val = X_flat.min(), X_flat.max()

        if min_val < -10 or max_val > 10:
            results["checks"].append({
                "name": f"{split} 集數值範圍",
                "status": "warning",
                "message": f"範圍 [{min_val:.2f}, {max_val:.2f}]（可能有異常值）"
            })
            print_status(f"{split} 集數值範圍", "warning", f"[{min_val:.2f}, {max_val:.2f}]")
            if results["status"] == "pass":
                results["status"] = "warning"
        else:
            print_status(f"{split} 集數值範圍", "pass", f"[{min_val:.2f}, {max_val:.2f}]")
            results["checks"].append({
                "name": f"{split} 集數值範圍",
                "status": "pass",
                "min": float(min_val),
                "max": float(max_val)
            })

        # 檢查標準化（訓練集應該接近 mean=0, std=1）
        if split == "train":
            X_flat_sample = X_flat[::10]  # 抽樣避免記憶體問題
            mean_val = np.mean(X_flat_sample)
            std_val = np.std(X_flat_sample)

            if abs(mean_val) > 0.1 or abs(std_val - 1.0) > 0.2:
                results["checks"].append({
                    "name": f"{split} 集標準化",
                    "status": "warning",
                    "message": f"mean={mean_val:.3f}, std={std_val:.3f}（應接近 0 和 1）"
                })
                print_status(f"{split} 集標準化", "warning", f"mean={mean_val:.3f}, std={std_val:.3f}")
                if results["status"] == "pass":
                    results["status"] = "warning"
            else:
                print_status(f"{split} 集標準化", "pass", f"mean={mean_val:.3f}, std={std_val:.3f}")
                results["checks"].append({
                    "name": f"{split} 集標準化",
                    "status": "pass",
                    "mean": float(mean_val),
                    "std": float(std_val)
                })

        # 檢查權重
        if w.size > 0:
            w_min, w_max = w.min(), w.max()
            w_mean = w.mean()

            # 權重應該正值且均值接近 1
            if w_min < 0:
                results["checks"].append({
                    "name": f"{split} 集權重正值",
                    "status": "fail",
                    "message": f"發現負權重（min={w_min:.3f}）"
                })
                print_status(f"{split} 集權重正值", "fail", f"min={w_min:.3f}")
                results["status"] = "fail"
            else:
                print_status(f"{split} 集權重正值", "pass", f"min={w_min:.3f}")

            # 檢查極端權重
            if w_max > 100:
                results["checks"].append({
                    "name": f"{split} 集權重極端值",
                    "status": "warning",
                    "message": f"最大權重 {w_max:.1f}（可能過大）"
                })
                print_status(f"{split} 集權重極端值", "warning", f"max={w_max:.1f}")
                if results["status"] == "pass":
                    results["status"] = "warning"
            else:
                print_status(f"{split} 集權重範圍", "pass", f"[{w_min:.3f}, {w_max:.3f}], mean={w_mean:.3f}")
                results["checks"].append({
                    "name": f"{split} 集權重範圍",
                    "status": "pass",
                    "min": float(w_min),
                    "max": float(w_max),
                    "mean": float(w_mean)
                })

    return results


def check_labels(splits: Dict) -> Dict[str, Any]:
    """標籤分布檢查"""
    results = {
        "status": "pass",
        "checks": [],
        "distributions": {}
    }

    print_section("3. 標籤分布檢查 (Label Distribution)")

    for split in ["train", "val", "test"]:
        if splits[split] is None:
            continue

        y = splits[split]["y"]

        # 檢查標籤有效性
        unique_labels = np.unique(y)
        expected_labels = {0, 1, 2}

        if not set(unique_labels).issubset(expected_labels):
            results["checks"].append({
                "name": f"{split} 集標籤有效性",
                "status": "fail",
                "message": f"發現無效標籤: {set(unique_labels) - expected_labels}"
            })
            print_status(f"{split} 集標籤有效", "fail", f"無效標籤: {set(unique_labels) - expected_labels}")
            results["status"] = "fail"
            continue

        # 計算標籤分布
        label_counts = np.bincount(y, minlength=3)
        label_pcts = label_counts / len(y) * 100

        results["distributions"][split] = {
            "class_0": int(label_counts[0]),
            "class_1": int(label_counts[1]),
            "class_2": int(label_counts[2]),
            "pct_0": float(label_pcts[0]),
            "pct_1": float(label_pcts[1]),
            "pct_2": float(label_pcts[2])
        }

        print(f"\n{Colors.BOLD}{split.upper()} 集標籤分布:{Colors.ENDC}")
        print(f"  Class 0 (下跌): {label_counts[0]:,} ({label_pcts[0]:.2f}%)")
        print(f"  Class 1 (持平): {label_counts[1]:,} ({label_pcts[1]:.2f}%)")
        print(f"  Class 2 (上漲): {label_counts[2]:,} ({label_pcts[2]:.2f}%)")

        # 檢查類別平衡（任何類別 < 20% 發出警告）
        min_pct = label_pcts.min()
        max_pct = label_pcts.max()

        if min_pct < 20:
            results["checks"].append({
                "name": f"{split} 集類別平衡",
                "status": "warning",
                "message": f"最小類別佔比 {min_pct:.1f}%（建議 > 20%）"
            })
            print_status(f"{split} 集類別平衡", "warning", f"最小類別 {min_pct:.1f}%")
            if results["status"] == "pass":
                results["status"] = "warning"
        else:
            print_status(f"{split} 集類別平衡", "pass", f"最小類別 {min_pct:.1f}%")
            results["checks"].append({
                "name": f"{split} 集類別平衡",
                "status": "pass",
                "min_pct": float(min_pct),
                "max_pct": float(max_pct)
            })

    return results


def check_training_suitability(splits: Dict, metadata: Dict) -> Dict[str, Any]:
    """訓練適用性檢查"""
    results = {
        "status": "pass",
        "checks": [],
        "statistics": {}
    }

    print_section("4. 訓練適用性檢查 (Training Suitability)")

    # 檢查訓練集大小（建議 > 100K 樣本）
    if splits["train"] is not None:
        train_size = len(splits["train"]["y"])
        results["statistics"]["train_size"] = train_size

        if train_size < 100000:
            results["checks"].append({
                "name": "訓練集大小",
                "status": "warning",
                "message": f"{train_size:,} 樣本（建議 > 100K）"
            })
            print_status("訓練集大小", "warning", f"{train_size:,} 樣本")
            if results["status"] == "pass":
                results["status"] = "warning"
        else:
            print_status("訓練集大小", "pass", f"{train_size:,} 樣本")
            results["checks"].append({
                "name": "訓練集大小",
                "status": "pass",
                "samples": train_size
            })

    # 檢查驗證集大小（建議 > 10K 樣本）
    if splits["val"] is not None:
        val_size = len(splits["val"]["y"])
        results["statistics"]["val_size"] = val_size

        if val_size < 10000:
            results["checks"].append({
                "name": "驗證集大小",
                "status": "warning",
                "message": f"{val_size:,} 樣本（建議 > 10K）"
            })
            print_status("驗證集大小", "warning", f"{val_size:,} 樣本")
            if results["status"] == "pass":
                results["status"] = "warning"
        else:
            print_status("驗證集大小", "pass", f"{val_size:,} 樣本")
            results["checks"].append({
                "name": "驗證集大小",
                "status": "pass",
                "samples": val_size
            })

    # 檢查股票覆蓋
    for split in ["train", "val", "test"]:
        if splits[split] is None or len(splits[split]["stock_ids"]) == 0:
            continue

        stock_ids = splits[split]["stock_ids"]
        unique_stocks = len(np.unique(stock_ids))
        results["statistics"][f"{split}_stocks"] = unique_stocks

        print_status(f"{split} 集股票數", "pass", f"{unique_stocks} 檔")
        results["checks"].append({
            "name": f"{split} 集股票數",
            "status": "pass",
            "stocks": unique_stocks
        })

    # 檢查資料切分比例
    if all(splits[s] is not None for s in ["train", "val", "test"]):
        total = sum(len(splits[s]["y"]) for s in ["train", "val", "test"])
        train_ratio = len(splits["train"]["y"]) / total
        val_ratio = len(splits["val"]["y"]) / total
        test_ratio = len(splits["test"]["y"]) / total

        results["statistics"]["split_ratios"] = {
            "train": float(train_ratio),
            "val": float(val_ratio),
            "test": float(test_ratio)
        }

        print(f"\n{Colors.BOLD}資料切分比例:{Colors.ENDC}")
        print(f"  Train: {train_ratio*100:.1f}%")
        print(f"  Val:   {val_ratio*100:.1f}%")
        print(f"  Test:  {test_ratio*100:.1f}%")

        # 檢查比例是否合理（訓練集應該 > 60%）
        if train_ratio < 0.6:
            results["checks"].append({
                "name": "切分比例",
                "status": "warning",
                "message": f"訓練集佔比 {train_ratio*100:.1f}%（建議 > 60%）"
            })
            print_status("切分比例", "warning", f"訓練集 {train_ratio*100:.1f}%")
            if results["status"] == "pass":
                results["status"] = "warning"
        else:
            print_status("切分比例", "pass", f"訓練集 {train_ratio*100:.1f}%")
            results["checks"].append({
                "name": "切分比例",
                "status": "pass"
            })

    # 檢查 metadata 配置
    if metadata is not None:
        print(f"\n{Colors.BOLD}V5 配置:{Colors.ENDC}")
        print(f"  波動率方法: {metadata.get('volatility', {}).get('method', 'unknown')}")

        tb = metadata.get('triple_barrier', {})
        print(f"  Triple-Barrier:")
        print(f"    - 止盈倍數: {tb.get('pt_multiplier', 'N/A')}")
        print(f"    - 止損倍數: {tb.get('sl_multiplier', 'N/A')}")
        print(f"    - 最大持有: {tb.get('max_holding', 'N/A')}")
        print(f"    - 最小報酬: {tb.get('min_return', 'N/A')}")

        sw = metadata.get('sample_weights', {})
        print(f"  樣本權重:")
        print(f"    - 啟用: {sw.get('enabled', False)}")
        print(f"    - Tau: {sw.get('tau', 'N/A')}")
        print(f"    - 類別平衡: {sw.get('balance_classes', False)}")

        results["statistics"]["config"] = {
            "volatility_method": metadata.get('volatility', {}).get('method'),
            "triple_barrier": tb,
            "sample_weights": sw
        }

    return results


def check_statistics(splits: Dict, verbose: bool = False) -> Dict[str, Any]:
    """統計分析"""
    results = {
        "status": "pass",
        "statistics": {}
    }

    print_section("5. 統計分析 (Statistical Analysis)")

    if splits["train"] is None:
        print_status("統計分析", "warning", "訓練集不存在，跳過")
        return results

    X_train = splits["train"]["X"]

    # 抽樣避免記憶體問題
    sample_size = min(10000, len(X_train))
    sample_indices = np.random.choice(len(X_train), sample_size, replace=False)
    X_sample = X_train[sample_indices]

    # 計算每個特徵的統計量
    X_2d = X_sample.reshape(-1, 20)  # (N*100, 20)

    feature_stats = []
    for i in range(20):
        feat = X_2d[:, i]
        stats = {
            "feature": i,
            "mean": float(np.mean(feat)),
            "std": float(np.std(feat)),
            "min": float(np.min(feat)),
            "max": float(np.max(feat)),
            "q25": float(np.percentile(feat, 25)),
            "q50": float(np.percentile(feat, 50)),
            "q75": float(np.percentile(feat, 75))
        }
        feature_stats.append(stats)

    results["statistics"]["features"] = feature_stats

    if verbose:
        print(f"\n{Colors.BOLD}特徵統計（前 10 個特徵）:{Colors.ENDC}")
        print(f"{'特徵':<6} {'均值':<8} {'標準差':<8} {'最小值':<8} {'最大值':<8}")
        print("-" * 46)
        for i in range(min(10, 20)):
            s = feature_stats[i]
            print(f"{i:<6} {s['mean']:<8.3f} {s['std']:<8.3f} {s['min']:<8.3f} {s['max']:<8.3f}")

    print_status("統計分析", "pass", f"已計算 20 個特徵的統計量")

    return results


def generate_summary(all_results: Dict[str, Any]) -> str:
    """生成總結報告"""
    print_section("總結報告 (Summary)")

    # 計算總體狀態
    statuses = [r.get("status", "pass") for r in all_results.values() if "status" in r]

    if "fail" in statuses:
        overall_status = "fail"
        status_symbol = f"{Colors.FAIL}❌ 不健康{Colors.ENDC}"
        recommendation = "資料存在嚴重問題，不建議直接用於訓練。請檢查失敗項目。"
    elif "warning" in statuses:
        overall_status = "warning"
        status_symbol = f"{Colors.WARNING}⚠️ 部分健康{Colors.ENDC}"
        recommendation = "資料大致可用，但存在一些問題。建議檢查警告項目並視情況調整。"
    else:
        overall_status = "pass"
        status_symbol = f"{Colors.OKGREEN}✅ 健康{Colors.ENDC}"
        recommendation = "資料品質良好，適合用於訓練模型。"

    print(f"\n{Colors.BOLD}整體健康狀態:{Colors.ENDC} {status_symbol}")
    print(f"\n{Colors.BOLD}建議:{Colors.ENDC}")
    print(f"  {recommendation}")

    # 統計各項檢查結果
    total_checks = sum(len(r.get("checks", [])) for r in all_results.values())
    passed_checks = sum(
        sum(1 for c in r.get("checks", []) if c.get("status") == "pass")
        for r in all_results.values()
    )
    warning_checks = sum(
        sum(1 for c in r.get("checks", []) if c.get("status") == "warning")
        for r in all_results.values()
    )
    failed_checks = sum(
        sum(1 for c in r.get("checks", []) if c.get("status") == "fail")
        for r in all_results.values()
    )

    print(f"\n{Colors.BOLD}檢查項目統計:{Colors.ENDC}")
    print(f"  總計: {total_checks} 項")
    print(f"  {Colors.OKGREEN}✅ 通過: {passed_checks}{Colors.ENDC}")
    print(f"  {Colors.WARNING}⚠️ 警告: {warning_checks}{Colors.ENDC}")
    print(f"  {Colors.FAIL}❌ 失敗: {failed_checks}{Colors.ENDC}")

    return overall_status


def save_report(all_results: Dict[str, Any], overall_status: str, output_path: str):
    """保存 JSON 報告"""
    report = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "overall_status": overall_status,
        "results": all_results
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n{Colors.OKGREEN}✅ 報告已保存: {output_path}{Colors.ENDC}")


# ============================================================
# 主程式
# ============================================================

def main():
    """主程式"""
    args = parse_args()

    print(f"{Colors.BOLD}{Colors.HEADER}")
    print("="*70)
    print("V5 資料健康檢查工具")
    print("="*70)
    print(f"{Colors.ENDC}\n")

    print(f"資料目錄: {args.data_dir}\n")

    # 載入資料
    splits = load_npz_data(args.data_dir)
    metadata = load_metadata(args.data_dir)

    # 執行檢查
    all_results = {}

    all_results["basic"] = check_basic(splits, metadata)
    all_results["quality"] = check_data_quality(splits)
    all_results["labels"] = check_labels(splits)
    all_results["suitability"] = check_training_suitability(splits, metadata)
    all_results["statistics"] = check_statistics(splits, args.verbose)

    # 生成總結
    overall_status = generate_summary(all_results)

    # 保存報告
    if args.save_report:
        report_path = os.path.join(args.data_dir, "health_report.json")
        save_report(all_results, overall_status, report_path)

    # 視覺化（可選）
    if args.plot:
        try:
            import matplotlib.pyplot as plt
            plot_visualizations(splits, args.data_dir)
        except ImportError:
            print(f"\n{Colors.WARNING}⚠️ matplotlib 未安裝，跳過視覺化{Colors.ENDC}")

    print("\n" + "="*70 + "\n")

    return 0 if overall_status != "fail" else 1


def plot_visualizations(splits: Dict, output_dir: str):
    """生成視覺化圖表"""
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    # 設定中文字體支援（避免中文顯示為方塊）
    try:
        # Windows 系統使用微軟正黑體
        mpl.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Arial Unicode MS']
        mpl.rcParams['axes.unicode_minus'] = False  # 解決負號顯示問題
    except Exception:
        # 如果設定失敗，忽略（圖表仍可正常生成，只是中文顯示為方塊）
        pass

    print_section("6. 視覺化 (Visualizations)")

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 標籤分布
    for idx, split in enumerate(["train", "val", "test"]):
        if splits[split] is None:
            continue

        y = splits[split]["y"]
        label_counts = np.bincount(y, minlength=3)

        ax = axes[0, idx]
        ax.bar([0, 1, 2], label_counts, color=['red', 'gray', 'green'])
        ax.set_title(f"{split.upper()} Label Distribution", fontsize=12, fontweight='bold')
        ax.set_xlabel("Class (0:Down, 1:Neutral, 2:Up)", fontsize=10)
        ax.set_ylabel("Sample Count", fontsize=10)
        ax.grid(True, alpha=0.3)
        # 添加數值標籤
        for i, count in enumerate(label_counts):
            ax.text(i, count, f'{count:,}', ha='center', va='bottom', fontsize=9)

    # 權重分布
    for idx, split in enumerate(["train", "val", "test"]):
        if splits[split] is None:
            continue

        w = splits[split]["weights"]

        ax = axes[1, idx]
        ax.hist(w, bins=50, alpha=0.7, edgecolor='black', color='steelblue')
        ax.set_title(f"{split.upper()} Weight Distribution", fontsize=12, fontweight='bold')
        ax.set_xlabel("Weight", fontsize=10)
        ax.set_ylabel("Frequency", fontsize=10)
        ax.axvline(w.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean={w.mean():.2f}')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    plot_path = os.path.join(output_dir, "health_visualizations.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"{Colors.OKGREEN}✅ 視覺化已保存: {plot_path}{Colors.ENDC}")
    plt.close()


if __name__ == "__main__":
    sys.exit(main())
