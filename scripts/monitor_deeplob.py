"""DeepLOB 訓練監控（從 TensorBoard 日誌讀取）

支援版本:
- V5: Triple-Barrier 標籤 + 樣本權重 + Stock Embeddings
- Generic/舊版: 標準 DeepLOB 訓練

使用方式:
    python scripts/monitor_deeplob.py                    # 自動偵測最新日誌
    python scripts/monitor_deeplob.py --log-dir logs/deeplob_v5  # 指定目錄
    python scripts/monitor_deeplob.py --export           # 匯出 CSV 供 AI 分析
    python scripts/monitor_deeplob.py --version v5       # 強制使用 V5 格式
"""

import argparse
import csv
from pathlib import Path
from collections import defaultdict

try:
    from tensorboard.backend.event_processing import event_accumulator
except ImportError:
    print("[ERROR] 需要安裝 tensorboard: pip install tensorboard")
    exit(1)


def read_tensorboard_logs(log_dir, version='auto'):
    """從 TensorBoard 日誌讀取訓練數據

    參數:
        log_dir: TensorBoard 日誌目錄
        version: 'auto', 'v5', 'generic'

    返回:
        Tuple[List[Dict], str]: (每個 epoch 的訓練數據, 偵測到的版本)
    """
    # 找到最新的日誌目錄
    log_base = Path(log_dir)
    if not log_base.exists():
        return [], 'unknown'

    # 找到最新的事件文件
    event_files = list(log_base.glob('**/events.out.tfevents.*'))
    if not event_files:
        return [], 'unknown'

    latest_event = max(event_files, key=lambda p: p.stat().st_mtime)

    # 讀取事件
    ea = event_accumulator.EventAccumulator(str(latest_event.parent))
    ea.Reload()

    # 提取標量數據
    data_by_epoch = defaultdict(dict)
    available_tags = ea.Tags()['scalars']

    # 自動偵測版本
    if version == 'auto':
        if 'Loss/val_weighted' in available_tags or 'F1/val_weighted' in available_tags:
            detected_version = 'v5'
        elif 'Loss/val' in available_tags:
            detected_version = 'generic'
        else:
            detected_version = 'unknown'
    else:
        detected_version = version

    # === V5 版本指標 ===
    if detected_version == 'v5':
        # 訓練指標
        if 'Loss/train' in available_tags:
            for event in ea.Scalars('Loss/train'):
                data_by_epoch[event.step]['train_loss'] = event.value

        if 'Accuracy/train' in available_tags:
            for event in ea.Scalars('Accuracy/train'):
                data_by_epoch[event.step]['train_acc'] = event.value

        # 驗證指標（V5 有加權和不加權兩種）
        if 'Loss/val_weighted' in available_tags:
            for event in ea.Scalars('Loss/val_weighted'):
                data_by_epoch[event.step]['val_loss_weighted'] = event.value

        if 'Accuracy/val_unweighted' in available_tags:
            for event in ea.Scalars('Accuracy/val_unweighted'):
                data_by_epoch[event.step]['val_acc'] = event.value

        if 'F1/val_weighted' in available_tags:
            for event in ea.Scalars('F1/val_weighted'):
                data_by_epoch[event.step]['val_f1_weighted'] = event.value

        if 'F1/val_unweighted' in available_tags:
            for event in ea.Scalars('F1/val_unweighted'):
                data_by_epoch[event.step]['val_f1_unweighted'] = event.value

        # 其他指標
        if 'LR' in available_tags:
            for event in ea.Scalars('LR'):
                data_by_epoch[event.step]['learning_rate'] = event.value

        if 'Grad/norm' in available_tags:
            for event in ea.Scalars('Grad/norm'):
                data_by_epoch[event.step]['grad_norm'] = event.value

    # === Generic/舊版指標 ===
    else:
        if 'Loss/train' in available_tags:
            for event in ea.Scalars('Loss/train'):
                data_by_epoch[event.step]['train_loss'] = event.value

        if 'Accuracy/train' in available_tags:
            for event in ea.Scalars('Accuracy/train'):
                data_by_epoch[event.step]['train_acc'] = event.value

        if 'Accuracy/val' in available_tags:
            for event in ea.Scalars('Accuracy/val'):
                data_by_epoch[event.step]['val_acc'] = event.value

        if 'Loss/val' in available_tags:
            for event in ea.Scalars('Loss/val'):
                data_by_epoch[event.step]['val_loss'] = event.value

        if 'Gradient/norm' in available_tags:
            for event in ea.Scalars('Gradient/norm'):
                data_by_epoch[event.step]['grad_norm'] = event.value

        if 'Learning_Rate' in available_tags:
            for event in ea.Scalars('Learning_Rate'):
                data_by_epoch[event.step]['learning_rate'] = event.value

    # 轉換為列表
    metrics = []
    for epoch in sorted(data_by_epoch.keys()):
        data = data_by_epoch[epoch]
        metric = {'epoch': int(epoch)}

        # 通用字段
        metric['train_loss'] = data.get('train_loss', 0)
        metric['train_acc'] = data.get('train_acc', 0)
        metric['val_acc'] = data.get('val_acc', 0)
        metric['learning_rate'] = data.get('learning_rate', 0)
        metric['grad_norm'] = data.get('grad_norm', 0)

        # V5 特有字段
        if detected_version == 'v5':
            metric['val_loss'] = data.get('val_loss_weighted', 0)
            metric['val_f1_weighted'] = data.get('val_f1_weighted', 0)
            metric['val_f1_unweighted'] = data.get('val_f1_unweighted', 0)
        else:
            metric['val_loss'] = data.get('val_loss', 0)

        metrics.append(metric)

    return metrics, detected_version


def export_to_csv(metrics, output_path, version='generic'):
    """匯出訓練數據到 CSV"""
    if not metrics:
        print("[WARNING] 沒有可匯出的數據")
        return

    # 根據版本選擇字段
    if version == 'v5':
        fieldnames = [
            'epoch', 'train_loss', 'val_loss', 'train_acc', 'val_acc',
            'val_f1_weighted', 'val_f1_unweighted', 'learning_rate', 'grad_norm'
        ]
    else:
        fieldnames = [
            'epoch', 'train_loss', 'val_loss', 'train_acc', 'val_acc',
            'learning_rate', 'grad_norm'
        ]

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(metrics)

    print(f"[OK] CSV 已匯出至: {output_path}")
    print(f"     共 {len(metrics)} 個 epoch 的數據")
    print(f"     版本: {version.upper()}")


def print_training_table(metrics, version='generic'):
    """打印訓練數據表格（Markdown 格式）"""
    if not metrics:
        return

    print("\n" + "=" * 120)
    print(f"訓練數據 ({version.upper()})")
    print("=" * 120)

    # 檢查是否為 V5 版本
    is_v5 = version == 'v5' and 'val_f1_weighted' in metrics[0]

    # Markdown 表格標題
    if is_v5:
        print("| Epoch | Train Loss | Val Loss | Train Acc | Val Acc    | Val F1 (W) | Val F1 (U) | LR        | Grad |")
        print("| ----- | ---------- | -------- | --------- | ---------- | ---------- | ---------- | --------- | ---- |")
    else:
        print("| Epoch | Train Loss | Val Loss | Train Acc | Val Acc    | Learning Rate | Grad Norm |")
        print("| ----- | ---------- | -------- | --------- | ---------- | ------------- | --------- |")

    # 找出最佳 val_acc 的 epoch
    best_val_acc = max(metrics, key=lambda x: x['val_acc'])
    best_val_epoch = best_val_acc['epoch']

    for m in metrics:
        val_acc_str = f"{m['val_acc']:.2f}%"

        # 標記最佳 val_acc
        if m['epoch'] == best_val_epoch:
            val_acc_str = f"**{val_acc_str}**"

        if is_v5:
            print(
                f"| {m['epoch']:<5} | "
                f"{m['train_loss']:<10.4f} | "
                f"{m['val_loss']:<8.4f} | "
                f"{m['train_acc']:<9.2f}% | "
                f"{val_acc_str:<10} | "
                f"{m.get('val_f1_weighted', 0):<10.4f} | "
                f"{m.get('val_f1_unweighted', 0):<10.4f} | "
                f"{m['learning_rate']:<9.6f} | "
                f"{m['grad_norm']:<4.2f} |"
            )
        else:
            print(
                f"| {m['epoch']:<5} | "
                f"{m['train_loss']:<10.4f} | "
                f"{m['val_loss']:<8.4f} | "
                f"{m['train_acc']:<9.2f}% | "
                f"{val_acc_str:<10} | "
                f"{m['learning_rate']:<13.6f} | "
                f"{m['grad_norm']:<9.2f} |"
            )

    print("\n註: ** 標記為最佳驗證準確率")
    if is_v5:
        print("     W = Weighted (加權), U = Unweighted (不加權)")


def print_summary(metrics, version='generic'):
    """打印訓練摘要"""
    if not metrics:
        return

    latest = metrics[-1]
    best_val = max(metrics, key=lambda x: x['val_acc'])

    is_v5 = version == 'v5' and 'val_f1_weighted' in latest

    print("\n" + "=" * 70)
    print(f"訓練摘要 ({version.upper()})")
    print("=" * 70)

    print(f"已完成 Epoch: {len(metrics)}")
    print(f"\n最新 Epoch {latest['epoch']}:")
    print(f"  Train Loss:    {latest['train_loss']:.4f}")
    print(f"  Val Loss:      {latest['val_loss']:.4f}")
    print(f"  Train Acc:     {latest['train_acc']:.2f}%")
    print(f"  Val Acc:       {latest['val_acc']:.2f}%")

    if is_v5:
        print(f"  Val F1 (W):    {latest.get('val_f1_weighted', 0):.4f}")
        print(f"  Val F1 (U):    {latest.get('val_f1_unweighted', 0):.4f}")

    print(f"  Learning Rate: {latest['learning_rate']:.6f}")
    print(f"  Grad Norm:     {latest['grad_norm']:.2f}")

    print(f"\n最佳驗證準確率: {best_val['val_acc']:.2f}% (Epoch {best_val['epoch']})")
    print(f"  Train Loss:    {best_val['train_loss']:.4f}")
    print(f"  Val Loss:      {best_val['val_loss']:.4f}")

    if is_v5 and 'val_f1_weighted' in best_val:
        print(f"  Val F1 (W):    {best_val.get('val_f1_weighted', 0):.4f}")
        print(f"  Val F1 (U):    {best_val.get('val_f1_unweighted', 0):.4f}")

    # 趨勢分析
    if len(metrics) >= 3:
        recent_3 = metrics[-3:]
        acc_change = recent_3[-1]['val_acc'] - recent_3[0]['val_acc']

        print("\n趨勢（最近 3 個 epoch）:")
        print(f"  Val Acc 變化: {acc_change:+.2f}%", end=" ")

        if acc_change > 0.5:
            print("[UP] 持續提升")
        elif acc_change > -0.5:
            print("[STABLE] 停滯")
        else:
            print("[DOWN] 下降")

        if is_v5:
            f1_change = (recent_3[-1].get('val_f1_unweighted', 0) -
                        recent_3[0].get('val_f1_unweighted', 0))
            print(f"  Val F1 (U) 變化: {f1_change:+.4f}", end=" ")

            if f1_change > 0.01:
                print("[UP] 持續提升")
            elif f1_change > -0.01:
                print("[STABLE] 停滯")
            else:
                print("[DOWN] 下降")


def main():
    parser = argparse.ArgumentParser(
        description='DeepLOB 訓練監控（從 TensorBoard 讀取）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用範例:
  python scripts/monitor_deeplob.py                          # 自動偵測最新日誌
  python scripts/monitor_deeplob.py --log-dir logs/deeplob_v5  # 指定 V5 目錄
  python scripts/monitor_deeplob.py --export                 # 匯出 CSV
  python scripts/monitor_deeplob.py --version v5             # 強制使用 V5 格式
        """)
    parser.add_argument('--export', action='store_true', help='匯出 CSV 供 AI 分析')
    parser.add_argument('--csv-path', type=str, default='checkpoints/training_data.csv',
                       help='CSV 匯出路徑 (預設: checkpoints/training_data.csv)')
    parser.add_argument('--log-dir', type=str, default='logs/deeplob_v5',
                       help='TensorBoard 日誌目錄 (預設: logs/deeplob_v5)')
    parser.add_argument('--version', type=str, default='auto', choices=['auto', 'v5', 'generic'],
                       help='指定版本 (預設: auto，自動偵測)')

    args = parser.parse_args()

    print("=" * 70)
    print("DeepLOB 訓練監控")
    print("=" * 70)

    # 從 TensorBoard 讀取數據
    print(f"\n讀取 TensorBoard 日誌: {args.log_dir}")

    log_base = Path(args.log_dir)
    if not log_base.exists():
        print(f"\n[ERROR] 日誌目錄不存在: {args.log_dir}")
        print("請確認訓練已開始並檢查路徑是否正確")
        print("\n可用的日誌目錄:")
        logs_root = Path('logs')
        if logs_root.exists():
            for d in logs_root.iterdir():
                if d.is_dir():
                    print(f"  - {d}")
        return

    # 找到最新的日誌子目錄
    subdirs = [d for d in log_base.iterdir() if d.is_dir()]
    if not subdirs:
        print("\n[ERROR] 找不到日誌文件")
        return

    latest_subdir = max(subdirs, key=lambda d: d.stat().st_mtime)
    print(f"使用日誌: {latest_subdir.name}")

    metrics, detected_version = read_tensorboard_logs(log_base, args.version)

    if not metrics:
        print("\n[WARNING] 無法讀取訓練數據")
        print("可能原因:")
        print("  1. 訓練尚未開始或未完成第一個 epoch")
        print("  2. TensorBoard 日誌格式不符")
        print("  3. 日誌目錄路徑錯誤")
        return

    print(f"成功讀取 {len(metrics)} 個 epoch 的數據")
    print(f"偵測到的版本: {detected_version.upper()}")

    # 匯出 CSV
    if args.export:
        print("\n" + "=" * 70)
        print("匯出 CSV")
        print("=" * 70)
        export_to_csv(metrics, args.csv_path, detected_version)

    # 顯示表格
    print_training_table(metrics, detected_version)

    # 顯示摘要
    print_summary(metrics, detected_version)

    print("\n" + "=" * 70)
    print("\n提示:")
    print("  python scripts/monitor_deeplob.py                    # 自動偵測並顯示")
    print("  python scripts/monitor_deeplob.py --export           # 匯出 CSV")
    print("  python scripts/monitor_deeplob.py --log-dir logs/... # 指定目錄")
    print()


if __name__ == "__main__":
    main()
