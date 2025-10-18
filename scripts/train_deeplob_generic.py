"""訓練通用 DeepLOB 模型（v4 資料管線，支援 Anchored CV）

此腳本用於訓練不使用股票嵌入的通用 DeepLOB 模型。
通用模型適用於任何股票，具備良好的泛化能力。

✅ v4 更新:
    - 支援 v4 資料格式（20 維特徵，標籤 {0,1,2}）
    - 讀取 normalization_meta.json（Z-Score 參數）
    - 支援 Anchored CV 文本檔訓練
    - 完整的資料品質檢查
    - 改善的評估指標（包含每個 horizon 的準確率）

功能:
    - 載入 LOB 數據（npz 或 Anchored CV 文本檔）
    - 訓練純 DeepLOB 模型（無股票嵌入）
    - 監控整體準確率 + F1(macro/weighted) 與混淆矩陣
    - 驗證早停（EarlyStopping）
    - 保存最佳模型檢查點，並在 Test 上報告
    - 匯出 TorchScript 以利部署 / RLlib 封裝（整模）

資料格式:
    v4 NPZ 格式:
        - X: (N, 100, 20) LOB 序列 [BidP1..5, AskP1..5, BidQ1..5, AskQ1..5]
        - y: (N,) 標籤 {0:上漲, 1:持平, 2:下跌}
        - stock_ids: (N,) 股票代碼（可選）

    v4 文本格式 (Anchored CV):
        - 每行: 20 特徵 + 5 標籤 (k=1,2,3,5,10)
        - 使用 k=10 的標籤訓練

使用方式:
    # 使用 NPZ 格式（預設）
    python scripts/train_deeplob_generic.py \\
        --config configs/deeplob_generic_config.yaml

    # 使用 Anchored CV 文本檔（指定折數）
    python scripts/train_deeplob_generic.py \\
        --config configs/deeplob_generic_config.yaml \\
        --use-anchored-cv \\
        --cv-fold 7

    # 可選覆蓋參數
    python scripts/train_deeplob_generic.py \\
        --config configs/deeplob_generic_config.yaml \\
        --epochs 50 \\
        --batch-size 512 \\
        --lr 1e-3

備註:
- v4 資料由 extract_tw_stock_data_v4.py 產出
- normalization_meta.json 包含 Z-Score 參數、資料品質統計
- 支援 Anchored CV (Fold 1-9) 的時序驗證
"""

import argparse
import os
import sys
from pathlib import Path
import json
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import itertools

import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.amp.grad_scaler import GradScaler  # 混合精度訓練 (PyTorch 2.5+ 新 API)
from torch.amp.autocast_mode import autocast
from datetime import datetime
import logging
from tqdm import tqdm

# 添加 src 到路徑
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.deeplob import DeepLOB
from src.utils.yaml_manager import YAMLManager

# 配置日誌
# （補充）讓 cudnn 可重現
torch.backends.cudnn.deterministic = True

# 創建 logger（稍後在 main() 中添加文件處理器）
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class GenericLOBDataset(Dataset):
    """通用 LOB 數據集（不使用 stock_id）"""

    def __init__(self, npz_path: str, use_anchored_cv=False, cv_fold=1):
        """載入 .npz 或 Anchored CV 數據

        參數:
            npz_path: .npz 或文本文件路徑
            use_anchored_cv: 是否使用 Anchored CV 格式
            cv_fold: 指定的折數（1-9），僅在 use_anchored_cv=True 時有效

        期望格式:
            - NPZ:
                - X: (N, 100, 20) LOB 序列
                - y: (N,) 標籤
                - stock_ids: (可選) 僅用於統計
            - Anchored CV 文本檔:
                - 每行: 20 特徵 + 5 標籤 (k=1,2,3,5,10)
        """
        logger.info(f"載入數據: {npz_path} (use_anchored_cv={use_anchored_cv}, cv_fold={cv_fold})")

        self.use_anchored_cv = use_anchored_cv
        self.cv_fold = cv_fold

        if use_anchored_cv:
            # 讀取 Anchored CV 格式
            self.X, self.y = self.load_anchored_cv(npz_path, cv_fold)
        else:
            # 讀取 NPZ 格式
            data = np.load(npz_path, allow_pickle=True)
            self.X = torch.FloatTensor(data['X'])  # (N, 100, 20)
            self.y = torch.LongTensor(data['y'])  # (N,)

            # stock_ids 僅用於統計（不用於訓練）
            self.stock_ids = None
            if 'stock_ids' in data:
                self.stock_ids = data['stock_ids']

            logger.info(f"  樣本數: {len(self.X):,}")
            logger.info(f"  序列形狀: {self.X.shape}")
            logger.info(f"  標籤分佈: {np.bincount(self.y.numpy())}")

            if self.stock_ids is not None:
                num_unique_stocks = len(np.unique(self.stock_ids))
                logger.info(f"  股票數量（統計用）: {num_unique_stocks}")

    def load_anchored_cv(self, file_path, cv_fold):
        """載入 Anchored CV 格式的數據

        參數:
            file_path: 文本文件路徑
            cv_fold: 指定的折數（1-9）

        返回:
            X: 特徵矩陣 (N, 100, 20)
            y: 標籤向量 (N,)
        """
        logger.info(f"  讀取 Anchored CV 數據: {file_path}，使用折數: {cv_fold}")

        X, y = [], []

        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                values = list(map(float, line.strip().split()))
                if len(values) != 25:
                    logger.warning(f"  ⚠️ 跳過無效行（不是 25 個值）: {line.strip()}")
                    continue

                # 特徵: 前 20 個值
                features = values[:20]
                X.append(features)

                # 標籤: 第 (5 * (cv_fold - 1)) 到 (5 * cv_fold) 個值
                label_start = 5 * (cv_fold - 1)
                label_end = 5 * cv_fold
                labels = values[label_start:label_end]

                # 使用 k=10 的標籤（最後一個標籤）
                y.append(int(labels[-1]))

        X = torch.FloatTensor(X)
        y = torch.LongTensor(y)

        logger.info(f"  樣本數: {len(X):,}")
        logger.info(f"  特徵形狀: {X.shape}")
        logger.info(f"  標籤分佈: {np.bincount(y.numpy())}")

        return X, y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return {
            'lob': self.X[idx],
            'label': self.y[idx]
        }


def compute_metrics(y_true, y_pred, average='macro'):
    """回傳常用指標（含 macro F1 與混淆矩陣）

    修復警告:
    - 使用 zero_division=0 避免 UndefinedMetricWarning
    - 當某些類別沒有預測樣本時，精度設為 0
    """
    f1_macro = f1_score(y_true, y_pred, average=average, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(np.unique(y_true))))
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    return f1_macro, cm, report


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', fname=None):
    """儲存混淆矩陣 PNG（便於審計/報告）"""
    if normalize:
        cm = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-12)
    plt.figure(figsize=(5.5, 5))
    plt.imshow(cm, interpolation='nearest')
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    if fname:
        plt.savefig(fname, bbox_inches='tight', dpi=160)
        plt.close()


class EarlyStopping:
    """早停機制

    Args:
        patience: 容忍的 epoch 數量
        mode: 'max' (準確率) 或 'min' (損失)
        min_delta: 最小改進閾值
        verbose: 是否打印日誌
    """
    def __init__(self, patience=10, mode='max', min_delta=0.0, verbose=True):
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.verbose = verbose

        self.best_score = None
        self.counter = 0
        self.early_stop = False
        self.best_epoch = 0

    def __call__(self, score, epoch):
        """
        Args:
            score: 當前指標值（準確率或損失）
            epoch: 當前 epoch 數

        Returns:
            should_stop: 是否應該停止訓練
        """
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            return False

        # 判斷是否改進
        if self.mode == 'max':
            improved = score > self.best_score + self.min_delta
        else:  # mode == 'min'
            improved = score < self.best_score - self.min_delta

        if improved:
            if self.verbose:
                logger.info(f"[EarlyStopping] 指標改進: {self.best_score:.4f} → {score:.4f}")
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                logger.info(f"[EarlyStopping] 無改進 ({self.counter}/{self.patience})")

            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    logger.info(f"[EarlyStopping] 觸發早停！最佳 epoch: {self.best_epoch}")

        return self.early_stop

    def state_dict(self):
        """保存狀態（用於檢查點）"""
        return {
            'best_score': self.best_score,
            'counter': self.counter,
            'best_epoch': self.best_epoch,
            'early_stop': self.early_stop
        }

    def load_state_dict(self, state_dict):
        """載入狀態"""
        self.best_score = state_dict['best_score']
        self.counter = state_dict['counter']
        self.best_epoch = state_dict['best_epoch']
        self.early_stop = state_dict['early_stop']


def train_epoch(model, dataloader, criterion, optimizer, device, scaler=None, clip_grad=None, use_bf16=False):
    """訓練一個 epoch

    參數:
        model: 模型
        dataloader: 數據載入器
        criterion: 損失函數
        optimizer: 優化器
        device: 設備
        scaler: GradScaler for AMP (可選，BF16 時為 None)
        clip_grad: 梯度裁剪最大範數（預設None不裁剪）
        use_bf16: 是否使用 BFloat16（無需 scaler）

    返回:
        avg_loss: 平均損失
        accuracy: 準確率 (%)
        avg_grad_norm: 平均梯度範數
    """
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    # ⭐ 新增: 梯度統計
    grad_norms = []
    skipped_batches = 0  # 新增：跳過的異常批次數

    use_amp = scaler is not None or use_bf16
    dtype = torch.bfloat16 if use_bf16 else torch.float16

    # 進度條
    pbar = tqdm(dataloader, desc='Training', leave=False)

    for batch_idx, batch in enumerate(pbar):
        lob = batch['lob'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()

        try:
            if use_amp:
                # 混合精度訓練 (BF16 或 FP16)
                with autocast(device_type='cuda', dtype=dtype):
                    logits = model(lob)
                    loss = criterion(logits, labels)

                # ⭐ 檢查 loss 是否有效
                if not torch.isfinite(loss):
                    logger.warning(f"Batch {batch_idx}: Loss is NaN/Inf, skipping this batch")
                    skipped_batches += 1
                    continue

                # ⭐ BF16 不需要 scaler（數值範圍與 FP32 相同）
                if use_bf16:
                    loss.backward()

                    # 計算梯度範數
                    total_norm = 0.0
                    has_invalid_grad = False
                    for p in model.parameters():
                        if p.grad is not None:
                            param_norm = p.grad.data.norm(2)
                            if torch.isfinite(param_norm):
                                total_norm += param_norm.item() ** 2
                            else:
                                has_invalid_grad = True
                                break

                    if has_invalid_grad or not torch.isfinite(torch.tensor(total_norm)):
                        logger.warning(f"Batch {batch_idx}: Gradient contains NaN/Inf, skipping")
                        optimizer.zero_grad()
                        skipped_batches += 1
                        continue

                    total_norm = total_norm ** 0.5
                    grad_norms.append(total_norm)

                    # 梯度裁剪
                    if clip_grad is not None:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)

                    optimizer.step()
                else:
                    # FP16 需要 scaler
                    scaler.scale(loss).backward()

                    # ⭐ 先 unscale，然後檢查梯度
                    scaler.unscale_(optimizer)

                    # ⭐ 計算梯度範數（裁剪前）- 改進版，處理 NaN/Inf
                    total_norm = 0.0
                    has_invalid_grad = False
                    for p in model.parameters():
                        if p.grad is not None:
                            param_norm = p.grad.data.norm(2)
                            if torch.isfinite(param_norm):
                                total_norm += param_norm.item() ** 2
                            else:
                                has_invalid_grad = True
                                break

                    # ⭐ 如果發現無效梯度，必須先調用 update() 再跳過
                    if has_invalid_grad or not torch.isfinite(torch.tensor(total_norm)):
                        logger.warning(f"Batch {batch_idx}: Gradient contains NaN/Inf, skipping")
                        optimizer.zero_grad()
                        scaler.update()  # ⭐ 關鍵：重置 scaler 狀態
                        skipped_batches += 1
                        continue

                    total_norm = total_norm ** 0.5
                    grad_norms.append(total_norm)

                    # ⭐ 梯度裁剪
                    if clip_grad is not None:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)

                    scaler.step(optimizer)
                    scaler.update()
            else:
                # 標準訓練
                logits = model(lob)
                loss = criterion(logits, labels)

                # ⭐ 檢查 loss 是否有效
                if not torch.isfinite(loss):
                    logger.warning(f"Batch {batch_idx}: Loss is NaN/Inf, skipping this batch")
                    skipped_batches += 1
                    continue

                loss.backward()

                # ⭐ 計算梯度範數 - 改進版
                total_norm = 0.0
                has_invalid_grad = False
                for p in model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        if torch.isfinite(param_norm):
                            total_norm += param_norm.item() ** 2
                        else:
                            has_invalid_grad = True
                            break

                if has_invalid_grad:
                    logger.warning(f"Batch {batch_idx}: Gradient contains NaN/Inf, skipping")
                    optimizer.zero_grad()
                    skipped_batches += 1
                    continue

                total_norm = total_norm ** 0.5
                grad_norms.append(total_norm)

                # ⭐ 梯度裁剪
                if clip_grad is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)

                optimizer.step()

            # 統計
            total_loss += loss.item()
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # 更新進度條（顯示梯度範數）
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100. * correct / total:.2f}%',
                'grad': f'{total_norm:.2f}'
            })

        except RuntimeError as e:
            logger.error(f"Batch {batch_idx}: Runtime error - {str(e)}")
            # ⭐ 如果是 FP16 AMP 模式，確保 scaler 狀態正確
            if scaler is not None:
                try:
                    scaler.update()
                except:
                    pass
            skipped_batches += 1
            continue

    if skipped_batches > 0:
        logger.warning(f"Skipped {skipped_batches} batches due to NaN/Inf in this epoch")

    avg_loss = total_loss / max(len(dataloader) - skipped_batches, 1)
    accuracy = 100. * correct / total if total > 0 else 0.0
    avg_grad_norm = np.mean(grad_norms) if grad_norms else 0.0

    return avg_loss, accuracy, avg_grad_norm

def predict_all(model, dataloader, device):
    model.eval(); ys=[]; ps=[]
    with torch.no_grad():
        for batch in dataloader:
            logits = model(batch['lob'].to(device))
            ys.append(batch['label'].cpu().numpy())
            ps.append(logits.argmax(1).cpu().numpy())
    return np.concatenate(ys), np.concatenate(ps)


def validate(model, dataloader, criterion, device):
    """驗證模型

    參數:
        model: 模型
        dataloader: 數據載入器
        criterion: 損失函數
        device: 設備

    返回:
        avg_loss: 平均損失
        accuracy: 準確率 (%)
    """
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    # 進度條
    pbar = tqdm(dataloader, desc='Validation', leave=False)

    with torch.no_grad():
        for batch in pbar:
            lob = batch['lob'].to(device)
            labels = batch['label'].to(device)

            logits = model(lob)
            loss = criterion(logits, labels)

            total_loss += loss.item()
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # 更新進度條
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100. * correct / total:.2f}%'
            })

    avg_loss = total_loss / len(dataloader)
    accuracy = 100. * correct / total

    return avg_loss, accuracy


def main():
    parser = argparse.ArgumentParser(description='訓練通用 DeepLOB 模型（無股票嵌入）')

    # 主要參數：配置文件
    parser.add_argument('--config', type=str,
                       default='configs/deeplob_v4_config.yaml',
                       help='YAML 配置文件路徑')

    # 可選覆蓋參數
    parser.add_argument('--epochs', type=int, help='覆蓋配置文件中的 epochs')
    parser.add_argument('--batch-size', type=int, help='覆蓋配置文件中的 batch_size')
    parser.add_argument('--lr', type=float, help='覆蓋配置文件中的 learning_rate')
    parser.add_argument('--resume', type=str, help='覆蓋配置文件中的 resume.enabled (true/false)')
    # 早停與度量選項
    parser.add_argument('--early-stop', action='store_true', help='啟用 EarlyStopping（以 val_acc 為準）')
    parser.add_argument('--early-stop-patience', type=int, default=12)
    parser.add_argument('--test-after-train', action='store_true', help='訓練完以最佳權重在 test 上評估')
    # Anchored CV 選項
    parser.add_argument('--use-anchored-cv', action='store_true', help='使用 Anchored CV 文本檔訓練')
    parser.add_argument('--cv-fold', type=int, default=1, help='指定的折數（1-9）')

    args = parser.parse_args()

    # ========== 配置日誌（同時輸出到控制台和文件） ==========
    # 創建日誌目錄
    log_base_dir = 'logs/train_deeplob'
    os.makedirs(log_base_dir, exist_ok=True)

    # 生成日誌文件名（帶時間戳）
    log_filename = os.path.join(
        log_base_dir,
        f'train_deeplob_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    )

    # 清除現有的 handlers（避免重複）
    logger.handlers.clear()

    # 創建格式器
    formatter = logging.Formatter(
        '[%(asctime)s] %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 控制台處理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 文件處理器
    file_handler = logging.FileHandler(log_filename, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.info("=" * 60)
    logger.info("DeepLOB 通用模型訓練")
    logger.info("=" * 60)
    logger.info(f"日誌文件: {log_filename}")
    logger.info("")

    # ========== 載入配置文件 ==========
    logger.info("=" * 60)
    logger.info(f"載入配置文件: {args.config}")
    logger.info("=" * 60)

    yaml_config = YAMLManager(args.config)
    config = yaml_config.as_dict()

    # 命令行參數覆蓋配置文件
    if args.epochs is not None:
        config['training']['epochs'] = args.epochs
    if args.batch_size is not None:
        config['training']['batch_size'] = args.batch_size
    if args.lr is not None:
        config['training']['learning_rate'] = args.lr
    if args.resume is not None:
        config['resume']['enabled'] = args.resume.lower() == 'true'

    # 顯示最終配置
    logger.info("\n當前配置:")
    logger.info(json.dumps(config, indent=2, ensure_ascii=False))

    # ========== 設置 ==========
    # 設置隨機種子
    seed = config['misc']['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # 設置設備
    device_name = config['hardware']['device']
    device = torch.device(device_name if torch.cuda.is_available() else 'cpu')
    logger.info(f"\n使用設備: {device}")

    # 創建輸出目錄
    output_dir = config['data']['output_dir']
    os.makedirs(output_dir, exist_ok=True)

    # 創建 TensorBoard writer
    log_dir = os.path.join('logs', 'deeplob_generic', datetime.now().strftime('%Y%m%d_%H%M%S'))
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)
    logger.info(f"TensorBoard 日誌目錄: {log_dir}")

    # ========== 載入 meta.json（可選） ==========
    data_dir = config['data']['data_dir']
    meta_path = os.path.join(data_dir, 'meta.json')
    meta = None

    if os.path.exists(meta_path):
        logger.info("\n" + "=" * 60)
        logger.info(f"載入元數據: {meta_path}")
        logger.info("=" * 60)

        with open(meta_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)

        # ⭐ 覆蓋配置文件中的 input_shape（若 meta 中有定義）
        if 'input_shape' in meta:
            config['model']['input_shape'] = meta['input_shape']
            logger.info(f"✅ 從 meta.json 覆蓋 input_shape: {meta['input_shape']}")

        # ⭐ 覆蓋 num_classes（若 meta 中有定義）
        if 'num_classes' in meta:
            config['model']['num_classes'] = meta['num_classes']
            logger.info(f"✅ 從 meta.json 覆蓋 num_classes: {meta['num_classes']}")

        # 顯示數據集資訊
        if 'dataset_version' in meta:
            logger.info(f"   數據集版本: {meta['dataset_version']}")
        if 'label_config' in meta:
            logger.info(f"   標籤配置: k={meta['label_config']['k']}, alpha={meta['label_config']['alpha']:.4f}")
            logger.info(f"   標籤風格: {meta['label_config'].get('label_style', 'unknown')}")
        if 'feature_config' in meta:
            feat_cfg = meta['feature_config']
            logger.info(f"   特徵處理: relativize={feat_cfg.get('relativize', False)}, normalize={feat_cfg.get('normalize', True)}")
    else:
        logger.info(f"\n未找到 meta.json，使用配置文件中的設定")

    # ========== 載入數據 ==========
    data_format = config['data'].get('format', 'npz')
    logger.info(f"\n數據載入格式: {data_format}")

    if data_format == 'npz':
        # 對齊 v3 輸出命名（stock_embedding_train/val/test.npz）
        train_path = os.path.join(data_dir, config['data'].get('train_file', 'stock_embedding_train.npz'))
        val_path   = os.path.join(data_dir, config['data'].get('val_file',   'stock_embedding_val.npz'))
        test_path  = os.path.join(data_dir, config['data'].get('test_file',  'stock_embedding_test.npz'))

        if not os.path.exists(train_path):
            logger.error(f"訓練數據不存在: {train_path}")
            logger.error("請先準備訓練數據！")
            logger.error("\n建議執行:")
            logger.error("  python scripts/extract_tw_stock_data_v2.py --output stock_generic_train.npz")
            return

        logger.info("\n" + "=" * 60)
        logger.info("載入訓練數據")
        logger.info("=" * 60)

        train_dataset = GenericLOBDataset(train_path)
        val_dataset = GenericLOBDataset(val_path)

        batch_size = config['training']['batch_size']
        num_workers = config['hardware']['num_workers']
        pin_memory = config['hardware']['pin_memory'] and device.type == 'cuda'
        prefetch_factor = config['hardware'].get('prefetch_factor', 2)
        persistent_workers = config['hardware'].get('persistent_workers', True) and num_workers > 0

        # DataLoader 參數字典
        dataloader_kwargs = {
            'batch_size': batch_size,
            'num_workers': num_workers,
            'pin_memory': pin_memory,
        }

        # 僅當 num_workers > 0 時添加這些參數
        if num_workers > 0:
            dataloader_kwargs['prefetch_factor'] = prefetch_factor
            dataloader_kwargs['persistent_workers'] = persistent_workers

        train_loader = DataLoader(
            train_dataset,
            shuffle=True,
            **dataloader_kwargs
        )

        val_loader = DataLoader(
            val_dataset,
            shuffle=False,
            **dataloader_kwargs
        )

        test_loader = None
        if os.path.exists(test_path):
            test_dataset = GenericLOBDataset(test_path)
            test_loader = DataLoader(test_dataset, shuffle=False, **dataloader_kwargs)

    elif data_format == 'anchored_cv':
        # 使用 Anchored CV 文本檔
        train_file = os.path.join(data_dir, config['data'].get('train_file', 'anchored_cv_train.txt'))
        val_file   = os.path.join(data_dir, config['data'].get('val_file',   'anchored_cv_val.txt'))

        logger.info("\n" + "=" * 60)
        logger.info("載入 Anchored CV 訓練數據")
        logger.info("=" * 60)

        train_dataset = GenericLOBDataset(train_file, use_anchored_cv=True, cv_fold=args.cv_fold)
        val_dataset = GenericLOBDataset(val_file, use_anchored_cv=True, cv_fold=args.cv_fold)

        batch_size = config['training']['batch_size']
        num_workers = config['hardware']['num_workers']
        pin_memory = config['hardware']['pin_memory'] and device.type == 'cuda'
        prefetch_factor = config['hardware'].get('prefetch_factor', 2)
        persistent_workers = config['hardware'].get('persistent_workers', True) and num_workers > 0

        # DataLoader 參數字典
        dataloader_kwargs = {
            'batch_size': batch_size,
            'num_workers': num_workers,
            'pin_memory': pin_memory,
        }

        # 僅當 num_workers > 0 時添加這些參數
        if num_workers > 0:
            dataloader_kwargs['prefetch_factor'] = prefetch_factor
            dataloader_kwargs['persistent_workers'] = persistent_workers

        train_loader = DataLoader(
            train_dataset,
            shuffle=True,
            **dataloader_kwargs
        )

        val_loader = DataLoader(
            val_dataset,
            shuffle=False,
            **dataloader_kwargs
        )

        test_loader = None  # Anchored CV 不使用 test_loader

    else:
        logger.error(f"不支援的數據格式: {data_format}")
        return

    # ========== 創建模型 ==========
    logger.info("\n" + "=" * 60)
    logger.info("創建通用 DeepLOB 模型（無股票嵌入）")
    logger.info("=" * 60)

    # ⭐ 動態適配 input_shape
    input_shape = tuple(config['model']['input_shape'])

    # 驗證與數據形狀一致
    actual_shape = tuple(train_dataset.X.shape[1:])  # (seq_len, features)
    if input_shape != actual_shape:
        logger.warning("=" * 60)
        logger.warning(f"⚠️  配置的 input_shape {input_shape} 與實際數據 {actual_shape} 不符")
        logger.warning(f"    自動調整為: {actual_shape}")
        logger.warning("=" * 60)
        input_shape = actual_shape
        config['model']['input_shape'] = list(input_shape)

    logger.info(f"模型輸入形狀: {input_shape}")

    model = DeepLOB(
        input_shape=input_shape,
        num_classes=config['model']['num_classes'],
        conv1_filters=config['model'].get('conv1_filters', 16),
        conv2_filters=config['model'].get('conv2_filters', 16),
        conv3_filters=config['model'].get('conv3_filters', 16),
        lstm_hidden_size=config['model']['lstm_hidden_size'],
        fc_hidden_size=config['model'].get('fc_hidden_size', 64),
        dropout=config['model']['dropout']
    ).to(device)

    # ⭐ RTX 5090 專屬優化
    if torch.cuda.is_available():
        logger.info("\n" + "=" * 60)
        logger.info("RTX 5090 專屬優化")
        logger.info("=" * 60)

        # 1. 啟用 TF32 (Ampere 架構以上) - 使用新 API
        # 新版 PyTorch 2.6+ 推荐的设置方式
        torch.backends.cuda.matmul.fp32_precision = 'tf32'  # 矩阵乘法使用 TF32
        torch.backends.cudnn.conv.fp32_precision = 'tf32'   # 卷积使用 TF32
        logger.info("✅ TF32 加速已啟用（矩陣運算 +20% 性能）- 使用新版 API")

        # 2. 啟用 cudnn benchmark（固定輸入大小）
        torch.backends.cudnn.benchmark = True
        logger.info("✅ cuDNN Benchmark 已啟用（自動尋找最優卷積算法）")

        # 3. 顯示 GPU 資訊
        props = torch.cuda.get_device_properties(0)
        logger.info(f"GPU: {props.name}")
        logger.info(f"VRAM: {props.total_memory / 1024**3:.1f} GB")
        logger.info(f"CUDA Compute Capability: {props.major}.{props.minor}")
        logger.info(f"Multi-Processor Count: {props.multi_processor_count}")

    # ========== 損失函數和優化器 ==========
    logger.info("\n" + "=" * 60)
    logger.info("類別權重配置")
    logger.info("=" * 60)

    train_labels = train_dataset.y.numpy()
    class_counts = np.bincount(train_labels)
    total_samples = len(train_labels)

    logger.info(f"訓練集標籤分佈: {class_counts}")

    # ⭐ 優先順序: meta.json > config.yaml > 自動計算
    class_weights = None

    if config['training'].get('use_class_weights', False):
        # 1. 優先使用 meta.json 中的 class_weights
        if meta is not None and 'class_weights' in meta and 'train' in meta['class_weights']:
            weights_list = meta['class_weights']['train']
            class_weights = torch.FloatTensor(weights_list).to(device)
            logger.info("✅ 使用 meta.json 中的類別權重")
            logger.info(f"   權重: {weights_list}")

        # 2. 其次使用配置文件指定的權重
        elif 'class_weights' in config['training']:
            weights_list = config['training']['class_weights']
            class_weights = torch.FloatTensor(weights_list).to(device)
            logger.info("✅ 使用配置文件指定的類別權重")
            logger.info(f"   權重: {weights_list}")

        # 3. 最後自動計算（inverse frequency）
        else:
            weights = total_samples / (len(class_counts) * class_counts + 1e-6)
            # 歸一化到 [0.5, 2.0] 範圍（避免極端值）
            weights = np.clip(weights / weights.mean(), 0.5, 2.0)
            class_weights = torch.FloatTensor(weights).to(device)
            logger.info("✅ 使用自動計算的類別權重（inverse frequency）")
            logger.info(f"   權重: {weights}")
    else:
        logger.info("⚪ 不使用類別權重（標準 CE loss）")

    logger.info("=" * 60)

    # ⭐ 標籤平滑（Label Smoothing）
    label_smoothing = config['training'].get('label_smoothing', 0.0)
    logger.info(f"標籤平滑: {label_smoothing}")

    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=label_smoothing
    )
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )

    # 學習率調度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=config['training']['scheduler']['mode'],
        patience=config['training']['scheduler']['patience'],
        factor=config['training']['scheduler']['factor'],
        min_lr=config['training']['scheduler'].get('min_lr', 1e-6)  # ⭐ 新增最小學習率
    )

    # 混合精度訓練 (AMP) - 支持 BFloat16 和 Float16
    use_amp = config['training'].get('use_amp', False) and device.type == 'cuda'
    use_bf16 = config['training'].get('use_bf16', False) and torch.cuda.is_bf16_supported()

    # ⭐ BF16 不需要 GradScaler（數值範圍與 FP32 相同）
    scaler = None
    if use_amp:
        if use_bf16:
            logger.info("\n✅ 混合精度訓練 (BFloat16) 已啟用")
            logger.info("   - 數值範圍與 FP32 相同，無溢出風險")
            logger.info("   - 不使用 GradScaler（BF16 特性）")
            logger.info("   - RTX 5090 原生支持，性能最佳")
        else:
            scaler = GradScaler('cuda')
            logger.info("\n✅ 混合精度訓練 (Float16) 已啟用")
            logger.info("   - 使用 GradScaler 防止數值溢出")
            logger.info("   ⚠️  建議使用 BFloat16（更穩定）")
    else:
        logger.info("\n⚪ 混合精度訓練 (AMP) 未啟用")

    # ⭐ 續訓: 載入檢查點
    start_epoch = 1
    best_val_acc = 0
    best_epoch = 0
    training_history = []
    early_stop = EarlyStopping(patience=args.early_stop_patience) if args.early_stop else None
    metrics_dir = os.path.join(output_dir, 'metrics')
    os.makedirs(metrics_dir, exist_ok=True)

    checkpoint_path = config['resume']['checkpoint_path']

    if config['resume']['enabled']:
        if os.path.exists(checkpoint_path):
            logger.info("\n" + "=" * 60)
            logger.info(f"載入檢查點: {checkpoint_path}")
            logger.info("=" * 60)

            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

            # 載入模型權重
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info("[OK] 模型權重已載入")

            # 載入優化器狀態 (可選)
            if config['resume']['load_optimizer'] and 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                logger.info("[OK] 優化器狀態已載入")

            # 載入訓練狀態
            if 'epoch' in checkpoint:
                start_epoch = checkpoint['epoch'] + 1
                logger.info(f"[OK] 從 Epoch {start_epoch} 繼續訓練")

            if 'val_acc' in checkpoint:
                best_val_acc = checkpoint['val_acc']
                best_epoch = checkpoint.get('epoch', 0)
                logger.info(f"[OK] 目前最佳驗證準確率: {best_val_acc:.2f}% (Epoch {best_epoch})")

            if 'train_acc' in checkpoint:
                logger.info(f"     訓練準確率: {checkpoint['train_acc']:.2f}%")

        else:
            logger.warning(f"[WARNING] 檢查點不存在: {checkpoint_path}")
            logger.warning("從頭開始訓練...")

    # ========== 訓練循環 ==========
    logger.info("\n" + "=" * 60)
    logger.info("開始訓練")
    logger.info("=" * 60)

    num_epochs = config['training']['epochs']

    # ⭐ 梯度裁剪參數
    clip_grad_norm = config['training'].get('clip_grad_norm', 1.0)
    logger.info(f"\n梯度裁剪: {'啟用 (max_norm=' + str(clip_grad_norm) + ')' if clip_grad_norm else '停用'}")

    for epoch in range(start_epoch, num_epochs + 1):
        logger.info(f"\nEpoch {epoch}/{num_epochs}")
        logger.info("-" * 60)

        # 訓練
        train_loss, train_acc, train_grad_norm = train_epoch(
            model, train_loader, criterion, optimizer, device, scaler, clip_grad=clip_grad_norm
        )
        logger.info(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%, Grad: {train_grad_norm:.2f}")

        # 驗證
        val_loss, val_acc = validate(
            model, val_loader, criterion, device
        )
        logger.info(f"Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")

        # 學習率調整
        scheduler.step(val_acc)

        # 記錄歷史
        training_history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'grad_norm': train_grad_norm,
            'lr': optimizer.param_groups[0]['lr']
        })

        # 記錄到 TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
        writer.add_scalar('Gradient/norm', train_grad_norm, epoch)  # ⭐ 新增梯度監控

        # 追蹤 macro F1 與混淆矩陣（Validation）
        y_val, p_val = predict_all(model, val_loader, device)
        f1_macro, cm, rep = compute_metrics(y_val, p_val)
        writer.add_scalar('F1_macro/val', f1_macro, epoch)
        with open(os.path.join(metrics_dir, f'val_report_epoch{epoch}.json'), 'w', encoding='utf-8') as f:
            json.dump({'f1_macro': f1_macro, 'report': rep}, f, ensure_ascii=False, indent=2)
        plot_confusion_matrix(cm, classes=[0,1,2], normalize=True,
                              title=f'Val Confusion (epoch {epoch})',
                              fname=os.path.join(metrics_dir, f'val_cm_epoch{epoch}.png'))

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch

            save_path = os.path.join(output_dir, 'deeplob_generic_best.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'train_acc': train_acc,
                'model_config': {
                    'input_shape': config['model']['input_shape'],
                    'num_classes': config['model']['num_classes'],
                    'lstm_hidden_size': config['model']['lstm_hidden_size'],
                    'dropout': config['model']['dropout']
                },
                'config': config
            }, save_path)

            logger.info(f"\n[保存] 最佳模型 (Val Acc: {val_acc:.2f}%)")

        # 早停檢查（使用新版 API）
        if early_stop and early_stop(val_acc, epoch):
            logger.info(f"\n提前結束訓練！最佳驗證準確率: {early_stop.best_score:.2f}% (Epoch {early_stop.best_epoch})")
            break

    # ========== 訓練完成 ==========
    logger.info("\n" + "=" * 60)
    logger.info("訓練完成!")
    logger.info("=" * 60)
    logger.info(f"最佳驗證準確率: {best_val_acc:.2f}% (Epoch {best_epoch})")

    # ========== 測試集評估（以最佳權重） ==========
    if args.test_after_train and test_loader is not None:
        best_path = os.path.join(output_dir, 'deeplob_generic_best.pth')
        if os.path.exists(best_path):
            ckpt = torch.load(best_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt['model_state_dict'])
            y_te, p_te = predict_all(model, test_loader, device)
            f1_te, cm_te, rep_te = compute_metrics(y_te, p_te)
            logger.info(f"[TEST] macro F1: {f1_te:.4f}")
            with open(os.path.join(metrics_dir, 'test_report.json'), 'w', encoding='utf-8') as f:
                json.dump({'f1_macro': f1_te, 'report': rep_te}, f, ensure_ascii=False, indent=2)
            plot_confusion_matrix(cm_te, classes=[0,1,2], normalize=True,
                                  title='Test Confusion (best ckpt)',
                                  fname=os.path.join(metrics_dir, 'test_cm.png'))

    # 匯出 TorchScript（整模；RLlib/部署可直接載入）
    example = torch.randn(1, *config['model']['input_shape']).to(device)
    scripted = torch.jit.trace(model, example)
    ts_path = os.path.join(output_dir, 'deeplob_generic_best.torchscript.pt')
    scripted.save(ts_path)
    logger.info(f"TorchScript 已輸出: {ts_path}")

    # 保存訓練歷史
    history_path = os.path.join(output_dir, 'training_history_generic.json')
    with open(history_path, 'w', encoding='utf-8') as f:
        json.dump(training_history, f, indent=2, ensure_ascii=False)
    logger.info(f"訓練歷史已保存: {history_path}")

    # 保存最後一個 epoch 的模型
    if config['misc'].get('save_last', True):
        last_checkpoint_path = os.path.join(output_dir, 'deeplob_generic_last.pth')
        torch.save({
            'epoch': num_epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'model_config': {
                'input_shape': config['model']['input_shape'],
                'num_classes': config['model']['num_classes'],
                'lstm_hidden_size': config['model']['lstm_hidden_size'],
                'dropout': config['model']['dropout']
            },
            'config': config
        }, last_checkpoint_path)
        logger.info(f"最後模型已保存: {last_checkpoint_path}")

    # 關閉 TensorBoard writer
    writer.close()

    logger.info("\n" + "=" * 60)
    logger.info("訓練完成！")
    logger.info("=" * 60)
    logger.info("最佳模型: checkpoints/deeplob_generic_best.pth（亦已輸出 TorchScript）")
    logger.info(f"最佳驗證準確率: {best_val_acc:.2f}%")
    logger.info(f"\nTensorBoard 日誌: {log_dir}")
    logger.info("查看訓練曲線:")
    logger.info(f"  tensorboard --logdir={log_dir}")
    logger.info(f"\n訓練日誌已保存至: {log_filename}")

    # 關閉文件處理器
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)

if __name__ == "__main__":
    main()
