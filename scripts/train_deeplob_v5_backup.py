"""DeepLOB V5 训练脚本
V5 为核心，支持 Triple-Barrier 标签 + 样本权重 + Stock Embeddings

核心特性：
- V5 标签：{0:下跌, 1:持平, 2:上涨}（已转换）
- 样本权重：训练/验证 loss 加权，测试输出加权/不加权指标
- Stock IDs：可选嵌入（泛化优先）
- 温度校准：降低加权 NLL/ECE
- 完整报表：混淆矩阵、per-class PR/RC、分组指标
- 安全检查：启动时验证数据一致性

使用方式:
    python scripts/train_deeplob_v5.py --config configs/train_v5.yaml
    python scripts/train_deeplob_v5.py --config configs/train_v5.yaml --override optim.lr=0.0005
"""

import argparse
import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from torch.amp import GradScaler, autocast
from sklearn.metrics import (
    accuracy_score, f1_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
from tqdm import tqdm
import itertools

# 添加 src 到路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.deeplob import DeepLOB
from src.utils.yaml_manager import YAMLManager

# ===================================================================
# 日志配置
# ===================================================================
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# ===================================================================
# V5 数据集类
# ===================================================================
class LOBV5Dataset(Dataset):
    """LOB V5 数据集（支持样本权重、stock_ids、数据验证）"""

    def __init__(
        self,
        npz_path: str,
        config: dict,
        split: str = "train",
        norm_meta: Optional[dict] = None
    ):
        """初始化 V5 数据集

        参数:
            npz_path: .npz 文件路径
            config: 配置字典
            split: "train"/"val"/"test"
            norm_meta: 正规化元数据（用于验证）
        """
        self.split = split
        self.config = config
        self.norm_meta = norm_meta

        logger.info(f"载入 {split} 数据: {npz_path}")

        # 载入数据
        data = np.load(npz_path, allow_pickle=True)

        # ===== 必备字段 =====
        required_keys = ['X', 'y']
        missing_keys = [k for k in required_keys if k not in data]

        if missing_keys and config['data']['fail_on_missing_keys']:
            raise KeyError(f"Missing required keys: {missing_keys}")

        self.X = torch.FloatTensor(data['X'])  # (N, 100, 20)
        self.y = torch.LongTensor(data['y'])   # (N,)

        # ===== 样本权重 =====
        if config['data']['use_sample_weights'] and 'weights' in data:
            self.weights = torch.FloatTensor(data['weights'])  # (N,)

            # 验证权重
            if config['safety_checks']['validate_weights']:
                self._validate_weights()

            # 归一化权重
            if config['data']['weights_normalize'] == "mean_to_1":
                mean_w = self.weights.mean().item()
                if mean_w > 0:
                    self.weights = self.weights / mean_w
                    logger.info(f"  权重已归一化（均值→1.0）")
        else:
            self.weights = torch.ones(len(self.X), dtype=torch.float32)
            if config['data']['use_sample_weights']:
                logger.warning(f"  ⚠️  缺少 'weights' 字段，使用全 1 权重")

        # ===== Stock IDs（可选） =====
        if config['data']['use_stock_ids'] and 'stock_ids' in data:
            self.stock_ids = data['stock_ids']  # (N,)
            logger.info(f"  载入 stock_ids: {len(np.unique(self.stock_ids))} 档股票")
        else:
            self.stock_ids = None

        # ===== y_raw（可选，用于验证） =====
        if 'y_raw' in data:
            self.y_raw = data['y_raw']
            if config['safety_checks']['validate_label_set']:
                unique_raw = np.unique(self.y_raw)
                expected_raw = {-1, 0, 1}
                if not set(unique_raw).issubset(expected_raw):
                    raise ValueError(f"y_raw 包含非法值: {unique_raw} (期望 {expected_raw})")
        else:
            self.y_raw = None

        # ===== 标签验证 =====
        if config['safety_checks']['validate_label_set']:
            self._validate_labels()

        # 打印统计
        logger.info(f"  样本数: {len(self.X):,}")
        logger.info(f"  序列形状: {self.X.shape}")

        label_counts = np.bincount(self.y.numpy())
        label_names = ['下跌(0)', '持平(1)', '上涨(2)']
        for i, (name, count) in enumerate(zip(label_names, label_counts)):
            pct = 100 * count / len(self.y)
            logger.info(f"  {name}: {count:,} ({pct:.1f}%)")

        if config['data']['use_sample_weights']:
            w_stats = self.weights.numpy()
            logger.info(f"  权重统计: mean={w_stats.mean():.4f}, "
                       f"std={w_stats.std():.4f}, "
                       f"min={w_stats.min():.4f}, max={w_stats.max():.4f}")

    def _validate_labels(self):
        """验证标签值域"""
        unique_y = np.unique(self.y.numpy())
        expected_y = {0, 1, 2}

        if not set(unique_y).issubset(expected_y):
            raise ValueError(f"y 包含非法值: {unique_y} (期望 {expected_y})")

    def _validate_weights(self):
        """验证权重为有限正值"""
        w = self.weights.numpy()

        if not np.all(np.isfinite(w)):
            raise ValueError(f"权重包含 NaN/Inf: {np.sum(~np.isfinite(w))} 个")

        if not np.all(w >= 0):
            raise ValueError(f"权重包含负值: {np.sum(w < 0)} 个")

        # 警告零权重
        if np.any(w == 0):
            n_zero = np.sum(w == 0)
            logger.warning(f"  ⚠️  {n_zero} 个样本权重为 0")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        item = {
            'lob': self.X[idx],
            'label': self.y[idx],
            'weight': self.weights[idx]
        }

        if self.stock_ids is not None:
            item['stock_id'] = self.stock_ids[idx]

        return item


# ===================================================================
# 温度校准模块
# ===================================================================
class TemperatureScaling(nn.Module):
    """温度缩放校准（Platt Scaling 的泛化版本）"""

    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, logits):
        """应用温度缩放"""
        return logits / self.temperature

    def fit(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        lr: float = 0.01,
        max_iter: int = 50
    ):
        """学习最优温度（在验证集上）

        参数:
            logits: (N, C) 模型输出 logits
            labels: (N,) 真实标签
            weights: (N,) 样本权重（可选）
            lr: 学习率
            max_iter: 最大迭代次数

        返回:
            best_temp: 最优温度值
            nll_before: 校准前的 NLL
            nll_after: 校准后的 NLL
        """
        if weights is None:
            weights = torch.ones(len(labels))

        # 计算校准前 NLL
        nll_criterion = nn.CrossEntropyLoss(reduction='none')
        with torch.no_grad():
            nll_before = (nll_criterion(logits, labels) * weights).sum() / weights.sum()

        # 优化温度
        optimizer = optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)

        def eval_loss():
            optimizer.zero_grad()
            scaled_logits = self.forward(logits)
            loss = (nll_criterion(scaled_logits, labels) * weights).sum() / weights.sum()
            loss.backward()
            return loss

        optimizer.step(eval_loss)

        # 计算校准后 NLL
        with torch.no_grad():
            scaled_logits = self.forward(logits)
            nll_after = (nll_criterion(scaled_logits, labels) * weights).sum() / weights.sum()

        best_temp = self.temperature.item()

        logger.info(f"[温度校准] T = {best_temp:.4f}")
        logger.info(f"  NLL: {nll_before:.4f} → {nll_after:.4f} "
                   f"(↓{nll_before - nll_after:.4f})")

        return best_temp, nll_before.item(), nll_after.item()


# ===================================================================
# 训练与评估函数
# ===================================================================
def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    config: dict,
    scaler: Optional[GradScaler] = None
) -> Dict[str, float]:
    """训练一个 epoch

    返回:
        metrics: {'loss': float, 'acc': float, 'grad_norm': float}
    """
    model.train()

    total_loss = 0
    total_correct = 0
    total_samples = 0
    total_weight = 0
    grad_norms = []

    use_amp = config['optim']['amp']
    grad_clip = config['optim']['grad_clip']

    pbar = tqdm(dataloader, desc='Training', leave=False)

    for batch in pbar:
        lob = batch['lob'].to(device)
        labels = batch['label'].to(device)
        weights = batch['weight'].to(device)

        optimizer.zero_grad()

        # 前向传播（混合精度）
        if use_amp:
            with autocast(device_type='cuda', dtype=torch.float16):
                logits = model(lob)
                loss_per_sample = nn.functional.cross_entropy(
                    logits, labels, reduction='none'
                )
                # 加权 loss
                loss = (loss_per_sample * weights).sum() / weights.sum()
        else:
            logits = model(lob)
            loss_per_sample = nn.functional.cross_entropy(
                logits, labels, reduction='none'
            )
            loss = (loss_per_sample * weights).sum() / weights.sum()

        # 反向传播
        if use_amp and scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            # 梯度裁剪
            if grad_clip > 0:
                total_norm = nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=grad_clip
                )
                grad_norms.append(total_norm.item())

            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()

            if grad_clip > 0:
                total_norm = nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=grad_clip
                )
                grad_norms.append(total_norm.item())

            optimizer.step()

        # 统计
        total_loss += loss.item() * weights.sum().item()
        total_weight += weights.sum().item()

        _, predicted = logits.max(1)
        total_correct += (predicted == labels).sum().item()
        total_samples += labels.size(0)

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100 * total_correct / total_samples:.2f}%'
        })

    metrics = {
        'loss': total_loss / total_weight,
        'acc': 100 * total_correct / total_samples,
        'grad_norm': np.mean(grad_norms) if grad_norms else 0.0
    }

    return metrics


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    config: dict,
    return_predictions: bool = False
) -> Dict:
    """评估模型（支持加权/不加权指标）

    返回:
        results: {
            'weighted': {'loss': float, 'f1_macro': float},
            'unweighted': {'acc': float, 'f1_macro': float,
                          'precision': array, 'recall': array},
            'confusion_matrix': array,
            'predictions': (可选) {'y_true': array, 'y_pred': array,
                                   'logits': array, 'weights': array}
        }
    """
    model.eval()

    all_logits = []
    all_labels = []
    all_weights = []
    all_preds = []

    total_loss = 0
    total_weight = 0

    criterion = nn.CrossEntropyLoss(reduction='none')

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating', leave=False):
            lob = batch['lob'].to(device)
            labels = batch['label'].to(device)
            weights = batch['weight'].to(device)

            logits = model(lob)
            loss_per_sample = criterion(logits, labels)

            # 加权 loss
            total_loss += (loss_per_sample * weights).sum().item()
            total_weight += weights.sum().item()

            _, predicted = logits.max(1)

            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())
            all_weights.append(weights.cpu())
            all_preds.append(predicted.cpu())

    # 合并所有批次
    all_logits = torch.cat(all_logits, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()
    all_weights = torch.cat(all_weights, dim=0).numpy()
    all_preds = torch.cat(all_preds, dim=0).numpy()

    # ===== 加权指标 =====
    weighted_loss = total_loss / total_weight

    # 加权 macro-F1（使用 sample_weight）
    weighted_f1_macro = f1_score(
        all_labels, all_preds, average='macro', sample_weight=all_weights
    )

    # ===== 不加权指标 =====
    unweighted_acc = accuracy_score(all_labels, all_preds)
    unweighted_f1_macro = f1_score(all_labels, all_preds, average='macro')

    # Per-class precision/recall
    precision, recall, _, _ = precision_recall_fscore_support(
        all_labels, all_preds, average=None, zero_division=0
    )

    # 混淆矩阵
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1, 2])

    results = {
        'weighted': {
            'loss': weighted_loss,
            'f1_macro': weighted_f1_macro
        },
        'unweighted': {
            'acc': unweighted_acc,
            'f1_macro': unweighted_f1_macro,
            'precision': precision,
            'recall': recall
        },
        'confusion_matrix': cm
    }

    if return_predictions:
        results['predictions'] = {
            'y_true': all_labels,
            'y_pred': all_preds,
            'logits': all_logits,
            'weights': all_weights
        }

    return results


# ===================================================================
# 辅助函数
# ===================================================================
def plot_confusion_matrix(
    cm: np.ndarray,
    save_path: str,
    normalize: bool = True,
    title: str = 'Confusion Matrix'
):
    """绘制并保存混淆矩阵"""
    if normalize:
        cm = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-12)

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title(title)
    plt.colorbar()

    classes = ['下跌(0)', '持平(1)', '上涨(2)']
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_label_mapping(output_dir: str):
    """保存标签映射 JSON"""
    label_mapping = {
        "format": "v5_labels",
        "mapping": {
            "0": "下跌 (down)",
            "1": "持平 (stationary/flat)",
            "2": "上涨 (up)"
        },
        "note": "V5 labels are already in {0,1,2} format (no conversion needed)"
    }

    path = os.path.join(output_dir, 'label_mapping.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(label_mapping, f, indent=2, ensure_ascii=False)

    logger.info(f"标签映射已保存: {path}")


class EarlyStopping:
    """早停机制"""

    def __init__(self, patience: int = 10, mode: str = 'max', min_delta: float = 0.0):
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0

    def __call__(self, score: float, epoch: int) -> bool:
        """检查是否应该早停

        返回:
            should_stop: True 表示应该停止训练
        """
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            return False

        # 判断是否改进
        if self.mode == 'max':
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta

        if improved:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                logger.info(f"[EarlyStopping] 触发早停！最佳 epoch: {self.best_epoch}")

        return self.early_stop


# ===================================================================
# 主训练函数
# ===================================================================
def main():
    parser = argparse.ArgumentParser(description='DeepLOB V5 训练脚本')
    parser.add_argument('--config', type=str, default='configs/train_v5.yaml',
                       help='YAML 配置文件路径')
    parser.add_argument('--override', type=str, nargs='*',
                       help='覆盖配置项（格式: key.subkey=value）')
    args = parser.parse_args()

    # ========== 载入配置 ==========
    logger.info("=" * 70)
    logger.info("DeepLOB V5 训练脚本")
    logger.info("=" * 70)
    logger.info(f"配置文件: {args.config}")

    yaml_manager = YAMLManager(args.config)
    config = yaml_manager.as_dict()

    # 应用命令行覆盖
    if args.override:
        for override in args.override:
            key, value = override.split('=')
            keys = key.split('.')

            # 嵌套字典设置
            d = config
            for k in keys[:-1]:
                d = d[k]

            # 类型转换
            try:
                d[keys[-1]] = eval(value)
            except:
                d[keys[-1]] = value

            logger.info(f"  覆盖: {key} = {value}")

    # ========== 设置随机种子 ==========
    seed = config['run']['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    logger.info(f"随机种子: {seed}")

    # ========== 设置设备 ==========
    device = torch.device(
        config['hardware']['device'] if torch.cuda.is_available() else 'cpu'
    )
    logger.info(f"设备: {device}")

    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA 版本: {torch.version.cuda}")

    # ========== 创建输出目录 ==========
    output_dir = config['misc']['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"输出目录: {output_dir}")

    # 创建 TensorBoard 日志
    log_dir = os.path.join(
        config['logging']['dir'],
        datetime.now().strftime('%Y%m%d_%H%M%S')
    )
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)
    logger.info(f"TensorBoard 日志: {log_dir}")

    # ========== 载入正规化元数据 ==========
    norm_meta_path = config['data']['norm_meta']
    with open(norm_meta_path, 'r', encoding='utf-8') as f:
        norm_meta = json.load(f)

    logger.info(f"正规化元数据: {norm_meta_path}")
    logger.info(f"  版本: {norm_meta.get('version', 'unknown')}")
    logger.info(f"  特征维度: {norm_meta.get('feature_dim', 'unknown')}")

    # 验证 input_shape
    expected_shape = tuple(config['model']['input']['shape'])
    actual_shape = (norm_meta.get('seq_len', 100), norm_meta.get('feature_dim', 20))

    if expected_shape != actual_shape:
        logger.warning(f"⚠️  input_shape 不匹配: 配置={expected_shape}, "
                      f"实际={actual_shape}")
        config['model']['input']['shape'] = list(actual_shape)

    # ========== 载入数据集 ==========
    logger.info("\n" + "=" * 70)
    logger.info("载入数据集")
    logger.info("=" * 70)

    train_dataset = LOBV5Dataset(
        config['data']['train'], config, split='train', norm_meta=norm_meta
    )
    val_dataset = LOBV5Dataset(
        config['data']['val'], config, split='val', norm_meta=norm_meta
    )
    test_dataset = LOBV5Dataset(
        config['data']['test'], config, split='test', norm_meta=norm_meta
    )

    # DataLoader 配置
    dataloader_kwargs = {
        'batch_size': config['dataloader']['batch_size'],
        'num_workers': config['dataloader']['num_workers'],
        'pin_memory': config['dataloader']['pin_memory']
    }

    if dataloader_kwargs['num_workers'] > 0:
        dataloader_kwargs['prefetch_factor'] = config['hardware']['prefetch_factor']
        dataloader_kwargs['persistent_workers'] = config['hardware']['persistent_workers']

    train_loader = DataLoader(train_dataset, shuffle=True, **dataloader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **dataloader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **dataloader_kwargs)

    # ========== 创建模型 ==========
    logger.info("\n" + "=" * 70)
    logger.info("创建模型")
    logger.info("=" * 70)

    model = DeepLOB(
        input_shape=tuple(config['model']['input']['shape']),
        num_classes=config['model']['num_classes'],
        conv1_filters=config['model']['conv1_filters'],
        conv2_filters=config['model']['conv2_filters'],
        conv3_filters=config['model']['conv3_filters'],
        lstm_hidden_size=config['model']['lstm_hidden_size'],
        fc_hidden_size=config['model']['fc_hidden_size'],
        dropout=config['model']['dropout']
    ).to(device)

    # 统计参数
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"模型参数量: {total_params:,}")

    # ========== 损失函数（类别权重） ==========
    class_weights = None
    if config['loss']['class_weights'] == 'auto':
        # 自动计算逆频率权重
        train_labels = train_dataset.y.numpy()
        class_counts = np.bincount(train_labels)
        total = len(train_labels)

        weights = total / (len(class_counts) * class_counts + 1e-6)
        weights = weights / weights.mean()  # 归一化到均值=1

        class_weights = torch.FloatTensor(weights).to(device)
        logger.info(f"\n类别权重（自动）: {weights}")

    # ========== 优化器 ==========
    if config['optim']['name'] == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config['optim']['lr'],
            weight_decay=config['optim']['weight_decay']
        )
    else:
        raise ValueError(f"不支持的优化器: {config['optim']['name']}")

    logger.info(f"优化器: {config['optim']['name']}")
    logger.info(f"  lr: {config['optim']['lr']}")
    logger.info(f"  weight_decay: {config['optim']['weight_decay']}")

    # ========== 学习率调度器 ==========
    num_epochs = config['train']['epochs']
    warmup_epochs = int(num_epochs * config['sched']['warmup_ratio'])

    if config['sched']['name'] == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=num_epochs - warmup_epochs,
            eta_min=1e-6
        )
    else:
        scheduler = None

    logger.info(f"学习率调度器: {config['sched']['name']}")
    logger.info(f"  warmup_epochs: {warmup_epochs}")

    # ========== 混合精度训练 ==========
    scaler = None
    if config['optim']['amp'] and device.type == 'cuda':
        scaler = GradScaler('cuda')
        logger.info("混合精度训练 (AMP): 已启用")

    # ========== 早停 ==========
    early_stop_metric = config['train']['early_stop']['metric']
    early_stop_patience = config['train']['early_stop']['patience']
    early_stop_mode = config['train']['early_stop']['mode']

    early_stopping = EarlyStopping(
        patience=early_stop_patience,
        mode=early_stop_mode
    )

    logger.info(f"早停配置: metric={early_stop_metric}, "
               f"patience={early_stop_patience}, mode={early_stop_mode}")

    # ========== 训练循环 ==========
    logger.info("\n" + "=" * 70)
    logger.info("开始训练")
    logger.info("=" * 70)

    best_score = -np.inf if early_stop_mode == 'max' else np.inf
    best_epoch = 0

    for epoch in range(1, num_epochs + 1):
        logger.info(f"\nEpoch {epoch}/{num_epochs}")
        logger.info("-" * 70)

        # ===== 训练 =====
        train_metrics = train_epoch(
            model, train_loader, None, optimizer, device, config, scaler
        )

        logger.info(f"Train - Loss: {train_metrics['loss']:.4f}, "
                   f"Acc: {train_metrics['acc']:.2f}%, "
                   f"Grad: {train_metrics['grad_norm']:.2f}")

        # ===== 验证 =====
        val_results = evaluate(model, val_loader, device, config)

        logger.info(f"Val   - Weighted Loss: {val_results['weighted']['loss']:.4f}, "
                   f"Weighted F1: {val_results['weighted']['f1_macro']:.4f}")
        logger.info(f"      - Unweighted Acc: {val_results['unweighted']['acc']:.4f}, "
                   f"Unweighted F1: {val_results['unweighted']['f1_macro']:.4f}")

        # Per-class metrics
        for i, (p, r) in enumerate(zip(
            val_results['unweighted']['precision'],
            val_results['unweighted']['recall']
        )):
            logger.info(f"      - Class {i}: P={p:.4f}, R={r:.4f}")

        # ===== 学习率调整（warmup 后） =====
        if epoch > warmup_epochs and scheduler is not None:
            scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']

        # ===== TensorBoard 记录 =====
        writer.add_scalar('Loss/train', train_metrics['loss'], epoch)
        writer.add_scalar('Loss/val_weighted', val_results['weighted']['loss'], epoch)
        writer.add_scalar('Accuracy/train', train_metrics['acc'], epoch)
        writer.add_scalar('Accuracy/val_unweighted',
                         val_results['unweighted']['acc'] * 100, epoch)
        writer.add_scalar('F1/val_weighted',
                         val_results['weighted']['f1_macro'], epoch)
        writer.add_scalar('F1/val_unweighted',
                         val_results['unweighted']['f1_macro'], epoch)
        writer.add_scalar('LR', current_lr, epoch)
        writer.add_scalar('Grad/norm', train_metrics['grad_norm'], epoch)

        # ===== 保存最佳模型 =====
        current_score = val_results['weighted']['f1_macro']

        if ((early_stop_mode == 'max' and current_score > best_score) or
            (early_stop_mode == 'min' and current_score < best_score)):
            best_score = current_score
            best_epoch = epoch

            save_path = os.path.join(output_dir, 'deeplob_v5_best.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_weighted_f1': current_score,
                'val_unweighted_acc': val_results['unweighted']['acc'],
                'config': config
            }, save_path)

            logger.info(f"\n[保存] 最佳模型 (Weighted F1: {current_score:.4f})")

        # ===== 早停检查 =====
        if early_stopping(current_score, epoch):
            logger.info(f"\n提前结束训练！最佳 epoch: {best_epoch}")
            break

    # ========== 训练完成 ==========
    logger.info("\n" + "=" * 70)
    logger.info("训练完成")
    logger.info("=" * 70)
    logger.info(f"最佳 epoch: {best_epoch}")
    logger.info(f"最佳 Weighted F1: {best_score:.4f}")

    # ========== 载入最佳模型并评估 ==========
    best_model_path = os.path.join(output_dir, 'deeplob_v5_best.pth')
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    logger.info("\n" + "=" * 70)
    logger.info("测试集评估（使用最佳模型）")
    logger.info("=" * 70)

    test_results = evaluate(
        model, test_loader, device, config, return_predictions=True
    )

    logger.info(f"Test - Weighted Loss: {test_results['weighted']['loss']:.4f}, "
               f"Weighted F1: {test_results['weighted']['f1_macro']:.4f}")
    logger.info(f"     - Unweighted Acc: {test_results['unweighted']['acc']:.4f}, "
               f"Unweighted F1: {test_results['unweighted']['f1_macro']:.4f}")

    for i, (p, r) in enumerate(zip(
        test_results['unweighted']['precision'],
        test_results['unweighted']['recall']
    )):
        logger.info(f"     - Class {i}: P={p:.4f}, R={r:.4f}")

    # ===== 保存混淆矩阵 =====
    plot_confusion_matrix(
        test_results['confusion_matrix'],
        save_path=os.path.join(output_dir, 'confusion_matrix_test.png'),
        normalize=True,
        title='Test Set Confusion Matrix (Best Model)'
    )

    # ===== 温度校准（如果启用） =====
    if config['calibration']['enabled']:
        logger.info("\n" + "=" * 70)
        logger.info("温度校准")
        logger.info("=" * 70)

        temp_scaler = TemperatureScaling().to(device)

        # 使用验证集学习温度
        val_preds = evaluate(
            model, val_loader, device, config, return_predictions=True
        )['predictions']

        best_temp, nll_before, nll_after = temp_scaler.fit(
            torch.FloatTensor(val_preds['logits']).to(device),
            torch.LongTensor(val_preds['y_true']).to(device),
            torch.FloatTensor(val_preds['weights']).to(device)
        )

        # 保存校准参数
        calib_path = os.path.join(output_dir, 'calibration.pt')
        torch.save({
            'temperature': best_temp,
            'nll_before': nll_before,
            'nll_after': nll_after
        }, calib_path)

        logger.info(f"校准参数已保存: {calib_path}")

    # ===== 保存工件 =====
    save_label_mapping(output_dir)

    # 保存 normalization_meta 副本
    import shutil
    shutil.copy(
        norm_meta_path,
        os.path.join(output_dir, 'normalization_meta.json')
    )

    # 保存完整配置
    with open(os.path.join(output_dir, 'config.yaml'), 'w', encoding='utf-8') as f:
        import yaml
        yaml.dump(config, f, allow_unicode=True, sort_keys=False)

    # 保存测试指标
    test_metrics = {
        'weighted': {
            'loss': float(test_results['weighted']['loss']),
            'f1_macro': float(test_results['weighted']['f1_macro'])
        },
        'unweighted': {
            'acc': float(test_results['unweighted']['acc']),
            'f1_macro': float(test_results['unweighted']['f1_macro']),
            'precision': test_results['unweighted']['precision'].tolist(),
            'recall': test_results['unweighted']['recall'].tolist()
        },
        'confusion_matrix': test_results['confusion_matrix'].tolist()
    }

    with open(os.path.join(output_dir, 'test_metrics.json'), 'w') as f:
        json.dump(test_metrics, f, indent=2)

    logger.info("\n所有工件已保存至: " + output_dir)

    writer.close()
    logger.info(f"\nTensorBoard 日志: {log_dir}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
