"""Training script for DeepLOB model."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from tqdm import tqdm
import logging

from src.utils import load_config, setup_logger
from src.data import FI2010Loader, LOBPreprocessor
from src.models import DeepLOB


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    use_amp: bool = True
) -> dict:
    """Train for one epoch."""
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc="Training")
    for batch_idx, (inputs, labels) in enumerate(pbar):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        # Mixed precision training
        if use_amp:
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        # Statistics
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # Update progress bar
        pbar.set_postfix({
            'loss': total_loss / (batch_idx + 1),
            'acc': 100. * correct / total
        })

    return {
        'loss': total_loss / len(train_loader),
        'accuracy': 100. * correct / total
    }


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> dict:
    """Validate model."""
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc="Validation"):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return {
        'loss': total_loss / len(val_loader),
        'accuracy': 100. * correct / total
    }


def main():
    parser = argparse.ArgumentParser(description="Train DeepLOB model")
    parser.add_argument("--config", type=str, default="configs/deeplob_config.yaml",
                        help="Path to config file")
    parser.add_argument("--data-dir", type=str, default="data/raw",
                        help="Data directory")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda or cpu)")
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Setup logging
    logger = setup_logger(
        "deeplob_training",
        log_file=f"{config['logging']['log_dir']}/training.log"
    )
    logger.info(f"Config: {config}")

    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Enable TF32 for RTX 5090
    if device.type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = config['hardware'].get('cudnn_benchmark', True)

    # Load and preprocess data
    logger.info("Loading data...")
    data_loader = FI2010Loader(args.data_dir)

    # Load all training files
    from pathlib import Path
    train_path = Path(args.data_dir) / "Training"
    train_files = sorted(train_path.glob("Train_Dst_NoAuction_ZScore_CF_*.txt"))

    logger.info(f"Found {len(train_files)} training files")

    # Load and concatenate all training data
    all_features = []
    all_labels = []

    for file_path in train_files:
        logger.info(f"Loading {file_path.name}...")
        features, labels = data_loader.load_from_file(file_path)
        all_features.append(features)
        all_labels.append(labels)

    # Concatenate all data
    features = np.concatenate(all_features, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    # Use first prediction horizon (k=10)
    labels = labels[:, 0]

    logger.info(f"Total data shape: {features.shape}, Labels shape: {labels.shape}")

    # Preprocess
    preprocessor = LOBPreprocessor(
        normalization_method=config['data']['normalization']['method'],
        sequence_length=config['data']['sequence_length'],
        prediction_horizon=config['data']['prediction_horizon']
    )

    # Normalize
    features_normalized = preprocessor.normalize(features, fit=True)

    # Create sequences
    sequences, sequence_labels = preprocessor.create_sequences(
        features_normalized,
        labels,
        stride=1
    )

    # Train/val/test split
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = \
        preprocessor.train_val_test_split(
            sequences,
            sequence_labels,
            train_ratio=config['data']['train_split'],
            val_ratio=config['data']['val_split'],
            test_ratio=config['data']['test_split']
        )

    logger.info(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    # Create data loaders
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.LongTensor(y_train)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val),
        torch.LongTensor(y_val)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['hardware']['num_workers'],
        pin_memory=config['hardware']['pin_memory']
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['hardware']['num_workers'],
        pin_memory=config['hardware']['pin_memory']
    )

    # Create model
    logger.info("Creating model...")
    model = DeepLOB(
        input_shape=tuple(config['model']['input_shape']),
        num_classes=config['model']['output_classes'],
        conv1_filters=config['model']['conv1']['out_channels'],
        conv2_filters=config['model']['conv2']['out_channels'],
        conv3_filters=config['model']['conv3']['out_channels'],
        lstm_hidden_size=config['model']['lstm']['hidden_size'],
        fc_hidden_size=config['model']['fc']['hidden_size'],
        dropout=config['model']['fc']['dropout']
    )
    model = model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=config['training']['scheduler']['mode'],
        factor=config['training']['scheduler']['factor'],
        patience=config['training']['scheduler']['patience'],
        min_lr=config['training']['scheduler']['min_lr']
    )

    # Mixed precision scaler
    scaler = GradScaler(enabled=config['training']['mixed_precision']['enabled'])

    # TensorBoard
    writer = SummaryWriter(config['logging']['log_dir'])

    # Training loop
    best_val_acc = 0.0
    epochs_no_improve = 0

    for epoch in range(config['training']['epochs']):
        logger.info(f"\nEpoch {epoch + 1}/{config['training']['epochs']}")

        # Train
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, scaler, device,
            use_amp=config['training']['mixed_precision']['enabled']
        )

        # Validate
        val_metrics = validate(model, val_loader, criterion, device)

        # Log metrics
        logger.info(
            f"Train Loss: {train_metrics['loss']:.4f}, "
            f"Train Acc: {train_metrics['accuracy']:.2f}%, "
            f"Val Loss: {val_metrics['loss']:.4f}, "
            f"Val Acc: {val_metrics['accuracy']:.2f}%"
        )

        writer.add_scalar('Loss/train', train_metrics['loss'], epoch)
        writer.add_scalar('Loss/val', val_metrics['loss'], epoch)
        writer.add_scalar('Accuracy/train', train_metrics['accuracy'], epoch)
        writer.add_scalar('Accuracy/val', val_metrics['accuracy'], epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)

        # Learning rate scheduling
        scheduler.step(val_metrics['loss'])

        # Save best model
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            save_path = Path(config['checkpoint']['save_dir']) / "best_model.pt"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), save_path)
            logger.info(f"Saved best model: {save_path}")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # Early stopping
        if config['training']['early_stopping']['enabled'] and \
           epochs_no_improve >= config['training']['early_stopping']['patience']:
            logger.info(f"Early stopping after {epoch + 1} epochs")
            break

    writer.close()
    logger.info(f"Training completed. Best val accuracy: {best_val_acc:.2f}%")


if __name__ == "__main__":
    main()
