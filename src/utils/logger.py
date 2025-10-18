"""Logging utilities."""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime


def setup_logger(
    name: str,
    log_file: Optional[str | Path] = None,
    level: int = logging.INFO,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Setup logger with console and file handlers.

    Args:
        name: Logger name
        log_file: Optional path to log file
        level: Logging level
        format_string: Custom format string

    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Remove existing handlers
    logger.handlers = []

    # Default format
    if format_string is None:
        format_string = '[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s'

    formatter = logging.Formatter(format_string, datefmt='%Y-%m-%d %H:%M:%S')

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_timestamp_str() -> str:
    """Get current timestamp as string for filenames."""
    return datetime.now().strftime('%Y%m%d_%H%M%S')


class MetricsLogger:
    """Logger for training metrics."""

    def __init__(self, log_dir: str | Path):
        """
        Initialize metrics logger.

        Args:
            log_dir: Directory to save metrics
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.metrics_file = self.log_dir / f"metrics_{get_timestamp_str()}.txt"
        self.metrics = []

    def log(self, step: int, metrics: dict) -> None:
        """
        Log metrics for a step.

        Args:
            step: Training step/epoch
            metrics: Dictionary of metric name -> value
        """
        log_entry = {'step': step, **metrics}
        self.metrics.append(log_entry)

        # Write to file
        with open(self.metrics_file, 'a', encoding='utf-8') as f:
            metric_str = f"Step {step}: " + ", ".join(
                f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                for k, v in metrics.items()
            )
            f.write(metric_str + '\n')

    def get_metrics(self) -> list[dict]:
        """Get all logged metrics."""
        return self.metrics.copy()
