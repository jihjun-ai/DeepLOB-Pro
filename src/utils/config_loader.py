"""Configuration loader utility."""

from pathlib import Path
from typing import Dict, Any, Optional
from .yaml_manager import YAMLManager


def load_config(config_path: str | Path) -> Dict[str, Any]:
    """
    Load YAML configuration file using YAMLManager.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Dictionary containing configuration (plain dict, not proxy)

    Raises:
        FileNotFoundError: If config file doesn't exist

    Note:
        Uses YAMLManager which properly handles:
        - Scientific notation (1e-4)
        - Preserves comments
        - Returns plain dict suitable for PyTorch configs
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # Load with YAMLManager and convert to plain dict
    yaml_manager = YAMLManager(config_path)
    config = yaml_manager.as_dict()

    return config


def save_config(config: Dict[str, Any], save_path: str | Path) -> None:
    """
    Save configuration to YAML file.

    Args:
        config: Configuration dictionary
        save_path: Path to save configuration
    """
    from ruamel.yaml import YAML

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    yaml = YAML()
    yaml.preserve_quotes = True
    yaml.indent(mapping=2, sequence=2, offset=0)

    with open(save_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f)


def merge_configs(base_config: Dict[str, Any],
                  override_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Merge two configuration dictionaries (deep merge).

    Args:
        base_config: Base configuration
        override_config: Configuration to override base values

    Returns:
        Merged configuration
    """
    if override_config is None:
        return base_config.copy()

    merged = base_config.copy()

    for key, value in override_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value

    return merged


def validate_config(config: Dict[str, Any], required_keys: list[str]) -> None:
    """
    Validate configuration has required keys.

    Args:
        config: Configuration dictionary
        required_keys: List of required keys (supports nested keys with dot notation)

    Raises:
        ValueError: If required keys are missing
    """
    missing_keys = []

    for key in required_keys:
        # Support nested keys like "model.lstm.hidden_size"
        keys = key.split('.')
        current = config

        for k in keys:
            if not isinstance(current, dict) or k not in current:
                missing_keys.append(key)
                break
            current = current[k]

    if missing_keys:
        raise ValueError(f"Missing required config keys: {missing_keys}")
