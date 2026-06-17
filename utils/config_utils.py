"""
Configuration Utilities

Provides utilities for loading and merging YAML configurations.
"""

import os
import yaml
from typing import Any, Dict, Optional

try:
    from omegaconf import OmegaConf
except ImportError:  # Optional dependency; plain YAML config loading still works.
    OmegaConf = None


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to YAML config file

    Returns:
        Dictionary with configuration
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config


def _require_omegaconf():
    if OmegaConf is None:
        raise ImportError(
            "OmegaConf is required for this helper. Install it with `pip install omegaconf` "
            "or use load_config() for plain YAML configs."
        )
    return OmegaConf


def load_config_omegaconf(config_path: str):
    """
    Load configuration using OmegaConf.

    OmegaConf provides more flexible configuration handling.

    Args:
        config_path: Path to YAML config file

    Returns:
        OmegaConf DictConfig
    """
    return _require_omegaconf().load(config_path)


def merge_config(base_config: Dict, override_config: Dict) -> Dict:
    """
    Merge override config into base config.

    Args:
        base_config: Base configuration
        override_config: Configuration to override with

    Returns:
        Merged configuration
    """
    result = base_config.copy()

    for key, value in override_config.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_config(result[key], value)
        else:
            result[key] = value

    return result


def merge_config_omegaconf(base, override):
    """
    Merge OmegaConf configurations.

    Args:
        base: Base configuration
        override: Configuration to override with

    Returns:
        Merged OmegaConf DictConfig
    """
    return _require_omegaconf().merge(base, override)


def save_config(config: Dict, save_path: str):
    """
    Save configuration to YAML file.

    Args:
        config: Configuration dictionary
        save_path: Path to save config
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def get_config_value(config: Dict, key_path: str, default: Any = None) -> Any:
    """
    Get value from nested config using dot notation.

    Args:
        config: Configuration dictionary
        key_path: Dot-separated path (e.g., 'model.d_model')
        default: Default value if key not found

    Returns:
        Configuration value
    """
    keys = key_path.split('.')
    value = config

    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default

    return value


def set_config_value(config: Dict, key_path: str, value: Any):
    """
    Set value in nested config using dot notation.

    Args:
        config: Configuration dictionary
        key_path: Dot-separated path (e.g., 'model.d_model')
        value: Value to set
    """
    keys = key_path.split('.')
    current = config

    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]

    current[keys[-1]] = value


class Config:
    """
    Configuration container with dot notation access.
    """

    def __init__(self, config_dict: Optional[Dict] = None):
        if config_dict is None:
            config_dict = {}
        self._config = config_dict

    def __getattr__(self, key: str) -> Any:
        if key.startswith('_'):
            return super().__getattribute__(key)

        if key in self._config:
            value = self._config[key]
            if isinstance(value, dict):
                return Config(value)
            return value

        raise AttributeError(f"Config has no key: {key}")

    def __setattr__(self, key: str, value: Any):
        if key.startswith('_'):
            super().__setattr__(key, value)
        else:
            self._config[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        """Get value with default."""
        try:
            return getattr(self, key)
        except AttributeError:
            return default

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return self._config

    def update(self, other: Dict):
        """Update config with another dictionary."""
        self._config.update(other)

    def __repr__(self) -> str:
        return f"Config({self._config})"


def create_config_from_args(args) -> Config:
    """
    Create Config object from command line arguments.

    Args:
        args: Argument namespace

    Returns:
        Config object
    """
    config = {}

    # Model config
    config['model'] = {
        'd_model': getattr(args, 'd_model', 256),
        'n_heads': getattr(args, 'n_heads', 8),
        'd_ffn': getattr(args, 'd_ffn', 1024),
        'n_layers': getattr(args, 'n_layers', 4),
        'dropout': getattr(args, 'dropout', 0.1),
    }

    # Data config
    config['data'] = {
        'data_root': getattr(args, 'data_root', 'data/'),
        'batch_size': getattr(args, 'batch_size', 8),
        'num_workers': getattr(args, 'num_workers', 4),
    }

    # Training config
    config['train'] = {
        'epochs': getattr(args, 'epochs', 500),
        'lr': getattr(args, 'lr', 1e-4),
        'weight_decay': getattr(args, 'weight_decay', 0.05),
    }

    # Other
    config['seed'] = getattr(args, 'seed', 42)
    config['output_dir'] = getattr(args, 'output_dir', 'outputs/')

    return Config(config)
