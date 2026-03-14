"""
Load Config Utility
===================

Load YAML configuration files with defaults and CLI overrides.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def load_config(
    config_path: Path | str | None,
    default_config_path: Path | str | None = None,
) -> dict[str, Any]:
    """Load configuration from YAML file.

    Parameters
    ----------
    config_path : Path | str | None
        Path to config file. If None, uses default_config_path.
    default_config_path : Path | str | None
        Default config path if config_path is None.

    Returns
    -------
    dict[str, Any]
        Configuration dictionary.
    """
    try:
        import yaml
    except ImportError:
        logger.warning("PyYAML not installed, using empty config")
        return {}

    # Determine which config file to load
    path = config_path or default_config_path
    if path is None:
        return {}

    path = Path(path)
    if not path.exists():
        logger.debug("Config file not found: %s", path)
        return {}

    with open(path, encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    logger.info("Loaded config from: %s", path)
    return config


def get_nested(config: dict, *keys: str, default: Any = None) -> Any:
    """Get nested value from config dictionary.

    Parameters
    ----------
    config : dict
        Configuration dictionary.
    *keys : str
        Nested keys to traverse.
    default : Any
        Default value if key not found.

    Returns
    -------
    Any
        Value at nested key path, or default.

    Example
    -------
    >>> config = {"canvas": {"width": 8000}}
    >>> get_nested(config, "canvas", "width", default=4000)
    8000
    """
    current = config
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current if current is not None else default


def merge_cli_args(config: dict, args: Any, mapping: dict[str, tuple]) -> dict:
    """Merge CLI arguments into config, CLI args take precedence.

    Parameters
    ----------
    config : dict
        Base configuration dictionary.
    args : Any
        Parsed argparse Namespace.
    mapping : dict[str, tuple]
        Mapping of CLI arg names to config paths.
        e.g., {"width": ("canvas", "width"), "random": ("random", "count")}

    Returns
    -------
    dict
        Merged configuration.
    """
    result = deep_copy(config)

    for arg_name, config_path in mapping.items():
        arg_value = getattr(args, arg_name, None)
        # Only override if CLI arg was explicitly provided (not None/default)
        if arg_value is not None:
            set_nested(result, config_path, arg_value)

    return result


def deep_copy(d: dict) -> dict:
    """Create a deep copy of a dictionary."""
    result = {}
    for k, v in d.items():
        if isinstance(v, dict):
            result[k] = deep_copy(v)
        elif isinstance(v, list):
            result[k] = v.copy()
        else:
            result[k] = v
    return result


def set_nested(d: dict, keys: tuple, value: Any) -> None:
    """Set a nested value in a dictionary, creating intermediate dicts as needed."""
    for key in keys[:-1]:
        if key not in d or not isinstance(d[key], dict):
            d[key] = {}
        d = d[key]
    d[keys[-1]] = value
