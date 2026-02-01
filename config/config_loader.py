"""
config_loader.py
-----------------
Singleton config loader. Reads config.yaml once and caches it.
All modules access configuration through this — never hardcoded values.
"""

import os
import yaml
from typing import Any, Dict


_CONFIG_CACHE: Dict[str, Any] = {}


def load_config(config_path: str | None = None) -> Dict[str, Any]:
    """
    Load and cache the YAML configuration file.

    Args:
        config_path: Path to config.yaml. Defaults to config/ relative to this file.

    Returns:
        Full config dictionary.
    """
    global _CONFIG_CACHE

    if _CONFIG_CACHE:
        return _CONFIG_CACHE

    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), "config.yaml")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r") as f:
        _CONFIG_CACHE = yaml.safe_load(f)

    return _CONFIG_CACHE


def get_recurring_detection_config() -> Dict[str, Any]:
    """Returns the recurring_detection block."""
    return load_config()["recurring_detection"]


def get_merchant_taxonomy() -> list[Dict[str, Any]]:
    """Returns the merchant taxonomy list."""
    return load_config()["merchant_taxonomy"]


def get_product_interpreter_config(product_type: str) -> Dict[str, Any]:
    """
    Returns interpreter config for a specific product type.

    Raises:
        KeyError: If product_type is not in the config.
    """
    interpreters = load_config()["product_interpreters"]
    if product_type not in interpreters:
        raise KeyError(
            f"No interpreter config for '{product_type}'. "
            f"Available: {list(interpreters.keys())}"
        )
    return interpreters[product_type]


def get_all_product_types() -> list[str]:
    """Returns all configured product types."""
    return list(load_config()["product_interpreters"].keys())


def get_confidence_tiers() -> Dict[str, Dict[str, float]]:
    """Returns confidence tier boundaries."""
    return load_config()["confidence_tiers"]


def get_drift_monitoring_config() -> Dict[str, Any]:
    """Returns drift monitoring config."""
    return load_config()["drift_monitoring"]


def reset_config() -> None:
    """Clears cached config. Useful for testing."""
    global _CONFIG_CACHE
    _CONFIG_CACHE = {}
