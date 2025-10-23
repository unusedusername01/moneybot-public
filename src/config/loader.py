# Copyright 2025 unusedusername01
# SPDX-License-Identifier: Apache-2.0

"""Configuration loader for MoneyBot.

Loads layered YAML configuration files and exposes utility helpers
for accessing values with fallbacks. The loader keeps the FastAPI
application defaults intact if configuration files are missing.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Optional

import logging

import yaml

_LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"
ENV_OVERRIDE_PREFIX = "MONEYBOT__"


def _read_yaml(path: Path) -> Dict[str, Any]:
    """Safely read a YAML file into a dictionary."""
    if not path.exists():
        return {}

    try:
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
    except yaml.YAMLError as exc:  # pragma: no cover - defensive logging
        raise ValueError(f"Failed to parse YAML file: {path}") from exc

    if not isinstance(data, dict):
        raise ValueError(f"Configuration file {path} must define a mapping at the root.")

    return data


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    """Recursively merge two dictionaries without side effects."""
    result: Dict[str, Any] = dict(base)

    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], Mapping)
            and isinstance(value, Mapping)
        ):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value

    return result


def _coerce_value(raw: str, current: Optional[Any]) -> Any:
    """Attempt to coerce environment override strings to the target type."""
    if current is None:
        return raw
    if isinstance(current, bool):
        lowered = raw.lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
        return current
    if isinstance(current, int):
        try:
            return int(raw)
        except ValueError:
            return current
    if isinstance(current, float):
        try:
            return float(raw)
        except ValueError:
            return current
    if isinstance(current, (list, tuple, set)):
        return [item.strip() for item in raw.split(",") if item.strip()]
    return raw


def _set_nested(mapping: MutableMapping[str, Any], path: Iterable[str], value: Any) -> None:
    """Set a nested key in a mapping based on an iterable path."""
    keys = list(path)
    current = mapping
    for key in keys[:-1]:
        if key not in current or not isinstance(current[key], MutableMapping):
            current[key] = {}
        current = current[key]  # type: ignore[assignment]
    current[keys[-1]] = value


def _apply_env_overrides(config: Dict[str, Any]) -> Dict[str, Any]:
    """Apply environment variable overrides using the MONEYBOT__ prefix."""
    overrides = {
        key[len(ENV_OVERRIDE_PREFIX) :]: value
        for key, value in os.environ.items()
        if key.startswith(ENV_OVERRIDE_PREFIX)
    }

    if not overrides:
        return config

    mutable = dict(config)
    for dotted_path, raw_value in overrides.items():
        path_parts = [part.strip() for part in dotted_path.split("__") if part.strip()]
        if not path_parts:
            continue

        current_value = get(mutable, path_parts, default=None)
        coerced = _coerce_value(raw_value, current_value)
        _set_nested(mutable, path_parts, coerced)

    return mutable


def load_config(preset: Optional[str] = None, *, fail_fast: bool = False) -> Dict[str, Any]:
    """Load MoneyBot configuration from YAML files.

    Order of precedence (later wins):
      1. config/base.yml if present
      2. config/<preset>.yml (explicit CLI preset or MONEYBOT_PRESET env)
      3. config/local.yml (optional developer overrides)
      4. Environment variable overrides MONEYBOT__section__key=value
    """
    if not CONFIG_DIR.exists():
        if fail_fast:
            raise FileNotFoundError("Configuration directory not found")
        return {}

    config: Dict[str, Any] = {}

    base_path = CONFIG_DIR / "base.yml"
    if base_path.exists():
        config = _deep_merge(config, _read_yaml(base_path))
    elif fail_fast and preset is None:
        raise FileNotFoundError("Missing required config/base.yml file")

    preset_name = preset or os.getenv("MONEYBOT_PRESET")
    if preset_name:
        preset_path = CONFIG_DIR / f"{preset_name}.yml"
        if not preset_path.exists():
            message = f"Configuration preset '{preset_name}' not found at {preset_path}"
            if fail_fast:
                raise FileNotFoundError(message)
            _LOGGER.warning(message)
        else:
            config = _deep_merge(config, _read_yaml(preset_path))

    local_path = CONFIG_DIR / "local.yml"
    if local_path.exists():
        config = _deep_merge(config, _read_yaml(local_path))

    config = _apply_env_overrides(config)

    return config


def get(config: Mapping[str, Any], path: Iterable[str] | str, default: Any = None) -> Any:
    """Retrieve a value from a configuration mapping using a dotted path."""
    if isinstance(path, str):
        parts = [part for part in path.split(".") if part]
    else:
        parts = list(path)

    current: Any = config
    for part in parts:
        if not isinstance(current, Mapping) or part not in current:
            return default
        current = current[part]
    return current


__all__ = [
    "CONFIG_DIR",
    "load_config",
    "get",
]
