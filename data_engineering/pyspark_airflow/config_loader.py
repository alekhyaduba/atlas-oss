"""
config_loader.py
Loads and validates run_config.yaml; exposes a typed Config object.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).parent / "run_config.yaml"


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


class Config:
    """Thin wrapper around the parsed YAML dict with dot-access helpers."""

    def __init__(self, data: dict[str, Any]):
        self._data = data

    def get(self, *keys: str, default: Any = None) -> Any:
        node = self._data
        for key in keys:
            if not isinstance(node, dict):
                return default
            node = node.get(key, default)
        return node

    # ── Convenience properties ───────────────────────────────

    @property
    def captioning_backend(self) -> str:
        return self.get("model", "captioning_backend", default="blip")

    @property
    def tagging_backend(self) -> str:
        return self.get("model", "tagging_backend", default="clip")

    @property
    def model_cfg(self) -> dict:
        return self.get("model", self.captioning_backend, default={})

    @property
    def clip_cfg(self) -> dict:
        return self.get("model", "clip", default={})

    @property
    def spark_cfg(self) -> dict:
        return self.get("spark", default={})

    @property
    def preprocessing_cfg(self) -> dict:
        return self.get("preprocessing", default={})

    @property
    def paths(self) -> dict:
        return self.get("paths", default={})

    @property
    def dvc_cfg(self) -> dict:
        return self.get("dvc", default={})

    @property
    def pipeline_version(self) -> str:
        return self.get("pipeline", "version", default="0.0.0")

    def __repr__(self) -> str:
        backend = self.captioning_backend
        return f"<Config pipeline={self.get('pipeline','name')} backend={backend}>"


@lru_cache(maxsize=1)
def load_config(config_path: str | None = None) -> Config:
    """
    Load config from YAML file, then overlay any env-var overrides.

    Environment variable overrides (examples):
        PIPELINE_CAPTIONING_BACKEND=blip2
        PIPELINE_SPARK_MASTER=spark://spark-master:7077
        PIPELINE_MODEL_DEVICE=cuda
    """
    path = Path(config_path or os.environ.get("PIPELINE_CONFIG_PATH", DEFAULT_CONFIG_PATH))
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as f:
        data = yaml.safe_load(f)

    # Apply env-var overrides ─────────────────────────────────
    overrides: dict[str, Any] = {}

    if backend := os.environ.get("PIPELINE_CAPTIONING_BACKEND"):
        overrides.setdefault("model", {})["captioning_backend"] = backend

    if device := os.environ.get("PIPELINE_MODEL_DEVICE"):
        for key in ("blip", "blip2", "git"):
            overrides.setdefault("model", {}).setdefault(key, {})["device"] = device

    if master := os.environ.get("PIPELINE_SPARK_MASTER"):
        overrides.setdefault("spark", {})["master"] = master

    if env := os.environ.get("PIPELINE_ENVIRONMENT"):
        overrides.setdefault("pipeline", {})["environment"] = env

    if overrides:
        data = _deep_merge(data, overrides)

    return Config(data)
