# Module 01 — Config-Driven Design
## `run_config.yaml` + `config_loader.py`

**Estimated time:** 30–45 minutes
**Difficulty:** ⭐☆☆☆☆ Beginner

---

## Why This Module Matters

One of the most common anti-patterns in data engineering is **hardcoded values in code**. When your model name, file paths, batch sizes, and device settings are buried in Python scripts, you need a redeploy every time a setting changes.

Config-driven design solves this:
- Settings live in a **single source of truth** (YAML file)
- Code reads config at runtime — no redeploy needed
- Environment variables let you **override per-environment** (dev vs prod)
- Trigger-time overrides let Airflow jobs **change behaviour dynamically**

---

## Concept 1 — YAML as a Configuration Language

YAML (Yet Another Markup Language) is the standard format for configuration in the data/DevOps world. It's human-readable and maps naturally to Python dicts.

```yaml
# Scalar values
name: "my_pipeline"
version: "1.0.0"
debug: false

# Nested structure (maps to Python dict of dicts)
model:
  captioning_backend: "blip"
  blip:
    model_name: "Salesforce/blip-image-captioning-base"
    device: "cpu"
    batch_size: 4

# Lists
tags:
  - "outdoor scene"
  - "portrait"
  - "urban"
```

**Reading YAML in Python:**

```python
import yaml

with open("run_config.yaml") as f:
    data = yaml.safe_load(f)   # returns a plain dict

# Access nested values
backend = data["model"]["captioning_backend"]   # "blip"
batch   = data["model"]["blip"]["batch_size"]   # 4
```

> **`safe_load` vs `load`** — Always use `safe_load`. `load` can execute arbitrary Python via YAML tags, which is a security risk.

---

## Concept 2 — Deep Merging Configs

Our pipeline needs to support **layered config**: a base YAML file, with environment-variable overrides on top. This is done with a recursive merge.

```python
def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base — nested keys are merged, not replaced."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)  # recurse into nested dicts
        else:
            result[key] = value                            # scalar: just overwrite
    return result
```

**Why not just `{**base, **override}`?**

Shallow merge would **wipe out entire nested sections** instead of updating individual keys:

```python
base     = {"model": {"blip": {"device": "cpu", "batch_size": 4}}}
override = {"model": {"blip": {"device": "cuda"}}}

# Shallow merge — WRONG: batch_size is lost!
{**base, **override}
# → {"model": {"blip": {"device": "cuda"}}}

# Deep merge — CORRECT: only device is overridden
_deep_merge(base, override)
# → {"model": {"blip": {"device": "cuda", "batch_size": 4}}}
```

---

## Concept 3 — The Config Wrapper Class

Rather than passing raw dicts everywhere (which is fragile and typo-prone), we wrap the config in a class with typed properties.

```python
class Config:
    def __init__(self, data: dict):
        self._data = data

    def get(self, *keys: str, default=None):
        """Safe nested access: cfg.get("model", "blip", "device")"""
        node = self._data
        for key in keys:
            if not isinstance(node, dict):
                return default
            node = node.get(key, default)
        return node

    @property
    def captioning_backend(self) -> str:
        return self.get("model", "captioning_backend", default="blip")

    @property
    def model_cfg(self) -> dict:
        # Returns the config block for whatever backend is active
        return self.get("model", self.captioning_backend, default={})
```

**Benefits over raw dicts:**

| Raw Dict | Config Class |
|----------|-------------|
| `data["model"]["blip"]["device"]` — KeyError if missing | `cfg.model_cfg.get("device", "cpu")` — safe with default |
| Typos silently return `None` | Properties are explicit and discoverable |
| No IDE autocomplete | IDE can complete property names |

---

## Concept 4 — Environment Variable Overrides

In containerized environments (Docker, Kubernetes), you inject configuration via **environment variables** rather than editing files. This is the [12-Factor App](https://12factor.net/config) principle.

```python
import os

# In load_config():
overrides = {}

# PIPELINE_CAPTIONING_BACKEND=blip2  →  overrides model.captioning_backend
if backend := os.environ.get("PIPELINE_CAPTIONING_BACKEND"):
    overrides.setdefault("model", {})["captioning_backend"] = backend

# PIPELINE_MODEL_DEVICE=cuda  →  overrides device for all model backends
if device := os.environ.get("PIPELINE_MODEL_DEVICE"):
    for key in ("blip", "blip2", "git"):
        overrides.setdefault("model", {}).setdefault(key, {})["device"] = device
```

The `:=` **walrus operator** (Python 3.8+) assigns and tests in one expression — perfect for "only override if env var is set."

**Usage examples:**

```bash
# Run with BLIP-2 on GPU
export PIPELINE_CAPTIONING_BACKEND=blip2
export PIPELINE_MODEL_DEVICE=cuda
python scripts/caption_images.py ...

# Run in mock mode for CI
export PIPELINE_CAPTIONING_BACKEND=mock
pytest tests/
```

---

## Concept 5 — `@lru_cache` for Singleton Config

Config should be loaded **once** per process, not re-read on every function call. Python's `functools.lru_cache` makes this effortless.

```python
from functools import lru_cache

@lru_cache(maxsize=1)
def load_config(config_path: str | None = None) -> Config:
    """Load once, cache forever. Subsequent calls return the same object."""
    ...
```

`lru_cache(maxsize=1)` keeps only the most recent result. Since we always call `load_config()` with the same argument, it behaves as a singleton.

> **Testing caveat:** `lru_cache` can cause test isolation issues. To reset: `load_config.cache_clear()`.

---

## Concept 6 — Config Structure Design Patterns

Our YAML is organized by **domain** (what the config controls), not by stage (where it's used). This means all model-related settings are together, all Spark settings are together, etc.

```
model:          ← all ML model settings
  captioning_backend
  blip: { ... }
  blip2: { ... }
  tagging_backend
  clip: { ... }

preprocessing:  ← all image processing settings
  opencv: { ... }
  target_size
  output_format

spark:          ← all Spark cluster settings

paths:          ← all filesystem paths

airflow:        ← all DAG scheduling settings
```

This structure makes it easy to:
- Find all settings for a domain in one place
- Override an entire domain at once
- Add a new backend without restructuring the file

---

## Summary

| Concept | Key Idea |
|---------|----------|
| YAML config | Single source of truth, human-readable |
| Deep merge | Layer env overrides without losing nested keys |
| Config class | Type-safe access, IDE-friendly properties |
| Env var overrides | Per-environment config without file edits |
| `lru_cache` | Load once, reuse everywhere (singleton pattern) |
| Domain-organized structure | Related settings grouped, easy to navigate |

---

## Exercises

**1. Basic** — Open `run_config.yaml` and change `model.blip.batch_size` from `4` to `8`. Run `python -c "from config.config_loader import load_config; cfg = load_config(); print(cfg.model_cfg)"` and verify the change.

**2. Intermediate** — Add a new field `preprocessing.grayscale: false` to the YAML. Then access it in Python: `cfg.get("preprocessing", "grayscale", default=False)`. What happens if you remove the field from YAML?

**3. Advanced** — Write a function `validate_config(cfg: Config) -> list[str]` that returns a list of validation errors (e.g., "model.blip.batch_size must be > 0", "paths.raw_images must exist"). What errors should be fatal vs warnings?

---

*Next → [Module 02: Image Preprocessing with OpenCV](./module_02_opencv.md)*
