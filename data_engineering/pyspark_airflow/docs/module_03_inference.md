# Module 03 — ML Inference Backends
## `caption_images.py`

**Estimated time:** 45–60 minutes
**Difficulty:** ⭐⭐⭐☆☆ Intermediate

---

## Why This Module Matters

In production data pipelines, you rarely commit to a single ML model forever. Models improve, requirements change, and different environments (GPU vs CPU, dev vs prod) need different backends. This module teaches the **Abstract Backend Pattern** — a software design approach that lets you swap ML models with a one-line config change.

---

## Concept 1 — Abstract Base Classes (ABC)

Python's `abc` module lets you define **interfaces**: classes that declare what methods must exist, without implementing them. Any class that inherits from an ABC *must* implement the abstract methods or Python raises a `TypeError`.

```python
from abc import ABC, abstractmethod
from typing import List

class CaptionBackend(ABC):
    """Interface: every caption backend must implement these two methods."""

    @abstractmethod
    def load(self) -> None:
        """Download/initialize the model. Called once before inference."""
        ...

    @abstractmethod
    def caption_batch(self, image_paths: List[str]) -> List[str]:
        """Caption a batch of images. Returns one caption per image."""
        ...

    # Concrete method — works for ALL backends without override
    def caption_single(self, image_path: str) -> str:
        return self.caption_batch([image_path])[0]
```

**Why ABCs instead of just convention?**

```python
# Without ABC — silently broken at runtime
class BadBackend:
    def load(self): pass
    # forgot caption_batch!

b = BadBackend()
b.caption_batch(["img.jpg"])  # AttributeError at runtime — discovered too late

# With ABC — caught at instantiation
class GoodBackend(CaptionBackend):
    def load(self): pass
    # forgot caption_batch!

b = GoodBackend()  # ← TypeError HERE: "Can't instantiate abstract class..."
                   # Fail fast, fail loudly
```

---

## Concept 2 — The Mock Backend (Test-First Engineering)

**Always build the mock backend first.** It lets you develop and test downstream pipeline stages before the real model is set up — no GPU, no internet, no waiting for model downloads.

```python
class MockCaptionBackend(CaptionBackend):
    TEMPLATES = [
        "a photograph showing {subject}",
        "an image depicting {subject}",
    ]
    SUBJECTS = ["a landscape", "an urban setting", "various objects"]

    def load(self) -> None:
        pass  # Nothing to load

    def caption_batch(self, image_paths: List[str]) -> List[str]:
        import hashlib
        results = []
        for path in image_paths:
            # Use MD5 of path for determinism — same path → same caption
            idx     = int(hashlib.md5(path.encode()).hexdigest(), 16)
            tmpl    = self.TEMPLATES[idx % len(self.TEMPLATES)]
            subject = self.SUBJECTS[idx % len(self.SUBJECTS)]
            results.append(tmpl.format(subject=subject))
        return results
```

**Key property: determinism.** The mock always returns the same caption for the same path. This makes tests **reproducible** — you can assert exact values.

```python
# In tests:
backend = MockCaptionBackend()
backend.load()
captions = backend.caption_batch(["test_0.jpg", "test_1.jpg"])
assert captions[0] == "a photograph showing a landscape"  # deterministic!
```

---

## Concept 3 — Loading HuggingFace Models

HuggingFace's `transformers` library provides a consistent API across hundreds of models. The key objects are a **Processor** (handles input formatting) and a **Model** (runs inference).

```python
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch

class BlipCaptionBackend(CaptionBackend):
    def __init__(self, cfg: dict):
        self.model_name = cfg.get("model_name", "Salesforce/blip-image-captioning-base")
        self.device     = cfg.get("device", "cpu")
        self._model     = None      # Loaded lazily in load()
        self._processor = None

    def load(self) -> None:
        # Processor: tokenizes/formats inputs for the model
        self._processor = BlipProcessor.from_pretrained(self.model_name)

        # Model: the neural network itself
        self._model = BlipForConditionalGeneration.from_pretrained(self.model_name)
        self._model = self._model.to(self.device)  # Move to GPU/CPU
        self._model.eval()  # ← Switch to inference mode (disables dropout, etc.)
```

**`model.eval()` — why it matters:**

```python
# Training mode (default):
# - Dropout randomly zeros activations (regularization)
# - BatchNorm uses batch statistics
# → Results are NON-DETERMINISTIC and slightly wrong for inference

# Eval mode:
# - Dropout is disabled (all neurons active)
# - BatchNorm uses running statistics
# → Results are DETERMINISTIC and correct for inference

model.eval()  # Always call before inference!
```

---

## Concept 4 — Batched Inference

Processing images one-by-one is 10–50× slower than processing them in batches. GPUs are designed for parallel computation — a batch of 4 images takes roughly the same time as 1 image.

```python
def caption_batch(self, image_paths: List[str]) -> List[str]:
    import torch

    results = []

    # Process in mini-batches of `batch_size`
    for i in range(0, len(image_paths), self.batch_size):
        batch_paths = image_paths[i : i + self.batch_size]

        # Open all images in the batch as PIL Images
        images = [Image.open(p).convert("RGB") for p in batch_paths]

        # Processor converts PIL Images → tensors the model can consume
        inputs = self._processor(images=images, return_tensors="pt")
        inputs = inputs.to(self.device)  # Move tensors to same device as model

        # torch.no_grad() — disable gradient tracking (saves memory, speeds up)
        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens
            )

        # Decode token IDs back to text strings
        captions = self._processor.batch_decode(output_ids, skip_special_tokens=True)
        results.extend(captions)

    return results
```

**`torch.no_grad()` — why it matters:**

```python
# During training, PyTorch tracks all computations to enable backpropagation.
# This "computational graph" consumes memory proportional to forward pass size.

# During inference we don't need backprop → don't need the graph → save memory

with torch.no_grad():          # Context manager: disables grad tracking inside
    output = model(inputs)     # Uses ~2x less memory than without no_grad

# Always use torch.no_grad() during inference!
```

---

## Concept 5 — Device Management (CPU vs GPU)

```python
import torch

# The device string controls where tensors/models live
device = "cuda"   # NVIDIA GPU — fastest
device = "mps"    # Apple Silicon (M1/M2) — fast on Mac
device = "cpu"    # CPU — universal but slowest

# Moving a model to a device:
model = model.to(device)

# Moving tensors to the same device as the model:
inputs = processor(images=images, return_tensors="pt").to(device)

# CRITICAL: model and inputs must be on the SAME device
# model on cuda + inputs on cpu → RuntimeError

# Check what's available:
print(torch.cuda.is_available())  # True if NVIDIA GPU present
print(torch.backends.mps.is_available())  # True on Apple Silicon
```

**Runtime device selection:**

```yaml
# In run_config.yaml — override with PIPELINE_MODEL_DEVICE=cuda
model:
  blip:
    device: "cpu"   # safe default for development
```

---

## Concept 6 — The Factory Function

The **Factory Pattern** is a function that creates the right object based on a configuration value. It centralizes the "which class to use?" decision in one place.

```python
def build_caption_backend(config: Config) -> CaptionBackend:
    backend = config.captioning_backend   # "blip" | "blip2" | "git" | "mock"
    model_cfg = config.model_cfg          # The config block for that backend

    if backend == "blip":
        return BlipCaptionBackend(model_cfg)
    if backend == "blip2":
        return Blip2CaptionBackend(model_cfg)
    if backend == "git":
        return GitCaptionBackend(model_cfg)
    if backend == "mock":
        return MockCaptionBackend()

    raise ValueError(f"Unknown captioning backend: {backend!r}")
```

**Benefits:**
- All stage code uses the same interface: `backend.load()`, `backend.caption_batch(...)`
- Adding a new backend only requires: 1) new class, 2) one new `if` in the factory
- Runtime backend selection with zero code changes in pipeline stages

**Usage in the stage:**

```python
def run_captioning(manifest_path, output_path, config_path, run_id):
    cfg     = load_config(config_path)
    backend = build_caption_backend(cfg)   # ← Factory selects the right class
    backend.load()                         # ← Same interface for all backends

    image_paths = pd.read_parquet(manifest_path)["processed_path"].tolist()
    captions    = backend.caption_batch(image_paths)  # ← Same interface
    ...
```

---

## Concept 7 — BLIP-2 and Mixed Precision

BLIP-2 is a larger, more accurate model but requires more careful memory management. **Mixed precision** (float16 instead of float32) halves memory usage with minimal accuracy loss.

```python
import torch

# BLIP-2 with float16 on GPU
dtype = torch.float16 if device == "cuda" else torch.float32

model = Blip2ForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype=dtype   # Load weights directly in float16
)
model = model.to(device)

# During inference, cast inputs to the same dtype
inputs = processor(images=images, return_tensors="pt")
inputs = inputs.to(device, dtype)  # Move device AND cast to float16
```

**float32 vs float16:**

| Property | float32 | float16 |
|----------|---------|---------|
| Memory per value | 4 bytes | 2 bytes |
| Precision | High | Lower (but fine for inference) |
| Speed on GPU | Standard | ~2× faster |
| CPU support | ✅ Full | ⚠️ Limited |

> **Rule of thumb:** float16 on CUDA only. Use float32 on CPU — float16 is not well-optimized for CPU inference.

---

## Summary

| Concept | Key Idea |
|---------|----------|
| Abstract Base Classes | Enforce interface contracts; fail at instantiation not runtime |
| Mock backend | Test-first: develop downstream stages before real model exists |
| HuggingFace Processor + Model | Processor formats inputs; Model runs inference |
| `model.eval()` | Disable dropout/training behavior before inference |
| Batched inference | Process N images at once — 10–50× faster than one-by-one |
| `torch.no_grad()` | Skip gradient tracking during inference — saves memory |
| Factory pattern | Centralize backend selection; stages use one interface |
| Mixed precision | float16 on GPU halves memory with minimal accuracy cost |

---

## Exercises

**1. Basic** — Set `PIPELINE_CAPTIONING_BACKEND=mock` and run `caption_images.py` on a directory of images. Print the captions from the output Parquet with Pandas.

**2. Intermediate** — Add a `caption_single_with_retry(path, retries=3)` method to the `CaptionBackend` ABC that retries on exception with exponential backoff. Implement it using `time.sleep`.

**3. Advanced** — Create a new `EnsembleBackend` that takes two backends, runs both, and returns the longer caption (as a heuristic for "more descriptive"). Make it work through the factory by adding `"ensemble"` as a valid backend name in the config.

---

*Next → [Module 04: Zero-Shot Tagging with CLIP](./module_04_tagging.md)*
