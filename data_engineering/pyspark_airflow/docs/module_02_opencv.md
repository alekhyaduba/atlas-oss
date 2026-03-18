# Module 02 — Image Preprocessing with OpenCV
## `preprocess_images.py`

**Estimated time:** 45–60 minutes
**Difficulty:** ⭐⭐☆☆☆ Beginner–Intermediate

---

## Why This Module Matters

Raw images from the real world are messy — they have sensor noise, inconsistent sizes, poor contrast, and varying color spaces. ML models trained on clean data perform poorly on dirty inputs.

This module covers how to build a **reproducible, config-driven OpenCV preprocessing pipeline** that transforms raw images into consistent, model-ready inputs, while tracking every transformation in a Pandas manifest.

---

## Concept 1 — Color Spaces: BGR vs RGB

OpenCV's most famous quirk: it reads images in **BGR** (Blue-Green-Red) order, not RGB. This catches every newcomer off guard.

```python
import cv2
import numpy as np

img = cv2.imread("photo.jpg")     # returns BGR numpy array, shape: (H, W, 3)

# ❌ Wrong: Pass BGR directly to a model expecting RGB
model_input = img  # colors will be shifted!

# ✅ Correct: Convert to RGB before passing to ML models
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# ✅ Correct: Convert back to BGR before cv2.imwrite
cv2.imwrite("output.jpg", cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
```

**Visual proof of the difference:**

```
BGR pixel [255, 0, 0] = blue
RGB pixel [255, 0, 0] = red
```

HuggingFace models (BLIP, CLIP) expect **RGB**. PIL `Image.open()` also returns RGB. Always convert at the boundary.

---

## Concept 2 — Denoising with NLMeans

Real-world images (especially from cameras, scans, or low-light) contain **noise** — random variation in pixel values that confuses ML models.

**Non-Local Means (NLM)** denoising works by finding similar patches in the image and averaging them. It's more effective than simple blurring because it preserves edges.

```python
# Fast NL-Means denoising (colored images)
denoised = cv2.fastNlMeansDenoisingColored(
    img,        # Source image (BGR)
    None,       # Destination (None = create new)
    h=10,       # Luminance filter strength — higher = more denoising, less detail
    hColor=10,  # Color component filter strength (usually same as h)
    templateWindowSize=7,   # Size of patches to compare (must be odd)
    searchWindowSize=21,    # Size of area to search for patches (must be odd)
)
```

**When to use which denoising method:**

| Method | Speed | Quality | Use case |
|--------|-------|---------|----------|
| `GaussianBlur` | Fast | Low | Quick tests |
| `medianBlur` | Fast | Medium | Salt-and-pepper noise |
| `fastNlMeansDenoisingColored` | Slow | High | Production quality |

**Tip:** `h=10` is a good starting point. For heavy noise (scanned documents, night photos), try `h=15–20`. Higher values create a "painted" look.

---

## Concept 3 — Sharpening with Unsharp Mask

After denoising, images can look soft. Sharpening enhances edges and fine details.

The **Unsharp Mask** technique (counterintuitively named — it sharpens, not blurs) works by subtracting a blurred version from the original to isolate edges, then adding that back amplified.

```python
def sharpen(img: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    # Step 1: Create a blurred version (captures the "unsharp" low frequencies)
    blurred = cv2.GaussianBlur(img, (kernel_size, kernel_size), sigmaX=0)

    # Step 2: addWeighted(src1, alpha, src2, beta, gamma)
    # result = img * 1.5 + blurred * (-0.5) + 0
    # = img + (img - blurred) * 0.5   ← adds the "edge signal" back
    sharpened = cv2.addWeighted(img, 1.5, blurred, -0.5, 0)
    return sharpened
```

**Decomposing `addWeighted`:**

```python
# cv2.addWeighted(src1, alpha, src2, beta, gamma)
# output[i] = src1[i] * alpha + src2[i] * beta + gamma
# Pixels are clipped to [0, 255] automatically

# For sharpening: alpha + beta should equal 1.0 (or slightly above)
# alpha=1.5, beta=-0.5 → 1.5 - 0.5 = 1.0 (no brightness change, only contrast)
```

---

## Concept 4 — CLAHE Auto-Contrast

**CLAHE** (Contrast Limited Adaptive Histogram Equalization) improves local contrast without blowing out bright regions or crushing dark ones — a common problem with global histogram equalization.

It works in the **LAB color space** where L = luminance (brightness) and A, B = color. We only equalize the L channel, leaving color unchanged.

```python
def auto_contrast(img_bgr: np.ndarray) -> np.ndarray:
    # Step 1: Convert BGR → LAB
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)

    # Step 2: Split into L, A, B channels
    l, a, b = cv2.split(lab)

    # Step 3: Apply CLAHE to L only
    clahe = cv2.createCLAHE(
        clipLimit=2.0,      # Max contrast amplification — prevents noise from being amplified
        tileGridSize=(8, 8) # Divide image into 8x8 tiles, equalize each independently
    )
    l_eq = clahe.apply(l)

    # Step 4: Recombine and convert back to BGR
    lab_eq = cv2.merge([l_eq, a, b])
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)
```

**Why LAB instead of HSV?**

LAB is **perceptually uniform** — equal changes in L values correspond to equal perceived brightness changes. HSV is better for color-based filtering, but LAB is superior for quality enhancement.

---

## Concept 5 — Resizing with LANCZOS4

ML models expect fixed input sizes. Resizing must be done carefully to avoid artifacts.

```python
def resize(img: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    return cv2.resize(
        img,
        (target_w, target_h),   # Note: (width, height) — opposite of shape which is (H, W)
        interpolation=cv2.INTER_LANCZOS4
    )
```

**Interpolation methods compared:**

| Method | Speed | Quality | Best for |
|--------|-------|---------|----------|
| `INTER_NEAREST` | Fastest | Lowest | Debug/masks |
| `INTER_LINEAR` | Fast | Good | Enlarging |
| `INTER_CUBIC` | Medium | Better | Enlarging |
| `INTER_LANCZOS4` | Slow | Best | Shrinking (our case) |
| `INTER_AREA` | Medium | Good | Shrinking |

> **`shape` vs resize args:** `img.shape` returns `(height, width, channels)`. `cv2.resize` takes `(width, height)`. Easy to mix up!

---

## Concept 6 — Image Hashing for Deduplication

We generate a **SHA-256 hash** of each processed image. This gives us:
1. **Deduplication** — detect when the same image is processed twice
2. **Integrity verification** — confirm a file wasn't corrupted during transfer
3. **Content addressing** — look up an image by what it contains, not its filename

```python
import hashlib

def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        # Read in 64KB chunks — handles large files without loading everything into RAM
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()  # Returns 64-char hex string

# Example output:
# "a665a45920422f9d417e4867efdc4fb8a04a1f3fff1fa07e998e86f7f7a27ae3"
```

**Why chunk-based reading?**

For a 10MB image, `f.read()` loads 10MB into RAM. Fine for one image. For 10,000 images in parallel, that's 100GB. Chunked reading keeps memory usage at a constant 64KB regardless of file size.

---

## Concept 7 — The Pandas Manifest Pattern

Every pipeline stage produces a **Parquet manifest** — a Pandas DataFrame saved to disk that records what was processed and all associated metadata. This is the backbone of traceability.

```python
import pandas as pd
from pathlib import Path

rows = []

for src_path in image_files:
    row = process_single(src_path, ...)   # returns a dict
    rows.append(row)

# Collect all rows into a DataFrame
df = pd.DataFrame(rows)

# Write to Parquet (columnar format, compressed, fast to read with Spark)
df.to_parquet("manifest.parquet", index=False)

# Read it back
df = pd.read_parquet("manifest.parquet")
print(df.columns.tolist())
# ['image_id', 'source_path', 'processed_path', 'status', 'image_hash', ...]
```

**Why Parquet instead of CSV?**

| Feature | CSV | Parquet |
|---------|-----|---------|
| Columnar storage | ❌ Row-based | ✅ Column-based |
| Compression | ❌ None by default | ✅ Snappy/zstd built-in |
| Schema enforcement | ❌ Everything is string | ✅ Types preserved |
| Spark compatibility | ⚠️ Works but slow | ✅ Native format |
| File size | Large | 3–10× smaller |

---

## Concept 8 — The `process_single` Pattern

Each image is processed by a **pure function** that takes a path and returns a metadata dict. This makes the pipeline:
- Easy to test (no side effects on class state)
- Easy to parallelize (no shared mutable state)
- Easy to debug (single function to trace)

```python
def process_single(src_path, dst_dir, preprocessor, run_id, ...) -> dict:
    t0 = time.time()

    img = load_image_bgr(src_path)
    if img is None:
        return {"source_path": src_path, "status": "error", "error": "unreadable"}

    processed = preprocessor.process(img)  # stateless transform
    cv2.imwrite(dst_path, processed)

    return {
        "image_id":   str(uuid.uuid4()),   # unique ID for this image
        "source_path": src_path,
        "processed_path": dst_path,
        "image_hash": sha256_file(dst_path),
        "status": "ok",
        "processing_duration_ms": int((time.time() - t0) * 1000),
        # ... other metadata
    }
```

> **Pattern:** Return a result dict (success or error) rather than raising exceptions. This lets the pipeline continue processing other images even when one fails. Errors are recorded in the manifest for later analysis.

---

## Putting It Together — The Processing Pipeline

```
load_image_bgr()
      │  BGR numpy array
      ▼
  denoise()          ← NL-Means, removes sensor noise
      │
      ▼
  sharpen()          ← Unsharp mask, enhances edges
      │
      ▼
  auto_contrast()    ← CLAHE on LAB L-channel
      │
      ▼
  resize()           ← LANCZOS4 to target_size
      │
      ▼
  convert_colorspace() ← BGR → RGB (for ML models)
      │
      ▼
  cv2.imwrite()      ← Back to BGR for file save
```

---

## Summary

| Concept | Key Idea |
|---------|----------|
| BGR vs RGB | OpenCV is BGR; ML models want RGB — convert at boundaries |
| NL-Means denoising | Patch-based noise removal, preserves edges |
| Unsharp mask | Edge enhancement via blurred subtraction |
| CLAHE | Local adaptive contrast in LAB color space |
| LANCZOS4 resize | Best quality interpolation for downscaling |
| SHA-256 hashing | Chunk-based file fingerprinting for dedup |
| Pandas manifest | Parquet-based traceability for every image |

---

## Exercises

**1. Basic** — Load an image with `cv2.imread`, print its `shape`, then convert to RGB and verify the first pixel changed. Display both with `cv2.imshow` if you have a GUI environment.

**2. Intermediate** — Modify `ImagePreprocessor.process()` to skip sharpening if the image is already sharp. Hint: compute the Laplacian variance (`cv2.Laplacian(img, cv2.CV_64F).var()`) and skip if it's above a threshold (try 100).

**3. Advanced** — Extend `process_single` to detect and reject **blurry images** (Laplacian variance < 50) by setting `status = "rejected_blurry"`. Update the manifest schema and check the results in `run_preprocessing`.

---

*Next → [Module 03: ML Inference Backends](./module_03_inference.md)*
