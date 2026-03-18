# Module 04 — Zero-Shot Tagging with CLIP
## `tag_images.py`

**Estimated time:** 45–60 minutes
**Difficulty:** ⭐⭐⭐☆☆ Intermediate

---

## Why This Module Matters

Traditional image classifiers need thousands of labeled examples per category. CLIP (Contrastive Language–Image Pretraining) enables **zero-shot classification**: you define categories as plain English text at runtime — no retraining needed. This module also introduces **structural tagging** — deriving metadata from image properties using pure logic, not ML.

---

## Concept 1 — What CLIP Is and How It Works

CLIP was trained on hundreds of millions of image-caption pairs from the internet. It learned a shared **embedding space** where images and text with similar meaning have similar vector representations.

```
"a dog running in a park"  →  text encoder  →  [0.2, -0.8, 0.5, ...]  ┐
                                                                         ├── similar vectors!
[image of a dog running]   →  image encoder →  [0.21, -0.79, 0.48, ...]┘

"a submarine"              →  text encoder  →  [-0.6, 0.3, -0.1, ...]   ← distant vector
```

**Zero-shot classification** exploits this:
1. Encode your candidate labels as text → get label vectors
2. Encode the query image → get image vector
3. Compute similarity between image and each label vector
4. The label with highest similarity is the predicted class

No training data needed. Just define your labels.

---

## Concept 2 — Cosine Similarity and L2 Normalization

CLIP uses **cosine similarity** to measure how similar two vectors are. Cosine similarity measures the *angle* between vectors, ignoring magnitude.

```
cosine_similarity(A, B) = (A · B) / (|A| × |B|)
                        = dot product / (magnitude of A × magnitude of B)

Range: [-1, 1]
  1.0  → identical direction (perfect match)
  0.0  → orthogonal (unrelated)
 -1.0  → opposite directions
```

If we **L2-normalize** vectors first (divide by their magnitude to make them unit vectors), then cosine similarity becomes just a dot product — which is much faster.

```python
import torch
import torch.nn.functional as F

# L2 normalize: divide each vector by its L2 norm
# Result: all vectors have length = 1 (unit vectors)
text_features  = F.normalize(text_features,  dim=-1)  # shape: (num_labels, embed_dim)
image_features = F.normalize(image_features, dim=-1)  # shape: (1, embed_dim)

# Cosine similarity = dot product (since both are unit vectors)
# @ is matrix multiplication: (1, embed_dim) @ (embed_dim, num_labels) = (1, num_labels)
similarities = (image_features @ text_features.T).squeeze(0)  # shape: (num_labels,)
```

---

## Concept 3 — The Full CLIP Tagging Pipeline

```python
from transformers import CLIPModel, CLIPProcessor

class ClipTaggingBackend(TaggingBackend):
    def load(self) -> None:
        self._processor = CLIPProcessor.from_pretrained(self.model_name)
        self._model     = CLIPModel.from_pretrained(self.model_name).to(self.device)
        self._model.eval()

    def tag_batch(self, image_paths, candidate_labels):
        import torch
        import torch.nn.functional as F

        # ── Step 1: Encode all text labels ONCE ──────────────────────────
        # This is the key optimization: encode labels once, reuse for every image
        text_inputs = self._processor(
            text=candidate_labels,
            return_tensors="pt",
            padding=True          # Pad shorter labels to match longest
        ).to(self.device)

        with torch.no_grad():
            text_features = self._model.get_text_features(**text_inputs)
            text_features = F.normalize(text_features, dim=-1)
            # shape: (num_labels, 512) — one embedding per label

        results = []
        for path in image_paths:
            # ── Step 2: Encode one image ──────────────────────────────────
            image = Image.open(path).convert("RGB")
            img_inputs = self._processor(images=image, return_tensors="pt").to(self.device)

            with torch.no_grad():
                img_features = self._model.get_image_features(**img_inputs)
                img_features = F.normalize(img_features, dim=-1)
                # shape: (1, 512)

                # ── Step 3: Compute similarity scores ────────────────────
                sims  = (img_features @ text_features.T).squeeze(0)
                # shape: (num_labels,)

                # ── Step 4: Softmax → probabilities that sum to 1 ────────
                probs = F.softmax(sims * 100, dim=0).cpu().numpy()
                # × 100 sharpens the distribution (CLIP's logit scale)

            # ── Step 5: Rank and filter ───────────────────────────────────
            ranked   = sorted(zip(candidate_labels, probs), key=lambda x: -x[1])
            filtered = [(label, float(round(score, 4)))
                        for label, score in ranked[:self.top_k]
                        if score >= self.threshold]
            results.append(filtered)

        return results
```

**Key optimization:** Text labels are encoded **once**, then reused for every image. If you have 15 labels and 1000 images, this saves 14,985 text encoding operations.

---

## Concept 4 — Softmax with Temperature Scaling

```python
probs = F.softmax(sims * 100, dim=0)
```

Why multiply by 100? This is **temperature scaling**.

```
softmax(x)[i] = exp(x[i]) / sum(exp(x[j]) for all j)
```

Without scaling, CLIP's raw similarity scores (usually 0.2–0.35) produce a very flat distribution — all labels get similar probabilities. The ×100 multiplier sharpens the distribution, making the top label clearly dominant.

```python
import torch
import torch.nn.functional as F

sims = torch.tensor([0.30, 0.25, 0.22, 0.20])  # raw CLIP similarities

F.softmax(sims, dim=0)
# → [0.27, 0.26, 0.24, 0.23]  ← almost uniform, not useful

F.softmax(sims * 100, dim=0)
# → [0.73, 0.07, 0.01, 0.01]  ← clear winner!
```

This matches CLIP's own architecture which uses a learned temperature parameter (logit scale).

---

## Concept 5 — Structural Tags: Rule-Based Enrichment

Not every tag needs ML. Many useful labels can be derived from **image metadata** with simple logic. These are cheap, fast, and perfectly accurate.

```python
def structural_tags(row: pd.Series) -> List[str]:
    """Derive rule-based tags from image dimensions — no ML needed."""
    tags = []
    w = row.get("processed_width",  0)
    h = row.get("processed_height", 0)

    if w and h:
        ratio = w / max(h, 1)

        # Aspect ratio tags
        if ratio > 1.5:
            tags.append("wide")             # panoramic / landscape orientation
        elif ratio < 0.67:
            tags.append("portrait_orientation")  # taller than wide
        else:
            tags.append("square_like")      # roughly square

        # Resolution tags
        if max(w, h) >= 1024:
            tags.append("high_res")
        elif max(w, h) <= 256:
            tags.append("low_res")

    return tags
```

**Applied to each row of the manifest:**

```python
struct_tags = df.apply(structural_tags, axis=1)  # Series of lists

# Merge semantic (CLIP) + structural tags, remove duplicates
for idx in df[ok_mask].index:
    existing = json.loads(df.at[idx, "tags"])        # e.g. ["outdoor scene", "landscape"]
    merged   = list(dict.fromkeys(existing + struct_tags.at[idx]))
    # dict.fromkeys preserves order while deduplicating
    df.at[idx, "tags"] = json.dumps(merged)
    # → '["outdoor scene", "landscape", "wide", "high_res"]'
```

---

## Concept 6 — Storing Structured Data in Parquet

Tags are a variable-length list — each image can have a different number. We store them as **JSON strings** in the Parquet column, which allows querying with both Pandas and SQL tools.

```python
import json

# Writing tags to a column
tags_list = [("outdoor scene", 0.82), ("landscape", 0.71)]
df.at[idx, "tags"]       = json.dumps(["outdoor scene", "landscape"])
df.at[idx, "tag_scores"] = json.dumps({"outdoor scene": 0.82, "landscape": 0.71})

# Reading tags back
tags = json.loads(df.at[idx, "tags"])
# → ["outdoor scene", "landscape"]

scores = json.loads(df.at[idx, "tag_scores"])
# → {"outdoor scene": 0.82, "landscape": 0.71}
```

**Querying in Spark later:**

```python
from pyspark.sql.functions import from_json, schema_of_json, col

# Parse JSON array column for filtering
df_spark.filter(col("tags").contains("landscape"))
```

---

## Concept 7 — Candidate Label Design

The quality of your tags depends heavily on your label vocabulary. Some principles:

```yaml
# ✅ Good labels: mutually exclusive, clear, at the right granularity
candidate_labels:
  - "outdoor scene"
  - "indoor scene"
  - "portrait"
  - "landscape"
  - "urban"
  - "nature"

# ❌ Problematic labels:
# Too overlapping:
  - "outdoor"        # overlaps with "outdoor scene"
  - "outside"        # same concept, different words — CLIP may split votes

# Too specific:
  - "golden retriever running in autumn leaves"   # too narrow
  - "JPEG artifact"                               # edge case, rarely useful

# Too abstract:
  - "good photo"     # subjective
  - "interesting"    # not grounded in visual features
```

**Tip:** When adding new labels, check if they overlap with existing ones. Run your pipeline on a test set and inspect the `tag_scores` column — if two labels frequently share high scores, they're redundant.

---

## Summary

| Concept | Key Idea |
|---------|----------|
| CLIP embedding space | Images and text share a vector space — proximity = semantic similarity |
| L2 normalization | Unit vectors make cosine similarity a simple dot product |
| Text encoding optimization | Encode labels once, reuse for all images |
| Temperature scaling (×100) | Sharpens softmax distribution for CLIP's similarity range |
| Structural tags | Rule-based metadata from image dimensions — no ML needed |
| JSON arrays in Parquet | Store variable-length lists as JSON strings for portability |
| Label vocabulary design | Distinct, visual, mid-granularity labels work best |

---

## Exercises

**1. Basic** — Run the tagging pipeline on 5 images. Load the output Parquet with Pandas and sort images by their top tag confidence score. Which images got the highest scores?

**2. Intermediate** — Add a new structural tag: `"blurry"` if the image's `processing_duration_ms` suggests it was a large file (> 500ms). How would you pass the Laplacian variance from the preprocessing stage to enable a proper blur check?

**3. Advanced** — Implement a `HierarchicalTaggingBackend` that runs CLIP twice: first with coarse labels (`["outdoor", "indoor", "abstract"]`), then with fine-grained labels specific to the winning coarse category (e.g., `outdoor → ["forest", "beach", "mountain", "city street"]`). Compare its results with the flat single-pass approach.

---

*Next → [Module 05: PySpark Data Versioning](./module_05_spark.md)*
