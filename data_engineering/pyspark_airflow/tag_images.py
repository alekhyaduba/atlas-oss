"""
tag_images.py
─────────────────────────────────────────────────────────────────
Stage 3 – Image Tagging
  • Reads the captioned manifest (Parquet)
  • Runs CLIP zero-shot classification to assign semantic tags
  • Also derives structural tags from image metadata (size, aspect ratio)
  • Writes tags as JSON array column back into the manifest

Supported backends (set via run_config.yaml → model.tagging_backend):
    clip  – OpenAI CLIP  (openai/clip-vit-base-patch32)
    mock  – Deterministic tags for CI / unit tests

Usage (standalone):
    python tag_images.py \
        --config   /path/to/run_config.yaml \
        --manifest /data/processed/captioned_manifest.parquet \
        --output   /data/processed/tagged_manifest.parquet \
        --run-id   abc123
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.config_loader import Config, load_config

logger = logging.getLogger(__name__)


# ── Abstract Tagging Backend ─────────────────────────────────

class TaggingBackend(ABC):
    @abstractmethod
    def load(self) -> None: ...

    @abstractmethod
    def tag_batch(
        self, image_paths: List[str], candidate_labels: List[str]
    ) -> List[List[Tuple[str, float]]]:
        """Return list of [(label, score)] lists, one per image."""
        ...


# ── Mock Backend ─────────────────────────────────────────────

class MockTaggingBackend(TaggingBackend):
    FIXED = [("outdoor scene", 0.82), ("landscape", 0.71), ("nature", 0.65)]

    def load(self) -> None:
        logger.info("[Mock] Tagging backend loaded.")

    def tag_batch(self, image_paths, candidate_labels):
        import hashlib
        results = []
        for p in image_paths:
            idx  = int(hashlib.md5(p.encode()).hexdigest(), 16)
            pool = [(candidate_labels[i % len(candidate_labels)], round(0.9 - i * 0.08, 3))
                    for i in range(3)]
            results.append(pool)
        return results


# ── CLIP Backend ─────────────────────────────────────────────

class ClipTaggingBackend(TaggingBackend):
    def __init__(self, cfg: dict):
        self.model_name  = cfg.get("model_name", "openai/clip-vit-base-patch32")
        self.device      = cfg.get("device", "cpu")
        self.top_k       = cfg.get("top_k_tags", 5)
        self.threshold   = cfg.get("confidence_threshold", 0.15)
        self._model      = None
        self._processor  = None

    def load(self) -> None:
        from transformers import CLIPModel, CLIPProcessor
        logger.info("Loading CLIP model: %s on %s", self.model_name, self.device)
        self._processor = CLIPProcessor.from_pretrained(self.model_name)
        self._model     = CLIPModel.from_pretrained(self.model_name).to(self.device)
        self._model.eval()
        logger.info("CLIP model loaded.")

    def tag_batch(
        self, image_paths: List[str], candidate_labels: List[str]
    ) -> List[List[Tuple[str, float]]]:
        import torch
        import torch.nn.functional as F

        results: List[List[Tuple[str, float]]] = []

        # Encode text labels once
        text_inputs = self._processor(
            text=candidate_labels, return_tensors="pt", padding=True
        ).to(self.device)
        with torch.no_grad():
            text_features = self._model.get_text_features(**text_inputs)
            text_features = F.normalize(text_features, dim=-1)

        for path in image_paths:
            try:
                image = Image.open(path).convert("RGB")
            except Exception as e:
                logger.warning("Cannot open %s: %s", path, e)
                results.append([])
                continue

            img_inputs = self._processor(images=image, return_tensors="pt").to(self.device)
            with torch.no_grad():
                img_features = self._model.get_image_features(**img_inputs)
                img_features = F.normalize(img_features, dim=-1)
                sims         = (img_features @ text_features.T).squeeze(0)
                probs        = F.softmax(sims * 100, dim=0).cpu().numpy()

            ranked = sorted(zip(candidate_labels, probs), key=lambda x: -x[1])
            filtered = [
                (label, float(round(score, 4)))
                for label, score in ranked[: self.top_k]
                if score >= self.threshold
            ]
            results.append(filtered)

        return results


# ── Structural Tagging Helpers ───────────────────────────────

def structural_tags(row: pd.Series) -> List[str]:
    """Derive rule-based tags from image dimensions."""
    tags: List[str] = []
    w = row.get("processed_width", 0)
    h = row.get("processed_height", 0)
    if w and h:
        ratio = w / max(h, 1)
        if ratio > 1.5:
            tags.append("wide")
        elif ratio < 0.67:
            tags.append("portrait_orientation")
        else:
            tags.append("square_like")
        if max(w, h) >= 1024:
            tags.append("high_res")
        elif max(w, h) <= 256:
            tags.append("low_res")
    return tags


# ── Factory ──────────────────────────────────────────────────

def build_tagging_backend(config: Config) -> TaggingBackend:
    backend = config.tagging_backend
    if backend == "clip":
        return ClipTaggingBackend(config.clip_cfg)
    if backend == "mock":
        return MockTaggingBackend()
    raise ValueError(f"Unknown tagging backend: {backend!r}")


# ── Core Stage ───────────────────────────────────────────────

def run_tagging(
    manifest_path: str,
    output_path: str,
    config_path: str | None,
    run_id: str,
) -> str:
    cfg     = load_config(config_path)
    backend = build_tagging_backend(cfg)
    backend.load()

    candidate_labels = cfg.clip_cfg.get("candidate_labels", [])

    df      = pd.read_parquet(manifest_path)
    ok_mask = df["status"] == "ok"
    image_paths = df.loc[ok_mask, "processed_path"].tolist()

    logger.info("Tagging %d images, backend=%s", len(image_paths), cfg.tagging_backend)

    t0        = time.time()
    tag_lists = backend.tag_batch(image_paths, candidate_labels)
    elapsed   = time.time() - t0

    # Flatten to JSON-serialisable strings (Parquet supports list<string> but JSON is portable)
    tags_json = [
        json.dumps([t for t, _ in pairs]) for pairs in tag_lists
    ]
    tag_scores_json = [
        json.dumps({t: s for t, s in pairs}) for pairs in tag_lists
    ]

    df.loc[ok_mask, "tags"]        = tags_json
    df.loc[ok_mask, "tag_scores"]  = tag_scores_json
    df.loc[~ok_mask, "tags"]       = json.dumps([])
    df.loc[~ok_mask, "tag_scores"] = json.dumps({})

    # Structural tags
    struct_tags = df.apply(structural_tags, axis=1)
    # Merge semantic + structural
    for idx in df[ok_mask].index:
        existing = json.loads(df.at[idx, "tags"])
        merged   = list(dict.fromkeys(existing + struct_tags.at[idx]))  # dedup, order preserved
        df.at[idx, "tags"] = json.dumps(merged)

    df["tagging_backend"]    = cfg.tagging_backend
    df["tagging_duration_ms"]= int(elapsed * 1000 / max(len(image_paths), 1))

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    logger.info("Tagged manifest written: %s", output_path)
    return output_path


# ── CLI ──────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Image tagging stage")
    p.add_argument("--config",   default=None)
    p.add_argument("--manifest", required=True)
    p.add_argument("--output",   required=True)
    p.add_argument("--run-id",   default=str(uuid.uuid4()))
    return p.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    run_tagging(args.manifest, args.output, args.config, args.run_id)
