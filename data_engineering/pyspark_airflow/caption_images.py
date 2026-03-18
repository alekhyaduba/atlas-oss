"""
caption_images.py
─────────────────────────────────────────────────────────────────
Stage 2 – Image Captioning Inference
  • Reads processed images from the preprocessing manifest (Parquet)
  • Runs the caption model selected in run_config.yaml
  • Writes captions back into the manifest and saves updated Parquet

Supported backends (set via run_config.yaml → model.captioning_backend):
    blip   – Salesforce BLIP  (lightweight, CPU-friendly)
    blip2  – Salesforce BLIP-2 (larger, GPU recommended)
    git    – Microsoft GIT
    mock   – Deterministic captions for CI / unit tests

Usage (standalone):
    python caption_images.py \
        --config   /path/to/run_config.yaml \
        --manifest /data/processed/manifest.parquet \
        --output   /data/processed/captioned_manifest.parquet \
        --run-id   abc123
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.config_loader import Config, load_config

logger = logging.getLogger(__name__)


# ── Abstract Caption Backend ─────────────────────────────────

class CaptionBackend(ABC):
    @abstractmethod
    def load(self) -> None: ...

    @abstractmethod
    def caption_batch(self, image_paths: List[str]) -> List[str]: ...

    def caption_single(self, image_path: str) -> str:
        return self.caption_batch([image_path])[0]


# ── Mock Backend (testing / CI) ──────────────────────────────

class MockCaptionBackend(CaptionBackend):
    TEMPLATES = [
        "a photograph showing {subject}",
        "an image depicting {subject}",
        "a scene featuring {subject}",
    ]
    SUBJECTS = ["a landscape", "an urban setting", "various objects", "natural scenery"]

    def load(self) -> None:
        logger.info("[Mock] Caption backend loaded.")

    def caption_batch(self, image_paths: List[str]) -> List[str]:
        import hashlib
        results = []
        for p in image_paths:
            idx = int(hashlib.md5(p.encode()).hexdigest(), 16)
            tmpl    = self.TEMPLATES[idx % len(self.TEMPLATES)]
            subject = self.SUBJECTS[idx % len(self.SUBJECTS)]
            results.append(tmpl.format(subject=subject))
        return results


# ── BLIP Backend ─────────────────────────────────────────────

class BlipCaptionBackend(CaptionBackend):
    def __init__(self, cfg: dict):
        self.model_name    = cfg.get("model_name", "Salesforce/blip-image-captioning-base")
        self.device        = cfg.get("device", "cpu")
        self.max_new_tokens= cfg.get("max_new_tokens", 50)
        self.batch_size    = cfg.get("batch_size", 4)
        self._model        = None
        self._processor    = None

    def load(self) -> None:
        from transformers import BlipForConditionalGeneration, BlipProcessor
        import torch
        logger.info("Loading BLIP model: %s on %s", self.model_name, self.device)
        cache = str(Path.home() / ".cache" / "huggingface")
        self._processor = BlipProcessor.from_pretrained(self.model_name, cache_dir=cache)
        self._model     = BlipForConditionalGeneration.from_pretrained(
            self.model_name, cache_dir=cache
        ).to(self.device)
        self._model.eval()
        logger.info("BLIP model loaded.")

    def caption_batch(self, image_paths: List[str]) -> List[str]:
        import torch
        results: List[str] = []
        for i in range(0, len(image_paths), self.batch_size):
            batch_paths = image_paths[i : i + self.batch_size]
            images = [Image.open(p).convert("RGB") for p in batch_paths]
            inputs = self._processor(images=images, return_tensors="pt").to(self.device)
            with torch.no_grad():
                out = self._model.generate(**inputs, max_new_tokens=self.max_new_tokens)
            captions = self._processor.batch_decode(out, skip_special_tokens=True)
            results.extend(captions)
        return results


# ── BLIP-2 Backend ───────────────────────────────────────────

class Blip2CaptionBackend(CaptionBackend):
    def __init__(self, cfg: dict):
        self.model_name    = cfg.get("model_name", "Salesforce/blip2-opt-2.7b")
        self.device        = cfg.get("device", "cuda")
        self.max_new_tokens= cfg.get("max_new_tokens", 100)
        self.batch_size    = cfg.get("batch_size", 2)
        self._model        = None
        self._processor    = None

    def load(self) -> None:
        from transformers import Blip2ForConditionalGeneration, Blip2Processor
        import torch
        logger.info("Loading BLIP-2 model: %s on %s", self.model_name, self.device)
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self._processor = Blip2Processor.from_pretrained(self.model_name)
        self._model     = Blip2ForConditionalGeneration.from_pretrained(
            self.model_name, torch_dtype=dtype
        ).to(self.device)
        self._model.eval()

    def caption_batch(self, image_paths: List[str]) -> List[str]:
        import torch
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        results: List[str] = []
        for i in range(0, len(image_paths), self.batch_size):
            batch_paths = image_paths[i : i + self.batch_size]
            images  = [Image.open(p).convert("RGB") for p in batch_paths]
            inputs  = self._processor(images=images, return_tensors="pt").to(self.device, dtype)
            with torch.no_grad():
                out = self._model.generate(**inputs, max_new_tokens=self.max_new_tokens)
            captions = self._processor.batch_decode(out, skip_special_tokens=True)
            results.extend([c.strip() for c in captions])
        return results


# ── GIT Backend ──────────────────────────────────────────────

class GitCaptionBackend(CaptionBackend):
    def __init__(self, cfg: dict):
        self.model_name    = cfg.get("model_name", "microsoft/git-base-coco")
        self.device        = cfg.get("device", "cpu")
        self.max_new_tokens= cfg.get("max_new_tokens", 50)
        self.batch_size    = cfg.get("batch_size", 4)
        self._model        = None
        self._processor    = None

    def load(self) -> None:
        from transformers import AutoModelForCausalLM, AutoProcessor
        import torch
        logger.info("Loading GIT model: %s on %s", self.model_name, self.device)
        self._processor = AutoProcessor.from_pretrained(self.model_name)
        self._model     = AutoModelForCausalLM.from_pretrained(self.model_name).to(self.device)
        self._model.eval()

    def caption_batch(self, image_paths: List[str]) -> List[str]:
        import torch
        results: List[str] = []
        for i in range(0, len(image_paths), self.batch_size):
            batch_paths = image_paths[i : i + self.batch_size]
            images  = [Image.open(p).convert("RGB") for p in batch_paths]
            inputs  = self._processor(images=images, return_tensors="pt").to(self.device)
            with torch.no_grad():
                out = self._model.generate(pixel_values=inputs.pixel_values,
                                           max_new_tokens=self.max_new_tokens)
            captions = self._processor.batch_decode(out, skip_special_tokens=True)
            results.extend(captions)
        return results


# ── Factory ──────────────────────────────────────────────────

def build_caption_backend(config: Config) -> CaptionBackend:
    backend = config.captioning_backend
    model_cfg = config.model_cfg

    if backend == "blip":
        return BlipCaptionBackend(model_cfg)
    if backend == "blip2":
        return Blip2CaptionBackend(model_cfg)
    if backend == "git":
        return GitCaptionBackend(model_cfg)
    if backend == "mock":
        return MockCaptionBackend()

    raise ValueError(f"Unknown captioning backend: {backend!r}")


# ── Core Stage ───────────────────────────────────────────────

def run_captioning(
    manifest_path: str,
    output_path: str,
    config_path: str | None,
    run_id: str,
) -> str:
    cfg     = load_config(config_path)
    backend = build_caption_backend(cfg)
    backend.load()

    df = pd.read_parquet(manifest_path)
    ok_mask = df["status"] == "ok"
    image_paths = df.loc[ok_mask, "processed_path"].tolist()

    logger.info("Captioning %d images with backend=%s", len(image_paths), cfg.captioning_backend)

    t0 = time.time()
    captions = backend.caption_batch(image_paths)
    elapsed  = time.time() - t0

    df.loc[ok_mask, "caption"]          = captions
    df.loc[~ok_mask, "caption"]         = None
    df["model_backend"]                 = cfg.captioning_backend
    df["model_name"]                    = cfg.model_cfg.get("model_name", "mock")
    df["caption_duration_ms"]           = int(elapsed * 1000 / max(len(image_paths), 1))

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    logger.info("Captioned manifest written: %s", output_path)
    return output_path


# ── CLI ──────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Image captioning stage")
    p.add_argument("--config",   default=None)
    p.add_argument("--manifest", required=True)
    p.add_argument("--output",   required=True)
    p.add_argument("--run-id",   default=str(uuid.uuid4()))
    return p.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    run_captioning(args.manifest, args.output, args.config, args.run_id)
