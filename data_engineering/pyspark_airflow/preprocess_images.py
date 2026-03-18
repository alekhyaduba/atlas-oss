"""
preprocess_images.py
─────────────────────────────────────────────────────────────────
Stage 1 – Image Preprocessing
  • Reads raw images from the input directory
  • Applies OpenCV pipeline: denoise → sharpen → CLAHE → resize
  • Writes processed images + a Pandas DataFrame manifest
  • The manifest is saved as Parquet for downstream Spark stages

Usage (standalone):
    python preprocess_images.py \
        --config /path/to/run_config.yaml \
        --input  /data/raw \
        --output /data/processed \
        --run-id  abc123
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.config_loader import load_config

logger = logging.getLogger(__name__)


# ── Helpers ──────────────────────────────────────────────────

def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def load_image_bgr(path: str) -> Optional[np.ndarray]:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        logger.warning("cv2.imread returned None for %s", path)
    return img


# ── OpenCV Pipeline ──────────────────────────────────────────

class ImagePreprocessor:
    """Stateless image preprocessor driven by run config."""

    def __init__(self, cfg: dict, preprocessing_cfg: dict):
        self.ocv = cfg  # opencv sub-config
        self.pre = preprocessing_cfg

    def denoise(self, img: np.ndarray) -> np.ndarray:
        if not self.ocv.get("denoise", True):
            return img
        h = self.ocv.get("denoise_h", 10)
        return cv2.fastNlMeansDenoisingColored(img, None, h, h, 7, 21)

    def sharpen(self, img: np.ndarray) -> np.ndarray:
        if not self.ocv.get("sharpen", True):
            return img
        k = self.ocv.get("sharpen_kernel_size", 3)
        blur = cv2.GaussianBlur(img, (k, k), 0)
        return cv2.addWeighted(img, 1.5, blur, -0.5, 0)

    def auto_contrast(self, img: np.ndarray) -> np.ndarray:
        if not self.ocv.get("auto_contrast", True):
            return img
        clip  = self.ocv.get("clahe_clip_limit", 2.0)
        grid  = tuple(self.ocv.get("clahe_tile_grid", [8, 8]))
        lab   = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=grid)
        l     = clahe.apply(l)
        return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    def resize(self, img: np.ndarray) -> np.ndarray:
        target = self.pre.get("target_size")
        if not target:
            return img
        w, h = target
        return cv2.resize(img, (w, h), interpolation=cv2.INTER_LANCZOS4)

    def convert_colorspace(self, img: np.ndarray) -> np.ndarray:
        cs = self.ocv.get("convert_colorspace", "RGB")
        if cs == "RGB":
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if cs == "GRAY":
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img  # keep BGR

    def process(self, img_bgr: np.ndarray) -> np.ndarray:
        img = self.denoise(img_bgr)
        img = self.sharpen(img)
        img = self.auto_contrast(img)
        img = self.resize(img)
        img = self.convert_colorspace(img)
        return img


# ── Core Processing ──────────────────────────────────────────

def process_single(
    src_path: str,
    dst_dir: str,
    preprocessor: ImagePreprocessor,
    run_id: str,
    pipeline_version: str,
    output_format: str = "JPEG",
    output_quality: int = 90,
) -> dict:
    """Process one image; return metadata row."""
    t0 = time.time()
    src = Path(src_path)

    img_bgr = load_image_bgr(src_path)
    if img_bgr is None:
        return {"source_path": src_path, "status": "error", "error": "unreadable"}

    orig_h, orig_w = img_bgr.shape[:2]
    processed = preprocessor.process(img_bgr)

    ext = "jpg" if output_format == "JPEG" else output_format.lower()
    dst_name = f"{src.stem}_processed.{ext}"
    dst_path = str(Path(dst_dir) / dst_name)

    encode_params: list[int] = []
    if output_format == "JPEG":
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, output_quality]
    elif output_format == "PNG":
        encode_params = [cv2.IMWRITE_PNG_COMPRESSION, 6]

    # cv2 needs BGR for write; convert back if we returned RGB
    write_img = cv2.cvtColor(processed, cv2.COLOR_RGB2BGR) if len(processed.shape) == 3 else processed
    cv2.imwrite(dst_path, write_img, encode_params)

    elapsed_ms = int((time.time() - t0) * 1000)
    file_size  = os.path.getsize(dst_path)
    img_hash   = sha256_file(dst_path)

    return {
        "image_id":              str(uuid.uuid4()),
        "source_path":           src_path,
        "processed_path":        dst_path,
        "original_width":        orig_w,
        "original_height":       orig_h,
        "processed_width":       processed.shape[1],
        "processed_height":      processed.shape[0],
        "file_size_bytes":       file_size,
        "image_hash":            img_hash,
        "run_id":                run_id,
        "pipeline_version":      pipeline_version,
        "status":                "ok",
        "processing_duration_ms": elapsed_ms,
    }


def run_preprocessing(
    input_dir: str,
    output_dir: str,
    config_path: str | None,
    run_id: str,
    manifest_path: str | None = None,
) -> str:
    cfg = load_config(config_path)
    pre_cfg = cfg.preprocessing_cfg
    ocv_cfg = pre_cfg.get("opencv", {})

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    preprocessor = ImagePreprocessor(ocv_cfg, pre_cfg)
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    image_files = [
        str(p) for p in Path(input_dir).rglob("*") if p.suffix.lower() in exts
    ]

    logger.info("Found %d images in %s", len(image_files), input_dir)
    if not image_files:
        logger.warning("No images found — exiting preprocessing stage.")
        return ""

    rows = []
    fmt  = pre_cfg.get("output_format", "JPEG")
    qual = pre_cfg.get("output_quality", 90)

    for src in image_files:
        row = process_single(
            src, output_dir, preprocessor,
            run_id, cfg.pipeline_version, fmt, qual
        )
        rows.append(row)
        logger.debug("Processed %s → %s", src, row.get("processed_path"))

    df = pd.DataFrame(rows)
    out_manifest = manifest_path or str(Path(output_dir) / "manifest.parquet")
    df.to_parquet(out_manifest, index=False)
    logger.info("Manifest written: %s  (%d rows)", out_manifest, len(df))
    return out_manifest


# ── CLI ──────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Image preprocessing stage")
    p.add_argument("--config",   default=None)
    p.add_argument("--input",    required=True)
    p.add_argument("--output",   required=True)
    p.add_argument("--run-id",   default=str(uuid.uuid4()))
    p.add_argument("--manifest", default=None)
    return p.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    run_preprocessing(args.input, args.output, args.config, args.run_id, args.manifest)
