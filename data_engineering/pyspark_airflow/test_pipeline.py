"""
test_pipeline.py
─────────────────────────────────────────────────────────────────
Unit tests for the image processing pipeline.
Run with:  pytest tests/test_pipeline.py -v
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

# Make project importable
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ["PIPELINE_CAPTIONING_BACKEND"] = "mock"
os.environ["PIPELINE_MODEL_DEVICE"]       = "cpu"

CONFIG_PATH = str(ROOT / "config" / "run_config.yaml")


# ── Fixtures ─────────────────────────────────────────────────

@pytest.fixture
def tmp_dirs():
    with tempfile.TemporaryDirectory() as d:
        raw  = Path(d) / "raw"
        proc = Path(d) / "processed"
        ver  = Path(d) / "versioned"
        raw.mkdir(); proc.mkdir(); ver.mkdir()
        yield {"raw": str(raw), "proc": str(proc), "ver": str(ver), "base": d}


@pytest.fixture
def sample_images(tmp_dirs):
    """Create 3 tiny synthetic images in the raw directory."""
    paths = []
    for i in range(3):
        arr = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        img = Image.fromarray(arr, "RGB")
        p   = Path(tmp_dirs["raw"]) / f"test_{i}.jpg"
        img.save(str(p))
        paths.append(str(p))
    return paths


@pytest.fixture
def run_id():
    return f"test_{uuid.uuid4().hex[:8]}"


# ── Config tests ─────────────────────────────────────────────

class TestConfig:
    def test_load(self):
        from config.config_loader import load_config
        cfg = load_config(CONFIG_PATH)
        assert cfg is not None

    def test_captioning_backend_override(self):
        from config.config_loader import load_config
        cfg = load_config(CONFIG_PATH)
        # env override was set in module-level setup
        assert cfg.captioning_backend == "mock"

    def test_pipeline_version(self):
        from config.config_loader import load_config
        cfg = load_config(CONFIG_PATH)
        assert isinstance(cfg.pipeline_version, str)
        assert len(cfg.pipeline_version) > 0

    def test_paths_present(self):
        from config.config_loader import load_config
        cfg = load_config(CONFIG_PATH)
        assert "raw_images" in cfg.paths


# ── Preprocessing tests ──────────────────────────────────────

class TestPreprocessing:
    def test_basic_run(self, tmp_dirs, sample_images, run_id):
        from scripts.preprocess_images import run_preprocessing
        manifest = run_preprocessing(
            tmp_dirs["raw"], tmp_dirs["proc"], CONFIG_PATH, run_id
        )
        assert Path(manifest).exists()

    def test_manifest_schema(self, tmp_dirs, sample_images, run_id):
        from scripts.preprocess_images import run_preprocessing
        manifest = run_preprocessing(
            tmp_dirs["raw"], tmp_dirs["proc"], CONFIG_PATH, run_id
        )
        df = pd.read_parquet(manifest)
        for col in ["image_id", "source_path", "processed_path", "status", "image_hash"]:
            assert col in df.columns, f"Missing column: {col}"

    def test_all_ok(self, tmp_dirs, sample_images, run_id):
        from scripts.preprocess_images import run_preprocessing
        manifest = run_preprocessing(
            tmp_dirs["raw"], tmp_dirs["proc"], CONFIG_PATH, run_id
        )
        df = pd.read_parquet(manifest)
        assert (df["status"] == "ok").all()

    def test_output_files_exist(self, tmp_dirs, sample_images, run_id):
        from scripts.preprocess_images import run_preprocessing
        manifest = run_preprocessing(
            tmp_dirs["raw"], tmp_dirs["proc"], CONFIG_PATH, run_id
        )
        df = pd.read_parquet(manifest)
        for p in df["processed_path"]:
            assert Path(p).exists(), f"Processed image missing: {p}"

    def test_empty_dir_returns_empty(self, tmp_dirs, run_id):
        from scripts.preprocess_images import run_preprocessing
        result = run_preprocessing(
            tmp_dirs["raw"], tmp_dirs["proc"], CONFIG_PATH, run_id
        )
        assert result == ""


# ── Captioning tests ─────────────────────────────────────────

class TestCaptioning:
    def _prep_manifest(self, tmp_dirs, sample_images, run_id):
        from scripts.preprocess_images import run_preprocessing
        return run_preprocessing(
            tmp_dirs["raw"], tmp_dirs["proc"], CONFIG_PATH, run_id
        )

    def test_mock_backend_captions(self, tmp_dirs, sample_images, run_id):
        manifest = self._prep_manifest(tmp_dirs, sample_images, run_id)
        out = str(Path(tmp_dirs["proc"]) / "captioned.parquet")
        from scripts.caption_images import run_captioning
        run_captioning(manifest, out, CONFIG_PATH, run_id)
        df = pd.read_parquet(out)
        ok = df[df["status"] == "ok"]
        assert ok["caption"].notna().all()
        assert (ok["caption"].str.len() > 0).all()

    def test_model_name_recorded(self, tmp_dirs, sample_images, run_id):
        manifest = self._prep_manifest(tmp_dirs, sample_images, run_id)
        out = str(Path(tmp_dirs["proc"]) / "captioned.parquet")
        from scripts.caption_images import run_captioning
        run_captioning(manifest, out, CONFIG_PATH, run_id)
        df = pd.read_parquet(out)
        assert "model_backend" in df.columns
        assert (df["model_backend"] == "mock").all()


# ── Tagging tests ─────────────────────────────────────────────

class TestTagging:
    def _prep_captioned(self, tmp_dirs, sample_images, run_id):
        from scripts.preprocess_images import run_preprocessing
        from scripts.caption_images import run_captioning
        m1 = run_preprocessing(tmp_dirs["raw"], tmp_dirs["proc"], CONFIG_PATH, run_id)
        out = str(Path(tmp_dirs["proc"]) / "captioned.parquet")
        run_captioning(m1, out, CONFIG_PATH, run_id)
        return out

    def test_tags_present(self, tmp_dirs, sample_images, run_id):
        captioned = self._prep_captioned(tmp_dirs, sample_images, run_id)
        tagged = str(Path(tmp_dirs["proc"]) / "tagged.parquet")
        from scripts.tag_images import run_tagging
        run_tagging(captioned, tagged, CONFIG_PATH, run_id)
        df = pd.read_parquet(tagged)
        assert "tags" in df.columns
        ok = df[df["status"] == "ok"]
        for tags_json in ok["tags"]:
            tags = json.loads(tags_json)
            assert isinstance(tags, list)

    def test_structural_tags_added(self, tmp_dirs, sample_images, run_id):
        captioned = self._prep_captioned(tmp_dirs, sample_images, run_id)
        tagged = str(Path(tmp_dirs["proc"]) / "tagged.parquet")
        from scripts.tag_images import run_tagging
        run_tagging(captioned, tagged, CONFIG_PATH, run_id)
        df = pd.read_parquet(tagged)
        ok = df[df["status"] == "ok"]
        structural = {"wide", "portrait_orientation", "square_like", "high_res", "low_res"}
        found = set()
        for tags_json in ok["tags"]:
            found |= set(json.loads(tags_json))
        assert found & structural, "Expected at least one structural tag"


# ── Full pipeline smoke test ──────────────────────────────────

class TestFullPipeline:
    def test_end_to_end_mock(self, tmp_dirs, sample_images, run_id):
        from scripts.preprocess_images import run_preprocessing
        from scripts.caption_images    import run_captioning
        from scripts.tag_images        import run_tagging

        m1 = run_preprocessing(tmp_dirs["raw"], tmp_dirs["proc"], CONFIG_PATH, run_id)
        m2 = str(Path(tmp_dirs["proc"]) / "captioned.parquet")
        run_captioning(m1, m2, CONFIG_PATH, run_id)
        m3 = str(Path(tmp_dirs["proc"]) / "tagged.parquet")
        run_tagging(m2, m3, CONFIG_PATH, run_id)

        df = pd.read_parquet(m3)
        assert len(df) == len(sample_images)
        assert "caption" in df.columns
        assert "tags"    in df.columns
        assert "image_hash" in df.columns

        # All hashes unique
        assert df["image_hash"].nunique() == len(df)
