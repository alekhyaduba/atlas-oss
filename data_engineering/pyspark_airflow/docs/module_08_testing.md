# Module 08 — Testing Data Pipelines
## `test_pipeline.py`

**Estimated time:** 45–60 minutes
**Difficulty:** ⭐⭐⭐☆☆ Intermediate

---

## Why This Module Matters

Data pipelines fail in subtle ways — wrong column types, missing rows, silent errors hidden by try/except blocks. Unlike web servers that crash loudly, data bugs produce incorrect outputs that can propagate through downstream systems undetected. **Testing data pipelines requires different thinking than testing application code.**

---

## Concept 1 — pytest Fundamentals

`pytest` is Python's testing framework. Tests are functions that start with `test_`, organized in classes that start with `Test`.

```python
# Simple test
def test_addition():
    assert 1 + 1 == 2

# Test class (groups related tests)
class TestPreprocessing:
    def test_basic_run(self):
        result = preprocess_images("input/", "output/")
        assert result is not None

    def test_manifest_schema(self):
        df = pd.read_parquet("output/manifest.parquet")
        assert "image_id" in df.columns
```

**Run tests:**

```bash
pytest tests/test_pipeline.py -v        # Verbose output
pytest tests/test_pipeline.py -v -k "TestConfig"  # Only TestConfig class
pytest tests/ --tb=short                # Short traceback on failure
pytest tests/ --cov=scripts             # Coverage report
```

**Reading pytest output:**

```
PASSED  tests/test_pipeline.py::TestConfig::test_load          ← success
FAILED  tests/test_pipeline.py::TestPreprocessing::test_all_ok ← failure
ERROR   tests/test_pipeline.py::TestTagging::test_tags_present  ← exception before assertion
```

---

## Concept 2 — Fixtures: Reusable Setup

Fixtures are functions that provide test data or setup. pytest automatically injects them by matching parameter names.

```python
import pytest
import tempfile
import numpy as np
from PIL import Image
from pathlib import Path

@pytest.fixture
def tmp_dirs():
    """Create a temporary directory tree for each test."""
    with tempfile.TemporaryDirectory() as d:
        raw  = Path(d) / "raw"
        proc = Path(d) / "processed"
        raw.mkdir()
        proc.mkdir()
        yield {"raw": str(raw), "proc": str(proc)}
    # ← Directory and all contents deleted here (cleanup is automatic)

@pytest.fixture
def sample_images(tmp_dirs):
    """Create 3 synthetic test images."""
    paths = []
    for i in range(3):
        arr = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        img = Image.fromarray(arr, "RGB")
        path = Path(tmp_dirs["raw"]) / f"test_{i}.jpg"
        img.save(str(path))
        paths.append(str(path))
    return paths

# Fixtures are injected by parameter name
class TestPreprocessing:
    def test_basic_run(self, tmp_dirs, sample_images):  # ← fixtures injected here
        result = run_preprocessing(tmp_dirs["raw"], tmp_dirs["proc"], ...)
        assert Path(result).exists()
```

**Fixture scope:** Control how often a fixture is created:

```python
@pytest.fixture(scope="function")  # Default: new instance per test (safe, isolated)
@pytest.fixture(scope="class")     # One instance per test class (faster)
@pytest.fixture(scope="module")    # One instance per file (fastest, less isolated)
@pytest.fixture(scope="session")   # One instance per entire test run
```

Use `scope="session"` for expensive setup like loading ML models. Use `scope="function"` for anything with mutable state (files, databases).

---

## Concept 3 — Testing Schemas, Not Just Values

A common data engineering bug: a column exists but has the wrong type, or is all-null. Schema testing catches these bugs.

```python
class TestPreprocessing:
    def test_manifest_schema(self, tmp_dirs, sample_images, run_id):
        manifest = run_preprocessing(...)
        df = pd.read_parquet(manifest)

        # Test that required columns exist
        required = ["image_id", "source_path", "processed_path", "status", "image_hash"]
        for col in required:
            assert col in df.columns, f"Missing column: {col}"

        # Test column types
        assert df["image_id"].dtype == object          # strings are "object" in Pandas
        assert df["file_size_bytes"].dtype == np.int64 # or int64

        # Test non-null constraints
        assert df["image_id"].notna().all(), "image_id should never be null"

        # Test uniqueness
        assert df["image_id"].is_unique, "image_id should be unique per image"
```

**Common schema assertions:**

```python
# Column exists
assert "my_col" in df.columns

# Correct dtype
assert df["count"].dtype == np.int64
assert df["name"].dtype == object   # string
assert df["score"].dtype == np.float64

# No nulls in required column
assert df["id"].notna().all()

# All values in valid set
assert df["status"].isin(["ok", "error"]).all()

# Unique
assert df["image_id"].is_unique

# Value range
assert (df["score"] >= 0).all() and (df["score"] <= 1).all()
```

---

## Concept 4 — The Mock Backend Enables Test Isolation

Test isolation means a test should not depend on external services (internet, GPU, real model weights). Our `MockCaptionBackend` and `MockTaggingBackend` enable fast, hermetic tests.

```python
import os

# Set mock mode at module level — affects all tests in this file
os.environ["PIPELINE_CAPTIONING_BACKEND"] = "mock"
os.environ["PIPELINE_MODEL_DEVICE"]       = "cpu"

class TestCaptioning:
    def test_mock_backend_captions(self, tmp_dirs, sample_images, run_id):
        # This test runs in milliseconds with no internet/GPU
        manifest = run_preprocessing(...)
        out = run_captioning(manifest, out_path, ...)

        df = pd.read_parquet(out)
        ok = df[df["status"] == "ok"]

        # Check captions exist and are non-empty
        assert ok["caption"].notna().all()
        assert (ok["caption"].str.len() > 0).all()

    def test_model_name_recorded(self, ...):
        ...
        df = pd.read_parquet(out)
        assert (df["model_backend"] == "mock").all()
```

**`lru_cache` trap:** `load_config()` uses `lru_cache`. If one test sets an env var and loads config, the next test gets the cached version. Fix:

```python
@pytest.fixture(autouse=True)
def reset_config_cache():
    """Clear config cache before each test to prevent bleed-through."""
    yield
    from config.config_loader import load_config
    load_config.cache_clear()
```

---

## Concept 5 — Dependent Fixtures (Fixture Chaining)

Fixtures can depend on other fixtures, building up complex setups in a readable way.

```python
@pytest.fixture
def tmp_dirs():
    with tempfile.TemporaryDirectory() as d:
        ...
        yield dirs

@pytest.fixture
def sample_images(tmp_dirs):          # depends on tmp_dirs
    # Creates images in tmp_dirs["raw"]
    ...

@pytest.fixture
def preprocessed_manifest(tmp_dirs, sample_images, run_id):  # chains both
    from scripts.preprocess_images import run_preprocessing
    return run_preprocessing(tmp_dirs["raw"], tmp_dirs["proc"], ...)

@pytest.fixture
def captioned_manifest(tmp_dirs, preprocessed_manifest, run_id):
    from scripts.caption_images import run_captioning
    out = str(Path(tmp_dirs["proc"]) / "captioned.parquet")
    run_captioning(preprocessed_manifest, out, ...)
    return out

class TestTagging:
    def test_tags_present(self, tmp_dirs, captioned_manifest, run_id):
        # We get a fully preprocessed + captioned manifest "for free"
        tagged = str(Path(tmp_dirs["proc"]) / "tagged.parquet")
        run_tagging(captioned_manifest, tagged, ...)
        df = pd.read_parquet(tagged)
        assert "tags" in df.columns
```

This pattern keeps individual tests short and focused — the setup complexity is in the fixtures.

---

## Concept 6 — Testing Data Contracts Between Stages

Each stage has an implicit **data contract**: it expects certain columns as input and guarantees certain columns as output. Test these contracts explicitly.

```python
class TestFullPipeline:
    def test_end_to_end_mock(self, tmp_dirs, sample_images, run_id):
        """Integration test: verify the contract across all stages."""
        # Run all stages
        m1 = run_preprocessing(...)
        m2 = str(Path(tmp_dirs["proc"]) / "captioned.parquet")
        run_captioning(m1, m2, ...)
        m3 = str(Path(tmp_dirs["proc"]) / "tagged.parquet")
        run_tagging(m2, m3, ...)

        df = pd.read_parquet(m3)

        # Contract: same number of rows as input images
        assert len(df) == len(sample_images)

        # Contract: columns from ALL stages are present
        assert "caption" in df.columns       # from stage 2
        assert "tags" in df.columns          # from stage 3
        assert "image_hash" in df.columns    # from stage 1

        # Contract: hashes are unique (no accidental duplicates)
        assert df["image_hash"].nunique() == len(df)

        # Contract: JSON columns are valid JSON
        for tags_json in df["tags"].dropna():
            parsed = json.loads(tags_json)   # Would raise if invalid JSON
            assert isinstance(parsed, list)
```

---

## Concept 7 — Property-Based Testing Mindset

Instead of testing specific values, test **properties that should always hold** regardless of inputs. This is more robust than example-based tests.

```python
class TestTagging:
    def test_tags_are_valid_json_arrays(self, tmp_dirs, sample_images, run_id):
        """Property: every 'tags' value must be a JSON array of strings."""
        # ... run pipeline ...
        df = pd.read_parquet(tagged)
        for tags_json in df[df["status"] == "ok"]["tags"]:
            tags = json.loads(tags_json)
            assert isinstance(tags, list)
            assert all(isinstance(t, str) for t in tags)   # all elements are strings

    def test_tag_scores_sum_to_approx_one(self, ...):
        """Property: softmax scores should sum to approximately 1."""
        for scores_json in df["tag_scores"].dropna():
            scores = json.loads(scores_json)
            if scores:
                total = sum(scores.values())
                assert 0.9 < total < 1.1, f"Scores sum to {total}, expected ~1.0"

    def test_hashes_are_deterministic(self, tmp_dirs, sample_images, run_id):
        """Property: processing the same image twice gives the same hash."""
        m1 = run_preprocessing(tmp_dirs["raw"], ...)
        m2 = run_preprocessing(tmp_dirs["raw"], ...)  # run again
        df1 = pd.read_parquet(m1)
        df2 = pd.read_parquet(m2)
        # Sort both by source_path to align rows
        hashes1 = set(df1["image_hash"])
        hashes2 = set(df2["image_hash"])
        assert hashes1 == hashes2
```

---

## Summary

| Concept | Key Idea |
|---------|----------|
| pytest structure | `test_*` functions, `Test*` classes, run with `pytest -v` |
| Fixtures | `@pytest.fixture` provides reusable setup; auto-cleanup with `yield` |
| Schema testing | Check column existence, types, nullability, uniqueness |
| Mock backends | Enables fast, hermetic tests without GPU/internet |
| `lru_cache` trap | Clear config cache between tests to prevent bleed-through |
| Fixture chaining | Fixtures can depend on fixtures — compose complex setups cleanly |
| Data contracts | Test that stage outputs fulfill downstream stage expectations |
| Property-based mindset | Test invariants that always hold, not just specific example values |

---

## Exercises

**1. Basic** — Add a test `test_no_duplicate_image_ids` to `TestPreprocessing` that asserts all `image_id` values are unique in the manifest. Run it and verify it passes.

**2. Intermediate** — Create a new fixture `@pytest.fixture(scope="module")` called `mock_model_dir` that creates a temporary directory with a fake `model_config.json` file. Use it in a test that checks the config loader can find it. Verify it's only created once per test module.

**3. Advanced** — Write a parametrized test using `@pytest.mark.parametrize` that runs the captioning stage with different numbers of images (1, 5, 10, 50) and asserts the output always has the same number of rows as input. Check performance: does caption duration scale linearly with image count?

```python
@pytest.mark.parametrize("n_images", [1, 5, 10])
def test_captioning_scales_linearly(self, tmp_dirs, n_images, run_id):
    # Create exactly n_images synthetic images
    # Run captioning
    # Assert len(df) == n_images
    ...
```

---

*Next → [Module 09: End-to-End Capstone](./module_09_capstone.md)*
