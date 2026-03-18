# 🗺️ Data Engineering Mini-Course
## Image Caption & Tag Pipeline — Learning Path

---

> **What you'll build:** A production-grade data pipeline that ingests raw images, cleans them with OpenCV, generates captions using ML models, tags them with CLIP, and versions everything with PySpark — all orchestrated in Apache Airflow and containerized with Docker.

---

## Who This Course Is For

- Developers who know Python and want to break into data engineering
- ML practitioners who want to productionize their pipelines
- Backend engineers learning the data/MLOps stack

**Prerequisites:** Python basics, comfort with the terminal, a little SQL intuition.

---

## The Stack at a Glance

```
┌─────────────────────────────────────────────────────────────────────┐
│  Layer              │  Tool                   │  What it does        │
├─────────────────────┼─────────────────────────┼──────────────────────┤
│  Orchestration      │  Apache Airflow          │  Schedules & runs    │
│  Image Processing   │  OpenCV                 │  Refines raw images  │
│  Data Wrangling     │  Pandas                 │  In-memory manifests │
│  Distributed Compute│  PySpark                │  Scale-out compute   │
│  ML Inference       │  HuggingFace Transformers│  Captions & tags     │
│  Containerization   │  Docker / Compose       │  Run anywhere        │
│  Configuration      │  YAML + Python          │  Runtime flexibility │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Course Modules

| # | Module | Key Skills | Files Covered |
|---|--------|------------|---------------|
| 00 | **This Overview** | Learning path, architecture | — |
| 01 | [Config-Driven Design](./module_01_config.md) | YAML config, env overrides, factory pattern | `run_config.yaml`, `config_loader.py` |
| 02 | [Image Preprocessing with OpenCV](./module_02_opencv.md) | CV2 pipeline, hashing, Pandas manifests | `preprocess_images.py` |
| 03 | [ML Inference Backends](./module_03_inference.md) | Abstract classes, backend factory, batching | `caption_images.py` |
| 04 | [Zero-Shot Tagging with CLIP](./module_04_tagging.md) | CLIP embeddings, cosine similarity, rule-based enrichment | `tag_images.py` |
| 05 | [PySpark Data Versioning](./module_05_spark.md) | SparkSession, schema enforcement, dedup, partitioned writes | `version_data.py` |
| 06 | [Airflow DAG Orchestration](./module_06_airflow.md) | DAGs, operators, XCom, trigger conf | `image_pipeline_dag.py` |
| 07 | [Dockerizing the Pipeline](./module_07_docker.md) | Dockerfile, Compose, volumes, service health | `Dockerfile`, `docker-compose.yml` |
| 08 | [Testing Data Pipelines](./module_08_testing.md) | pytest fixtures, mock backends, schema assertions | `test_pipeline.py` |
| 09 | [End-to-End Capstone](./module_09_capstone.md) | Putting it all together, scaling tips, extensions | All files |

---

## The Pipeline Flow

```
RAW IMAGES
     │
     ▼
① validate_inputs ─── Are there images? How many?
     │
     ▼
② preprocess_images ── OpenCV: denoise → sharpen → CLAHE → resize
     │                  Output: manifest.parquet
     ▼
③ caption_images ───── BLIP / BLIP-2 / GIT / mock inference
     │                  Output: captioned_manifest.parquet
     ▼
④ tag_images ────────── CLIP zero-shot + structural rule tags
     │                  Output: tagged_manifest.parquet
     ▼
⑤ version_data ──────── PySpark: dedup → partition write + lineage
     │                  Output: versioned/dataset/run_id=XXX/
     ▼
⑥ notify_success ────── Log summary (extend to Slack/email)
```

---

## How to Use This Course

1. **Read each module in order** — later modules build on earlier ones.
2. **Run the code snippets** in a local Python environment or Jupyter.
3. **Experiment with the config** — change backends, tweak preprocessing, observe results.
4. **Complete the exercises** at the end of each module before moving on.
5. **Finish with the capstone** by triggering the full pipeline end-to-end.

---

## Quick Environment Setup

```bash
# Clone the project and install dependencies
pip install -r requirements.txt

# Set mock mode so you can run without GPU/internet
export PIPELINE_CAPTIONING_BACKEND=mock

# Run a quick smoke test to verify everything works
pytest tests/test_pipeline.py -v
```

---

*Start with → [Module 01: Config-Driven Design](./module_01_config.md)*
