# Image Caption & Tag Pipeline

End-to-end data engineering pipeline built on **Apache Airflow**, **PySpark**, **Pandas**, and **OpenCV** that ingests raw images, refines them, runs caption inference, tags them, and versions the outputs for reproducible data management.

---

## Architecture

```
Raw Images
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│  AIRFLOW DAG  –  image_caption_tag_pipeline                     │
│                                                                 │
│  ① validate_inputs                                              │
│        │  checks raw dir is non-empty                           │
│        ▼                                                        │
│  ② preprocess_images          (OpenCV)                          │
│        │  denoise → sharpen → CLAHE → resize → manifest.parquet │
│        ▼                                                        │
│  ③ caption_images             (BLIP / BLIP-2 / GIT / mock)     │
│        │  batch inference → captioned_manifest.parquet          │
│        ▼                                                        │
│  ④ tag_images                 (CLIP zero-shot + structural)     │
│        │  semantic tags + structural tags → tagged_manifest     │
│        ▼                                                        │
│  ⑤ version_data               (PySpark)                         │
│        │  dedup → partition-aware Parquet write + lineage JSON  │
│        ▼                                                        │
│  ⑥ notify_success                                               │
│        │  logs summary (extend to Slack / email / metrics)      │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
data/versioned/
  dataset/run_id=<run>/  ← queryable Parquet partitions
  lineage/<run>.json      ← full lineage record
```

---

## Project Structure

```
image_pipeline/
├── config/
│   ├── run_config.yaml        ← model selection, paths, Spark, DVC settings
│   └── config_loader.py       ← typed config loader with env-var overrides
├── scripts/
│   ├── preprocess_images.py   ← Stage 1: OpenCV preprocessing
│   ├── caption_images.py      ← Stage 2: caption inference (BLIP/BLIP-2/GIT/mock)
│   ├── tag_images.py          ← Stage 3: CLIP tagging + structural tags
│   └── version_data.py        ← Stage 4: PySpark DVC-style versioning
├── dags/
│   └── image_pipeline_dag.py  ← Airflow DAG definition
├── docker/
│   ├── Dockerfile             ← production image
│   ├── docker-compose.yml     ← full local dev stack
│   └── requirements.txt
└── tests/
    └── test_pipeline.py       ← pytest unit + smoke tests
```

---

## Quick Start

### 1 – Local dev with Docker Compose

```bash
# Build & start everything
cd docker
docker compose up --build -d

# Open Airflow UI → http://localhost:8080  (admin / admin)

# Drop some images into the raw volume
docker compose cp /path/to/your/images airflow-webserver:/opt/airflow/data/raw/

# Trigger the DAG from the UI, or via CLI:
docker compose exec airflow-webserver \
  airflow dags trigger image_caption_tag_pipeline

# Tail logs
docker compose logs -f airflow-worker
```

### 2 – Run stages standalone (no Airflow)

```bash
pip install -r docker/requirements.txt

python scripts/preprocess_images.py \
    --input  data/raw \
    --output data/processed/run01 \
    --run-id run01

python scripts/caption_images.py \
    --manifest data/processed/run01/manifest.parquet \
    --output   data/processed/run01/captioned_manifest.parquet \
    --run-id   run01

python scripts/tag_images.py \
    --manifest data/processed/run01/captioned_manifest.parquet \
    --output   data/processed/run01/tagged_manifest.parquet \
    --run-id   run01

python scripts/version_data.py \
    --manifest data/processed/run01/tagged_manifest.parquet \
    --output   data/versioned \
    --run-id   run01
```

### 3 – Run tests

```bash
pytest tests/test_pipeline.py -v
```

---

## Runtime Configuration

All tunable settings live in `config/run_config.yaml`.

### Switch caption model at runtime

| Method | Example |
|--------|---------|
| Edit YAML | `model.captioning_backend: blip2` |
| Env var | `PIPELINE_CAPTIONING_BACKEND=blip2` |
| Airflow trigger conf | `{"captioning_backend": "blip2", "model_device": "cuda"}` |

### Supported caption backends

| Backend | Model | Notes |
|---------|-------|-------|
| `mock`  | deterministic | CI / unit tests, no GPU |
| `blip`  | Salesforce/blip-image-captioning-base | CPU-friendly |
| `blip2` | Salesforce/blip2-opt-2.7b | GPU recommended |
| `git`   | microsoft/git-base-coco | CPU-friendly |

### Supported tag backends

| Backend | Model | Notes |
|---------|-------|-------|
| `mock`  | deterministic | CI / unit tests |
| `clip`  | openai/clip-vit-base-patch32 | configurable candidate labels |

---

## Versioned Output Schema

Each run writes a Parquet partition at `data/versioned/dataset/run_id=<run>/` with columns:

| Column | Type | Description |
|--------|------|-------------|
| `image_id` | string | UUID per image |
| `source_path` | string | Original file path |
| `processed_path` | string | Post-preprocessing path |
| `caption` | string | Generated caption |
| `tags` | string (JSON array) | Semantic + structural tags |
| `tag_scores` | string (JSON object) | Tag → confidence score |
| `model_backend` | string | Caption model used |
| `model_name` | string | HuggingFace model ID |
| `image_hash` | string | SHA-256 of processed file |
| `pipeline_version` | string | Version from config |
| `timestamp` | string | ISO-8601 UTC |
| `file_size_bytes` | long | Processed file size |
| `processing_duration_ms` | int | Wall-clock time |

---

## Scaling to Production

1. **Spark cluster** – change `spark.master` in config to `spark://spark-master:7077` and add workers in `docker-compose.yml`.
2. **GPU captioning** – set `model.blip.device: cuda` and use a CUDA-enabled base image.
3. **Remote storage** – set `dvc.storage_backend: s3` and configure bucket credentials.
4. **Parallel workers** – increase `airflow-worker` replicas in Compose or use Kubernetes Executor.
5. **Model caching** – uncomment the `CACHE_MODELS` block in the Dockerfile to bake models into the image.
