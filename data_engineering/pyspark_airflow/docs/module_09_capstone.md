# Module 09 — End-to-End Capstone
## Putting It All Together

**Estimated time:** 90–120 minutes
**Difficulty:** ⭐⭐⭐⭐⭐ Advanced

---

## What This Module Covers

You've learned each component in isolation. Now you'll wire them together, understand how information flows through the full pipeline, and extend it with real-world enhancements. This module also maps each concept to its industry application.

---

## Part 1 — The Complete Data Flow

Trace a single image from raw input to versioned output, understanding exactly what happens at each byte.

```
1. RAW IMAGE: /data/raw/sunset.jpg  (1920×1080, 2.3MB, BGR, noisy)
   │
   │ Stage 1 – preprocess_images.py
   │   • cv2.imread()      → numpy array (H, W, 3) BGR
   │   • fastNlMeans()     → denoised array
   │   • addWeighted()     → sharpened array
   │   • CLAHE in LAB      → contrast-enhanced array
   │   • cv2.resize()      → (512, 512) array
   │   • cvtColor BGR→RGB  → model-ready array
   │   • cv2.imwrite()     → /data/processed/run01/sunset_processed.jpg
   │   • sha256_file()     → "a3b4c5..."
   │   • append to rows[]  → dict with 12 metadata fields
   ↓
   manifest.parquet row:
   {
     "image_id":      "7f3a-...",
     "source_path":   "/data/raw/sunset.jpg",
     "processed_path": "/data/processed/run01/sunset_processed.jpg",
     "image_hash":    "a3b4c5...",
     "status":        "ok",
     "processed_width": 512,
     "processed_height": 512,
     ...
   }
   │
   │ Stage 2 – caption_images.py
   │   • load_config()          → Config(captioning_backend="blip")
   │   • build_caption_backend()→ BlipCaptionBackend
   │   • backend.load()         → model weights loaded
   │   • PIL.Image.open()       → RGB PIL Image
   │   • processor(images=...)  → input tensors
   │   • model.generate()       → token IDs [3, 1045, 4, ...]
   │   • batch_decode()         → "a beautiful sunset over mountains"
   ↓
   adds to row:
   {
     "caption":       "a beautiful sunset over mountains",
     "model_backend": "blip",
     "model_name":    "Salesforce/blip-image-captioning-base",
     ...
   }
   │
   │ Stage 3 – tag_images.py
   │   • CLIP.get_text_features(labels)   → (15, 512) text embeddings
   │   • CLIP.get_image_features(image)   → (1, 512) image embedding
   │   • F.normalize() on both            → unit vectors
   │   • (img @ text.T)                   → (15,) similarities
   │   • F.softmax(× 100)                 → (15,) probabilities
   │   • top-5 above threshold 0.15       → [("landscape", 0.73), ...]
   │   • structural_tags()                → ["wide", "high_res"]
   │   • merge + deduplicate              → ["landscape", "wide", "high_res"]
   ↓
   adds to row:
   {
     "tags":       '["landscape", "wide", "high_res"]',
     "tag_scores": '{"landscape": 0.73, "outdoor scene": 0.12, ...}',
     ...
   }
   │
   │ Stage 4 – version_data.py
   │   • SparkSession.read.parquet()       → DataFrame (all images)
   │   • enforce_schema()                  → types cast, nulls filled
   │   • Window.partitionBy("image_hash")  → groups by hash
   │   • row_number().over(w) == 1         → keep newest per hash
   │   • write.partitionBy("run_id")       → creates directory structure
   │   • write_lineage()                   → lineage/run01.json
   ↓
   data/versioned/
   ├── dataset/
   │   └── run_id=run01/
   │       └── part-00000.snappy.parquet  ← contains our sunset row
   └── lineage/
       └── run01.json  ← {run_id, model_name, total: 1, ok: 1, ...}
```

---

## Part 2 — How Config Flows Through Everything

The config file is read at the start of each stage. Trace how one change propagates.

```
run_config.yaml
  model:
    captioning_backend: "blip2"   ← Change this one value
    blip2:
      device: "cuda"
      batch_size: 2
           │
           │ config_loader.load_config()
           ▼
      Config object
      .captioning_backend → "blip2"
      .model_cfg          → {"model_name": "Salesforce/blip2-opt-2.7b", ...}
           │
           │ build_caption_backend(config)
           ▼
      Blip2CaptionBackend(cfg)
      .model_name = "Salesforce/blip2-opt-2.7b"
      .device     = "cuda"
      .batch_size = 2
           │
           │ Writes to manifest:
           ▼
      row["model_backend"] = "blip2"
      row["model_name"]    = "Salesforce/blip2-opt-2.7b"
           │
           │ Lineage file records:
           ▼
      lineage.json["captioning_backend"] = "blip2"
      lineage.json["config_hash"]        = "f7a3..."  ← changes if config changes
```

---

## Part 3 — The Airflow Execution Model

```
You trigger the DAG
         │
         ▼
SchedulerJob reads image_pipeline_dag.py
  → Discovers tasks and dependencies
  → Creates a DagRun with run_id="manual__2024-01-01..."
  → Creates TaskInstances for each task
         │
         ▼
Task "validate_inputs" → state: queued
         │  scheduler enqueues
         ▼
CeleryWorker picks up "validate_inputs"
  → Calls validate_inputs(**context)
  → Pushes XCom: image_count=42
  → state: success
         │
         ▼
Task "preprocess_images" → state: queued (dependency met)
         │
         ▼
CeleryWorker picks up "preprocess_images"
  → Pulls XCom from "validate_inputs" (not needed here)
  → Runs OpenCV preprocessing
  → Pushes XCom: manifest_path, run_id, processed_dir
  → state: success
         │
         ▼ (continues through caption → tag → version → notify)
         │
         ▼
All tasks success → DagRun state: success
```

---

## Part 4 — Capstone Challenge: Extend the Pipeline

Choose one or more of these extensions to build on the existing codebase:

### Challenge A — Add an Image Quality Score

Add a new field `quality_score` to the preprocessing manifest based on:
- Laplacian variance (sharpness): `cv2.Laplacian(img, cv2.CV_64F).var()`
- Normalize to [0, 1]: `min(variance / 1000.0, 1.0)`

Then add a Spark aggregation in `version_data.py` that reports the average quality score per run in the lineage file.

### Challenge B — Add a New Caption Backend

Implement `LlavaBackend(CaptionBackend)` using the `llava-hf/llava-1.5-7b-hf` model from HuggingFace. Register it in `build_caption_backend()` with key `"llava"`. Add it to `run_config.yaml`.

### Challenge C — Parallel Stage Execution

Refactor the Airflow DAG so `caption_images` and `tag_images` run in parallel (both reading from `preprocess_images`, both writing separate manifests). Add a `merge_manifests` task that joins the two manifests on `image_id` before versioning.

### Challenge D — S3 Storage Backend

Modify `preprocess_images.py` to write processed images to S3 using `boto3`, when `dvc.storage_backend: "s3"` is set in config. Hint: use `s3_client.upload_file(local_path, bucket, key)`. Pass the S3 URI as `processed_path` in the manifest.

---

## Part 5 — Concept-to-Industry Mapping

| Course Concept | Where You'll See It in Industry |
|----------------|--------------------------------|
| Config-driven design | Every production ML system (MLflow, Kubeflow, SageMaker) |
| Pandas manifest (Parquet) | Delta Lake, Apache Iceberg, data lakehouses |
| OpenCV preprocessing | Any computer vision data pipeline |
| Abstract backend + factory | Model serving platforms, plugin architectures |
| CLIP zero-shot | Content moderation, product categorization, search |
| PySpark partitioned writes | Every data warehouse ETL (dbt + Spark, Databricks) |
| Window functions | Top-N per group, sessionization, slowly changing dimensions |
| Airflow DAGs | Used by Airbnb, Twitter, Lyft, NASA, thousands of companies |
| XCom for path passing | Any event-driven pipeline (Kafka → Spark → S3) |
| Docker + Compose | CI/CD pipelines, MLOps platforms, cloud deployments |
| Health checks | Kubernetes readiness probes, load balancer configuration |
| Idempotent tasks | Exactly-once processing, safe retries in distributed systems |
| Lineage records | Data observability tools (Monte Carlo, Great Expectations) |
| Schema enforcement | Data contracts, API versioning, data quality gates |
| Mock backends | TDD in ML systems, CI without GPU |

---

## Part 6 — What to Learn Next

You've now built a complete data engineering pipeline from scratch. Here are natural next steps:

### Immediate Extensions (1–2 weeks each)
- **dbt** — Transform data in the warehouse with SQL + version control
- **Great Expectations** — Automated data quality validation
- **MLflow** — Track experiments, model versions, and deployments

### Scale-Up Skills (1–2 months each)
- **Kubernetes + Helm** — Container orchestration at scale
- **Apache Kafka** — Real-time event streaming pipelines
- **Delta Lake / Apache Iceberg** — ACID transactions on data lakes
- **Terraform** — Infrastructure-as-code for cloud resources

### Specialization Paths
- **ML Engineering:** FastAPI model serving, Triton Inference Server, ONNX
- **Platform Engineering:** Airflow on K8s, Spark on EMR/Databricks
- **Data Quality:** Soda Core, Elementary, Anomalo
- **Vector Databases:** Pinecone, Weaviate — store CLIP embeddings for image search

---

## Final Checklist

Before considering this course complete, verify you can:

- [ ] Explain why `yaml.safe_load` is safer than `yaml.load`
- [ ] Write a deep merge function from memory
- [ ] List the 4 stages of the OpenCV preprocessing pipeline in order
- [ ] Explain why `model.eval()` and `torch.no_grad()` matter for inference
- [ ] Describe how CLIP's embedding space enables zero-shot classification
- [ ] Explain what `Window.partitionBy().orderBy()` does and give a use case
- [ ] Draw the Airflow task state machine (queued → running → success/fail)
- [ ] Explain why `catchup=False` is almost always correct
- [ ] Distinguish bind mounts from named volumes in Docker
- [ ] Write a pytest fixture that creates test images and cleans up after itself
- [ ] Explain what "idempotent" means and why it matters for retries

---

*Congratulations on completing the course!*

*Go build something real — fork this project, add your own backend, point it at real images, and observe what it produces. The best way to cement these skills is to extend working code, not start from scratch.*
