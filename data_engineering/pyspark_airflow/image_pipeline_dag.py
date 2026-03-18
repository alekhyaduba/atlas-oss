"""
image_pipeline_dag.py
─────────────────────────────────────────────────────────────────
Airflow DAG – Image Caption & Tag Pipeline

Stages
──────
  validate_inputs        → Check raw image directory is non-empty
  preprocess_images      → OpenCV denoise / sharpen / CLAHE / resize
  caption_images         → BLIP / BLIP-2 / GIT caption inference
  tag_images             → CLIP zero-shot image tagging
  version_data           → PySpark dedup + partition-aware Parquet write
  notify_success         → Log summary stats (extend to Slack / email)

Runtime config is read from /opt/airflow/config/run_config.yaml.
Override individual settings at trigger time via Airflow conf:
  {"captioning_backend": "blip2", "model_device": "cuda"}
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago

# ── Paths resolved inside the container ─────────────────────
BASE_DIR    = Path(os.environ.get("PIPELINE_BASE_DIR", "/opt/airflow"))
CONFIG_PATH = str(BASE_DIR / "config" / "run_config.yaml")
SCRIPTS_DIR = BASE_DIR / "scripts"
DATA_RAW    = str(BASE_DIR / "data" / "raw")
DATA_PROC   = str(BASE_DIR / "data" / "processed")
DATA_VER    = str(BASE_DIR / "data" / "versioned")

# Make scripts importable inside tasks
import sys
sys.path.insert(0, str(BASE_DIR))

logger = logging.getLogger(__name__)


# ── Default DAG args ─────────────────────────────────────────

default_args = {
    "owner":            "data-engineering",
    "depends_on_past":  False,
    "email_on_failure": False,
    "email_on_retry":   False,
    "retries":          2,
    "retry_delay":      timedelta(minutes=5),
    "execution_timeout":timedelta(minutes=60),
}


# ── Task Callables ────────────────────────────────────────────

def _get_run_id(context: dict) -> str:
    """Stable run ID derived from DAG run ID."""
    dag_run_id = context["run_id"]
    return dag_run_id.replace(":", "_").replace("+", "_")[:64]


def validate_inputs(**context):
    raw_dir = context["dag_run"].conf.get("raw_images_dir", DATA_RAW)
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    images = [p for p in Path(raw_dir).rglob("*") if p.suffix.lower() in exts]
    if not images:
        raise FileNotFoundError(f"No images found in {raw_dir}")
    logger.info("Validation passed: %d images found in %s", len(images), raw_dir)
    context["task_instance"].xcom_push(key="image_count", value=len(images))
    context["task_instance"].xcom_push(key="raw_dir",     value=raw_dir)


def preprocess_images(**context):
    # Resolve any trigger-time overrides
    dag_conf      = context["dag_run"].conf or {}
    run_id        = _get_run_id(context)
    raw_dir       = dag_conf.get("raw_images_dir", DATA_RAW)
    proc_dir      = str(Path(DATA_PROC) / run_id)
    manifest_path = str(Path(proc_dir) / "manifest.parquet")

    # Apply env-var overrides from dag_conf
    if backend := dag_conf.get("captioning_backend"):
        os.environ["PIPELINE_CAPTIONING_BACKEND"] = backend
    if device := dag_conf.get("model_device"):
        os.environ["PIPELINE_MODEL_DEVICE"] = device

    from scripts.preprocess_images import run_preprocessing
    out = run_preprocessing(raw_dir, proc_dir, CONFIG_PATH, run_id, manifest_path)
    context["task_instance"].xcom_push(key="manifest_path",    value=out)
    context["task_instance"].xcom_push(key="processed_dir",    value=proc_dir)
    context["task_instance"].xcom_push(key="run_id",           value=run_id)
    logger.info("Preprocessing complete: manifest=%s", out)


def caption_images(**context):
    ti         = context["task_instance"]
    manifest   = ti.xcom_pull(task_ids="preprocess_images", key="manifest_path")
    run_id     = ti.xcom_pull(task_ids="preprocess_images", key="run_id")
    proc_dir   = ti.xcom_pull(task_ids="preprocess_images", key="processed_dir")
    out_path   = str(Path(proc_dir) / "captioned_manifest.parquet")

    from scripts.caption_images import run_captioning
    out = run_captioning(manifest, out_path, CONFIG_PATH, run_id)
    ti.xcom_push(key="captioned_manifest", value=out)
    logger.info("Captioning complete: %s", out)


def tag_images(**context):
    ti       = context["task_instance"]
    manifest = ti.xcom_pull(task_ids="caption_images", key="captioned_manifest")
    run_id   = ti.xcom_pull(task_ids="preprocess_images", key="run_id")
    proc_dir = ti.xcom_pull(task_ids="preprocess_images", key="processed_dir")
    out_path = str(Path(proc_dir) / "tagged_manifest.parquet")

    from scripts.tag_images import run_tagging
    out = run_tagging(manifest, out_path, CONFIG_PATH, run_id)
    ti.xcom_push(key="tagged_manifest", value=out)
    logger.info("Tagging complete: %s", out)


def version_data(**context):
    ti       = context["task_instance"]
    manifest = ti.xcom_pull(task_ids="tag_images", key="tagged_manifest")
    run_id   = ti.xcom_pull(task_ids="preprocess_images", key="run_id")

    from scripts.version_data import run_versioning
    out = run_versioning(manifest, DATA_VER, CONFIG_PATH, run_id)
    ti.xcom_push(key="versioned_output", value=out)
    logger.info("Versioning complete: %s", out)


def notify_success(**context):
    ti        = context["task_instance"]
    run_id    = ti.xcom_pull(task_ids="preprocess_images", key="run_id")
    img_count = ti.xcom_pull(task_ids="validate_inputs",   key="image_count")
    ver_out   = ti.xcom_pull(task_ids="version_data",      key="versioned_output")

    lineage_file = Path(DATA_VER) / "lineage" / f"{run_id}.json"
    summary = {
        "run_id":        run_id,
        "image_count":   img_count,
        "versioned_to":  ver_out,
        "lineage_file":  str(lineage_file),
        "status":        "SUCCESS",
        "completed_at":  datetime.now(timezone.utc).isoformat(),
    }
    logger.info("Pipeline summary:\n%s", json.dumps(summary, indent=2))
    # Extend here: post to Slack, send email, write to monitoring DB, etc.


# ── DAG Definition ────────────────────────────────────────────

with DAG(
    dag_id="image_caption_tag_pipeline",
    description="End-to-end image preprocessing → caption → tag → DVC pipeline",
    default_args=default_args,
    schedule_interval="@hourly",
    start_date=days_ago(1),
    max_active_runs=1,
    catchup=False,
    tags=["image-processing", "ml-inference", "data-versioning"],
    doc_md=__doc__,
) as dag:

    t_validate = PythonOperator(
        task_id="validate_inputs",
        python_callable=validate_inputs,
        provide_context=True,
    )

    t_preprocess = PythonOperator(
        task_id="preprocess_images",
        python_callable=preprocess_images,
        provide_context=True,
    )

    t_caption = PythonOperator(
        task_id="caption_images",
        python_callable=caption_images,
        provide_context=True,
    )

    t_tag = PythonOperator(
        task_id="tag_images",
        python_callable=tag_images,
        provide_context=True,
    )

    t_version = PythonOperator(
        task_id="version_data",
        python_callable=version_data,
        provide_context=True,
    )

    t_notify = PythonOperator(
        task_id="notify_success",
        python_callable=notify_success,
        provide_context=True,
    )

    # ── Linear DAG chain ────────────────────────────────────
    t_validate >> t_preprocess >> t_caption >> t_tag >> t_version >> t_notify
