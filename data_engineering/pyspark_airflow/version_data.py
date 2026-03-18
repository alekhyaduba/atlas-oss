"""
version_data.py
─────────────────────────────────────────────────────────────────
Stage 4 – Data Version Control with PySpark
  • Reads the tagged manifest Parquet
  • Uses PySpark for scalable deduplication, schema enforcement,
    and partition-aware writes
  • Produces a versioned DVC-style Parquet dataset keyed by run_id
  • Writes a JSON lineage file alongside each run partition

Usage (standalone):
    python version_data.py \
        --config   /path/to/run_config.yaml \
        --manifest /data/processed/tagged_manifest.parquet \
        --output   /data/versioned \
        --run-id   abc123
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.config_loader import load_config

logger = logging.getLogger(__name__)


# ── Schema Enforcement ───────────────────────────────────────

REQUIRED_COLUMNS = {
    "image_id":          "string",
    "source_path":       "string",
    "processed_path":    "string",
    "caption":           "string",
    "tags":              "string",
    "tag_scores":        "string",
    "model_backend":     "string",
    "model_name":        "string",
    "pipeline_version":  "string",
    "run_id":            "string",
    "image_hash":        "string",
    "processed_width":   "int",
    "processed_height":  "int",
    "file_size_bytes":   "long",
    "status":            "string",
}


def _build_spark_session(spark_cfg: dict):
    """Build a SparkSession from config, isolated so import errors are contained."""
    from pyspark.sql import SparkSession

    builder = (
        SparkSession.builder
        .appName(spark_cfg.get("app_name", "ImagePipeline"))
        .master(spark_cfg.get("master", "local[*]"))
        .config("spark.executor.memory",  spark_cfg.get("executor_memory", "2g"))
        .config("spark.driver.memory",    spark_cfg.get("driver_memory",   "2g"))
        .config("spark.executor.cores",   str(spark_cfg.get("executor_cores", 2)))
        .config("spark.sql.shuffle.partitions",
                spark_cfg.get("extra_configs", {}).get("spark.sql.shuffle.partitions", "8"))
        .config("spark.serializer",
                spark_cfg.get("extra_configs", {}).get(
                    "spark.serializer", "org.apache.spark.serializer.KryoSerializer"))
    )
    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel(spark_cfg.get("log_level", "WARN"))
    return spark


def enforce_schema(spark, df_spark, timestamp: str):
    """Add missing columns with NULL defaults; cast types."""
    from pyspark.sql import functions as F
    from pyspark.sql.types import IntegerType, LongType, StringType

    type_map = {"string": StringType(), "int": IntegerType(), "long": LongType()}
    for col_name, col_type in REQUIRED_COLUMNS.items():
        if col_name not in df_spark.columns:
            df_spark = df_spark.withColumn(col_name, F.lit(None).cast(type_map[col_type]))
        else:
            df_spark = df_spark.withColumn(col_name, F.col(col_name).cast(type_map[col_type]))

    # Enrich with timestamp
    df_spark = df_spark.withColumn("timestamp", F.lit(timestamp))
    return df_spark


def deduplicate(spark, df_spark):
    """Remove duplicate image hashes, keeping the latest."""
    from pyspark.sql import Window
    from pyspark.sql import functions as F

    w = Window.partitionBy("image_hash").orderBy(F.col("timestamp").desc())
    return (
        df_spark
        .withColumn("_rank", F.row_number().over(w))
        .filter(F.col("_rank") == 1)
        .drop("_rank")
    )


def write_versioned_parquet(df_spark, output_dir: str, run_id: str, num_partitions: int):
    """Write Parquet partitioned by run_id."""
    from pyspark.sql import functions as F

    out_path = str(Path(output_dir) / "dataset")
    (
        df_spark
        .repartition(num_partitions)
        .write
        .mode("append")
        .partitionBy("run_id")
        .parquet(out_path)
    )
    logger.info("Versioned Parquet written to %s", out_path)
    return out_path


def write_lineage(df_spark, output_dir: str, run_id: str, config, manifest_path: str):
    """Write a JSON lineage/metadata file for this run."""
    from pyspark.sql import functions as F

    stats = df_spark.agg(
        F.count("*").alias("total_images"),
        F.sum(F.when(F.col("status") == "ok", 1).otherwise(0)).alias("ok_count"),
        F.sum(F.when(F.col("status") != "ok", 1).otherwise(0)).alias("error_count"),
    ).collect()[0]

    lineage = {
        "run_id":           run_id,
        "pipeline_version": config.pipeline_version,
        "timestamp":        datetime.now(timezone.utc).isoformat(),
        "captioning_backend": config.captioning_backend,
        "model_name":       config.model_cfg.get("model_name", "mock"),
        "tagging_backend":  config.tagging_backend,
        "input_manifest":   manifest_path,
        "total_images":     int(stats["total_images"]),
        "ok_count":         int(stats["ok_count"]),
        "error_count":      int(stats["error_count"]),
        "config_hash":      hashlib.md5(
            json.dumps(config._data, sort_keys=True).encode()
        ).hexdigest(),
    }

    lineage_path = Path(output_dir) / "lineage" / f"{run_id}.json"
    lineage_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lineage_path, "w") as f:
        json.dump(lineage, f, indent=2)

    logger.info("Lineage written: %s", lineage_path)
    return str(lineage_path)


# ── Core Stage ───────────────────────────────────────────────

def run_versioning(
    manifest_path: str,
    output_dir: str,
    config_path: str | None,
    run_id: str,
) -> str:
    cfg       = load_config(config_path)
    spark_cfg = cfg.spark_cfg
    timestamp = datetime.now(timezone.utc).isoformat()

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    spark = _build_spark_session(spark_cfg)

    logger.info("Reading manifest: %s", manifest_path)
    df_spark = spark.read.parquet(manifest_path)
    df_spark = enforce_schema(spark, df_spark, timestamp)
    df_spark = deduplicate(spark, df_spark)

    n = df_spark.count()
    logger.info("Records after deduplication: %d", n)

    out_path = write_versioned_parquet(
        df_spark, output_dir, run_id,
        spark_cfg.get("num_partitions", 8),
    )
    lineage_path = write_lineage(df_spark, output_dir, run_id, cfg, manifest_path)

    spark.stop()
    logger.info("Versioning complete. Output: %s", out_path)
    return out_path


# ── CLI ──────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Data versioning stage (PySpark)")
    p.add_argument("--config",   default=None)
    p.add_argument("--manifest", required=True)
    p.add_argument("--output",   required=True)
    p.add_argument("--run-id",   default=str(uuid.uuid4()))
    return p.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    run_versioning(args.manifest, args.output, args.config, args.run_id)
