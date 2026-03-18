# Module 05 — PySpark Data Versioning
## `version_data.py`

**Estimated time:** 60–75 minutes
**Difficulty:** ⭐⭐⭐⭐☆ Intermediate–Advanced

---

## Why This Module Matters

Pandas is great for in-memory processing, but it can't handle datasets larger than your RAM. PySpark distributes computation across many machines, processes data in parallel, and is the standard tool for **large-scale data engineering**. This module introduces Spark's core concepts through the lens of a real problem: deduplicating and versioning an image dataset.

---

## Concept 1 — What PySpark Is and When to Use It

PySpark is the Python API for **Apache Spark** — a distributed compute engine. A Spark job can run on a single machine (`local[*]`) or a cluster of hundreds of machines with the exact same code.

```
When to use Pandas:          When to use PySpark:
─────────────────────        ───────────────────────
< 1GB of data                > 1GB of data
Single machine               Multiple machines
Prototyping                  Production at scale
Interactive exploration      Scheduled batch jobs
```

**In our pipeline:** The preprocessing and captioning stages use Pandas (fast, simple for per-image operations). The versioning stage uses Spark because it's designed for data warehouse operations that scale.

---

## Concept 2 — SparkSession: The Entry Point

Everything in Spark starts with a `SparkSession`. Think of it as the "database connection" for distributed compute.

```python
from pyspark.sql import SparkSession

spark = (
    SparkSession.builder
    .appName("ImagePipeline")           # Job name in Spark UI
    .master("local[*]")                 # local[*] = use all local CPUs
                                        # spark://host:7077 = remote cluster
    .config("spark.executor.memory", "2g")
    .config("spark.driver.memory",   "2g")
    .config("spark.sql.shuffle.partitions", "8")
    .getOrCreate()                      # Reuse existing session if one exists
)

spark.sparkContext.setLogLevel("WARN")  # Reduce noisy output

# Always stop Spark when done to free resources
spark.stop()
```

**`local[*]` vs cluster:**

```
local[*]  → All computation runs in-process, uses all CPU cores
            No cluster setup needed — great for development

spark://master:7077  → Submits job to a remote Spark cluster
                       Workers run on separate machines
                       Same code, distributed execution
```

---

## Concept 3 — DataFrames: Spark vs Pandas

Spark DataFrames look similar to Pandas DataFrames but behave very differently under the hood.

```python
# Pandas DataFrame — data lives in RAM, operations execute immediately
import pandas as pd
pdf = pd.read_parquet("manifest.parquet")
pdf["new_col"] = pdf["col_a"] + pdf["col_b"]  # Executes NOW

# Spark DataFrame — operations build a logical plan, execute lazily
df_spark = spark.read.parquet("manifest.parquet")
df_spark = df_spark.withColumn("new_col", col("col_a") + col("col_b"))
# ↑ Nothing has executed yet! Spark builds a query plan.

# Execution is triggered by "actions" like count(), show(), write()
n = df_spark.count()  # NOW it executes the plan and counts
```

**Lazy evaluation benefits:**
- Spark can **optimize the plan** before running (e.g., push filters early)
- Operations are combined into efficient stages
- No wasted work until you actually need results

---

## Concept 4 — Schema Enforcement

Raw data from upstream stages may have missing columns, wrong types, or inconsistent nulls. Schema enforcement ensures downstream code always gets what it expects.

```python
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, LongType, StringType

REQUIRED_COLUMNS = {
    "image_id":         "string",
    "caption":          "string",
    "tags":             "string",
    "processed_width":  "int",
    "file_size_bytes":  "long",
    # ...
}

def enforce_schema(spark, df, timestamp: str):
    type_map = {
        "string": StringType(),
        "int":    IntegerType(),
        "long":   LongType()
    }

    for col_name, col_type in REQUIRED_COLUMNS.items():
        if col_name not in df.columns:
            # Column missing entirely → add it as NULL
            df = df.withColumn(col_name, F.lit(None).cast(type_map[col_type]))
        else:
            # Column exists → cast to correct type (silently converts "42" → 42)
            df = df.withColumn(col_name, F.col(col_name).cast(type_map[col_type]))

    # Add timestamp enrichment
    df = df.withColumn("timestamp", F.lit(timestamp))
    return df
```

**`F.lit(None)` — adding null columns:**

```python
# F.lit() creates a constant column
F.lit("hello")    # every row gets the string "hello"
F.lit(42)         # every row gets the integer 42
F.lit(None)       # every row gets NULL
F.lit(None).cast(StringType())  # NULL with explicit type for schema
```

---

## Concept 5 — Window Functions for Deduplication

Images may be processed multiple times (retries, re-runs). We want to keep **only the latest record for each unique image hash**. This is the classic **deduplication by latest** pattern, solved elegantly with **Window Functions**.

```python
from pyspark.sql import Window
from pyspark.sql import functions as F

def deduplicate(spark, df):
    # A Window defines a "group + order" for row-level functions
    w = (
        Window
        .partitionBy("image_hash")      # Group rows with the same hash
        .orderBy(F.col("timestamp").desc())  # Within each group, newest first
    )

    return (
        df
        .withColumn("_rank", F.row_number().over(w))
        # row_number() = 1 for newest record in each group, 2 for second-newest, etc.

        .filter(F.col("_rank") == 1)    # Keep only the newest per hash

        .drop("_rank")                  # Remove the helper column
    )
```

**Window function mechanics:**

```
image_hash  | timestamp            | caption      | _rank
────────────┼──────────────────────┼──────────────┼───────
abc123      | 2024-01-02 10:00:00  | "a dog..."   |  1   ← KEEP (newest)
abc123      | 2024-01-01 08:00:00  | "a dog..."   |  2   ← DROP (older)
def456      | 2024-01-01 09:00:00  | "a cat..."   |  1   ← KEEP (only record)
```

---

## Concept 6 — Aggregations with `.agg()`

Spark's `.agg()` computes multiple aggregations in one pass — far more efficient than multiple `.count()` calls.

```python
from pyspark.sql import functions as F

stats = df.agg(
    F.count("*").alias("total_images"),

    # Conditional count: count rows where status == "ok"
    F.sum(
        F.when(F.col("status") == "ok", 1).otherwise(0)
    ).alias("ok_count"),

    # Count errors
    F.sum(
        F.when(F.col("status") != "ok", 1).otherwise(0)
    ).alias("error_count"),

).collect()   # collect() brings results from cluster to driver (Python)

row = stats[0]   # .collect() returns a list of Row objects
print(f"Total: {row['total_images']}, OK: {row['ok_count']}, Errors: {row['error_count']}")
```

**`F.when(...).otherwise(...)` — conditional expressions:**

```python
# Equivalent SQL: CASE WHEN status = 'ok' THEN 1 ELSE 0 END
F.when(condition, true_value).otherwise(false_value)

# Chainable:
F.when(F.col("score") > 0.8, "high")
 .when(F.col("score") > 0.5, "medium")
 .otherwise("low")
```

---

## Concept 7 — Partitioned Parquet Writes

Writing data partitioned by a key enables **partition pruning** — when you query for a specific run_id, Spark reads only that partition, not the entire dataset.

```python
(
    df_spark
    .repartition(8)          # Distribute data into 8 partitions before writing
                             # More partitions = more parallelism (set to num cores)
    .write
    .mode("append")          # "append" | "overwrite" | "error" | "ignore"
    .partitionBy("run_id")   # Create subdirectories: dataset/run_id=abc123/
    .parquet(output_path)
)
```

**Physical output structure:**

```
data/versioned/dataset/
├── run_id=run_20240101/
│   ├── part-00000.snappy.parquet
│   ├── part-00001.snappy.parquet
│   └── ...
├── run_id=run_20240102/
│   └── ...
└── _SUCCESS
```

**Reading with partition pruning:**

```python
# Reads ALL partitions
df = spark.read.parquet("data/versioned/dataset/")

# Reads ONLY run_id=run_20240101 (fast — skips all other directories)
df = spark.read.parquet("data/versioned/dataset/").filter(
    F.col("run_id") == "run_20240101"
)
```

---

## Concept 8 — Data Lineage

A **lineage record** answers the question: "Where did this data come from and how was it produced?" It's the cornerstone of reproducible data engineering.

```python
import json, hashlib
from datetime import datetime, timezone

def write_lineage(df_spark, output_dir, run_id, config, manifest_path):
    stats = df_spark.agg(...).collect()[0]

    lineage = {
        "run_id":             run_id,
        "pipeline_version":   config.pipeline_version,
        "timestamp":          datetime.now(timezone.utc).isoformat(),
        "captioning_backend": config.captioning_backend,
        "model_name":         config.model_cfg.get("model_name"),
        "input_manifest":     manifest_path,       # What went in
        "total_images":       int(stats["total_images"]),
        "ok_count":           int(stats["ok_count"]),
        # Fingerprint the entire config so you can reproduce this exact run
        "config_hash":        hashlib.md5(
                                  json.dumps(config._data, sort_keys=True).encode()
                              ).hexdigest(),
    }

    path = Path(output_dir) / "lineage" / f"{run_id}.json"
    with open(path, "w") as f:
        json.dump(lineage, f, indent=2)
```

**Lineage enables:**
- Reproduce any past run by looking up its `config_hash`
- Debug "why did the model output change?" by comparing lineage files
- Audit which model version processed which images

---

## Summary

| Concept | Key Idea |
|---------|----------|
| SparkSession | Single entry point; `local[*]` for dev, cluster URL for prod |
| Lazy evaluation | Operations build a plan; actions trigger execution |
| Schema enforcement | Add missing columns as NULL; cast types explicitly |
| `F.lit(None)` | Add a NULL column with typed schema |
| Window functions | `row_number().over(Window.partitionBy().orderBy())` for dedup |
| `.agg()` with conditions | `F.when().otherwise()` for conditional aggregation |
| Partitioned Parquet writes | `partitionBy("run_id")` enables fast partition-pruned reads |
| Lineage records | Config hash + input path + stats = reproducibility |

---

## Exercises

**1. Basic** — Load the versioned Parquet output into a Spark DataFrame. Count how many images have more than 3 tags. Hint: use `F.size(F.from_json(col("tags"), ...))` or filter the `tags` column as a string.

**2. Intermediate** — Add a `compute_tag_stats(df_spark)` function that returns a DataFrame of `(tag_name, count, avg_confidence)` — one row per unique tag across all images. Use `F.explode` after parsing the `tag_scores` JSON.

**3. Advanced** — Modify `write_versioned_parquet` to also partition by `model_backend` in addition to `run_id` (i.e., `partitionBy("run_id", "model_backend")`). Then write a query that reads only `blip` captions from a specific run. What are the tradeoffs of more vs fewer partition columns?

---

*Next → [Module 06: Airflow DAG Orchestration](./module_06_airflow.md)*
