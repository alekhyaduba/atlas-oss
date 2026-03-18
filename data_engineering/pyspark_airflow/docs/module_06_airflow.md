# Module 06 — Airflow DAG Orchestration
## `image_pipeline_dag.py`

**Estimated time:** 60–75 minutes
**Difficulty:** ⭐⭐⭐⭐☆ Intermediate–Advanced

---

## Why This Module Matters

Writing scripts that process data is only half the job. The other half is **scheduling, monitoring, retrying, and connecting** those scripts reliably. Apache Airflow is the industry-standard tool for this. Understanding how Airflow thinks about workflows is a foundational data engineering skill.

---

## Concept 1 — What is a DAG?

**DAG** = Directed Acyclic Graph. In Airflow, a DAG is a collection of tasks with defined dependencies, where:
- **Directed** = dependencies flow in one direction
- **Acyclic** = no cycles (a task can't depend on its own output)

```
validate_inputs
      │
      ▼
preprocess_images
      │
      ▼
caption_images
      │
      ▼
tag_images
      │
      ▼
version_data
      │
      ▼
notify_success
```

Airflow reads your Python file, discovers the DAG object, and manages scheduling, execution, retries, and history.

---

## Concept 2 — Defining a DAG

```python
from airflow import DAG
from airflow.utils.dates import days_ago
from datetime import timedelta

default_args = {
    "owner":             "data-engineering",
    "retries":           2,
    "retry_delay":       timedelta(minutes=5),
    "execution_timeout": timedelta(minutes=60),
    "email_on_failure":  False,
}

with DAG(
    dag_id="image_caption_tag_pipeline",    # Unique identifier in Airflow
    description="End-to-end image pipeline",
    default_args=default_args,
    schedule_interval="@hourly",            # Cron string or preset
    start_date=days_ago(1),                 # How far back to backfill
    max_active_runs=1,                      # Only 1 run at a time
    catchup=False,                          # Don't backfill missed runs
    tags=["image-processing", "ml-inference"],
) as dag:
    # Tasks defined here inherit default_args
    ...
```

**Schedule presets:**

| Preset | Equivalent cron | Meaning |
|--------|-----------------|---------|
| `@once` | (run once) | Manual trigger |
| `@hourly` | `0 * * * *` | Top of every hour |
| `@daily` | `0 0 * * *` | Midnight every day |
| `@weekly` | `0 0 * * 0` | Sunday midnight |
| `0 */6 * * *` | Custom | Every 6 hours |

**`catchup=False`** is important: without it, if your pipeline was offline for a week, Airflow would try to run 168 hourly catch-up jobs when it restarts. Almost always set this to `False`.

---

## Concept 3 — PythonOperator

The simplest and most flexible Airflow operator: executes a Python function as a task.

```python
from airflow.operators.python import PythonOperator

def my_task_function(**context):
    # context contains Airflow execution metadata
    print(f"Running DAG: {context['dag'].dag_id}")
    print(f"Execution date: {context['execution_date']}")
    return "done"   # Return value is stored in XCom automatically

task = PythonOperator(
    task_id="my_task",              # Unique within the DAG
    python_callable=my_task_function,
    provide_context=True,           # Inject context dict into the function
)
```

**`**context` — the context dictionary contains:**

```python
context = {
    "dag":            <DAG object>,
    "dag_run":        <DagRun object>,     # The specific run instance
    "run_id":         "manual__2024-01-01T00:00:00+00:00",
    "execution_date": datetime(2024, 1, 1),
    "task":           <Task object>,
    "task_instance":  <TaskInstance object>,  # Key for XCom
    # ...
}
```

---

## Concept 4 — XCom: Passing Data Between Tasks

**XCom** (Cross-Communication) is Airflow's mechanism for passing small pieces of data between tasks. Think of it as a key-value store scoped to a DAG run.

```python
def task_a(**context):
    result = "data/processed/manifest.parquet"
    # Push a value to XCom
    context["task_instance"].xcom_push(key="manifest_path", value=result)
    return result   # Return value is also stored as XCom key="return_value"

def task_b(**context):
    ti = context["task_instance"]
    # Pull the value from the previous task
    manifest = ti.xcom_pull(
        task_ids="task_a",      # Which task produced it
        key="manifest_path"     # Which key to retrieve
    )
    print(f"Received: {manifest}")
    # → "data/processed/manifest.parquet"
```

**XCom limits and best practices:**

| Store in XCom | Don't store in XCom |
|---------------|---------------------|
| File paths | File contents |
| Run IDs | DataFrames |
| Record counts | Model weights |
| Status strings | Large JSON blobs |

XCom is stored in Airflow's database. Keep values small (< 1MB). For large data, write to a file and pass the path via XCom.

---

## Concept 5 — Trigger-Time Configuration

Airflow DAGs can accept runtime overrides via `dag_run.conf`. This is how you change behaviour without modifying code or YAML.

```python
def preprocess_images(**context):
    dag_conf = context["dag_run"].conf or {}   # {} if no conf provided

    # Use trigger conf if present, fall back to default
    raw_dir = dag_conf.get("raw_images_dir", DATA_RAW)

    # Apply model overrides via environment variables
    # (config_loader reads these at runtime)
    if backend := dag_conf.get("captioning_backend"):
        os.environ["PIPELINE_CAPTIONING_BACKEND"] = backend

    if device := dag_conf.get("model_device"):
        os.environ["PIPELINE_MODEL_DEVICE"] = device
    ...
```

**Triggering with conf:**

```bash
# CLI trigger
airflow dags trigger image_caption_tag_pipeline \
    --conf '{"captioning_backend": "blip2", "model_device": "cuda"}'

# Airflow UI → Trigger DAG → "Trigger DAG w/ config" → paste JSON
```

---

## Concept 6 — Task Dependencies

Airflow uses the `>>` operator (bitshift) to define dependency chains.

```python
# Sequential chain
t_validate >> t_preprocess >> t_caption >> t_tag >> t_version >> t_notify

# Fan-out (parallel tasks)
t_validate >> [t_preprocess_a, t_preprocess_b]

# Fan-in (wait for multiple tasks)
[t_caption, t_tag] >> t_version

# Explicit set_downstream / set_upstream (equivalent)
t_validate.set_downstream(t_preprocess)
t_preprocess.set_upstream(t_validate)
```

**Our pipeline is linear** because each stage depends on the previous stage's output. A more advanced version could parallelize caption and tag inference if they didn't share state.

---

## Concept 7 — Retries and Error Handling

```python
default_args = {
    "retries":     2,                   # Retry up to 2 times on failure
    "retry_delay": timedelta(minutes=5),# Wait 5 min between retries
    "execution_timeout": timedelta(minutes=60),  # Kill task after 60min
}
```

**Task states in Airflow:**

```
queued → running → success
                  ↓
                failed → up_for_retry → running → success
                                                  ↓
                                                failed (max retries reached)
```

**Idempotency** — critical for retries to be safe:

```python
# ❌ Non-idempotent: running twice appends data twice
df.to_parquet("manifest.parquet", mode="append")

# ✅ Idempotent: running twice produces the same result
# Use a run_id-specific path so reruns write to a fresh location
out_path = f"data/processed/{run_id}/manifest.parquet"
df.to_parquet(out_path)  # Overwrite if it exists
```

A task is **idempotent** if running it multiple times produces the same result as running it once. This is essential when Airflow retries failed tasks.

---

## Concept 8 — Run ID as a Correlation Key

Every pipeline run needs a **stable, unique identifier** that ties all outputs together. We derive it from Airflow's `run_id` to make it traceable back to the scheduler.

```python
def _get_run_id(context: dict) -> str:
    """Derive a stable, filesystem-safe run ID from Airflow's run ID."""
    dag_run_id = context["run_id"]
    # e.g. "manual__2024-01-01T10:30:00+00:00"
    # → "manual__2024-01-01T10_30_00+00_00"
    return dag_run_id.replace(":", "_").replace("+", "_")[:64]
```

This run ID:
- Prefixes output directories (`data/processed/{run_id}/`)
- Is embedded in every row of the manifest
- Names the lineage JSON file (`lineage/{run_id}.json`)
- Becomes the Parquet partition key (`run_id=manual__...`)

With this, you can trace any output file back to the exact Airflow run that produced it.

---

## Summary

| Concept | Key Idea |
|---------|----------|
| DAG | Directed Acyclic Graph — tasks + dependencies, no cycles |
| `schedule_interval` | Cron strings or presets; `catchup=False` prevents backfill storms |
| PythonOperator | Execute any Python function as a task |
| `**context` | Airflow injects execution metadata; access via `context["dag_run"]` etc. |
| XCom | Pass file paths and IDs between tasks — not large data |
| Trigger conf | Runtime overrides via `dag_run.conf` — change model without redeploying |
| `>>` operator | Define task dependencies declaratively |
| Idempotency | Safe retries require the same result on re-run |
| Run ID | Stable correlation key linking all outputs from one DAG run |

---

## Exercises

**1. Basic** — In the Airflow UI (after running `docker-compose up`), trigger the DAG manually. Navigate to the Graph View and observe each task's state. Click on a task and inspect its XCom values.

**2. Intermediate** — Add a new task `validate_outputs` between `version_data` and `notify_success` that reads the lineage JSON and asserts `ok_count > 0`. If zero images were processed successfully, raise `AirflowException("No images processed successfully")`.

**3. Advanced** — Refactor the pipeline so `caption_images` and `tag_images` run **in parallel** (both depend on `preprocess_images`, and `version_data` depends on both). You'll need to change how manifests are passed via XCom, and merge the two manifest DataFrames before versioning. Draw the new DAG graph first.

---

*Next → [Module 07: Dockerizing the Pipeline](./module_07_docker.md)*
