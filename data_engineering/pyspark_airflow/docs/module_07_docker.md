# Module 07 — Dockerizing the Pipeline
## `Dockerfile` + `docker-compose.yml`

**Estimated time:** 45–60 minutes
**Difficulty:** ⭐⭐⭐☆☆ Intermediate

---

## Why This Module Matters

"It works on my machine" is the oldest problem in software. Docker solves it by packaging your code, runtime, system libraries, and dependencies into a single **container image** that runs identically everywhere — your laptop, a cloud VM, or a Kubernetes cluster.

---

## Concept 1 — What Docker Does

Docker creates **containers** — isolated processes that share the host OS kernel but have their own filesystem, networking, and process space.

```
WITHOUT Docker:
  Dev machine:  Python 3.11, OpenCV 4.9, PyTorch 2.2  ✅ Works
  Prod server:  Python 3.8,  OpenCV 4.5, PyTorch 1.9  ❌ Fails silently

WITH Docker:
  Container image = Python 3.11 + OpenCV 4.9 + PyTorch 2.2
  Dev machine:  runs container  ✅ Works
  Prod server:  runs same container  ✅ Works identically
```

**Key concepts:**
- **Image** — a snapshot of a filesystem, like a class in OOP
- **Container** — a running instance of an image, like an object
- **Dockerfile** — instructions for building an image
- **Registry** — a repository for storing and distributing images (Docker Hub, ECR)

---

## Concept 2 — Dockerfile Anatomy

```dockerfile
# ── Base image ────────────────────────────────────────────────
# Start from Apache Airflow 2.8 with Python 3.11 already installed
FROM apache/airflow:2.8.1-python3.11

# ── System dependencies ───────────────────────────────────────
USER root  # ← Switch to root to install system packages

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \    # OpenCV runtime dependency (display)
    libglib2.0-0 \       # OpenCV runtime dependency (GLib)
    default-jdk-headless \ # Java — required for PySpark
    curl \
    && rm -rf /var/lib/apt/lists/*  # ← Clear apt cache to shrink image size

# ── Python dependencies ───────────────────────────────────────
USER airflow  # ← Drop back to non-root for security

COPY --chown=airflow:root requirements.txt /opt/airflow/requirements.txt

RUN pip install --no-cache-dir -r /opt/airflow/requirements.txt
# --no-cache-dir: don't cache pip downloads (smaller image)

# ── Application code ──────────────────────────────────────────
COPY --chown=airflow:root config/   /opt/airflow/config/
COPY --chown=airflow:root scripts/  /opt/airflow/scripts/
COPY --chown=airflow:root dags/     /opt/airflow/dags/

# ── Environment variables ─────────────────────────────────────
ENV PIPELINE_BASE_DIR=/opt/airflow
ENV TRANSFORMERS_CACHE=/opt/airflow/model_cache
ENV PYTHONPATH="/opt/airflow:${PYTHONPATH}"
```

---

## Concept 3 — Layer Caching: Building Images Fast

Every `RUN`, `COPY`, and `ADD` instruction creates a **layer**. Docker caches layers and only rebuilds from the first changed layer down. This makes rebuilds fast if you structure the Dockerfile correctly.

```dockerfile
# ❌ WRONG order — code changes invalidate the slow pip install layer
COPY . /opt/airflow/           # If ANY file changes...
RUN pip install -r requirements.txt  # ...this runs again (slow!)

# ✅ CORRECT order — dependencies before code
COPY requirements.txt /opt/airflow/requirements.txt
RUN pip install -r requirements.txt  # Only reruns if requirements.txt changes

COPY config/   /opt/airflow/config/    # These layers are fast —
COPY scripts/  /opt/airflow/scripts/   # code changes only rebuild these
COPY dags/     /opt/airflow/dags/
```

**Rule:** Copy files that change rarely (dependencies) **before** files that change often (code).

---

## Concept 4 — Docker Compose: Multi-Service Orchestration

A real Airflow deployment needs multiple services: a database, a message broker, a scheduler, and workers. Docker Compose defines all of these in one YAML file.

```yaml
# Shared config for all Airflow services using YAML anchors
x-airflow-common: &airflow-common
  image: image-pipeline:latest
  environment: &airflow-common-env
    AIRFLOW__CORE__EXECUTOR: CeleryExecutor
    AIRFLOW__DATABASE__SQL_ALCHEMY_CONN: postgresql+psycopg2://airflow:airflow@postgres/airflow
  volumes:
    - ../dags:/opt/airflow/dags   # Mount local dags/ into container
    - raw-data:/opt/airflow/data/raw

services:
  postgres:            # Airflow's metadata database
    image: postgres:15
    ...

  redis:               # Celery's message broker
    image: redis:7-alpine
    ...

  airflow-webserver:   # The UI
    <<: *airflow-common    # Merge shared config
    command: webserver
    ports:
      - "8080:8080"

  airflow-scheduler:   # Reads DAGs, schedules tasks
    <<: *airflow-common
    command: scheduler

  airflow-worker:      # Executes tasks
    <<: *airflow-common
    command: celery worker
```

**`<<: *airflow-common`** is a **YAML anchor merge** — it injects the shared config block at that point, DRY (Don't Repeat Yourself) for Compose files.

---

## Concept 5 — Volumes: Data Persistence

Containers are ephemeral — when a container stops, its filesystem changes are lost. **Volumes** provide persistent storage that survives container restarts.

```yaml
volumes:
  raw-data:       # Docker-managed named volume
  processed-data:
  model-cache:    # Model weights are large — persist them to avoid re-downloading

services:
  airflow-worker:
    volumes:
      # Bind mount: syncs local directory into container
      - ../dags:/opt/airflow/dags      # local/dags ↔ /opt/airflow/dags

      # Named volume: Docker manages the storage location
      - raw-data:/opt/airflow/data/raw
      - model-cache:/opt/airflow/model_cache
```

**Bind mount vs named volume:**

| Type | Definition | Use case |
|------|------------|----------|
| Bind mount | `./local/path:/container/path` | Development — edit code and see changes instantly |
| Named volume | `volume_name:/container/path` | Production data — managed by Docker, survives `docker-compose down` |

---

## Concept 6 — Health Checks and Service Dependencies

Services start in an order, but "started" doesn't mean "ready". Postgres may be up but still initializing. Health checks let you define what "ready" means.

```yaml
postgres:
  image: postgres:15
  healthcheck:
    test: ["CMD", "pg_isready", "-U", "airflow"]  # Command to check health
    interval: 5s    # Check every 5 seconds
    retries: 5      # Give up after 5 failures

airflow-webserver:
  depends_on:
    postgres:
      condition: service_healthy    # Wait until postgres passes its health check
    redis:
      condition: service_healthy
```

Without health checks, `airflow-webserver` would start before Postgres is ready to accept connections, causing it to crash with a "connection refused" error.

---

## Concept 7 — Environment Variables in Compose

Airflow's entire configuration can be controlled via environment variables using the `AIRFLOW__SECTION__KEY` naming convention.

```yaml
environment:
  # AIRFLOW__SECTION__KEY format
  AIRFLOW__CORE__EXECUTOR: CeleryExecutor
  # Equivalent to airflow.cfg: [core] executor = CeleryExecutor

  AIRFLOW__DATABASE__SQL_ALCHEMY_CONN: postgresql+psycopg2://user:pass@host/db
  # Equivalent to: [database] sql_alchemy_conn = ...

  AIRFLOW__CELERY__BROKER_URL: redis://:@redis:6379/0

  # Custom pipeline env vars
  PIPELINE_CAPTIONING_BACKEND: mock
  PIPELINE_MODEL_DEVICE: cpu
```

This means you can configure a complete Airflow deployment without ever editing `airflow.cfg`.

---

## Concept 8 — Celery Executor: Scaling Workers

Our Compose file uses **CeleryExecutor** — Airflow's distributed task execution mode.

```
SchedulerJob          ← Reads DAGs, determines what to run, when
     │  task to queue
     ▼
  Redis (broker)      ← Message queue: task slots waiting to be claimed
     │  task claimed
     ▼
CeleryWorker(s)       ← Execute the actual task functions
     │  result
     ▼
Postgres (backend)    ← Store task results and status
```

**Scaling workers horizontally:**

```bash
# Run 3 workers instead of 1
docker compose up --scale airflow-worker=3

# Each worker picks up tasks from Redis independently
# Tasks run in parallel across all workers
```

This is how you go from processing 100 images on one machine to 10,000 images across many machines — the same DAG code, just more workers.

---

## Summary

| Concept | Key Idea |
|---------|----------|
| Images vs containers | Image = snapshot (class); container = running instance (object) |
| Layer caching | Dependencies first, code last — avoid cache invalidation |
| `--no-cache-dir` | Smaller images by not storing pip download cache |
| YAML anchors (`&`, `*`, `<<:`) | DRY shared config across Compose services |
| Named volumes | Persistent storage for data and model weights |
| Health checks + `depends_on` | Wait for services to be truly ready, not just started |
| `AIRFLOW__SECTION__KEY` | Configure Airflow entirely via environment variables |
| CeleryExecutor | Scale workers horizontally by adding more containers |

---

## Exercises

**1. Basic** — Run `docker compose up -d` and open http://localhost:8080. Navigate to Admin → Variables and add a variable. Then add code to your DAG to read it with `from airflow.models import Variable; val = Variable.get("my_key")`.

**2. Intermediate** — Modify the `Dockerfile` to add a build argument `ARG CACHE_MODELS=false` and a conditional `RUN` that pre-downloads the BLIP model when `CACHE_MODELS=true`. Build with `docker build --build-arg CACHE_MODELS=true .` and measure image size difference.

**3. Advanced** — Add a second `airflow-worker` service in `docker-compose.yml` with different environment variables (e.g., `PIPELINE_CAPTIONING_BACKEND=blip`). Then modify the DAG to use `queue="gpu_workers"` on the `caption_images` task so it routes only to a designated worker. This is how you build heterogeneous worker pools.

---

*Next → [Module 08: Testing Data Pipelines](./module_08_testing.md)*
