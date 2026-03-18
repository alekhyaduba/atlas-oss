# ============================================================
# Image Caption & Tag Pipeline – Docker Image
# Base: Apache Airflow 2.8 with Python 3.11
# ============================================================

FROM apache/airflow:2.8.1-python3.11

# Switch to root for system-level installs
USER root

# ── System dependencies ──────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        # OpenCV runtime libs
        libgl1-mesa-glx \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libgomp1 \
        # Java for Spark
        default-jdk-headless \
        # Utilities
        curl \
        git \
    && rm -rf /var/lib/apt/lists/*

# ── Java / Spark env ─────────────────────────────────────────
ENV JAVA_HOME=/usr/lib/jvm/default-java
ENV PATH="${JAVA_HOME}/bin:${PATH}"

# ── Python deps ──────────────────────────────────────────────
USER airflow

COPY --chown=airflow:root requirements.txt /opt/airflow/requirements.txt

RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r /opt/airflow/requirements.txt

# ── Project files ─────────────────────────────────────────────
COPY --chown=airflow:root config/   /opt/airflow/config/
COPY --chown=airflow:root scripts/  /opt/airflow/scripts/
COPY --chown=airflow:root dags/     /opt/airflow/dags/

# ── Data directories ─────────────────────────────────────────
RUN mkdir -p \
    /opt/airflow/data/raw \
    /opt/airflow/data/processed \
    /opt/airflow/data/versioned \
    /opt/airflow/model_cache \
    /opt/airflow/dvc_store

# Pre-download HuggingFace models at build time (optional).
# Comment this out to download lazily at runtime instead.
# ARG CACHE_MODELS=false
# RUN if [ "$CACHE_MODELS" = "true" ]; then \
#       python -c "from transformers import BlipProcessor, BlipForConditionalGeneration; \
#                  BlipProcessor.from_pretrained('Salesforce/blip-image-captioning-base', cache_dir='/opt/airflow/model_cache'); \
#                  BlipForConditionalGeneration.from_pretrained('Salesforce/blip-image-captioning-base', cache_dir='/opt/airflow/model_cache')"; \
#     fi

# ── Environment ──────────────────────────────────────────────
ENV PIPELINE_BASE_DIR=/opt/airflow
ENV PIPELINE_CONFIG_PATH=/opt/airflow/config/run_config.yaml
ENV TRANSFORMERS_CACHE=/opt/airflow/model_cache
ENV HF_HOME=/opt/airflow/model_cache
ENV PYTHONPATH="/opt/airflow:${PYTHONPATH}"

# Airflow uses its own entrypoint; no CMD needed.
