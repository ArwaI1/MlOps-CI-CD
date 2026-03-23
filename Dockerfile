# ─────────────────────────────────────────────────────────────────────────────
# Dockerfile - Sign Language Classifier Model Server
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.10-slim

# Build-time arguments
ARG RUN_ID
ARG MLFLOW_TRACKING_URI
ARG MLFLOW_TRACKING_USERNAME
ARG MLFLOW_TRACKING_PASSWORD

# Expose them as env vars so mlflow CLI can read them
ENV RUN_ID=${RUN_ID}
ENV MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI}
ENV MLFLOW_TRACKING_USERNAME=${MLFLOW_TRACKING_USERNAME}
ENV MLFLOW_TRACKING_PASSWORD=${MLFLOW_TRACKING_PASSWORD}

WORKDIR /app

# Install MLflow and runtime deps
RUN pip install --no-cache-dir mlflow torch

# Download the model artifact from the MLflow tracking server.
# Real command: mlflow artifacts download --run-id <RUN_ID>
#               --artifact-path model      (path inside the run)
#               --dst-path /app/model      (local destination)
RUN if [ -n "${RUN_ID}" ] && [ -n "${MLFLOW_TRACKING_URI}" ]; then \
        echo "Downloading model for Run ID: ${RUN_ID} from ${MLFLOW_TRACKING_URI}" && \
        mlflow artifacts download \
            --run-id "${RUN_ID}" \
            --artifact-path model \
            --dst-path /app/model; \
    else \
        echo "RUN_ID or MLFLOW_TRACKING_URI not set - running in mock mode." && \
        mkdir -p /app/model && \
        echo "${RUN_ID}" > /app/model/run_id.txt; \
    fi

COPY . /app

# Default command - replace with your real inference server entrypoint
CMD ["python", "-c", "import os; print('Model server running. Run ID:', os.environ.get('RUN_ID', 'N/A'))"]
