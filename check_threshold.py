"""
check_threshold.py
------------------
Reads the MLflow Run ID from model_info.txt, queries the MLflow tracking
server for the final 'accuracy' metric, and exits with a non-zero code
(failing the pipeline) if accuracy < THRESHOLD.
"""

import mlflow
import sys
import os

THRESHOLD = 0.85


def main():
    # 1. Read the Run ID produced by train.py
    run_id_file = "model_info.txt"
    if not os.path.exists(run_id_file):
        print("ERROR: model_info.txt not found. Did the validate job succeed?")
        sys.exit(1)

    with open(run_id_file, "r") as f:
        run_id = f.read().strip()

    print(f"Run ID: {run_id}")

    # 2. Fetch the run from MLflow
    client = mlflow.tracking.MlflowClient()

    try:
        run = client.get_run(run_id)
    except Exception as exc:
        print(f"ERROR: Could not fetch run from MLflow: {exc}")
        sys.exit(1)

    # 3. Extract the accuracy metric
    metrics = run.data.metrics
    accuracy = metrics.get("accuracy")

    if accuracy is None:
        print("ERROR: Metric 'accuracy' not found in this run.")
        sys.exit(1)

    print(f"Accuracy reported by MLflow: {accuracy:.4f}")
    print(f"Required threshold:          {THRESHOLD:.4f}")

    # 4. Gate on threshold
    if accuracy < THRESHOLD:
        print(
            f"\nDEPLOYMENT BLOCKED: accuracy {accuracy:.4f} "
            f"is below threshold {THRESHOLD}."
        )
        sys.exit(1)

    print(
        f"\nAccuracy {accuracy:.4f} meets the threshold. "
        "Proceeding with deployment."
    )


if __name__ == "__main__":
    main()
