#!/usr/bin/env bash
set -euo pipefail

BACKEND_STORE_URI="${MLFLOW_BACKEND_STORE_URI:-sqlite:///mlflow.db}"
HOST="${MLFLOW_HOST:-127.0.0.1}"
PORT="${MLFLOW_PORT:-5000}"

echo "Launching MLflow UI"
echo "Backend store: ${BACKEND_STORE_URI}"
echo "URL: http://${HOST}:${PORT}"

mlflow ui \
  --backend-store-uri "${BACKEND_STORE_URI}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --allowed-hosts "*" \
  --cors-allowed-origins "*"