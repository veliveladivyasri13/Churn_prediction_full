# MLflow CI/CD Configuration & Remote Tracking Guide

## 1. Overview

This document details the configuration of **MLflow Tracking & Model Registry** across local developer environments and remote Continuous Integration (CI) workflows.

---

## 2. Tracking Backends

### A. Local Development (Default)
In local development, the project uses a self-contained SQLite database:
```yaml
# config.yaml
mlflow:
  tracking_uri: "sqlite:///mlflow.db"
  experiment_name: "Customer_Churn_Prediction"
```
- **Backend Store**: SQLite (`mlflow.db`)
- **Artifact Store**: Local directory (`mlruns/`)
- **UI Launch Command**:
  ```powershell
  mlflow ui --backend-store-uri sqlite:///mlflow.db
  ```

### B. Continuous Integration (CI / Production)
In CI workflows (`.github/workflows/lab8_ci.yml` and `lab9_failure_tests.yml`), MLflow dynamically respects the `MLFLOW_TRACKING_URI` environment variable:
```python
tracking_uri = os.getenv("MLFLOW_TRACKING_URI") or config["mlflow"].get("tracking_uri")
if tracking_uri:
    mlflow.set_tracking_uri(tracking_uri)
```

If an external tracking server (e.g. hosted on AWS EC2, Azure ML, or DagsHub) is configured:
1. Navigate to your GitHub repository: **Settings > Secrets and variables > Actions**.
2. Add a new secret named `MLFLOW_TRACKING_URI` (e.g., `https://mlflow.example.com`).
3. Add authentication secrets if applicable:
   - `MLFLOW_TRACKING_USERNAME`
   - `MLFLOW_TRACKING_PASSWORD`
4. The workflow passes these securely:
   ```yaml
   env:
     MLFLOW_TRACKING_URI: ${{ secrets.MLFLOW_TRACKING_URI }}
     MLFLOW_TRACKING_USERNAME: ${{ secrets.MLFLOW_TRACKING_USERNAME }}
     MLFLOW_TRACKING_PASSWORD: ${{ secrets.MLFLOW_TRACKING_PASSWORD }}
   ```

If no secret is set, the CI job automatically falls back to local SQLite within the GitHub Actions runner, ensuring tests always pass cleanly without external network dependencies.

---

## 3. Model Registry Integration

The automated lifecycle manager (`src/mlops_pipeline/08_automate_lifecycle.py`) handles model promotion programmatically:
- Newly registered versions start with stage `None`.
- The script transitions candidate models to `Staging`.
- The challenger model's primary metric (`f1_score`) is compared against the active `Production` model.
- If superior, the challenger is promoted to `Production`, and previous versions are transitioned to `Archived`.
