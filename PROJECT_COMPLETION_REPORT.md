# Churn Prediction MLOps Project — Final Completion and Verification Report

**Project folder:** `Churn_Prediction-241FA18343`  
**Scope:** Labs 3–13  
**Reference:** `churn-prediction-full-project-files` and its `FILE_PLACEMENT_MAP.csv`  
**Audit date:** 6 November 2026

## 1. Result

The project has been consolidated into one clean project root and renamed to
`Churn_Prediction-241FA18343`.

The reference placement map contains 146 rows. One row is a malformed
placeholder (`}`), not a usable file path. All 145 valid mapped paths are
present in the consolidated project. The project also contains additional
working files and local MLflow run artifacts.

The older nested `churn-prediction/` copy and the previous `.git/` history
were excluded from the deliverable. The nested copy caused duplicate Python
test module names during test collection.

## 2. Verification performed in this audit

| Check | Result | Evidence / limitation |
|---|---|---|
| Reference file placement | PASS | 145 of 145 valid mapped paths found |
| Python syntax compilation | PASS | `compileall` completed with exit code 0 |
| Automated project tests | PASS | `40 passed` |
| API, model and performance tests | PASS | Included in the 40-test pytest run |
| YAML / JSON configuration review | See project files | Syntax and key paths reviewed; external services not exercised |
| Docker image build and container run | NOT RUN | Docker executable/engine unavailable in the audit environment |
| Kubernetes deployment and rollout | NOT RUN | Kubernetes cluster unavailable in the audit environment |
| GitHub Actions scheduled run | NOT VERIFIED | Requires the repository and configured self-hosted Windows runner |
| Fresh-batch automatic retraining | NOT RUN end-to-end | Pipeline code and artifacts are present; a new, unique batch must be supplied |

The pytest run emitted scikit-learn `InconsistentVersionWarning` messages:
the bundled serialized estimators report scikit-learn 1.9.0, while the audit
runtime had scikit-learn 1.8.0. The test suite passed, but use the pinned
project requirements to avoid estimator-version compatibility risk.

## 3. Lab coverage inventory

| Lab | Implementation present | Verification status |
|---|---|---|
| Lab 3 — Baseline model | Yes | Tests and artifacts present |
| Lab 4 — MLflow tracking | Yes | Scripts and MLflow artifacts present |
| Lab 5 — Preprocessing pipeline | Yes | Pipeline scripts and processed data present |
| Lab 6 — Model registry | Yes | Registry scripts and model artifacts present |
| Lab 7 — End-to-end pipeline | Yes | Orchestrator and stage scripts present |
| Lab 8 — CI workflow | Yes | Workflow and tests present |
| Lab 9 — Failure handling | Yes | Failure-injection scripts and reports present |
| Lab 10 — FastAPI serving | Yes | API source, deployment bundle and tests present |
| Lab 11 — Docker/Kubernetes | Yes | Dockerfile, deployment scripts and manifests present; live rollout remains environment-dependent |
| Lab 12 — Monitoring/retraining | Yes | Drift, trend, monitoring, retraining and registry scripts present |
| Lab 13 — Automation | Yes | Monitoring and runner workflows present; hosted run requires configured runner |

“Implementation present” means the corresponding project files and artifacts
exist. It does not mean every external deployment or scheduled workflow has
been executed in this audit.

## 4. Important project structure

- `api/` — FastAPI application, schemas, preprocessing and model loader
- `artifacts/` — validation, model and pipeline evidence
- `data/` — raw, processed and drift-monitoring datasets
- `deployment/` — production model bundle and deployment automation
- `docs/` — architecture, lab reports and runbooks
- `k8s/` — Kubernetes Deployment and Service manifests
- `models/` — trained and retrained model artifacts
- `notebooks/` — project notebooks
- `pipelines/` — Labs 3–6 entry points
- `reports/` — evaluation, quality, lifecycle and API reports
- `src/mlops_pipeline/` — Labs 7–13 pipeline stages
- `tests/` — API, pipeline and performance tests
- `.github/workflows/` — CI, failure-test and Lab 13 workflows

## 5. Run locally on Windows

Use Python 3.11 for consistency with the Docker image and GitHub Actions
workflow. Open PowerShell in the project root:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pytest -q
```

To run the main pipeline:

```powershell
python src\mlops_pipeline\churn_prediction_pipeline.py
```

To start the API after confirming `deployment/model.pkl`,
`deployment/preprocessor.pkl`, and `deployment/model_metadata.json` exist:

```powershell
python -m pip install -r requirements-api.txt
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

API health check: `http://127.0.0.1:8000/health`

## 6. Remaining environment-dependent steps

1. Install and start Docker Desktop, then build and run the Docker image.
2. Enable a Kubernetes cluster, apply `k8s/deployment.yaml` and
   `k8s/service.yaml`, and verify rollout and service connectivity.
3. Push the project to GitHub and configure a self-hosted Windows runner
   with Python, Docker and kubectl for Lab 13.
4. Place a genuinely new, unique customer batch at `data/raw/new_churn.csv`,
   then run the Lab 12/13 monitoring trigger and verify that ingestion,
   retraining, registry update and promotion/deployment gates are recorded.

## 7. Notes

- The ZIP keeps the project’s local `mlflow.db` and `mlruns/` artifacts.
  Treat these as local tracking state; a clean machine may need to configure
  its own MLflow tracking store.
- The bundled model artifacts were serialized with scikit-learn 1.9.0.
  Install the pinned requirements before loading them.
- Do not delete the raw data, processed data, deployment bundle, or model
  files merely to reduce ZIP size; several pipeline stages depend on them.
