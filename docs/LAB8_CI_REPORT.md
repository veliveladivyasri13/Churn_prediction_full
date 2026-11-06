# Lab 8: Continuous Integration (CI) Report

## 1. Objective

Lab 8 integrates Continuous Integration (CI) into the Customer Churn Prediction project via GitHub Actions. Every push or pull request touching core pipeline code, data schemas, or configurations triggers an automated workflow that runs the complete MLOps pipeline and archives generated artifacts.

---

## 2. CI Workflow Specification

The workflow is defined in `.github/workflows/lab8_ci.yml`.

### Triggers:
- Push to branches: `main`, `master`
- Pull requests targeting `main`, `master`
- Monitored paths:
  - `config.yaml`
  - `src/mlops_pipeline/**`
  - `requirements.txt`
  - `data/raw/churn.csv`
  - `.github/workflows/lab8_ci.yml`
- Manual trigger: `workflow_dispatch`

### Execution Matrix & Steps:
1. **Runner Environment**: `ubuntu-latest`
2. **Repository Checkout**: `actions/checkout@v4`
3. **Python Runtime Setup**: `actions/setup-python@v5` (Python 3.11)
4. **Dependency Installation**: `python -m pip install -r requirements.txt`
5. **Pipeline Execution**: `python src/mlops_pipeline/churn_prediction_pipeline.py`
6. **Artifact Archival**: `actions/upload-artifact@v4` packaging `artifacts/**`, `reports/**`, `logs/**`, `outputs/**`, `models/**`, `mlruns/**`.

---

## 3. Local Verification of CI Commands

To guarantee workflow reliability before remote execution, all individual commands were validated locally in the project environment:

| Step / Command | Local Execution Output | Status |
|---|---|---|
| `python -m pip install -r requirements.txt` | All dependencies satisfied | **PASSED** |
| `python src/mlops_pipeline/churn_prediction_pipeline.py` | 10/10 stages completed successfully | **PASSED** |
| Artifact validation | Reports and models generated in `reports/`, `artifacts/`, `models/` | **PASSED** |

### Local vs Remote Execution Notes:
- **Local Run**: Validated on Windows 11 PowerShell using local SQLite tracking (`sqlite:///mlflow.db`).
- **Remote CI Run**: Uses repository secrets (`${{ secrets.MLFLOW_TRACKING_URI }}`) when connecting to an external tracking server, falling back to local SQLite inside the container runner.
- **Path Portability**: All scripts utilize `pathlib.Path` and relative path resolutions, ensuring 100% cross-platform compatibility between local Windows and remote Linux runners.
