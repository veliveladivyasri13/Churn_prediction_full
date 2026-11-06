# Lab 9: Failure Injection and Recovery Testing Report

## 1. Executive Summary

Lab 9 evaluates the resilience and self-healing capabilities of the Customer Churn Prediction MLOps pipeline. Controlled faults and catastrophic failures are systematically injected into data files, schema configurations, preprocessing artifacts, and model files to verify that:
1. Failures are detected immediately before corrupting downstream processes.
2. Informative diagnostic logs and failure reports are generated.
3. Automated recovery restores the pipeline to operational status without human intervention.
4. Results are consistent across repeated verification runs (3 full repetitions).

---

## 2. Tested Scenarios & Real Execution Results

Execution of `src/mlops_pipeline/10_failure_injection.py` evaluates 5 distinct scenarios across 3 repetitions:

| # | Failure Scenario | Injected Condition | Expected Behavior | Actual Outcome | Status |
|---|---|---|---|---|---|
| **1** | **Missing Data** | `data/raw/churn.csv` temporarily unlinked | Validation halts gracefully with descriptive error | File absence caught; exit code non-zero | **PASSED** (3/3 runs) |
| **2** | **Invalid Datatype** | Non-numeric string inserted into `MonthlyCharges` | Pandera schema check detects type violation | Failure logged to `artifacts/baseline_validation.csv`; pipeline halted | **PASSED** (3/3 runs) |
| **3** | **Schema Mismatch** | Required column `Contract` removed | Pandera strict schema check flags missing column | Failure case logged; pipeline halted | **PASSED** (3/3 runs) |
| **4** | **Preprocessing Inconsistency** | Synthetic `NaN` injected into `X_train_final.npy` | `04_validate_outputs.py` detects NaN | Error recorded in `artifacts/preprocessing_summary_report.json`; execution halted | **PASSED** (3/3 runs) |
| **5** | **Missing Model + Recovery** | Local model `models/churn_model.pkl` deleted | Evaluation fails, triggers recovery from MLflow Registry, and re-evaluates | Model recovered from `models:/CustomerChurnClassifier/Production`; re-evaluation succeeded | **PASSED** (3/3 runs) |

---

## 3. Failure Report Evidence

The tests output structured diagnostic reports to `reports/failure_report.json`:
- **Total Repetitions**: 3
- **Scenarios Evaluated Per Repetition**: 5
- **Total Scenario Runs**: 15
- **Passed Scenario Runs**: 15
- **Overall Status**: **PASSED**

---

## 4. Execution Commands

From the target project root `C:\MLOPS\Churn_Prediction-241FA18343`:

```powershell
# Run the complete Lab 9 failure injection suite:
& "C:\MLOPS\.venv\Scripts\python.exe" src\mlops_pipeline\10_failure_injection.py

# Inspect results:
Get-Content reports\failure_report.json
```
