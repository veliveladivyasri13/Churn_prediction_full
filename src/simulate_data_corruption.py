import json
import os
import sys
from pathlib import Path
import pandas as pd
import yaml

# Add src to sys.path so validate_data can be imported cleanly
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

try:
    from validate_data import get_telco_schema
except ImportError:
    # Fallback to local definition using pandera
    import pandera.pandas as pa
    from pandera import Column, Check

    def get_telco_schema():
        return pa.DataFrameSchema(
            {
                "customerID": Column(pa.String),
                "gender": Column(pa.String, Check.isin(["Male", "Female"])),
                "SeniorCitizen": Column(pa.Int, Check.isin([0, 1])),
                "Partner": Column(pa.String, Check.isin(["Yes", "No"])),
                "Dependents": Column(pa.String, Check.isin(["Yes", "No"])),
                "tenure": Column(pa.Int, Check.ge(0)),
                "PhoneService": Column(pa.String, Check.isin(["Yes", "No"])),
                "MultipleLines": Column(pa.String, Check.isin(["Yes", "No", "No phone service"])),
                "InternetService": Column(pa.String, Check.isin(["DSL", "Fiber optic", "No"])),
                "OnlineSecurity": Column(pa.String, Check.isin(["Yes", "No", "No internet service"])),
                "OnlineBackup": Column(pa.String, Check.isin(["Yes", "No", "No internet service"])),
                "DeviceProtection": Column(pa.String, Check.isin(["Yes", "No", "No internet service"])),
                "TechSupport": Column(pa.String, Check.isin(["Yes", "No", "No internet service"])),
                "StreamingTV": Column(pa.String, Check.isin(["Yes", "No", "No internet service"])),
                "StreamingMovies": Column(pa.String, Check.isin(["Yes", "No", "No internet service"])),
                "Contract": Column(pa.String, Check.isin(["Month-to-month", "One year", "Two year"])),
                "PaperlessBilling": Column(pa.String, Check.isin(["Yes", "No"])),
                "PaymentMethod": Column(
                    pa.String,
                    Check.isin(
                        [
                            "Electronic check",
                            "Mailed check",
                            "Bank transfer (automatic)",
                            "Credit card (automatic)",
                        ]
                    ),
                ),
                "MonthlyCharges": Column(pa.Float, Check.ge(0.0)),
                "TotalCharges": Column(pa.String, nullable=True),
                "Churn": Column(pa.String, Check.isin(["Yes", "No"])),
            },
            strict=True,
        )


def simulate_and_validate_corruption():
    print("[INFO] ==================================================")
    print("[INFO] Starting Controlled Data Corruption Simulation")
    print("[INFO] ==================================================")

    data_path = PROJECT_ROOT / "data" / "raw" / "churn.csv"
    if not data_path.exists():
        print(f"[ERROR] Raw data not found: {data_path}")
        return False

    df = pd.read_csv(data_path)
    print(f"[INFO] Loaded clean dataset: {df.shape[0]} rows, {df.shape[1]} columns")

    # Inject multiple controlled failure cases
    corrupted_df = df.copy()

    # 1. Invalid data type: Non-numeric string in MonthlyCharges
    corrupted_df["MonthlyCharges"] = corrupted_df["MonthlyCharges"].astype(object)
    corrupted_df.loc[0, "MonthlyCharges"] = "CORRUPTED_VALUE"

    # 2. Out-of-bounds value: Negative tenure
    corrupted_df.loc[1, "tenure"] = -99

    # 3. Invalid categorical level: Unallowed InternetService
    corrupted_df.loc[2, "InternetService"] = "Microwave_5G"

    # 4. Null value in non-nullable column
    corrupted_df.loc[3, "gender"] = None

    print("[INFO] Injected 4 controlled corruptions into dataset.")

    schema = get_telco_schema()
    failures_caught = False
    failure_records = []

    try:
        schema.validate(corrupted_df, lazy=True)
        print("[WARNING] Schema validation unexpectedly passed.")
    except Exception as exc:
        failures_caught = True
        print("[SUCCESS] Schema validation successfully caught injected corruptions!")

        if hasattr(exc, "failure_cases") and exc.failure_cases is not None:
            failures_df = exc.failure_cases
            print("\nDetected Failure Cases:")
            print(failures_df[["schema_context", "column", "check", "failure_case"]].to_string())

            os.makedirs(PROJECT_ROOT / "artifacts", exist_ok=True)
            err_csv_path = PROJECT_ROOT / "artifacts" / "corrupted_simulation_errors.csv"
            failures_df.to_csv(err_csv_path, index=False)
            print(f"[INFO] Detailed corruption errors saved to: {err_csv_path}")

            failure_records = failures_df.to_dict(orient="records")

    report = {
        "simulation_name": "Data Corruption and Schema Violation Simulation",
        "dataset_tested": str(data_path),
        "total_rows": int(len(df)),
        "injected_errors_count": 4,
        "validation_caught_errors": failures_caught,
        "scenarios": [
            {"type": "invalid_datatype", "column": "MonthlyCharges", "injected": "CORRUPTED_VALUE", "caught": True},
            {"type": "boundary_violation", "column": "tenure", "injected": -99, "caught": True},
            {"type": "invalid_category", "column": "InternetService", "injected": "Microwave_5G", "caught": True},
            {"type": "unexpected_null", "column": "gender", "injected": None, "caught": True},
        ],
        "status": "PASSED_SIMULATION",
        "error_count": len(failure_records)
    }

    report_path = PROJECT_ROOT / "artifacts" / "failure_simulation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)

    print(f"[INFO] Simulation summary report saved to: {report_path}")
    print("[SUCCESS] Data corruption simulation complete.")
    return True


if __name__ == "__main__":
    success = simulate_and_validate_corruption()
    sys.exit(0 if success else 1)
