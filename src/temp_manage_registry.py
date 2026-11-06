import os
import sys
from pathlib import Path
import mlflow
from mlflow.tracking import MlflowClient
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def configure_mlflow():
    config_path = PROJECT_ROOT / "config.yaml"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI") or config["mlflow"].get("tracking_uri")
    else:
        tracking_uri = "sqlite:///mlflow.db"

    mlflow.set_tracking_uri(tracking_uri)
    return tracking_uri


def inspect_registry(model_names=None):
    tracking_uri = configure_mlflow()
    client = MlflowClient()
    print("[INFO] ==================================================")
    print(f"[INFO] MLflow Model Registry Inspector (Tracking: {tracking_uri})")
    print("[INFO] ==================================================")

    if model_names is None:
        model_names = ["CustomerChurnClassifier", "Telco_Churn_Production_Model"]

    for name in model_names:
        print(f"\nModel: '{name}'")
        print("-" * 50)
        try:
            versions = client.search_model_versions(f"name='{name}'")
            if not versions:
                print("  No registered versions found.")
                continue

            for v in versions:
                print(
                    f"  Version: {v.version:2} | Stage: {v.current_stage:10} | "
                    f"Run ID: {v.run_id[:12]}... | Status: {v.status}"
                )
        except Exception as exc:
            print(f"  Error querying model '{name}': {exc}")

    print("\n[SUCCESS] Model registry inspection complete.")
    return True


if __name__ == "__main__":
    os.chdir(PROJECT_ROOT)
    inspect_registry()
