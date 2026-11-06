import sys
from mlflow import MlflowClient

client = MlflowClient()

model_name = "Telco_Churn_Production_Model"

if len(sys.argv) > 1:
    version = str(sys.argv[1])
else:
    # Default to current Production version
    prod_versions = [v for v in client.search_model_versions(f"name='{model_name}'") if v.current_stage == "Production"]
    version = prod_versions[0].version if prod_versions else "1"

mv = client.get_model_version(model_name, version)

print("MODEL NAME:", mv.name)
print("VERSION:", mv.version)
print("STAGE:", mv.current_stage)
print("RUN ID:", mv.run_id)
print("SOURCE:", mv.source)
print("STATUS:", mv.status)
print("CREATION TIMESTAMP:", mv.creation_timestamp)
