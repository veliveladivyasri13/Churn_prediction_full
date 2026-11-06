import sys
from mlflow import MlflowClient

client = MlflowClient()

if len(sys.argv) > 1:
    run_id = sys.argv[1]
else:
    # Default to current Production model version's run_id, or first available run
    try:
        prod_versions = [v for v in client.search_model_versions("name='Telco_Churn_Production_Model'") if v.current_stage == "Production"]
        if prod_versions:
            run_id = prod_versions[0].run_id
        else:
            run_id = "a6e6705dfc7f40f7aab961c9bd5fa741"
    except Exception:
        run_id = "a6e6705dfc7f40f7aab961c9bd5fa741"

run = client.get_run(run_id)

print("RUN ID:", run.info.run_id)
print("STATUS:", run.info.status)
print("ARTIFACT URI:", run.info.artifact_uri)

print("\nARTIFACTS:")

artifacts = client.list_artifacts(run_id)

for artifact in artifacts:
    print(f"{artifact.path} | is_dir={artifact.is_dir}")

    if artifact.is_dir:
        children = client.list_artifacts(run_id, artifact.path)

        for child in children:
            print(f"  |-- {child.path} | is_dir={child.is_dir}")
