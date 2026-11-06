import json
import os
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[1]

class TestMLOpsPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.config_path = PROJECT_ROOT / "config.yaml"
        with open(cls.config_path, "r", encoding="utf-8") as f:
            cls.config = yaml.safe_load(f)

    def test_01_config_structure(self):
        """Validate config.yaml has all necessary sections and paths."""
        self.assertIn("data", self.config)
        self.assertIn("split", self.config)
        self.assertIn("model", self.config)
        self.assertIn("mlflow", self.config)
        self.assertIn("registry", self.config)
        self.assertIn("quality_gate", self.config)
        self.assertIn("paths", self.config)
        self.assertEqual(self.config["data"]["target_column"], "Churn")

    def test_02_raw_dataset_exists_and_schema(self):
        """Verify raw dataset exists, has required columns, and non-empty."""
        raw_path = PROJECT_ROOT / self.config["data"]["raw_path"]
        self.assertTrue(raw_path.exists(), f"Raw dataset not found at {raw_path}")
        df = pd.read_csv(raw_path)
        self.assertGreater(len(df), 7000)
        self.assertIn("customerID", df.columns)
        self.assertIn("Churn", df.columns)

    def test_03_processed_artifacts_exist(self):
        """Verify preprocessing artifacts (numpy arrays, raw splits, metadata) exist."""
        proc_dir = PROJECT_ROOT / self.config["data"]["processed_dir"]
        for fname in ["X_train_final.npy", "X_test_final.npy", "y_train.npy", "y_test.npy", "X_train_raw.csv", "X_test_raw.csv", "dataset_metadata.json"]:
            path = proc_dir / fname
            self.assertTrue(path.exists(), f"Missing processed artifact: {path}")

    def test_04_processed_data_integrity(self):
        """Verify processed matrices have no NaNs and matching dimensions."""
        proc_dir = PROJECT_ROOT / self.config["data"]["processed_dir"]
        X_train = np.load(proc_dir / "X_train_final.npy")
        X_test = np.load(proc_dir / "X_test_final.npy")
        y_train = np.load(proc_dir / "y_train.npy")
        y_test = np.load(proc_dir / "y_test.npy")

        self.assertEqual(np.isnan(X_train).sum(), 0, "X_train contains NaNs")
        self.assertEqual(np.isnan(X_test).sum(), 0, "X_test contains NaNs")
        self.assertEqual(X_train.shape[1], X_test.shape[1], "Feature dimension mismatch")
        self.assertEqual(X_train.shape[0], y_train.shape[0], "Train sample count mismatch")
        self.assertEqual(X_test.shape[0], y_test.shape[0], "Test sample count mismatch")

    def test_05_saved_model_and_preprocessor(self):
        """Ensure trained model and preprocessor serialize and deserialize properly."""
        model_path = PROJECT_ROOT / self.config["paths"]["model"]
        preproc_path = PROJECT_ROOT / self.config["paths"]["preprocessor"]
        self.assertTrue(model_path.exists(), "Model pickle does not exist")
        self.assertTrue(preproc_path.exists(), "Preprocessor pickle does not exist")

        model = joblib.load(model_path)
        preprocessor = joblib.load(preproc_path)
        self.assertTrue(hasattr(model, "predict"), "Model object missing predict method")
        self.assertTrue(hasattr(preprocessor, "transform"), "Preprocessor object missing transform method")

    def test_06_reports_and_logs(self):
        """Verify generated reports exist and contain valid results."""
        eval_report_path = PROJECT_ROOT / self.config["paths"]["evaluation_report"]
        quality_report_path = PROJECT_ROOT / self.config["paths"]["quality_gate_report"]
        self.assertTrue(eval_report_path.exists(), "Evaluation report missing")
        self.assertTrue(quality_report_path.exists(), "Quality gate report missing")

        with open(eval_report_path, "r", encoding="utf-8") as f:
            eval_data = json.load(f)
            self.assertIn("metrics", eval_data)
            self.assertGreater(eval_data["metrics"]["f1_score"], 0.5)

        with open(quality_report_path, "r", encoding="utf-8") as f:
            q_data = json.load(f)
            self.assertEqual(q_data["status"], "PASSED")

    def test_07_docs_and_workflow_files(self):
        """Verify all required documentation and git workflow files exist and non-empty."""
        for doc_file in [
            PROJECT_ROOT / "docs" / "LAB7_PIPELINE.md",
            PROJECT_ROOT / "docs" / "LAB8_CI_REPORT.md",
            PROJECT_ROOT / "docs" / "LAB9_FAILURE_HANDLING_REPORT.md",
            PROJECT_ROOT / "docs" / "MLFLOW_CI_SETUP.md",
            PROJECT_ROOT / "GIT_WORKFLOW.md",
            PROJECT_ROOT / ".github" / "workflows" / "lab8_ci.yml",
            PROJECT_ROOT / ".github" / "workflows" / "lab9_failure_tests.yml"
        ]:
            self.assertTrue(doc_file.exists(), f"Missing required file: {doc_file}")
            self.assertGreater(doc_file.stat().st_size, 50, f"File appears empty: {doc_file}")

    def test_08_additional_models_and_threshold(self):
        """Verify full pipeline, XGBoost pipeline, and decision threshold artifacts."""
        for mpath in [
            PROJECT_ROOT / "models" / "churn_full_pipeline.pkl",
            PROJECT_ROOT / "models" / "churn_xgboost_pipeline.pkl",
            PROJECT_ROOT / "models" / "churn_decision_threshold.pkl"
        ]:
            self.assertTrue(mpath.exists(), f"Missing model artifact: {mpath}")

        pipe_rf = joblib.load(PROJECT_ROOT / "models" / "churn_full_pipeline.pkl")
        self.assertTrue(hasattr(pipe_rf, "predict"), "churn_full_pipeline missing predict method")
        thresh = joblib.load(PROJECT_ROOT / "models" / "churn_decision_threshold.pkl")
        self.assertIn("optimal_threshold", thresh)

    def test_09_new_data_and_simulation_artifacts(self):
        """Verify new_churn.csv and corruption simulation outputs."""
        for path in [
            PROJECT_ROOT / "data" / "raw" / "new_churn.csv",
            PROJECT_ROOT / "artifacts" / "new_data_validation.csv",
            PROJECT_ROOT / "artifacts" / "corrupted_simulation_errors.csv",
            PROJECT_ROOT / "artifacts" / "failure_simulation_report.json",
            PROJECT_ROOT / "artifacts" / "pipeline_reproducibility_report.json"
        ]:
            self.assertTrue(path.exists(), f"Missing required artifact: {path}")

    def test_10_notebooks_validity(self):
        """Verify notebooks exist and contain valid JSON."""
        for nb_path in [
            PROJECT_ROOT / "notebooks" / "churn_prediction.ipynb",
            PROJECT_ROOT / "notebooks" / "project_implementation.ipynb"
        ]:
            self.assertTrue(nb_path.exists(), f"Notebook missing: {nb_path}")
            with open(nb_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.assertIn("cells", data)
                self.assertGreater(len(data["cells"]), 5)

    def test_11_outputs_and_reports(self):
        """Verify Lab 7 inference and error analysis outputs."""
        for out_file in [
            PROJECT_ROOT / "outputs" / "lab7_inference_predictions.csv",
            PROJECT_ROOT / "outputs" / "lab7_false_positives.csv",
            PROJECT_ROOT / "outputs" / "lab7_false_negatives.csv",
            PROJECT_ROOT / "logs" / "pipeline_run_log.json"
        ]:
            self.assertTrue(out_file.exists(), f"Missing output file: {out_file}")


if __name__ == "__main__":
    unittest.main()
