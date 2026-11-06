import os
import sys
from pathlib import Path

import pandas as pd
import yaml

# Add this script's directory to path so validation module is importable
_DIR = Path(__file__).resolve().parent
if str(_DIR) not in sys.path:
    sys.path.insert(0, str(_DIR))

from validation import validate_schema


def main():
    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    data_path = config["data"]["raw_path"]

    if not os.path.exists(data_path):
        print(f"[ERROR] Target file not found at: {data_path}")
        sys.exit(1)

    raw_df = pd.read_csv(data_path)
    passed = validate_schema(raw_df, output_report_name="baseline_validation.csv")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
