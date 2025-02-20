"""Copies and zips all instances from an experiment that match the
given criteria."""

from pathlib import Path

import pandas as pd
import shutil

# Source and target directory
EXPERIMENT_DIR = "path/to/input/dir"
OUT_DIR = Path("out/exports")

# Filter criteria
PREDICTION_IS_CORRECT = False


experiment_name = Path(EXPERIMENT_DIR).name
OUT_DIR /= experiment_name
OUT_DIR.mkdir(parents=True, exist_ok=True)

fc_dir = Path(EXPERIMENT_DIR) / "fact-checks"
predictions_path = Path(EXPERIMENT_DIR) / "predictions.csv"
df = pd.read_csv(predictions_path)

filtered_ids = df["sample_index"]

if PREDICTION_IS_CORRECT is not None:
    is_correct = df["correct"]
    filtered_ids = df[is_correct == PREDICTION_IS_CORRECT]["sample_index"]

fc_dirs = [fc_dir / f"{idx}" for idx in filtered_ids]

# Copy each filtered fact-check directory to the target dir
for fc_dir in fc_dirs:
    shutil.copytree(fc_dir, OUT_DIR / "raw" / fc_dir.name)

# Zip all the dirs
shutil.make_archive((OUT_DIR / experiment_name).as_posix(), 'zip', OUT_DIR / "raw")
