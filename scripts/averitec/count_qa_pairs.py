from defame.common.logger import Logger
from pathlib import Path
import json

experiment_dir = Path("")

averitec_out_path = experiment_dir / Logger.averitec_out_filename

with open(averitec_out_path, "r") as f:
    averitec_out = json.load(f)

qa_pair_count = 0
for instance in averitec_out:
    evidence = instance["evidence"]
    qa_pair_count += len(evidence)

print(f"{qa_pair_count} QA pairs counted.")
