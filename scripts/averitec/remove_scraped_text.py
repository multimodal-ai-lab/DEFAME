from defame.common.logger import Logger
from pathlib import Path
import json

experiment_dir = Path("")

averitec_out_path = experiment_dir / Logger.averitec_out_filename

with open(averitec_out_path, "r") as f:
    averitec_out = json.load(f)

for instance in averitec_out:
    evidence = instance["evidence"]
    for qa_pair in evidence:
        qa_pair["scraped_text"] = ""

# Modify path
new_averitec_out_path = averitec_out_path.with_stem("averitec_out_no_scraped")

with open(new_averitec_out_path, "w") as f:
    json.dump(averitec_out, f, indent=4)
