from defame.common.logger import Logger
from pathlib import Path
import json

experiment_dir = Path("")

averitec_out_path = experiment_dir / Logger.averitec_out_filename

with open(averitec_out_path, "r") as averitec_out_file:
    averitec_out = json.load(averitec_out_file)

averitec_out.sort(key=lambda x: x["claim_id"])

with open(averitec_out_path, "w") as averitec_out_file:
    json.dump(averitec_out, averitec_out_file, indent=4)
