"""Verifies if the generated Averitec output file is complete and consistent."""

import json
from defame.utils.utils import load_experiment_parameters
from defame.evidence_retrieval.integrations.search.knowledge_base import N_CLAIMS
from defame.eval.averitec.benchmark import AVeriTeC

path = ""

experiment_params = load_experiment_parameters(path)
averitec = AVeriTeC(**experiment_params["benchmark_kwargs"])

n_claims = N_CLAIMS[averitec.variant]

with open(path + "/averitec_out_no_scraped.json", "r") as f:
    results = json.load(f)
checked_claims = {result["claim_id"]: result["claim"] for result in results}
complete = consistent = True
for i in range(n_claims):
    if i not in checked_claims.keys():
        print(f"Claim {i} is missing!")
        complete = False
    elif checked_claims[i] != str(averitec.get_by_id(str(i))["input"]):
        consistent = False
        print(f"Claim {i} is different from original claim!")

if complete:
    print("✔ The output file is complete.")
if consistent:
    print("✔ The output file is consistent.")
claim_ids = list(checked_claims.keys())
if sorted(claim_ids) == claim_ids:
    print("✔ The output file is sorted.")
