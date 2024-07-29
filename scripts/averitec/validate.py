"""Verifies if the generated Averitec output file is complete and consistent."""

import json
from src.utils.utils import load_experiment_parameters
from src.tools.search.knowledge_base import N_CLAIMS
from src.eval.benchmark import AVeriTeC

path = "/pfss/mlde/workspaces/mlde_wsp_Rohrbach/users/mr74vahu/MAFC/out/averitec/2024-07-28_10-24_gpt_4o"

experiment_params = load_experiment_parameters(path)
averitec = AVeriTeC(**experiment_params["benchmark_kwargs"])

n_claims = N_CLAIMS[averitec.variant]

with open(path + "/averitec_out.json", "r") as f:
    results = json.load(f)
checked_claims = {result["claim_id"]: result["claim"] for result in results}
complete = consistent = True
for i in range(n_claims):
    if i not in checked_claims.keys():
        print(f"Claim {i} is missing!")
        complete = False
    elif checked_claims[i] != averitec.get_by_id(i)["content"].text:
        consistent = False
        print(f"Claim {i} is different from original claim!")

if complete:
    print("The output file is complete!")
if consistent:
    print("The output file is consistent!")
