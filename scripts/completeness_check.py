"""Verifies if the generated Averitec output file is complete."""

import json

N_CLAIMS = 2215

path = "/pfss/mlde/workspaces/mlde_wsp_Rohrbach/users/mr74vahu/MAFC/out/averitec/2024-07-28_10-24_gpt_4o"

with open(path + "/averitec_out.json", "r") as f:
    results = json.load(f)

claim_ids = [result["claim_id"] for result in results]
complete = True
for i in range(N_CLAIMS):
    if i not in claim_ids:
        print(f"Claim {i} is missing!")
        complete = False

if complete:
    print("The output file is complete!")
