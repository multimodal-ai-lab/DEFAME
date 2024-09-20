"""Shared configuration across all project code."""

import yaml

# Directories
data_base_dir = ""  # Where the datasets are stored
result_base_dir = ""  # Where outputs are to be saved
temp_dir = ""  # Where caches etc. are saved

embedding_model = "Alibaba-NLP/gte-base-en-v1.5"  # used for semantic search in FEVER and Averitec knowledge bases
manipulation_detection_model = "/pfss/mlde/workspaces/mlde_wsp_Rohrbach/users/tb17xity/InFact/third_party/TruFor/weights/trufor.pth.tar"

api_keys = yaml.safe_load(open("config/api_keys.yaml"))
