"""Shared configuration across all project code."""

import yaml

data_base_dir = ""  # Where the datasets are stored
result_base_dir = ""  # Where outputs are to be saved
embedding_model = "Alibaba-NLP/gte-base-en-v1.5"  # used for semantic search in FEVER and Averitec knowledge bases

api_keys = yaml.safe_load(open("config/api_keys.yaml"))
