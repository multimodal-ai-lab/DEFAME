"""Shared configuration across all project code."""

import yaml
from pathlib import Path

# Directories
working_dir = Path.cwd() # working_dir should be DEFAME
data_base_dir = ""  # Where the datasets are stored
result_base_dir = working_dir / "out/"  # Where outputs are to be saved
temp_dir = result_base_dir / "temp/" # Where caches etc. are saved

embedding_model = "Alibaba-NLP/gte-base-en-v1.5"  # used for semantic search in FEVER and Averitec knowledge bases
manipulation_detection_model = working_dir / "third_party/TruFor/weights/trufor.pth.tar" 

api_keys = yaml.safe_load(open("config/api_keys.yaml"))

random_seed = 42 # used for sub-sampling in partial dataset testing
