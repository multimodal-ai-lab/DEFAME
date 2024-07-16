"""Shared configuration across all project code."""

import yaml

search_engine_options = {
    'google',
    'wiki_dump',
    'duckduckgo',
    'averitec_kb'
}
path_to_data = ""  # Where the datasets are stored
path_to_result = ""  # Where outputs are to be saved
embedding_model = "Alibaba-NLP/gte-base-en-v1.5"  # used for semantic search in FEVER and Averitec knowledge bases

api_keys = yaml.safe_load(open("config/api_keys.yaml"))
