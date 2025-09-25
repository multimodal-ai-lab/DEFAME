import yaml
from pathlib import Path

# Build the path to 'api_keys.yaml' in the project root directory
# Path(__file__) is this file: .../DEFAME/defame/config.py
# .parent is the 'defame' directory
# .parent is the project root 'DEFAME'
config_path = Path(__file__).parent.parent / 'api_keys.yaml'

api_keys = {}
if config_path.is_file():
    with open(config_path, 'r') as f:
        api_keys = yaml.safe_load(f) or {}
else:
    # If the file doesn't exist, api_keys will be an empty dict,
    # preventing crashes on import.
    print(f"Warning: Configuration file not found at {config_path}")
