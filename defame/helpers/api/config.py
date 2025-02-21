import yaml
from pathlib import Path

config_path = Path(__file__).parent / "config.yaml"  # must lie in same folder
config = yaml.safe_load(open(config_path))

host = config["host"]
port = config["port"]
api_key = config["api_key"]
save_dir = Path(config["save_dir"])
fact_checker_kwargs = config["fact_checker"]
