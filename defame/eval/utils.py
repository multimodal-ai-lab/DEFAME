import os
from multiprocessing import set_start_method
from pathlib import Path

from defame.eval.evaluate import evaluate
from defame.utils.utils import load_config

set_start_method("spawn")


def run_experiment(config_file_path: str):
    """Runs the experiment as configured by the specified configuration file."""
    config_path = Path(config_file_path)
    experiment_params = load_config(config_path)
    evaluate(experiment_name=config_path.stem, **experiment_params)


def run_multiple_experiments(config_dir: str):
    """Runs multiple experiments sequentially as configured by the configuration files inside
    the specified directory. Deletes each config file after it was read."""
    config_dir = Path(config_dir)

    while configs := os.listdir(config_dir):
        config_path = config_dir / configs[0]
        experiment_params = load_config(config_path)
        evaluate(experiment_name=config_path.stem,
                 **experiment_params)
        os.remove(config_path)

    print("Finished! No more experiments to run.")
