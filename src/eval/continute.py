from .evaluate import evaluate

import yaml


def continue_evaluation(experiment_dir: str):
    config_path = experiment_dir + "config.yaml"
    with open(config_path, "r") as f:
        experiment_params = yaml.safe_load(f)
    experiment_params["continue_experiment_dir"] = experiment_dir
    evaluate(**experiment_params)
