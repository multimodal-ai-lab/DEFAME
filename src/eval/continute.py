from .evaluate import evaluate
from src.utils.utils import load_experiment_parameters


def continue_evaluation(experiment_dir: str):
    experiment_params = load_experiment_parameters(experiment_dir)
    experiment_params["continue_experiment_dir"] = experiment_dir
    evaluate(**experiment_params)
