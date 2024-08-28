from .evaluate import evaluate
from infact.utils.utils import load_experiment_parameters


def continue_evaluation(experiment_dir: str, **kwargs):
    experiment_params = load_experiment_parameters(experiment_dir)
    experiment_params["continue_experiment_dir"] = experiment_dir
    if kwargs:
        experiment_params.update(kwargs)
    evaluate(**experiment_params)
