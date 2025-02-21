from .evaluate import evaluate
from defame.utils.utils import load_experiment_parameters


def continue_evaluation(experiment_dir: str, **kwargs):
    experiment_params = load_experiment_parameters(experiment_dir)
    experiment_params["continue_experiment_dir"] = experiment_dir
    if kwargs:
        experiment_params.get_status(kwargs)
    evaluate(**experiment_params)
