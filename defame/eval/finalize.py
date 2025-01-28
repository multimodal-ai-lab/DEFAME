from .evaluate import finalize_evaluation as finalize
from defame.utils.utils import load_experiment_parameters
from defame.eval import load_benchmark


def finalize_evaluation(experiment_dir: str):
    experiment_params = load_experiment_parameters(experiment_dir)
    benchmark_name = experiment_params["benchmark_name"]
    benchmark_kwargs = experiment_params["benchmark_kwargs"]
    benchmark = load_benchmark(benchmark_name, **benchmark_kwargs)
    finalize(experiment_dir, benchmark)
