from src.eval.benchmark import AVeriTeC
from src.eval.evaluate import finalize_evaluation
from src.utils.utils import load_experiment_parameters

experiment_dir = ""

experiment_params = load_experiment_parameters(experiment_dir)
averitec = AVeriTeC(**experiment_params["benchmark_kwargs"])
finalize_evaluation(experiment_dir, benchmark=averitec, duration=0)
