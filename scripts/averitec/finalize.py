from eval.averitec.benchmark import AVeriTeC
from infact.eval.evaluate import finalize_evaluation
from infact.utils.utils import load_experiment_parameters

experiment_dir = ""

experiment_params = load_experiment_parameters(experiment_dir)
averitec = AVeriTeC(**experiment_params["benchmark_kwargs"])
finalize_evaluation(experiment_dir, benchmark=averitec)
