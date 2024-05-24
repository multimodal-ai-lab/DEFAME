import numpy as np
import os
import time
from itertools import product
from safe.fact_checker import FactChecker
from eval.benchmark import AVeriTeC, FEVER
from common.console import green, red, bold
from eval.logging import setup_logging, log_model_config, log_testing_result, print_log
from common.shared_config import model_abbr
from evaluate import evaluate


hyperparameters = {'model': ["huggingface:meta-llama/Meta-Llama-3-70B-Instruct"],
                   'multimodal_model': ["huggingface:llava-hf/llava-1.5-7b-hf"],
                   'search_engine': ["duckduck", "google"],
                   'benchmark' : [AVeriTeC("dev")],
                   'extract_claims': [True, False],
}

combinations = product(
    hyperparameters['model'],
    hyperparameters['multimodal_model'],
    hyperparameters['search_engine'],
    hyperparameters['benchmark'],
    hyperparameters['extract_claims']
)

if __name__ == "__main__":
    for combination in combinations:
        evaluate(*combination)