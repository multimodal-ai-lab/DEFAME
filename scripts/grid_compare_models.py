from itertools import product
from eval.benchmark import AVeriTeC
from eval.evaluate import evaluate


hyperparameters = {'model': ["huggingface:meta-llama/Meta-Llama-3-70B-Instruct"],
                   'multimodal_model': ["huggingface:llava-hf/llava-1.5-7b-hf"],
                   'search_engine': ["duckduck", "google"],
                   'benchmark' : [AVeriTeC("dev")],
                   'n': [None],
                   'extract_claims': [True, False],
}

combinations = product(
    hyperparameters['model'],
    hyperparameters['multimodal_model'],
    hyperparameters['search_engine'],
    hyperparameters['benchmark'],
    hyperparameters['n'],
    hyperparameters['extract_claims']
)

if __name__ == "__main__":
    for combination in combinations:
        evaluate(*combination)
