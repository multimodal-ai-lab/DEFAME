import numpy as np
import os
import time
from itertools import product
from safe.fact_checker import FactChecker
from eval.benchmark import AVeriTeC, FEVER
from common.console import green, red, bold
from eval.logging import setup_logging, log_model_config, log_testing_result, print_log
from common.shared_config import model_abbr

def evaluate(
        model, 
        multimodal_model, 
        search_engine, 
        benchmark, 
        extract_claims, 
        verbose = False, 
        logging = True
) -> float:
    ################################################################################
    #                                   LOGGING
    if logging:
        os.makedirs('log', exist_ok=True)
        dataset_abbr = benchmark.name
        model_ab = model_abbr[model]

        # Setup logging
        config_logger, testing_logger, print_logger = setup_logging(dataset_abbr, model_ab)

        log_model_config(config_logger, {
            "LLM": model, 
            "MLLM": multimodal_model, 
            "Search Engine": search_engine, 
            "Benchmark": benchmark.name,
            "Extract Claims": extract_claims,
            "Full Dataset": True
        })
        start_time = time.time()
    ################################################################################
    n = len(benchmark)
    print(f"Loaded {benchmark.name} containing {n} instances.")
    print(f"Evaluating: ")    

    fc = FactChecker(
        model=model, 
        multimodal_model=multimodal_model, 
        search_engine=search_engine, 
        extract_claims=extract_claims, 
    )
    predictions = []
    for i, instance in enumerate(benchmark):
        content = instance["content"]
        if logging:
            prediction = fc.check(content, verbose=verbose, logger=print_logger)
            log_message = {
            "sample_index": i + 1,
            "target": instance["label"].value,
            "predicted": prediction.value,
            "correct": instance["label"] == prediction
            }
            log_testing_result(testing_logger, log_message)
            if instance["label"] == prediction:
                print_log(print_logger, bold(green("CORRECT")))
            else:
                print_log(print_logger, bold(red("WRONG - Ground truth: " + instance["label"].value)))
        else:
            prediction = fc.check(content, verbose=verbose)

        predictions.append(prediction)
        if instance["label"] == prediction:
            print(bold(green("CORRECT")))
        else:
            print(bold(red("WRONG - Ground truth: " + instance["label"].value)))

        if len(predictions) == n:
            break

    # Compute metrics
    ground_truth = benchmark.get_labels()[:n]
    correct_predictions = np.array(predictions) == np.array(ground_truth)
    accuracy = np.sum(correct_predictions) / n
    print(f"Accuracy: {accuracy*100:.1f} %")

    ################################################################################
    #                                   LOGGING
    if logging:
        end_time = time.time()
        log_testing_result(testing_logger, {
            "Accuracy": f"{accuracy*100:.1f} %",
            "Correct Predictions": correct_predictions.tolist(),
            "Incorrect Predictions": (n - correct_predictions.sum()).tolist(),
            "Duration of Run": f'{end_time - start_time} seconds'
        })
    ################################################################################

    return accuracy


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