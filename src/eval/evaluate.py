import time

import numpy as np

from common.console import green, red, bold, gray
from common.shared_config import model_abbr
from eval.benchmark import load_benchmark
from eval.logging import EvaluationLogger
from safe.fact_checker import FactChecker


# TODO The following comments should be inserted in the README.md
# For multimodal usage turn image into a tensor by either:
# 1) pulling it from link:
#    image_url = "https://llava-vl.github.io/static/images/view.jpg"
#    image = Image.open(requests.get(image_url, stream=True).raw)
#   or
# 2) pulling it from path
#    image_path = path_to_data + "MAFC_test/image_claims/00000.png"
#    image = Image.open(image_path)
#
# Hand the tensor as second argument to Factchecker.check


def evaluate(
        model: str,
        search_engine: str,
        benchmark_name: str,
        benchmark_kwargs: dict = None,
        multimodal_model: str = None,
        n: int = None,
        extract_claims: bool = True,
        verbose: bool = False,
        logging: bool = True
) -> float:
    benchmark = load_benchmark(benchmark_name, **benchmark_kwargs)
    logger = EvaluationLogger(benchmark.name, model_abbr[model]) if logging else None

    if logging:
        config = {
            "LLM": model,
            "MLLM": multimodal_model,
            "Search Engine": search_engine,
            "Benchmark": benchmark.name,
            "Extract Claims": extract_claims,
            "Full Dataset": True if n == len(benchmark) else f'{n} samples'
        }
        logger.save_config(config)
        start_time = time.time()

    summary = f"LLM: {model}, " \
              f"MLLM: {multimodal_model}, " \
              f"Search Engine: {search_engine}, " \
              f"Benchmark: {benchmark.name}\n"

    print(bold(gray(summary)))

    if logging:
        logger.log(summary)

    fc = FactChecker(
        model=model,
        multimodal_model=multimodal_model,
        search_engine=search_engine,
        extract_claims=extract_claims,
    )

    if not n:
        n = len(benchmark)

    predictions = []
    for i, instance in enumerate(benchmark):
        print(f"\nEvaluating on claim {i + 1} of {n}:")
        content = instance["content"]

        prediction = fc.check(content, verbose=verbose, logger=logger)
        prediction_is_correct = instance["label"] == prediction

        if logging:
            logger.save_next_prediction(sample_index=i + 1, target=instance["label"], predicted=prediction)
            if prediction_is_correct:
                logger.log("CORRECT")
            else:
                logger.log("WRONG - Ground truth: " + instance["label"].value)

        if prediction_is_correct:
            print(bold(green("CORRECT")))
        else:
            print(bold(red("WRONG - Ground truth: " + instance["label"].value)))

        predictions.append(prediction)
        if len(predictions) == n:
            break

    # Compute metrics
    ground_truth = benchmark.get_labels()[:n]
    correct_predictions = np.array(predictions) == np.array(ground_truth)
    accuracy = np.sum(correct_predictions) / n
    print(f"Accuracy: {accuracy * 100:.1f} %\n\n")

    if logging:
        end_time = time.time()
        results = {
            "Accuracy": f"{accuracy * 100:.1f} %",
            "Correct Predictions": correct_predictions.tolist(),
            "Incorrect Predictions": (n - correct_predictions.sum()).tolist(),
            "Duration of Run": f'{end_time - start_time} seconds'
        }
        logger.save_aggregated_results(results)

    return accuracy
