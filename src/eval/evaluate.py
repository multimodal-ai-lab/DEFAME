import time
import csv
import inspect

from common.console import green, red, bold, gray
from common.label import Label
from common.plot import plot_confusion_matrix
from common.shared_config import model_abbr
from eval.benchmark import load_benchmark
from eval.logger import EvaluationLogger
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
        model_kwargs: dict = None,
        benchmark_kwargs: dict = None,
        multimodal_model: str = None,
        n_samples: int = None,
        sample_ids: list[int] = None,
        random_sampling: bool = False,
        extract_claims: bool = True,
        verbose: bool = False,
) -> float:
    assert n_samples is None or sample_ids is None

    benchmark = load_benchmark(benchmark_name, **benchmark_kwargs)

    logger = EvaluationLogger(benchmark.name, model_abbr[model], verbose=verbose)

    # Save hyperparams based on the signature of evaluate()
    signature = inspect.signature(evaluate)
    logger.save_config(signature, locals())
    start_time = time.time()

    fc = FactChecker(
        model=model,
        multimodal_model=multimodal_model,
        search_engines=[search_engine],
        extract_claims=extract_claims,
        logger=logger,
        classes=benchmark.get_classes(),
    )

    if not n_samples:
        n_samples = len(benchmark) if not sample_ids else len(sample_ids)

    if random_sampling:
        benchmark.shuffle()

    samples_to_evaluate = [benchmark.get_by_id(i) for i in sample_ids] if sample_ids else benchmark

    # Run the evaluation for each instance individually
    predictions = []
    for i, instance in enumerate(samples_to_evaluate):
        print(f"\nEvaluating on claim {i + 1} of {n_samples} (#{instance['id']}):")
        content = instance["content"]

        prediction = fc.check(content)
        prediction_is_correct = instance["label"] == prediction

        logger.save_next_prediction(sample_index=i + 1, target=instance["label"], predicted=prediction)
        if prediction_is_correct:
            logger.log(bold(green("CORRECT")))
        else:
            logger.log(bold(red("WRONG - Ground truth: " + instance["label"].value +"\n\n")))

        predictions.append(prediction)
        if len(predictions) == n_samples:
            break

    # Compute and save evaluation results
    ground_truth = benchmark.get_labels()[:n_samples]
    search_summary = {name: searcher.total_searches for name, searcher in fc.searcher.search_apis.items() if searcher}
    end_time = time.time()
    accuracy = logger.save_results(predictions, ground_truth,
                                   duration=end_time - start_time,
                                   search_summary=search_summary)
    plot_confusion_matrix(predictions,
                          ground_truth,
                          benchmark.get_classes(),
                          benchmark_name=benchmark.name,
                          save_dir=logger.target_dir)

    return accuracy


def load_results(path: str):
    ground_truth = []
    predictions = []
    for _, target, predicted, _ in next_result(path):
        ground_truth.append(Label[target])
        predictions.append(Label[predicted])
    return ground_truth, predictions


def next_result(path: str):
    with open(path) as f:
        reader = csv.reader(f)
        next(reader)  # skip header line
        for row in reader:
            yield row
