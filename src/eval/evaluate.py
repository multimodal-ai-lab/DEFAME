import csv
import inspect
import time
import pandas as pd

from common.console import green, red, bold
from common.label import Label
from common.plot import plot_confusion_matrix
from common.modeling import model_full_name_to_shorthand, AVAILABLE_MODELS, Model
from eval.benchmark import load_benchmark, AVeriTeC
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
        max_iterations: int = 3,
        verbose: bool = False,
) -> float:
    assert n_samples is None or sample_ids is None

    benchmark = load_benchmark(benchmark_name, **benchmark_kwargs)
    lookup = {value.value: key for key, value in benchmark.class_mapping.items()}

    model = model_full_name_to_shorthand(model) if model not in AVAILABLE_MODELS["Shorthand"].values else model
    logger = EvaluationLogger(benchmark.name, model, verbose=verbose)

    # Save hyperparams based on the signature of evaluate()
    signature = inspect.signature(evaluate)
    logger.save_config(signature, locals())
    start_time = time.time()

    # Initialize model
    model = Model(model, **model_kwargs)

    fc = FactChecker(
        model=model,
        multimodal_model=multimodal_model,
        search_engines=[search_engine],
        extract_claims=extract_claims,
        max_iterations=max_iterations,
        logger=logger,
        # Benchmark specifics:
        class_definitions=benchmark.class_definitions,
        extra_prepare_rules=benchmark.extra_prepare_rules,
        extra_plan_rules=benchmark.extra_plan_rules,
        extra_judge_rules=benchmark.extra_judge_rules,
    )

    if random_sampling:
        benchmark.shuffle()

    if n_samples:
        samples_to_evaluate = benchmark[:n_samples]
    else:
        if sample_ids:
            samples_to_evaluate = [benchmark.get_by_id(i) for i in sample_ids]
            n_samples = len(sample_ids)
        else:
            samples_to_evaluate = benchmark
            n_samples = len(benchmark)


    eval_log = []
    # Run the evaluation for each instance individually
    predictions = []
    for i, instance in enumerate(samples_to_evaluate):
        logger.log(f"Evaluating claim {i + 1} of {n_samples} (#{instance['id']}):")
        content = instance["content"]

        doc = fc.check(content)

        prediction = doc.verdict
        if prediction == Label.CHERRY_PICKING:  # AVeriTeC combines these two classes into one class (here CONFLICTING)
            prediction = Label.CONFLICTING

        eval_log.append({"claim": content, "evidence": "", "pred_label": prediction.name})
        prediction_is_correct = instance["label"] == prediction

        logger.save_next_prediction(sample_index=instance['id'], target=instance["label"], predicted=prediction)
        logger.save_fc_doc(doc, instance['id'])
        if prediction_is_correct:
            logger.log(bold(green("CORRECT\n")))
        else:
            logger.log(bold(red("WRONG - Ground truth: " + instance["label"].value + "\n")))

        predictions.append(prediction)
        if len(predictions) == n_samples:
            break

    benchmark_classes = benchmark.get_classes()
    if isinstance(benchmark, AVeriTeC):
        benchmark_classes.remove(Label.CHERRY_PICKING)

    # Compute and save evaluation results
    ground_truth = [s["label"] for s in samples_to_evaluate]
    search_summary = {name: searcher.total_searches
                      for name, searcher in fc.actor.searcher.search_apis.items()
                      if searcher}
    end_time = time.time()
    accuracy = logger.save_results(predictions, ground_truth,
                                   duration=end_time - start_time,
                                   search_summary=search_summary)
    plot_confusion_matrix(predictions,
                          ground_truth,
                          benchmark_classes,
                          benchmark_name=benchmark.name,
                          save_dir=logger.target_dir)

    return accuracy, eval_log, benchmark


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


def compute_accuracy(predictions: pd.DataFrame) -> float:
    correct_stats = predictions["correct"].value_counts()
    prediction_stats = predictions["predicted"].value_counts()
    n_refused = prediction_stats["REFUSED_TO_ANSWER"] if "REFUSED_TO_ANSWER" in list(prediction_stats.keys()) else 0
    accuracy = correct_stats[True] / (len(predictions) - n_refused)
    return accuracy
