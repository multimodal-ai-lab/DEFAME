import csv
import inspect
import json
import time
from multiprocessing import Pool, Queue
from pathlib import Path
from queue import Empty
from typing import Sequence

import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm

from infact.common.label import Label
from infact.common.logger import Logger
from infact.common.modeling import model_specifier_to_shorthand, AVAILABLE_MODELS, make_model
from infact.eval.averitec.compute_score import compute_averitec_score
from infact.eval.benchmark import load_benchmark, AVeriTeC, Benchmark
from infact.fact_checker import FactChecker
from infact.tools import initialize_tools, Searcher
from infact.tools.search.knowledge_base import KnowledgeBase
from infact.utils.console import green, red, bold, sec2hhmmss, sec2mmss, num2text
from infact.utils.plot import plot_confusion_matrix
from infact.utils.utils import unroll_dict


def evaluate(
        llm: str,
        benchmark_name: str,
        tools_config: dict[str, dict],
        fact_checker_kwargs: dict = None,
        llm_kwargs: dict = None,
        benchmark_kwargs: dict = None,
        mllm: str = None,
        mllm_kwargs: dict = None,
        n_samples: int = None,
        sample_ids: list[int] = None,
        random_sampling: bool = False,
        print_log_level: str = False,
        continue_experiment_dir: str = None,
        n_workers: int = None,
):
    assert not n_samples or not sample_ids

    if llm_kwargs is None:
        llm_kwargs = dict()
    if mllm_kwargs is None:
        mllm_kwargs = dict()

    benchmark = load_benchmark(benchmark_name, **benchmark_kwargs)
    is_test = benchmark.variant == "test"

    llm = model_specifier_to_shorthand(llm) if llm not in AVAILABLE_MODELS["Shorthand"].values else llm
    procedure_variant = None if fact_checker_kwargs is None else fact_checker_kwargs.get("procedure_variant")
    logger = Logger(benchmark_name=benchmark.shorthand,
                    procedure_name=procedure_variant,
                    model_name=llm,
                    print_log_level=print_log_level,
                    target_dir=continue_experiment_dir)

    is_resumed = continue_experiment_dir is not None

    status_verb = "Resuming" if is_resumed else "Starting"
    print(bold(f"{status_verb} evaluation for {benchmark.name}."))

    n_devices = torch.cuda.device_count()
    if n_workers is None:
        match llm:
            case "llama3_8b":
                n_workers = 8
            case "llama3_70b":
                n_workers = 3  # only 3 copies fit on 8 A100 GPUs
            case _:
                n_workers = n_devices * 2  # 2 workers per GPU

    # Save hyperparams based on the signature of evaluate()
    if not is_resumed:
        signature = inspect.signature(evaluate)
        logger.save_config(signature, locals())

    # Load the tools and verify if they are allowed
    tools = initialize_tools(tools_config, logger=logger)
    if benchmark.available_actions is not None:
        for tool in tools:
            for action in tool.actions:
                assert action in benchmark.available_actions, \
                    f"Action {action} not available for benchmark {benchmark.name}."
    del tools

    if random_sampling:
        benchmark.shuffle()

    if n_samples:
        samples = benchmark[:n_samples]
    elif sample_ids:
        samples = [benchmark.get_by_id(i) for i in sample_ids]
    else:
        samples = benchmark

    # Exclude already existing samples (relevant if evaluation is resumed)
    if is_resumed:
        samples_to_evaluate = []
        # Retrieve the IDs of already checked claims
        predictions_path = continue_experiment_dir + "/predictions.csv"
        df = pd.read_csv(predictions_path)
        checked_claim_ids = df["sample_index"].to_numpy()

        # Only keep samples that haven't been checked yet
        for sample in samples:
            if sample["id"] not in checked_claim_ids:
                samples_to_evaluate.append(sample)

    else:
        samples_to_evaluate = samples

    # Update number of to-be-checked samples
    n_samples = len(samples_to_evaluate)

    if n_samples == 0:
        raise RuntimeError("Nothing to evaluate.")

    is_averitec = isinstance(benchmark, AVeriTeC)

    start_time = time.time()

    input_queue = Queue()
    output_queue = Queue()
    devices_queue = Queue()

    fact_checker_kwargs.update(dict(
        class_definitions=benchmark.class_definitions,
        extra_prepare_rules=benchmark.extra_prepare_rules,
        extra_plan_rules=benchmark.extra_plan_rules,
        extra_judge_rules=benchmark.extra_judge_rules,
    ))

    logger_kwargs = dict(
        print_log_level=print_log_level,
        target_dir=logger.target_dir
    )

    worker_args = (llm, llm_kwargs, mllm, mllm_kwargs, fact_checker_kwargs,
                   tools_config, logger_kwargs, is_averitec, input_queue, output_queue, devices_queue)

    print(f"Evaluating {n_samples} samples using {n_workers} workers...")

    with Pool(n_workers, fact_check, worker_args):
        # Initialize workers by assigning them a GPU device
        for d in range(n_workers):
            devices_queue.put(d % n_devices)

        # Fill the input queue with benchmark instances
        for instance in samples_to_evaluate:
            content = instance["content"]
            input_queue.put(content)

        # Gather worker results by reading the output queue
        for _ in tqdm(range(n_samples), smoothing=0.02):
            try:
                doc, meta = output_queue.get(timeout=30 * 60)  # 30 minutes timeout
            except Empty as e:
                # Happens if some worker died during execution, causing an
                # incomplete number of instances
                break
            content = doc.claim.original_context
            claim_id = content.id_number
            instance = benchmark.get_by_id(claim_id)
            prediction = doc.verdict

            if is_averitec:
                if prediction == Label.CHERRY_PICKING:
                    # Merge cherry-picking and conflicting label
                    prediction = Label.CONFLICTING

                pred_label = benchmark.get_class_name(prediction)
                averitec_out_instance = {
                    "claim_id": claim_id,
                    "claim": content.text,
                    "pred_label": pred_label
                }

                if "q_and_a" in meta:
                    averitec_out_instance["evidence"] = meta["q_and_a"]

                logger.save_next_averitec_out(averitec_out_instance)

            logger.save_next_prediction(
                sample_index=claim_id,
                claim=doc.claim.text,
                target=instance.get("label"),
                justification=doc.justification,
                predicted=prediction,
                gt_justification=instance.get("justification")
            )
            logger.save_fc_doc(doc, instance['id'])
            logger.save_next_instance_stats(meta["Statistics"], instance['id'])

            if not is_test:
                prediction_is_correct = instance["label"] == prediction
                if prediction_is_correct:
                    logger.log(bold(green("CORRECT\n")))
                else:
                    logger.log(bold(red("WRONG - Ground truth: " + instance["label"].value + "\n")))

    end_time = time.time()
    duration = end_time - start_time

    stats = {
        "Number of workers": n_workers,
        "Total run duration": duration,
    }

    finalize_evaluation(stats, logger.target_dir, benchmark)


def aggregate_stats(instance_stats: pd.DataFrame, category: str) -> dict[str, float]:
    """Sums the values for all instances for all the columns the name of
    which begin with 'category'."""
    aggregated_stats = dict()
    columns = list(instance_stats.columns)
    for column in columns:
        if column.startswith(category):
            aggregated = instance_stats[column].sum()
            if isinstance(aggregated, np.integer):
                aggregated = int(aggregated)
            elif isinstance(aggregated, np.floating):
                aggregated = float(aggregated)
            aggregated_stats[column] = aggregated
    return unroll_dict(aggregated_stats)


def finalize_evaluation(stats: dict,
                        experiment_dir: str | Path,
                        benchmark: Benchmark):
    """Takes a dictionary of experiment statistics, computes experiment metrics, and saves
    all the values in a YAML file. Also plots a confusion matrix if ground truth is available."""
    experiment_dir = Path(experiment_dir)
    is_averitec = isinstance(benchmark, AVeriTeC)
    is_test = benchmark.variant == "test"

    instance_stats = pd.read_csv(experiment_dir / Logger.instance_stats_filename)

    # Add aggregated statistics from individual claims
    stats.update({"Time per claim": instance_stats["Duration"].mean()})
    stats.update(aggregate_stats(instance_stats, category="Model"))
    stats.update(aggregate_stats(instance_stats, category="Tools"))

    # Retrieve predictions and ground truth
    df = pd.read_csv(experiment_dir / Logger.predictions_filename)
    predicted_labels = df["predicted"].to_numpy()
    ground_truth_labels = None if is_test else df["target"].to_numpy()

    # Compute metrics and save them along with the other stats
    metric_stats = compute_metrics(predicted_labels, ground_truth_labels)
    stats["Predictions"] = metric_stats
    save_stats(stats, target_dir=experiment_dir)

    if not is_test:
        benchmark_classes = benchmark.get_classes()
        if is_averitec:
            benchmark_classes.remove(Label.CHERRY_PICKING)

        plot_confusion_matrix(predicted_labels,
                              ground_truth_labels,
                              benchmark_classes,
                              benchmark_name=benchmark.name,
                              save_dir=experiment_dir)

        if is_averitec:
            averitec_out_path = experiment_dir / Logger.averitec_out_filename
            scores = compute_averitec_score(benchmark.file_path, averitec_out_path)
            scores_path = experiment_dir / "averitec_scores.yaml"
            with open(scores_path, "w") as f:
                yaml.dump(scores, f, sort_keys=False)


def compute_metrics(predicted_labels: Sequence[Label],
                    ground_truth_labels: Sequence[Label] = None):
    n_samples = len(predicted_labels)
    n_refused = np.count_nonzero(np.array(predicted_labels) == Label.REFUSED_TO_ANSWER)

    metric_summary = {
        "Total": n_samples,
        "Refused": int(n_refused),
    }

    if ground_truth_labels is not None:
        correct_predictions = np.asarray(np.array(predicted_labels) == np.array(ground_truth_labels))
        n_correct_predictions = np.sum(correct_predictions)
        n_wrong_predictions = n_samples - n_correct_predictions - n_refused
        accuracy = n_correct_predictions / (n_samples - n_refused)

        metric_summary.update({
            "Correct": int(n_correct_predictions),
            "Wrong": int(n_wrong_predictions),
            "Accuracy": accuracy,
        })

    return metric_summary


def save_stats(stats: dict, target_dir: Path):
    """Writes two files: one machine-readable file 'stats.json' and one
    human-readable file 'stats.yaml'."""
    # Save machine-readable stats
    with open(target_dir / 'results.json', "w") as f:
        json.dump(stats, f, sort_keys=False)

    # Create a human-readable version
    stats_human_readable = stats.copy()
    stats_human_readable["Total run duration"] = sec2hhmmss(stats["Total run duration"])
    stats_human_readable["Time per claim"] = sec2mmss(stats["Time per claim"])
    acc = stats["Predictions"].get("Accuracy")
    if acc is not None:
        stats_human_readable["Predictions"]["Accuracy"] = f"{acc * 100:.1f} %"
    model = stats_human_readable["Model"].copy()
    model["Input tokens"] = num2text(model["Input tokens"])
    model["Output tokens"] = num2text(model["Output tokens"])
    model["Input tokens cost"] = "$" + num2text(model["Input tokens cost"])
    model["Output tokens cost"] = "$" + num2text(model["Output tokens cost"])
    model["Total cost"] = "$" + num2text(model["Total cost"])
    stats_human_readable["Model"] = model

    # Save the human-readable statistics and print them
    with open(target_dir / 'results.yaml', "w") as f:
        stats_str = yaml.dump(stats_human_readable, sort_keys=False)
        f.write(stats_str)

    print("Results:\n" + stats_str)


def fact_check(llm: str, llm_kwargs: dict, mllm: str, mllm_kwargs: dict,
               fact_checker_kwargs: dict, tools_config: dict, logger_kwargs: dict,
               is_averitec: bool, input_queue: Queue, output_queue: Queue, devices_queue: Queue):
    device = f"cuda:{devices_queue.get()}"

    logger = Logger(**logger_kwargs)

    tools = initialize_tools(tools_config, logger=logger, device=device)

    # Initialize model(s)
    llm = make_model(llm, logger=logger, device=device, **llm_kwargs)

    # Setup fact-checker
    fc = FactChecker(
        llm=llm,
        tools=tools,
        logger=logger,
        **fact_checker_kwargs,
    )

    # Get the knowledge base object
    if is_averitec:
        searcher = tools[0]
        assert isinstance(searcher, Searcher)
        kb = searcher.search_apis["averitec_kb"]
        assert isinstance(kb, KnowledgeBase)
    else:
        kb = None

    # Run fact-checks as long as there is work to do
    while True:
        content = input_queue.get()
        if is_averitec:
            # Restrict the KB to the current claim's resources
            kb.current_claim_id = content.id_number
        logger.set_current_fc_id(content.id_number)
        _, docs, metas = fc.check_content(content)
        doc = docs[0]
        meta = metas[0]
        output_queue.put((doc, meta))


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


def naive_evaluate(model: str, model_kwargs: dict = None, benchmark_name: str = "fever1", n_samples: int = None,
                   **kwargs) -> float:
    benchmark = load_benchmark(benchmark_name)
    model = make_model(model, **model_kwargs)
    samples_to_evaluate = benchmark[:n_samples] if n_samples else benchmark

    eval_log = []
    predictions = []
    for instance in samples_to_evaluate:
        query = f"Check if the following claim is 'supported', 'not enough information', or 'refuted' using your available knowledge. Answer with only one of the three options. Claim: {instance['content']}"
        prediction = model.generate(query).replace("'", "").replace(".", "").lower()
        if prediction not in ['supported', 'not enough information', 'refuted']:
            print(instance["id"], prediction)
        eval_log.append({"claim": instance["content"], "pred_label": prediction})
        prediction_is_correct = instance["label"].value == prediction
        predictions.append(prediction_is_correct)
    accuracy = np.average(predictions)

    return accuracy, eval_log


def bold_print_dict(dictionary: dict):
    for key, value in dictionary.items():
        print(f"\t{bold(str(key))}: {value}")
