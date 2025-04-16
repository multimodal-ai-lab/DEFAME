import csv
import inspect
import json
import re
import time
import traceback
from multiprocessing import Process
from pathlib import Path
from queue import Empty
from typing import Sequence, Optional

import nltk
import numpy as np
import pandas as pd
import torch
import yaml
# from rouge_score import rouge_scorer
# from datasets import load_metric
from nltk.tokenize.treebank import TreebankWordDetokenizer
from prettytable import PrettyTable
from sklearn.metrics import precision_score, recall_score, f1_score
from tqdm import tqdm

from defame.common import Label, logger, Action
from defame.common.modeling import model_specifier_to_shorthand, AVAILABLE_MODELS, make_model
from defame.eval import load_benchmark
from defame.eval.averitec.benchmark import AVeriTeC
from defame.eval.averitec.compute_score import compute_averitec_score
from defame.eval.benchmark import Benchmark
from defame.eval.mocheg.benchmark import MOCHEG
from defame.evidence_retrieval.tools import initialize_tools
from defame.fact_checker import FactChecker
from defame.helpers.parallelization.pool import Pool
from defame.helpers.parallelization.task import Task
from defame.utils.console import bold, sec2hhmmss, sec2mmss, num2text
from defame.utils.plot import plot_confusion_matrix
from defame.utils.utils import unroll_dict


def evaluate(
        llm: str,
        benchmark_name: str,
        tools_config: dict[str, dict],
        experiment_name: str = None,
        fact_checker_kwargs: dict = None,
        llm_kwargs: dict = None,
        benchmark_kwargs: dict = None,
        allowed_actions: list[str] = None,
        n_samples: int = None,
        sample_ids: list[int | str] = None,
        random_sampling: bool = False,
        print_log_level: str = "log",
        continue_experiment_dir: str = None,
        n_workers: int = None,
):
    assert not n_samples or not sample_ids

    if llm_kwargs is None:
        llm_kwargs = dict()

    if fact_checker_kwargs is None:
        fact_checker_kwargs = dict()

    logger.set_log_level(print_log_level)

    benchmark = load_benchmark(benchmark_name, **benchmark_kwargs)

    is_resumed = continue_experiment_dir is not None
    status_verb = "Resuming" if is_resumed else "Starting"
    exp_name_str = f" '{bold(experiment_name)}'" if experiment_name else ""
    logger.info(f"{status_verb} evaluation{exp_name_str} on {benchmark.name}.")

    llm = model_specifier_to_shorthand(llm) if llm not in AVAILABLE_MODELS["Shorthand"].values else llm

    procedure_variant = fact_checker_kwargs.get("procedure_variant", FactChecker.default_procedure)

    logger.set_experiment_dir(path=continue_experiment_dir,
                              benchmark_name=benchmark.shorthand,
                              procedure_name=procedure_variant,
                              model_name=llm,
                              experiment_name=experiment_name)
    logger.log("Saving all outputs to:", logger.target_dir.as_posix())

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

    if allowed_actions is None:
        allowed_actions = benchmark.available_actions
    else:
        allowed_actions = [a for a in benchmark.available_actions if a.name in allowed_actions]

    # Sanity check
    p = Process(target=validate_config, args=(tools_config, allowed_actions))
    p.start()
    p.join()

    if random_sampling:
        benchmark.shuffle()

    if n_samples:
        assert 0 < n_samples <= len(benchmark), f"{n_samples} specified but only {len(benchmark)} samples available."
        samples = benchmark[:n_samples]
    elif sample_ids:
        samples = [benchmark.get_by_id(str(i)) for i in sample_ids]
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
            # sample["id"] should be convertable into a number type for indexing
            if int(sample["id"]) not in checked_claim_ids:
                samples_to_evaluate.append(sample)

        stats_file_path = logger.target_dir / 'results.json'
        if stats_file_path.exists():
            with open(stats_file_path, "r") as f:
                stats = json.load(f)
        else:
            stats = dict()

    else:
        samples_to_evaluate = samples
        stats = dict()

    # Update number of to-be-checked samples
    n_samples = len(samples_to_evaluate)

    if n_samples == 0:
        raise RuntimeError("Nothing to evaluate.")

    n_workers = min(n_workers, n_samples)

    is_averitec = isinstance(benchmark, AVeriTeC)

    start_time = time.time()

    print(f"Evaluating {n_samples} samples using {n_workers} workers...")

    pool = Pool(n_workers=n_workers,
                llm=llm,
                llm_kwargs=llm_kwargs,
                tools_config=tools_config,
                available_actions=allowed_actions,
                class_definitions=benchmark.class_definitions,
                extra_prepare_rules=benchmark.extra_prepare_rules,
                extra_plan_rules=benchmark.extra_plan_rules,
                extra_judge_rules=benchmark.extra_judge_rules,
                print_log_level=print_log_level,
                target_dir=logger.target_dir,
                **fact_checker_kwargs)

    # Turn each sample into a task and add it to the pool's task queue
    for instance in samples_to_evaluate:
        task = Task(instance["input"], id=instance["id"])
        pool.add_task(task)

    progress = tqdm(range(n_samples), smoothing=0.02)

    pool.wait_until_ready()

    try:
        while progress.n + pool.n_failed_tasks < n_samples:
            try:
                output = pool.get_result(timeout=60)
                benchmark.process_output(output)
                progress.update(1)

            except Empty as e:
                if not pool.is_running():
                    logger.warning("Worker pool stopped running early. Terminating evaluation.")
                    break

    except Exception as e:
        logger.critical(f"An unexpected error occurred in the main process:")
        logger.critical(traceback.format_exc())

    end_time = time.time()
    duration = end_time - start_time

    stats.update({
        "Number of workers": n_workers,
        "Total run duration": duration + stats.get("Total run duration", 0)
    })

    finalize_evaluation(logger.target_dir, benchmark, stats)


def validate_config(tools_config: dict[str, dict], allowed_actions: Sequence[Action]):
    """Run this within in a subprocess to avoid errors with CUDA (which doesn't
    like forking subprocesses, but spawning doesn't work either)."""

    # Load the tools
    tools = initialize_tools(tools_config, llm=None)

    # Verify that each tool is relevant (i.e. offers at least one allowed action)
    for tool in tools:
        for action in tool.actions:
            if action in allowed_actions:
                break
        else:
            logger.info(f"Tool {tool.name} offers only forbidden actions. You may exclude this tool.")

    # Verify that each allowed action has a corresponding tool
    for action in allowed_actions:
        for tool in tools:
            if action in tool.actions:
                break
        else:
            logger.warning(f"No Tool available for action {action.name}.")

    # Print a convenient table to see if available actions match with allowed actions
    logger.log(bold("Action Summary:"))
    table = PrettyTable()
    table.align = "l"
    table.field_names = ["Action", "Available", "Allowed"]
    offered_actions = {action for tool in tools for action in tool.actions}
    allowed_actions = set(allowed_actions)
    for action in offered_actions | allowed_actions:
        is_available = action in offered_actions
        is_allowed = action in allowed_actions
        table.add_row([action.name,
                       "✅ Yes" if is_available else "❌ No",
                       "✅ Yes" if is_allowed else "❌ No"])
    logger.log(table.__repr__())


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


def finalize_evaluation(experiment_dir: str | Path,
                        benchmark: Benchmark,
                        stats: dict = None):
    """Takes a dictionary of experiment statistics, computes experiment metrics, and saves
    all the values in a YAML file. Also plots a confusion matrix if ground truth is available."""
    experiment_dir = Path(experiment_dir)
    is_averitec = isinstance(benchmark, AVeriTeC)
    is_mocheg = isinstance(benchmark, MOCHEG)
    is_test = benchmark.variant == "test"
    try:
        instance_stats = pd.read_csv(experiment_dir / logger.instance_stats_filename)
    except Exception:
        print("Terminated before instance_stats.csv was created. ")
        return

    # If stats are not specified, check if there already exist some
    if stats is None:
        stats_file_path = experiment_dir / 'results.json'
        if stats_file_path.exists():
            with open(stats_file_path, "r") as f:
                stats = json.load(f)

    if stats is None:
        stats = dict()

    # Add aggregated statistics from individual claims
    stats.update({"Time per claim": instance_stats["Duration"].mean()})
    stats.update(aggregate_stats(instance_stats, category="Model"))
    stats.update(aggregate_stats(instance_stats, category="Tools"))

    # Retrieve predictions and ground truth
    df = pd.read_csv(experiment_dir / logger.predictions_filename)
    # Sort by 'sample_index' column
    df = df.sort_values(by="sample_index").reset_index(drop=True)
    df.to_csv(experiment_dir / logger.predictions_filename, index=False)

    predicted_labels = df["predicted"].to_numpy()
    if is_averitec:
        ground_truth_labels = None if is_test else df["target"].to_numpy()
    else:
        # Assuming that the test set also has target labels.
        ground_truth_labels = df["target"].to_numpy()

    predicted_justifications = df["justification"].apply(remove_urls_and_brackets)
    ground_truth_justifications = df["gt_justification"].apply(remove_urls_and_brackets)

    # Compute metrics and save them along with the other stats
    metric_stats = compute_metrics(predicted_labels,
                                   ground_truth_labels,
                                   predicted_justifications=predicted_justifications,
                                   ground_truth_justifications=ground_truth_justifications,
                                   is_mocheg=is_mocheg)
    stats["Predictions"] = metric_stats
    save_stats(stats, target_dir=experiment_dir)

    logger.info(f"All outputs saved in {experiment_dir.as_posix()}.")

    benchmark_classes = benchmark.get_classes()
    plot_confusion_matrix(predicted_labels,
                          ground_truth_labels,
                          benchmark_classes,
                          benchmark_name=benchmark.name,
                          save_dir=experiment_dir)

    if is_averitec:
        averitec_out_path = experiment_dir / logger.averitec_out_filename
        scores = compute_averitec_score(benchmark.file_path, averitec_out_path)
        scores_path = experiment_dir / "averitec_scores.yaml"
        with open(scores_path, "w") as f:
            yaml.dump(scores, f, sort_keys=False)


def compute_metrics(predicted_labels: np.ndarray,
                    ground_truth_labels: Optional[np.ndarray] = None,
                    predicted_justifications: Optional[Sequence[str]] = None,
                    ground_truth_justifications: Optional[Sequence[str]] = None,
                    is_mocheg: bool = False):
    n_samples = len(predicted_labels)
    n_refused = np.count_nonzero(np.array(predicted_labels) == "REFUSED_TO_ANSWER")

    metrics = dict()
    metric_summary = {
        "Total": n_samples,
        "Refused": int(n_refused),
        "Metrics": metrics
    }

    # Classification Metrics
    try:
        labels = np.unique(np.append(ground_truth_labels, predicted_labels))
        precision = precision_score(ground_truth_labels, predicted_labels, labels=labels, average=None)
        recall = recall_score(ground_truth_labels, predicted_labels, labels=labels, average=None)
        f1_scores = f1_score(ground_truth_labels, predicted_labels, labels=labels, average=None)

        macro_f1 = f1_score(ground_truth_labels, predicted_labels, labels=labels, average='macro')

        for label, p, r, f1 in zip(labels, precision, recall, f1_scores):
            metrics.update({
                f"{label}_Precision": float(round(p, 3)),
                f"{label}_Recall": float(round(r, 3)),
                f"{label}_F1_Score": float(round(f1, 3)),
            })

        metric_summary[f"Macro-Averaged F1-Score"] = float(round(macro_f1, 2))

    except Exception as e:
        print(f"There was an error computing classification metrics: {str(e)}")

    # Generation Metrics
    try:
        if is_mocheg and (ground_truth_justifications is not None) and (predicted_justifications is not None):
            nltk.download('punkt')

            # Load the metrics from `datasets`
            bertscore_metric = load_metric("bertscore")
            bleu_metric_datasets = load_metric("bleu")
            rouge_metric = load_metric("rouge")

            # Post-process justifications for metric computation
            processed_preds, processed_labels = postprocess_text(predicted_justifications, ground_truth_justifications)

            # Compute scores
            bleu_datasets = compute_metrics_with_text(processed_preds, processed_labels, bleu_metric_datasets, "bleu")
            bertscore = compute_metrics_with_text(processed_preds, processed_labels, bertscore_metric, "bertscore")
            rouge_scores = compute_metrics_with_text(processed_preds, processed_labels, rouge_metric, "rouge")

            # Aggregate metrics into Generation dictionary
            generation_metrics = {
                "BLEU": bleu_datasets["bleu"],
                "ROUGE1": float(rouge_scores.get("rouge1", 0)),
                "ROUGE2": float(rouge_scores.get("rouge2", 0)),
                "ROUGE_L": float(rouge_scores.get("rougeL", 0)),
                "BERTScore": bertscore["bertscore"],
            }

            # Update metric_summary with generation metrics
            metric_summary.update({"Generation": generation_metrics})

    except Exception as e:
        print(f"There was an error computing MOCHEG generation metrics: {str(e)}")

    # Final accuracy calculation
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
    if "Total run duration" in stats:
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


def postprocess_text(preds, labels, num_limit=None):
    """
    Postprocessing for BERTScore evaluation for MOCHEG dataset
    """
    if num_limit:
        preds = [TreebankWordDetokenizer().detokenize(pred.split()[:num_limit]) for pred in preds]
    preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in labels]
    return preds, labels


def remove_urls_and_brackets(text):
    if pd.isna(text):
        return ''
    else:
        return re.sub(r'\[.*?\]\(.*?\)', '', text)


def compute_metric_with_text_bleu(decoded_preds, decoded_labels, metric):
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
    decoded_preds_bleu = [pred.split(' ') for pred in decoded_preds]
    decoded_labels_bleu = [[pred.split(' ')] for pred in decoded_labels]
    result = metric.compute(predictions=decoded_preds_bleu, references=decoded_labels_bleu)
    result["bleu"] = round(result["bleu"] * 100, 4)
    return result


def compute_metric_with_text_bertscore(decoded_preds, decoded_labels, metric):
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
    all_result = metric.compute(predictions=decoded_preds, references=decoded_labels, lang="en")
    avg_result = sum(all_result["f1"]) / len(all_result["f1"]) * 100
    return {"bertscore": round(avg_result, 4)}


def compute_metrics_with_text(decoded_preds, decoded_labels, metric, metric_name):
    if metric_name == "bleu":
        return compute_metric_with_text_bleu(decoded_preds, decoded_labels, metric)
    elif metric_name == "bertscore":
        return compute_metric_with_text_bertscore(decoded_preds, decoded_labels, metric)
    else:
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
        result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        result = {key: round(value.mid.fmeasure * 100, 4) for key, value in result.items()}
        return result
