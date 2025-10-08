import csv
import inspect
import json
import re
import time
import traceback
from multiprocessing import Process, set_start_method
from pathlib import Path
from queue import Empty
from typing import Sequence, Optional

import nltk
import numpy as np
import pandas as pd
import torch
import yaml
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


# ---------------------- UTF-8 helpers (Windows-safe) ----------------------

def _read_csv_robust(path: Path) -> pd.DataFrame:
    encodings = ["utf-8", "utf-8-sig", "latin-1"]
    last_err = None
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as e:
            last_err = e
    raise last_err


def _write_csv_utf8(df: pd.DataFrame, path: Path):
    df.to_csv(path, index=False, encoding="utf-8")


# ---------------------- Evaluate ----------------------

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

    benchmark = load_benchmark(benchmark_name, **(benchmark_kwargs or {}))

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
                n_workers = max(1, n_devices * 2)  # 2 workers per GPU

    # Save hyperparams based on the signature of evaluate()
    if not is_resumed:
        signature = inspect.signature(evaluate)
        logger.save_config(signature, locals())

    if allowed_actions is None:
        allowed_actions = benchmark.available_actions
    else:
        allowed_actions = [a for a in benchmark.available_actions if a.name in allowed_actions]

    # Sanity check for tool config
    try:
        set_start_method("spawn")
    except RuntimeError:
        # Already set in this process; it's fine.
        pass
    p = Process(target=validate_config, args=(tools_config, allowed_actions))
    p.start()
    p.join()

    if random_sampling:
        benchmark.shuffle()

    # Select samples
    if n_samples:
        assert 0 < n_samples <= len(benchmark), f"{n_samples} specified but only {len(benchmark)} samples available."
        samples = benchmark[:n_samples]
    elif sample_ids:
        samples = [benchmark.get_by_id(str(i)) for i in sample_ids]
    else:
        samples = benchmark

    # For "only draw confusion matrix when fully done" we track exactly which IDs we intend to run:
    selected_ids = {str(s["id"]) for s in samples}

    # Exclude already existing samples if resuming
    if is_resumed:
        samples_to_evaluate = []
        predictions_path = Path(continue_experiment_dir) / "predictions.csv"
        if predictions_path.exists():
            try:
                df_prev = _read_csv_robust(predictions_path)
                checked_claim_ids = set(df_prev["sample_index"].astype(str).tolist())
            except Exception:
                checked_claim_ids = set()
        else:
            checked_claim_ids = set()

        for sample in samples:
            if str(sample["id"]) not in checked_claim_ids:
                samples_to_evaluate.append(sample)

        stats_file_path = logger.target_dir / 'results.json'
        if stats_file_path.exists():
            with open(stats_file_path, "r", encoding="utf-8") as f:
                stats = json.load(f)
        else:
            stats = dict()
    else:
        samples_to_evaluate = samples
        stats = dict()

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

    # Queue tasks
    for instance in samples_to_evaluate:
        task = Task(instance["input"], id=instance["id"])
        pool.add_task(task)

    progress = tqdm(range(n_samples), smoothing=0.02)

    # ----------- Make the loop resilient: skip problematic examples -----------
    try:
        while progress.n + pool.n_failed_tasks < n_samples:
            try:
                output = pool.get_result(timeout=60)
            except Empty:
                if not pool.is_running():
                    logger.warning("Worker pool stopped early. Terminating evaluation loop.")
                    break
                # keep waiting
                continue
            except Exception as e:
                # Unexpected pool error for this item; skip & continue
                logger.warning(f"Skipping one example due to get_result error: {e}")
                continue

            # Process a successful result, but be defensive
            try:
                benchmark.process_output(output)
                progress.update(1)
            except Exception as e:
                logger.warning(f"Skipping one example due to process_output error: {e}")
                logger.debug(traceback.format_exc())
                # do NOT update progress here (no row saved)
                continue

    except Exception:
        logger.critical("An unexpected error occurred in the main process:")
        logger.critical(traceback.format_exc())

    end_time = time.time()
    duration = end_time - start_time

    stats.update({
        "Number of workers": n_workers,
        "Total run duration": duration + stats.get("Total run duration", 0)
    })

    # Finalize (write metrics, add fake_cls, maybe confusion matrix)
    finalize_evaluation(
        logger.target_dir,
        benchmark,
        stats,
        selected_ids=selected_ids  # pass intended ID set for "full completion" check
    )


def validate_config(tools_config: dict[str, dict], allowed_actions: Sequence[Action]):
    """Run this within in a subprocess to avoid errors with CUDA."""
    tools = initialize_tools(tools_config, llm=None)

    for tool in tools:
        for action in tool.actions:
            if action in allowed_actions:
                break
        else:
            logger.info(f"Tool {tool.name} offers only forbidden actions. You may exclude this tool.")

    for action in allowed_actions:
        for tool in tools:
            if action in tool.actions:
                break
        else:
            logger.warning(f"No Tool available for action {action.name}.")

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
    """Sums the values for columns whose names begin with 'category'."""
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
                        stats: dict = None,
                        selected_ids: Optional[set[str]] = None):
    """
    Finalization:
      • Save aggregated stats
      • Add DGM4 fake_cls column to predictions.csv
      • Compute metrics
      • Plot confusion matrix ONLY if all selected IDs have predictions
    """
    experiment_dir = Path(experiment_dir)
    is_averitec = isinstance(benchmark, AVeriTeC)
    is_mocheg = isinstance(benchmark, MOCHEG)
    is_test = getattr(benchmark, "variant", None) == "test"

    # Load instance stats
    try:
        instance_stats = _read_csv_robust(experiment_dir / logger.instance_stats_filename)
    except Exception:
        print("Terminated before instance_stats.csv was created.")
        return

    # Restore / init stats dict
    if stats is None:
        stats_file_path = experiment_dir / 'results.json'
        if stats_file_path.exists():
            with open(stats_file_path, "r", encoding="utf-8") as f:
                stats = json.load(f)
    if stats is None:
        stats = dict()

    # Add aggregated statistics
    if "Duration" in instance_stats.columns:
        stats.update({"Time per claim": instance_stats["Duration"].mean()})
    stats.update(aggregate_stats(instance_stats, category="Model"))
    stats.update(aggregate_stats(instance_stats, category="Tools"))

    # Retrieve predictions and ground truth
    pred_path = experiment_dir / logger.predictions_filename
    if not pred_path.exists():
        # Nothing was saved (e.g., all tasks failed)
        save_stats(stats, target_dir=experiment_dir)
        logger.info(f"All outputs saved in {experiment_dir.as_posix()}.")
        return

    df = _read_csv_robust(pred_path)

    # Sort by sample_index if present
    sort_col = "sample_index" if "sample_index" in df.columns else (None)
    if sort_col:
        df = df.sort_values(by=sort_col).reset_index(drop=True)

    # ------------------ ADD fake_cls column (DGM4 only) ------------------
    try:
        if getattr(benchmark, "shorthand", "").lower() == "dgm4":
            # Prefer the fast map built in your DGM4.__init__ / _load_data()
            if hasattr(benchmark, "id2fake_cls") and isinstance(benchmark.id2fake_cls, dict):
                if "sample_index" in df.columns:
                    df["fake_cls"] = df["sample_index"].astype(str).map(benchmark.id2fake_cls)
                else:
                    # If sample_index is missing, keep a blank column to avoid KeyError elsewhere
                    if "fake_cls" not in df.columns:
                        df["fake_cls"] = pd.NA
            else:
                # Fall back: create the column if absent (blank)
                if "fake_cls" not in df.columns:
                    df["fake_cls"] = pd.NA
        else:
            # Non-DGM4: do not add; ensure no stray column gets added
            if "fake_cls" not in df.columns:
                df["fake_cls"] = pd.NA  # harmless, but uniform for downstream code if needed
    except Exception as e:
        print(f"Warning: could not attach fake_cls to predictions.csv: {e}")
        if "fake_cls" not in df.columns:
            df["fake_cls"] = pd.NA

    # Persist predictions with UTF-8
    _write_csv_utf8(df, pred_path)

    # Prepare arrays
    predicted_labels = df["predicted"].to_numpy() if "predicted" in df.columns else np.array([])
    if is_averitec:
        ground_truth_labels = None if is_test else (df["target"].to_numpy() if "target" in df.columns else None)
    else:
        ground_truth_labels = df["target"].to_numpy() if "target" in df.columns else None

    predicted_justifications = (
        df["justification"].apply(remove_urls_and_brackets)
        if "justification" in df.columns else pd.Series([""] * len(df))
    )
    ground_truth_justifications = (
        df["gt_justification"].apply(remove_urls_and_brackets)
        if "gt_justification" in df.columns else pd.Series([""] * len(df))
    )

    # Metrics
    metric_stats = compute_metrics(predicted_labels,
                                   ground_truth_labels,
                                   predicted_justifications=predicted_justifications,
                                   ground_truth_justifications=ground_truth_justifications,
                                   is_mocheg=is_mocheg)
    stats["Predictions"] = metric_stats
    save_stats(stats, target_dir=experiment_dir)
    logger.info(f"All outputs saved in {experiment_dir.as_posix()}.")

    # ------------- Only plot confusion matrix if FULLY completed -------------
    try:
        can_plot = False
        if ground_truth_labels is not None:
            if selected_ids:
                # All intended IDs must be present
                have_ids = set(df["sample_index"].astype(str).tolist()) if "sample_index" in df.columns else set()
                can_plot = selected_ids.issubset(have_ids)
            else:
                # Fallback: if no selection set provided, require no NaNs in preds/targets
                can_plot = ("predicted" in df.columns and "target" in df.columns
                            and df["predicted"].notna().all() and df["target"].notna().all())

        if can_plot:
            benchmark_classes = benchmark.get_classes()
            plot_confusion_matrix(predicted_labels,
                                  ground_truth_labels,
                                  benchmark_classes,
                                  benchmark_name=benchmark.name,
                                  save_dir=experiment_dir)
        else:
            logger.info("Skipping confusion matrix (evaluation not 100% complete for selected items).")
    except Exception as e:
        print(f"Warning while plotting confusion matrix: {e}")

    # AVeriTeC score when applicable
    if isinstance(benchmark, AVeriTeC):
        try:
            averitec_out_path = experiment_dir / logger.averitec_out_filename
            scores = compute_averitec_score(benchmark.file_path, averitec_out_path)
            scores_path = experiment_dir / "averitec_scores.yaml"
            with open(scores_path, "w", encoding="utf-8") as f:
                yaml.dump(scores, f, sort_keys=False, allow_unicode=True)
        except Exception as e:
            print(f"Warning computing AVeriTeC score: {e}")


# ---------------------- Metrics & utils ----------------------

def compute_metrics(predicted_labels: np.ndarray,
                    ground_truth_labels: Optional[np.ndarray] = None,
                    predicted_justifications: Optional[Sequence[str]] = None,
                    ground_truth_justifications: Optional[Sequence[str]] = None,
                    is_mocheg: bool = False):
    n_samples = len(predicted_labels)
    n_refused = np.count_nonzero(np.array(predicted_labels) == "REFUSED_TO_ANSWER")

    metrics = dict()
    metric_summary = {
        "Total": int(n_samples),
        "Refused": int(n_refused),
        "Metrics": metrics
    }

    # Classification Metrics
    try:
        if ground_truth_labels is not None:
            labels = np.unique(np.append(ground_truth_labels, predicted_labels))
            precision = precision_score(ground_truth_labels, predicted_labels, labels=labels, average=None, zero_division=0)
            recall = recall_score(ground_truth_labels, predicted_labels, labels=labels, average=None, zero_division=0)
            f1_scores = f1_score(ground_truth_labels, predicted_labels, labels=labels, average=None, zero_division=0)
            macro_f1 = f1_score(ground_truth_labels, predicted_labels, labels=labels, average='macro', zero_division=0)

            for label, p, r, f1 in zip(labels, precision, recall, f1_scores):
                metrics.update({
                    f"{label}_Precision": float(round(p, 3)),
                    f"{label}_Recall": float(round(r, 3)),
                    f"{label}_F1_Score": float(round(f1, 3)),
                })

            metric_summary["Macro-Averaged F1-Score"] = float(round(macro_f1, 2))
    except Exception as e:
        print(f"There was an error computing classification metrics: {str(e)}")

    # Generation Metrics (only for MOCHEG)
    try:
        if is_mocheg and (ground_truth_justifications is not None) and (predicted_justifications is not None):
            nltk.download('punkt')
            # If you actually use datasets.load_metric, re-enable and import appropriately
            # bertscore_metric = load_metric("bertscore")
            # bleu_metric_datasets = load_metric("bleu")
            # rouge_metric = load_metric("rouge")
            # processed_preds, processed_labels = postprocess_text(predicted_justifications, ground_truth_justifications)
            # bleu_datasets = compute_metrics_with_text(processed_preds, processed_labels, bleu_metric_datasets, "bleu")
            # bertscore = compute_metrics_with_text(processed_preds, processed_labels, bertscore_metric, "bertscore")
            # rouge_scores = compute_metrics_with_text(processed_preds, processed_labels, rouge_metric, "rouge")
            # generation_metrics = {
            #     "BLEU": bleu_datasets["bleu"],
            #     "ROUGE1": float(rouge_scores.get("rouge1", 0)),
            #     "ROUGE2": float(rouge_scores.get("rouge2", 0)),
            #     "ROUGE_L": float(rouge_scores.get("rougeL", 0)),
            #     "BERTScore": bertscore["bertscore"],
            # }
            # metric_summary.update({"Generation": generation_metrics})
    except Exception as e:
        print(f"There was an error computing MOCHEG generation metrics: {str(e)}")

    # Final accuracy
    if ground_truth_labels is not None and len(ground_truth_labels) == len(predicted_labels):
        correct_predictions = np.asarray(np.array(predicted_labels) == np.array(ground_truth_labels))
        n_correct_predictions = int(np.sum(correct_predictions))
        n_wrong_predictions = int(n_samples - n_correct_predictions - n_refused)
        denom = (n_samples - n_refused)
        accuracy = (n_correct_predictions / denom) if denom > 0 else 0.0

        metric_summary.update({
            "Correct": n_correct_predictions,
            "Wrong": n_wrong_predictions,
            "Accuracy": accuracy,
        })

    return metric_summary


def save_stats(stats: dict, target_dir: Path):
    """Writes machine-readable results.json and a human-friendly results.yaml (UTF-8)."""
    with open(target_dir / 'results.json', "w", encoding="utf-8") as f:
        json.dump(stats, f, sort_keys=False, ensure_ascii=False)

    stats_hr = stats.copy()
    if "Total run duration" in stats_hr:
        stats_hr["Total run duration"] = sec2hhmmss(stats_hr["Total run duration"])
    if "Time per claim" in stats_hr:
        stats_hr["Time per claim"] = sec2mmss(stats_hr["Time per claim"])

    if "Predictions" in stats_hr and isinstance(stats_hr["Predictions"], dict):
        acc = stats_hr["Predictions"].get("Accuracy")
        if acc is not None:
            stats_hr["Predictions"]["Accuracy"] = f"{acc * 100:.1f} %"

    if "Model" in stats_hr and isinstance(stats_hr["Model"], dict):
        model = stats_hr["Model"].copy()
        if "Input tokens" in model:
            model["Input tokens"] = num2text(model["Input tokens"])
        if "Output tokens" in model:
            model["Output tokens"] = num2text(model["Output tokens"])
        if "Input tokens cost" in model:
            model["Input tokens cost"] = "$" + num2text(model["Input tokens cost"])
        if "Output tokens cost" in model:
            model["Output tokens cost"] = "$" + num2text(model["Output tokens cost"])
        if "Total cost" in model:
            model["Total cost"] = "$" + num2text(model["Total cost"])
        stats_hr["Model"] = model

    with open(target_dir / 'results.yaml', "w", encoding="utf-8") as f:
        stats_str = yaml.dump(stats_hr, sort_keys=False, allow_unicode=True)
        f.write(stats_str)
    print("Results:\n" + stats_str)


# ---------------------- (Optional) helpers used above ----------------------

def bold_print_dict(dictionary: dict):
    for key, value in dictionary.items():
        print(f"\t{bold(str(key))}: {value}")


def postprocess_text(preds, labels, num_limit=None):
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
