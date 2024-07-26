import csv
import inspect
import time
import yaml

import numpy as np
import pandas as pd

from src.common.label import Label
from src.common.modeling import model_full_name_to_shorthand, AVAILABLE_MODELS, MLLM, LLM
from src.eval.averitec.compute_score import compute_averitec_score
from src.eval.benchmark import load_benchmark, AVeriTeC
from src.eval.logger import EvaluationLogger
from src.fact_checker import FactChecker
from src.tools import initialize_tools, Searcher
from src.tools.search.knowledge_base import KnowledgeBase
from src.utils.console import green, red, bold
from src.utils.plot import plot_confusion_matrix


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
        verbose: bool = False,
) -> float:
    assert not n_samples or not sample_ids

    if llm_kwargs is None:
        llm_kwargs = dict()
    if mllm_kwargs is None:
        mllm_kwargs = dict()

    benchmark = load_benchmark(benchmark_name, **benchmark_kwargs)

    llm = model_full_name_to_shorthand(llm) if llm not in AVAILABLE_MODELS["Shorthand"].values else llm
    logger = EvaluationLogger(benchmark.shorthand, llm, verbose=verbose)

    # Save hyperparams based on the signature of evaluate()
    signature = inspect.signature(evaluate)
    logger.save_config(signature, locals())

    # Load the tools and verify if they are allowed
    tools = initialize_tools(tools_config, logger=logger)
    if benchmark.available_actions is not None:
        for tool in tools:
            for action in tool.actions:
                assert action in benchmark.available_actions, \
                    f"Action {action} not available for benchmark {benchmark.name}."

    # Initialize model(s)
    llm = LLM(llm, logger=logger, **llm_kwargs)
    if mllm is not None:
        mllm = MLLM(name=mllm, logger=logger, **mllm_kwargs)

    fc = FactChecker(
        llm=llm,
        mllm=mllm,
        tools=tools,
        logger=logger,
        **fact_checker_kwargs,
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

    is_averitec = isinstance(benchmark, AVeriTeC)

    if is_averitec:
        searcher = tools[0]
        assert isinstance(searcher, Searcher)
        kb = searcher.search_apis["averitec_kb"]
        assert isinstance(kb, KnowledgeBase)
    else:
        kb = None

    start_time = time.time()

    predictions, averitec_out = [], []
    for i, instance in enumerate(samples_to_evaluate):
        logger.log(f"Evaluating claim {i + 1} of {n_samples} (#{instance['id']}):")
        content = instance["content"]

        # Update the current claim to restrict the KB to the current claim's resources
        if is_averitec:
            kb.current_claim_id = instance['id']

        _, docs, q_and_a = fc.check(content)

        doc = docs[0]
        prediction = doc.verdict
        if is_averitec and prediction == Label.CHERRY_PICKING:  # Needed for Averitec
            prediction = Label.CONFLICTING
        pred_label = benchmark.get_class_name(prediction)
        averitec_output = {
            "claim_id": instance['id'],
            "claim": instance["content"].text,
            "evidence": q_and_a,
            "pred_label": pred_label
        }

        averitec_out.append(averitec_output)
        prediction_is_correct = instance["label"] == prediction

        logger.save_next_prediction(
            sample_index=instance['id'],
            claim=doc.claim.text,
            target=instance["label"],
            justification=doc.justification,
            predicted=prediction,
            gt_justification=instance["justification"]
        )
        logger.save_fc_doc(doc, instance['id'])
        if prediction_is_correct:
            logger.log(bold(green("CORRECT\n")))
        else:
            logger.log(bold(red("WRONG - Ground truth: " + instance["label"].value + "\n")))

        predictions.append(prediction)
        if len(predictions) == n_samples:
            break

    benchmark_classes = benchmark.get_classes()
    if is_averitec:
        benchmark_classes.remove(Label.CHERRY_PICKING)

    ground_truth = [s["label"] for s in samples_to_evaluate]
    search_summary = {
        name: searcher.total_searches
        for tool in fc.actor.tools if isinstance(tool, Searcher)
        for name, searcher in tool.search_apis.items()
    }
    end_time = time.time()
    accuracy = logger.save_results(predictions, ground_truth, averitec_out,
                                   duration=end_time - start_time,
                                   search_summary=search_summary)
    plot_confusion_matrix(predictions,
                          ground_truth,
                          benchmark_classes,
                          benchmark_name=benchmark.name,
                          save_dir=logger.target_dir)

    if is_averitec:
        scores = compute_averitec_score(benchmark.file_path, logger.averitec_out)
        scores_path = logger.target_dir + "averitec_scores.yaml"
        with open(scores_path, "w") as f:
            yaml.dump(scores, f, sort_keys=False)

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


def compute_accuracy(predictions: pd.DataFrame) -> float:
    correct_stats = predictions["correct"].value_counts()
    prediction_stats = predictions["predicted"].value_counts()
    n_refused = prediction_stats["REFUSED_TO_ANSWER"] if "REFUSED_TO_ANSWER" in list(prediction_stats.keys()) else 0
    accuracy = correct_stats[True] / (len(predictions) - n_refused)
    return accuracy


def naive_evaluate(model: str, model_kwargs: dict = None, benchmark_name: str = "fever1", n_samples: int = None,
                   **kwargs) -> float:
    benchmark = load_benchmark(benchmark_name)
    model = LLM(model, **model_kwargs)
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
