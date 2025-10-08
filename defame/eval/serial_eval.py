# defame/eval/serial_eval.py
from __future__ import annotations

import inspect, json, io, time
from pathlib import Path
from typing import Sequence, Optional

import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm

from defame.common import Label, logger, Action
from defame.common.modeling import model_specifier_to_shorthand, AVAILABLE_MODELS
from defame.eval import load_benchmark
from defame.eval.averitec.benchmark import AVeriTeC
from defame.eval.averitec.compute_score import compute_averitec_score
from defame.eval.benchmark import Benchmark
from defame.eval.mocheg.benchmark import MOCHEG
from defame.evidence_retrieval.tools import initialize_tools
from defame.fact_checker import FactChecker
from defame.utils.console import bold, num2text, sec2hhmmss, sec2mmss
from defame.utils.plot import plot_confusion_matrix


# ---------------------------
# Resilient Tool Proxy
# ---------------------------
class _ResilientTool:
    """
    Wraps a tool and catches exceptions on ANY callable attribute.
    Returns None on failure so the FC can proceed with other tools.
    """
    def __init__(self, tool, claim_id_getter=lambda: None):
        self._tool = tool
        self.name = getattr(tool, "name", tool.__class__.__name__)
        self.actions = getattr(tool, "actions", [])
        self._get_claim_id = claim_id_getter

    def __repr__(self):
        return f"<ResilientTool {self.name}>"

    def __getattr__(self, attr):
        orig = getattr(self._tool, attr)
        if callable(orig):
            def wrapped(*args, **kwargs):
                try:
                    return orig(*args, **kwargs)
                except Exception as e:
                    cid = self._get_claim_id()
                    msg = f"[tool-soft-fail] tool={self.name} attr={attr} claim_id={cid} error={e}"
                    logger.warning(msg)
                    # Return a benign value that the FC can treat as "no evidence"
                    return None
            return wrapped
        return orig


def _wrap_tools_resilient(tools: list, claim_id_getter):
    wrapped = []
    for t in tools:
        try:
            wrapped.append(_ResilientTool(t, claim_id_getter=claim_id_getter))
        except Exception as e:
            logger.warning(f"[tool-wrap-fail] Could not wrap tool {getattr(t, 'name', t)}: {e}")
    return wrapped


def validate_config_inline(tools_config: dict[str, dict], allowed_actions: Sequence[Action]):
    """Same as validate_config but inline (no subprocess)."""
    tools = initialize_tools(tools_config, llm=None)
    # force tool import/initialization to surface early errors
    _ = {a for t in tools for a in t.actions}
    # no further checks here; FactChecker will filter actions itself.


def evaluate_serial(
    llm: str,
    benchmark_name: str,
    tools_config: dict[str, dict],
    experiment_name: str | None = None,
    fact_checker_kwargs: dict | None = None,
    llm_kwargs: dict | None = None,
    benchmark_kwargs: dict | None = None,
    allowed_actions: list[str] | None = None,
    n_samples: int | None = None,
    sample_ids: list[int | str] | None = None,
    random_sampling: bool = False,
    print_log_level: str = "log",
    soft_fail_tools: bool = True,           # <— NEW: enable per-tool soft-fail
    retry_without_tools_on_error: bool = True,  # <— NEW: fallback if claim still errors
):
    assert not n_samples or not sample_ids

    if llm_kwargs is None: llm_kwargs = {}
    if fact_checker_kwargs is None: fact_checker_kwargs = {}
    if benchmark_kwargs is None: benchmark_kwargs = {}

    logger.set_log_level(print_log_level)

    benchmark = load_benchmark(benchmark_name, **benchmark_kwargs)
    llm = model_specifier_to_shorthand(llm) if llm not in AVAILABLE_MODELS["Shorthand"].values else llm

    procedure_variant = fact_checker_kwargs.get("procedure_variant", FactChecker.default_procedure)

    logger.set_experiment_dir(
        path=None,
        benchmark_name=benchmark.shorthand,
        procedure_name=procedure_variant,
        model_name=llm,
        experiment_name=experiment_name,
    )
    logger.log("Saving all outputs to:", logger.target_dir.as_posix())

    # Save hyperparams (like evaluate does)
    signature = inspect.signature(evaluate_serial)
    logger.save_config(signature, locals())

    # Tools / actions
    if allowed_actions is None:
        allowed_actions = benchmark.available_actions
    else:
        # keep only those the benchmark allows
        allowed_actions = [a for a in benchmark.available_actions if a.name in allowed_actions]

    # Validation (raises if a tool can't even import)
    validate_config_inline(tools_config, allowed_actions)

    # Sampling
    if random_sampling:
        benchmark.shuffle()

    if n_samples:
        assert 0 < n_samples <= len(benchmark), f"{n_samples} specified but only {len(benchmark)} samples available."
        samples = benchmark[:n_samples]
    elif sample_ids:
        samples = [benchmark.get_by_id(str(i)) for i in sample_ids]
    else:
        samples = benchmark

    if len(samples) == 0:
        raise RuntimeError("Nothing to evaluate.")

    # Build raw tools once (we may reuse and/or wrap per-claim)
    raw_tools = initialize_tools(tools_config, llm=None)

    # Helper so wrappers know the current claim id for logging
    _current_claim_id = {"val": None}
    def _get_claim_id(): return _current_claim_id["val"]

    # Construct a base FactChecker kwargs bundle (shared for main + fallback)
    fc_kwargs_common = dict(
        llm=llm,
        llm_kwargs=llm_kwargs,
        available_actions=allowed_actions,
        procedure_variant=procedure_variant,
        interpret=fact_checker_kwargs.get("interpret", False),
        decompose=fact_checker_kwargs.get("decompose", False),
        decontextualize=fact_checker_kwargs.get("decontextualize", False),
        filter_check_worthy=fact_checker_kwargs.get("filter_check_worthy", False),
        max_iterations=fact_checker_kwargs.get("max_iterations", 3),
        max_result_len=fact_checker_kwargs.get("max_result_len", 64_000),
        restrict_results_to_claim_date=fact_checker_kwargs.get("restrict_results_to_claim_date", True),
        allow_fact_checking_sites=fact_checker_kwargs.get("allow_fact_checking_sites", True),
        classes=list(benchmark.class_definitions.keys()),
        class_definitions=benchmark.class_definitions,
        extra_prepare_rules=benchmark.extra_prepare_rules,
        extra_plan_rules=benchmark.extra_plan_rules,
        extra_judge_rules=benchmark.extra_judge_rules,
    )

    start = time.time()
    progress = tqdm(samples, smoothing=0.02)

    skipped = 0
    recovered = 0
    tool_soft_fail_enabled = "ON" if soft_fail_tools else "OFF"
    progress.set_description_str(f"soft-fail={tool_soft_fail_enabled}")

    skipped_rows = []  # optional: keep a small log of skipped sample ids + reasons

    for instance in progress:
        # Some benchmarks store id here; fall back gracefully if not present
        sample_id = instance.get("id") if isinstance(instance, dict) else None
        _current_claim_id["val"] = sample_id

        # Build FactChecker for this claim:
        #  - either use resilient-wrapped tools
        #  - or raw tools (if soft_fail_tools=False)
        if soft_fail_tools:
            tools_for_claim = _wrap_tools_resilient(raw_tools, claim_id_getter=_get_claim_id)
        else:
            tools_for_claim = list(raw_tools)

        fc = FactChecker(
            tools=tools_for_claim,
            tools_config=None,   # we pass instantiated tools directly
            **fc_kwargs_common
        )

        try:
            # Run full fact-check for this sample with resilient tools
            doc, meta = fc.verify_claim(instance["input"])
            benchmark.process_output((doc, meta))

        except Exception as e:
            # Something still bubbled up despite per-tool soft-fails.
            # Optionally retry the *same claim* with tools removed (LLM-only).
            if retry_without_tools_on_error:
                logger.warning(f"[serial_eval] Claim {sample_id}: error despite soft-fail. Retrying without tools. Error: {e}")
                try:
                    fc_no_tools = FactChecker(
                        tools=[],           # <- LLM-only fallback
                        tools_config=None,
                        **fc_kwargs_common
                    )
                    doc, meta = fc_no_tools.verify_claim(instance["input"])
                    benchmark.process_output((doc, meta))
                    recovered += 1
                    progress.set_description_str(f"soft-fail={tool_soft_fail_enabled} recovered={recovered} skipped={skipped}")
                    continue
                except Exception as e2:
                    # Even LLM-only failed — skip this claim.
                    skipped += 1
                    progress.set_description_str(f"soft-fail={tool_soft_fail_enabled} recovered={recovered} skipped={skipped}")
                    logger.warning(f"[serial_eval] Skipping sample id={sample_id} after fallback failed: {e2}")
                    skipped_rows.append({"id": sample_id, "error": f"primary={e}; fallback={e2}"})
                    continue
            else:
                # No fallback requested — skip
                skipped += 1
                progress.set_description_str(f"soft-fail={tool_soft_fail_enabled} skipped={skipped}")
                logger.warning(f"[serial_eval] Skipping sample id={sample_id} due to error: {e}")
                skipped_rows.append({"id": sample_id, "error": str(e)})
                continue

    duration = time.time() - start
    stats = {
        "Number of workers": 1,
        "Total run duration": duration,
        "Skipped samples": int(skipped),
        "Recovered w/o tools": int(recovered),
        "Soft-fail tools": bool(soft_fail_tools),
        "Fallback without tools": bool(retry_without_tools_on_error),
    }

    # Optionally write a CSV of skipped samples for debugging
    try:
        if skipped_rows:
            pd.DataFrame(skipped_rows).to_csv(logger.target_dir / "skipped_samples.csv", index=False)
    except Exception:
        pass  # never let bookkeeping fail the run

    # Finalize exactly like the original (with some robustness)
    _finalize_serial(logger.target_dir, benchmark, stats)



def _finalize_serial(experiment_dir: str | Path, benchmark: Benchmark, stats: dict | None = None):
    experiment_dir = Path(experiment_dir)
    is_averitec = isinstance(benchmark, AVeriTeC)
    is_mocheg = isinstance(benchmark, MOCHEG)
    is_test = benchmark.variant == "test"

    # ---------- instance_stats.csv (robust) ----------
    inst_path = experiment_dir / logger.instance_stats_filename
    instance_stats = None
    if inst_path.exists():
        try:
            instance_stats = pd.read_csv(inst_path)
        except Exception:
            instance_stats = None

    if stats is None:
        stats_file_path = experiment_dir / 'results.json'
        if stats_file_path.exists():
            try:
                with open(stats_file_path, "r") as f:
                    stats = json.load(f)
            except Exception:
                stats = None
    if stats is None:
        stats = {}

    # Aggregate model/tool timing only if we have instance_stats
    if instance_stats is not None and not instance_stats.empty:
        def _aggregate(df: pd.DataFrame, prefix: str) -> dict:
            out = {}
            for c in df.columns:
                if c.startswith(prefix):
                    v = df[c].sum()
                    out[c] = (float(v) if isinstance(v, np.floating)
                              else int(v) if isinstance(v, np.integer)
                              else v)
            return out

        stats.update({"Time per claim": float(instance_stats["Duration"].mean())})
        stats.update({"Model": _aggregate(instance_stats, "Model")})
        stats.update({"Tools": _aggregate(instance_stats, "Tools")})
    else:
        # Graceful defaults when no rows processed (e.g., all skipped)
        stats.setdefault("Time per claim", 0.0)
        stats.setdefault("Model", {})
        stats.setdefault("Tools", {})

    # ---------- predictions.csv (robust) ----------
    pred_path = experiment_dir / logger.predictions_filename

    def read_csv_robust(path: Path) -> pd.DataFrame:
        for enc in ("utf-8", "utf-8-sig", "cp1252", "latin1"):
            try:
                return pd.read_csv(path, encoding=enc)
            except UnicodeDecodeError:
                continue
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return pd.read_csv(io.StringIO(f.read()))

    if pred_path.exists():
        df = read_csv_robust(pred_path)
        if not df.empty and "sample_index" in df.columns:
            df = df.sort_values(by="sample_index").reset_index(drop=True)
        df.to_csv(pred_path, index=False)
    else:
        # Create an empty predictions file so downstream code won’t crash
        df = pd.DataFrame(columns=["sample_index", "predicted", "target"])
        df.to_csv(pred_path, index=False)

    predicted_labels = df["predicted"].to_numpy() if "predicted" in df else np.array([])
    if is_averitec:
        ground_truth_labels = None if is_test else (df["target"].to_numpy() if "target" in df else np.array([]))
    else:
        ground_truth_labels = df["target"].to_numpy() if "target" in df else np.array([])

    # ---------- metrics ----------
    n_samples = len(predicted_labels)
    n_refused = int(np.count_nonzero(np.array(predicted_labels) == "REFUSED_TO_ANSWER"))
    metrics = {"Total": n_samples, "Refused": n_refused, "Metrics": {}}

    if ground_truth_labels is not None and len(ground_truth_labels) == n_samples and n_samples > 0:
        from sklearn.metrics import precision_score, recall_score, f1_score
        labels = np.unique(np.append(ground_truth_labels, predicted_labels))
        precision = precision_score(ground_truth_labels, predicted_labels, labels=labels, average=None, zero_division=0)
        recall = recall_score(ground_truth_labels, predicted_labels, labels=labels, average=None, zero_division=0)
        f1s = f1_score(ground_truth_labels, predicted_labels, labels=labels, average=None, zero_division=0)
        macro_f1 = f1_score(ground_truth_labels, predicted_labels, labels=labels, average='macro', zero_division=0)
        for lab, p, r, f in zip(labels, precision, recall, f1s):
            metrics["Metrics"].update({
                f"{lab}_Precision": float(round(p, 3)),
                f"{lab}_Recall": float(round(r, 3)),
                f"{lab}_F1_Score": float(round(f, 3)),
            })
        correct = int(np.sum(np.asarray(predicted_labels) == np.asarray(ground_truth_labels)))
        denom = max(n_samples - n_refused, 1)
        accuracy = correct / denom
        metrics.update({"Correct": correct, "Wrong": int(n_samples - correct - n_refused), "Accuracy": accuracy})
    stats["Predictions"] = metrics

    # ---------- save machine-readable ----------
    with open(experiment_dir / 'results.json', "w") as f:
        json.dump(stats, f, sort_keys=False)

    # ---------- save human-readable ----------
    human = stats.copy()
    human["Total run duration"] = sec2hhmmss(stats.get("Total run duration", 0))
    human["Time per claim"] = sec2mmss(stats.get("Time per claim", 0))
    if (acc := stats.get("Predictions", {}).get("Accuracy")) is not None:
        human["Predictions"]["Accuracy"] = f"{acc*100:.1f} %"
    if "Model" in human:
        m = human["Model"].copy()
        for k in ["Input tokens", "Output tokens", "Input tokens cost", "Output tokens cost", "Total cost"]:
            if k in m:
                if "cost" in k.lower():
                    m[k] = "$" + num2text(m[k])
                else:
                    m[k] = num2text(m[k])
        human["Model"] = m

    with open(experiment_dir / 'results.yaml', "w") as f:
        f.write(yaml.dump(human, sort_keys=False))

    # ---------- confusion matrix (only if we have targets) ----------
    if ground_truth_labels is not None and len(ground_truth_labels) == n_samples and n_samples > 0:
        plot_confusion_matrix(predicted_labels, ground_truth_labels, benchmark.get_classes(),
                              benchmark_name=benchmark.name, save_dir=experiment_dir)

    # ---------- AVeriTeC post-processing ----------
    if is_averitec and (experiment_dir / logger.averitec_out_filename).exists():
        averitec_out_path = experiment_dir / logger.averitec_out_filename
        scores = compute_averitec_score(benchmark.file_path, averitec_out_path)
        with open(experiment_dir / "averitec_scores.yaml", "w") as f:
            yaml.dump(scores, f, sort_keys=False)
