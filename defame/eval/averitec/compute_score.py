import json
from pathlib import Path

import numpy as np

from defame.eval.averitec.score import AVeriTeCEvaluator, print_with_space

import matplotlib.pyplot as plt

scorer = AVeriTeCEvaluator()
metric = scorer.metric


def compute_averitec_score(dataset_path: str | Path, results_path: str | Path) -> dict:
    """Computes the overall Averitec score."""
    scores = {}  # will be returned

    # Load the JSON file contents
    with open(dataset_path) as f_data:
        dataset = json.load(f_data)
    with open(results_path) as f_res:
        results = json.load(f_res)

    # Question-only score
    q_only_score, all_q_scores = scorer.evaluate_questions_only(results, dataset)
    print(f"\nQuestion-only score (HU-{metric}): {q_only_score:.4f}")
    scores["Question-only score"] = float(np.round(q_only_score, 4))

    # Visualize metric distribution
    plt.hist(all_q_scores, bins=20)
    plt.axvline(x=0.25, linestyle='dashed', color='orange')
    plt.xlim(0, 1)
    plt.title("Hungarian Meteor score distribution for questions only")
    plt.show()

    # Show 5 worst scored questions (and the corresponding ground truth)
    print("\nWorst scored instances:")
    q_scores_sorted = np.argsort(all_q_scores)
    for i in q_scores_sorted[:5]:
        result = results[i]
        print(f"\nClaim {i}: {all_q_scores[i]:.4f}")
        print(result["claim"])
        print("Generated questions:")
        if "evidence" in result:
            for evidence in result["evidence"]:
                print("\t" + evidence["question"])
        print("Gold questions:")
        for evidence in dataset[i]["questions"]:
            print("\t" + evidence["question"])

    # Q&A score
    q_and_a_score, all_qa_scores = scorer.evaluate_questions_and_answers(results, dataset)
    print(f"Question-answer score (HU-{metric}): {q_and_a_score:.4f}")
    scores["Question-answer score"] = float(np.round(q_and_a_score, 4))

    # Visualize metric distribution
    plt.hist(all_qa_scores, bins=20)
    plt.axvline(x=0.25, linestyle='dashed', color='orange')
    plt.xlim(0, 1)
    plt.title("Hungarian Meteor score distribution for Q&A")
    plt.show()

    # Veracity F1 score
    veracity_scores = scorer.evaluate_veracity(results, dataset)
    print("\nVeracity F1 scores:")
    veracity_scores_summary = {}
    for k, v in veracity_scores.items():
        print(f" * {k}: {v:.4f}")
        veracity_scores_summary[k] = float(np.round(v, 4))
    scores["Veracity F1 scores"] = veracity_scores_summary

    # AVeriTeC scores
    print("\nAVeriTeC scores:")
    averitec_scores = scorer.evaluate_averitec_score(results, dataset)
    averitec_scores_summary = {}
    for i, level in enumerate(scorer.averitec_reporting_levels):
        print(f" * Veracity score ({metric} @ {level}): {averitec_scores[i]:.4f}")
        averitec_scores_summary[level] = float(np.round(averitec_scores[i], 4))
    scores[f"AVeriTeC scores by {metric}"] = averitec_scores_summary

    # AVeriTeC scores for threshold = 0.25
    print("\nAVeriTeC scores by type @ 0.25:")
    type_scores = scorer.evaluate_averitec_veracity_by_type(
        results, dataset, threshold=0.25
    )
    averitec_scores_summary = {}
    for t, v in type_scores.items():
        print(f" * Veracity scores ({t}): {v:.4f}")
        averitec_scores_summary[t] = float(np.round(v, 4))
    scores["AVeriTeC scores by type at 0.25"] = averitec_scores_summary

    return scores


def compute_single_claim_score(prediction: dict, reference: dict):
    """Prints out the score statistics for a single given prediction-reference pair."""

    for evidence in prediction["evidence"]:
        print("\n___________Prediction: ____________")
        print(evidence["question"])
        print(evidence["answer"])
        print()

    print("\n___________Reference: ____________")
    for question in reference["questions"]:
        print(question["question"])
        print(question["answers"])
        print()

    q_score = scorer.evaluate_questions_only([prediction], [reference])
    print_with_space("Question-only score (HU-" + scorer.metric + "):", str(q_score))
    p_score = scorer.evaluate_questions_and_answers([prediction], [reference])
    print_with_space("Question-answer score (HU-" + scorer.metric + "):", str(p_score))
    print("====================")
    v_score = scorer.evaluate_veracity([prediction], [reference])
    print("Veracity F1 scores:")
    for k, v in v_score.items():
        print_with_space(" * " + k + ":", str(v))
    print("--------------------")
    print("AVeriTeC scores:")
    v_score = scorer.evaluate_averitec_score([prediction], [reference])
    for i, level in enumerate(scorer.averitec_reporting_levels):
        print_with_space(
            " * Veracity scores (" + scorer.metric + " @ " + str(level) + "):",
            str(v_score[i]),
        )
    print("--------------------")
    print("AVeriTeC scores by type @ 0.25:")
    type_scores = scorer.evaluate_averitec_veracity_by_type(
        [prediction], [reference], threshold=0.25
    )
    for t, v in type_scores.items():
        print_with_space(" * Veracity scores (" + t + "):", str(v))
    print("\n_________________________")
    print("_________________________")
