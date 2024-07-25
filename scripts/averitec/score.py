import json
import sys
from contextlib import redirect_stdout

from src.eval.averitec.score import AVeriTeCEvaluator, print_with_space
from src.eval.evaluate import evaluate

output_file_path = "/pfss/mlde/workspaces/mlde_wsp_Rohrbach/users/tb17xity/MAFC/out/averitec/2024-07-21_20-11_llama3_70b/averitec_score.txt"

# Open the output file in write mode
with open(output_file_path, 'w') as f:
    # Redirect standard output to the file
    with redirect_stdout(f):
        # Load references and predictions
        with open("/pfss/mlde/workspaces/mlde_wsp_Rohrbach/data/raw/AVeriTeC/dev.json") as f_ref:
            references = json.load(f_ref)
        with open("/pfss/mlde/workspaces/mlde_wsp_Rohrbach/users/tb17xity/MAFC/out/averitec/2024-07-21_20-11_llama3_70b/eval.json") as f_pred:
            predictions = json.load(f_pred)

        scorer = AVeriTeCEvaluator()

        for claim_id, prediction in enumerate(predictions):
            print(f"\nClaim: {claim_id}")

            reference = references[claim_id]

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

        print("\n____________Overall Score: _____________")

        q_score = scorer.evaluate_questions_only(predictions, references)
        print_with_space("Question-only score (HU-" + scorer.metric + "):", str(q_score))
        p_score = scorer.evaluate_questions_and_answers(predictions, references)
        print_with_space("Question-answer score (HU-" + scorer.metric + "):", str(p_score))
        print("====================")
        v_score = scorer.evaluate_veracity(predictions, references)
        print("Veracity F1 scores:")
        for k, v in v_score.items():
            print_with_space(" * " + k + ":", str(v))
        print("--------------------")
        print("AVeriTeC scores:")
        v_score = scorer.evaluate_averitec_score(predictions, references)
        for i, level in enumerate(scorer.averitec_reporting_levels):
            print_with_space(
                " * Veracity scores (" + scorer.metric + " @ " + str(level) + "):",
                str(v_score[i]),
            )
        print("--------------------")
        print("AVeriTeC scores by type @ 0.25:")
        type_scores = scorer.evaluate_averitec_veracity_by_type(
            predictions, references, threshold=0.25
        )
        for t, v in type_scores.items():
            print_with_space(" * Veracity scores (" + t + "):", str(v))
