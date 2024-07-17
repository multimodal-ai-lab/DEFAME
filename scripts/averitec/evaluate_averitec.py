import json

from src.eval.averitec.score import AVeriTeCEvaluator, print_with_space
from src.eval.evaluate import evaluate

accuracy, eval_log, evidence_log, benchmark = evaluate(
    llm="llama3_70b",
    tools_config=dict(searcher=dict(
        search_engines=["averitec_kb"],
        limit_per_search=5
    )),
    fact_checker_kwargs=dict(
        interpret=False,
        decompose=False,
        decontextualize=False,
        filter_check_worthy=False,
        max_iterations=3,
    ),
    llm_kwargs=dict(temperature=0.01),
    benchmark_name="averitec",
    benchmark_kwargs=dict(variant="dev"),
    # sample_ids=[300],
    n_samples=10,
    random_sampling=False,
    verbose=True,
)

predictions = eval_log
with open(benchmark.file_path) as f:
    references = json.load(f)
# potentially references = benchmark.data is enough here
scorer = AVeriTeCEvaluator()
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
