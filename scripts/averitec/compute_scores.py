import yaml

from defame.eval.averitec.compute_score import compute_averitec_score

dataset_path = ""
experiment_dir = ""
results_path = experiment_dir + "eval.json"

scores = compute_averitec_score(dataset_path, results_path)
scores_path = experiment_dir + "averitec_scores.yaml"
with open(scores_path, "w") as f:
    yaml.dump(scores, f, sort_keys=False)
