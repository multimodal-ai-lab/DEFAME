"""Loads results from multiple experiments and generates plots to compare them."""

from pathlib import Path

import yaml
import json

from defame.utils.plot import plot_grouped_bar_chart, COLOR_PALETTE, plot_histogram_comparison
from defame.eval.averitec.score import AVeriTeCEvaluator
from tqdm import tqdm

to_compare = [
    {
        "legend_label": "Llama 3 (70B) SI",
        "experiment_dir": "/pfss/mlde/workspaces/mlde_wsp_Rohrbach/users/mr74vahu/MAFC/out/averitec/2024-07-25_15-22_llama3_70b",
        "color": "light-gray",
    },
    {
        "legend_label": "Llama 3 (70B) MI",
        "experiment_dir": "/pfss/mlde/workspaces/mlde_wsp_Rohrbach/users/mr74vahu/MAFC/out/averitec/2024-08-02_13-38_llama3_70b",
        "color": "mid-gray",
    },
    {
        "legend_label": "GPT-4o mini SI",
        "experiment_dir": "/pfss/mlde/workspaces/mlde_wsp_Rohrbach/users/mr74vahu/MAFC/out/averitec/2024-07-25_15-23_gpt_4o_mini",
        "color": "orange",
    },
    {
        "legend_label": "GPT-4o mini MI",
        "experiment_dir": "/pfss/mlde/workspaces/mlde_wsp_Rohrbach/users/mr74vahu/MAFC/out/averitec/2024-08-02_08-57_gpt_4o_mini",
        "color": "yellow-orange",
    },
    {
        "legend_label": "GPT-4o SI",
        "experiment_dir": "/pfss/mlde/workspaces/mlde_wsp_Rohrbach/users/mr74vahu/MAFC/out/averitec/2024-07-25_15-23_gpt_4o",
        "color": "dark-blue",
    },
    {
        "legend_label": "GPT-4o MI",
        "experiment_dir": "/pfss/mlde/workspaces/mlde_wsp_Rohrbach/users/mr74vahu/MAFC/out/averitec/2024-07-29_14-33_gpt_4o",
        "color": "light-blue",
    },
]


def load_averitec_results(path: Path) -> dict:
    with open(path, "r") as f:
        averitec_results = yaml.safe_load(f)
    return averitec_results


# Load the files
all_results = {}
for instance in to_compare:
    exp_dir = instance["experiment_dir"]
    name = instance["legend_label"]
    results = load_averitec_results(Path(exp_dir) / "averitec_scores.yaml")
    all_results[name] = results

# Extract the relevant Averitec scores and plot them
all_scores = {}
for name, results in all_results.items():
    scores_by_meteor_raw = results["AVeriTeC scores by meteor"]
    scores_by_meteor = [score for score in scores_by_meteor_raw.values()]
    all_scores[name] = scores_by_meteor

colors = [COLOR_PALETTE[instance["color"]] for instance in to_compare]

plot_grouped_bar_chart(
    x_labels=[0.1, 0.2, 0.25, 0.3, 0.4, 0.5],
    values=all_scores,
    title="AVeriTeC Scores",
    x_label="Hungarian METEOR threshold value",
    y_label="AVeriTeC score",
    show_values=False,
    colors=colors,
    save_path="/pfss/mlde/workspaces/mlde_wsp_Rohrbach/users/mr74vahu/MAFC/out/averitec/score_comparison.pdf"
)

# Plot Hungarian METEOR distribution comparison

# Load the dec set
scorer = AVeriTeCEvaluator()
dataset_path = "/pfss/mlde/workspaces/mlde_wsp_Rohrbach/data/raw/AVeriTeC/dev.json"
with open(dataset_path) as f:
    dev_set = json.load(f)

# Recompute the METEOR scores
data_rows = []
print("Recomputing METEOR scores...")
for experiment in tqdm(to_compare):
    name = experiment["legend_label"]
    out_path = Path(experiment["experiment_dir"]) / "averitec_out.json"
    with open(out_path) as f:
        results = json.load(f)
    q_meteors = scorer.get_questions_only_meteor(results, dev_set)
    data_rows.append(q_meteors)

plot_histogram_comparison(
    data_rows,
    title="Hungarian METEOR Score Distribution",
    labels=3 * ["SI", "MI"],
    y_label="Hungarian METEOR score",
    hist_range=(0, 1),
    colors=colors,
    secondary_labels=["Llama 3 (70B)", "GPT-4o mini", "GPT 4o"],
    save_path="/pfss/mlde/workspaces/mlde_wsp_Rohrbach/users/mr74vahu/MAFC/out/averitec/meteor_comparison.pdf",
    h_line_at=0.25,
)
