import pandas as pd
from common.plot import plot_confusion_matrix
from eval.benchmark import AVeriTeC, FEVER

path = "insert_path"

df = pd.read_csv(path)
correct_stats = df["correct"].value_counts()
prediction_stats = df["predicted"].value_counts()
n_refused = prediction_stats["REFUSED_TO_ANSWER"] if "REFUSED_TO_ANSWER" in list(prediction_stats.keys()) else 0
accuracy = correct_stats[True] / (len(df) - n_refused)
print("Accuracy: %.1f" % (accuracy * 100))
plot_confusion_matrix(df["predicted"],
                      df["target"],
                      classes=AVeriTeC().get_classes(),
                      benchmark_name="Averitec Dev",
                      save_dir=None)
