import pandas as pd
from utils.plot import plot_confusion_matrix
from eval.benchmark import AVeriTeC
from eval.evaluate import compute_accuracy

path = "insert_path"

df = pd.read_csv(path)
accuracy = compute_accuracy(df)
print("Accuracy: %.1f" % (accuracy * 100))
plot_confusion_matrix(df["predicted"],
                      df["target"],
                      classes=AVeriTeC().get_classes(),
                      benchmark_name="Averitec Dev",
                      save_dir=None)
