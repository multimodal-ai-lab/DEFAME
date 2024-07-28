import pandas as pd
from src.utils.plot import plot_confusion_matrix
from src.eval.benchmark import AVeriTeC
from src.eval.evaluate import compute_accuracy
from src.common.label import Label

experiment_dir = "/pfss/mlde/workspaces/mlde_wsp_Rohrbach/users/mr74vahu/MAFC/out/averitec/2024-07-28_09-54_gpt_4o/"

df = pd.read_csv(experiment_dir + "predictions.csv")
accuracy = compute_accuracy(df)
print("Accuracy: %.1f" % (accuracy * 100))
classes = AVeriTeC().get_classes()
classes.remove(Label.CHERRY_PICKING)
plot_confusion_matrix(df["predicted"],
                      df["target"],
                      classes=classes,
                      benchmark_name="Averitec Dev",
                      save_dir=experiment_dir)
