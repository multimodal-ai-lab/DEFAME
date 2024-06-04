import pandas as pd

path = "insert_path"

df = pd.read_csv(path)
prediction_stats = df["correct"].value_counts()
accuracy = prediction_stats[True] / len(df)
print("Accuracy: %.1f" % (accuracy * 100))
