import pandas as pd
import numpy as np
from eval.evaluate import compute_accuracy

predictions_prev_path = "/pfss/mlde/workspaces/mlde_wsp_Rohrbach/users/mr74vahu/MAFC/out/2024-06-13 V0.3 FEVER GPT-4o/predictions.csv"
predictions_new_path = "/pfss/mlde/workspaces/mlde_wsp_Rohrbach/users/mr74vahu/MAFC/out/2024-06-13 V0.3.1 FEVER GPT-4o/predictions.csv"

predictions_prev = pd.read_csv(predictions_prev_path)
predictions_new = pd.read_csv(predictions_new_path)

assert len(predictions_prev) == len(predictions_new)

predictions_prev.sort_values(by='sample_index', inplace=True)
predictions_new.sort_values(by='sample_index', inplace=True)

# Check the differences between the predictions
accuracy_prev = compute_accuracy(predictions_prev)
accuracy_new = compute_accuracy(predictions_new)
print("Previous accuracy: %.1f" % (accuracy_prev * 100))
print("New accuracy: %.1f" % (accuracy_new * 100))

# Compute the prediction consistency
concurrent_predictions = predictions_prev["predicted"].values == predictions_new["predicted"].values
n_concurrent = np.count_nonzero(concurrent_predictions)
print(f"Prediction consistency: {n_concurrent / len(predictions_new) * 100:.1f}")
