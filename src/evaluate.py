import numpy as np

from safe.fact_checker import FactChecker
from eval.benchmark import AVeriTeC

n = 5
model = "OPENAI:gpt-3.5-turbo-0125"
benchmark = AVeriTeC("dev")
assert n <= len(benchmark)

print(f"Loaded {benchmark.name} containing {len(benchmark)} instances.")
print(f"Evaluating on {n} samples.")

fc = FactChecker(model=model)

# For each single instance in the benchmark, predict its veracity
predictions = []
for instance in benchmark:
    content = instance["content"]
    prediction = fc.check(content)
    predictions.append(prediction)
    if len(predictions) == 5:
        break

# Compute metrics
ground_truth = benchmark.get_labels()[:n]
correct_predictions = np.array(predictions) == np.array(ground_truth)  # TODO: translate labels
accuracy = np.sum(correct_predictions) / n
print(f"Accuracy: {accuracy*100:.1f} %")
