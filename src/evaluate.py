import numpy as np

from safe.fact_checker import FactChecker
from eval.benchmark import AVeriTeC, FEVER


model = "OPENAI:gpt-3.5-turbo-0125"
# model = "huggingface:meta-llama/Meta-Llama-3-70B-Instruct"
search_tool = "wiki"  # "serper" or "wiki"
benchmark = FEVER("dev")
n = 5
extract_claims = False


assert n <= len(benchmark)

print(f"Loaded {benchmark.name} containing {len(benchmark)} instances.")
print(f"Evaluating on {n} samples.")

fc = FactChecker(model=model, search_tool=search_tool, extract_claims=extract_claims)

# For each single instance in the benchmark, predict its veracity
predictions = []
for instance in benchmark:
    content = instance["content"]
    prediction = fc.check(content,verbose=False)
    predictions.append(prediction)
    if len(predictions) == 5:
        break

# Compute metrics
ground_truth = benchmark.get_labels()[:n]
correct_predictions = np.array(predictions) == np.array(ground_truth)
accuracy = np.sum(correct_predictions) / n
print(f"Accuracy: {accuracy*100:.1f} %")
