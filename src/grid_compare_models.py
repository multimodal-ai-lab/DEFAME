import numpy as np

from safe.fact_checker import FactChecker
from eval.benchmark import AVeriTeC
from common.console import green, bold
import time

n = 5

models = ["huggingface:meta-llama/Meta-Llama-3-70B-Instruct",
          "OPENAI:gpt-3.5-turbo-0125",
          "huggingface:meta-llama/Meta-Llama-3-8B-Instruct",
          "huggingface:mistralai/Mixtral-8x7B-Instruct-v0.1",]

benchmark = AVeriTeC("dev")
assert n <= len(benchmark)

print(f"Loaded {benchmark.name} containing {len(benchmark)} instances.")
print(f"Evaluating on {n} samples.")

for model in models:
    print(bold(f"\n\nModel: {green(model)}\n\n"))
    start_time = time.time()
    fc = FactChecker(model=model)

    # For each single instance in the benchmark, predict its veracity
    predictions = []
    for instance in benchmark:
        content = instance["content"]
        prediction = fc.check(content, verbose=False)
        predictions.append(prediction)
        if len(predictions) == 5:
            break

    # Compute metrics
    ground_truth = benchmark.get_labels()[:n]
    correct_predictions = np.array(predictions) == np.array(ground_truth)
    accuracy = np.sum(correct_predictions) / n
    end_time = time.time()
    print(bold(f"\n\nModel: {green(model)}\n"))
    print(f"Execution time: {end_time - start_time} seconds")
    print(f"Accuracy: {accuracy*100:.1f} %")
    print("\n\n\nNEXT MODEL\n\n\n")
