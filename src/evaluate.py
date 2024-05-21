import numpy as np

from safe.fact_checker import FactChecker
from eval.benchmark import AVeriTeC, FEVER


# model = "OPENAI:gpt-3.5-turbo-0125"
model = "huggingface:meta-llama/Meta-Llama-3-70B-Instruct"
multimodal_model = "huggingface:llava-hf/llava-1.5-7b-hf"
search_engine = "google"  # "google" or "wiki"
benchmark = AVeriTeC("dev")
n = 5
extract_claims = False


assert n <= len(benchmark)

print(f"Loaded {benchmark.name} containing {len(benchmark)} instances.")
print(f"Evaluating on {n} samples.")

fc = FactChecker(model=model, multimodal_model=multimodal_model, search_engine=search_engine, extract_claims=extract_claims)

# For multimodal usage turn image into a tensor by either:
# 1) pulling it from link: 
#    image_url = "https://llava-vl.github.io/static/images/view.jpg"
#    image = Image.open(requests.get(image_url, stream=True).raw)
#   or
# 2) pulling it from path
#    image_path = path_to_data + "MAFC_test/image_claims/00000.png"
#    image = Image.open(image_path)
#
# Hand the tensor as second argument to Factchecker.check 

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
print(f"Accuracy: {accuracy*100:.1f} %")
