from eval.benchmark import FEVER
from eval.evaluate import evaluate

model = "OPENAI:gpt-3.5-turbo-0125"  # "OPENAI:gpt-3.5-turbo-0125" | check shared_config for all available models
multimodal_model = None
search_engine = "wiki"  # "duckduck" or "google" or "wiki" | check shared_config.py for all available search_engines
benchmark = FEVER("dev")
n_samples = 20
extract_claims = False
assert n_samples <= len(benchmark)
verbose = True
logging = False

print(f"Loaded {benchmark.name} containing {len(benchmark)} instances.")
print(f"Evaluating on {n_samples} samples.")

evaluate(
    model=model,
    multimodal_model=multimodal_model,
    search_engine=search_engine,
    benchmark=benchmark,
    n=n_samples,
    extract_claims=extract_claims,
    verbose=verbose,
    logging=logging
)
