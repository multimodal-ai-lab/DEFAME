from eval.evaluate import evaluate

model = "huggingface:meta-llama/Meta-Llama-3-70B-Instruct"  # "OPENAI:gpt-3.5-turbo-0125" | check shared_config for all available models
multimodal_model = "huggingface:llava-hf/llava-1.5-7b-hf"
search_engine = "duckduck"  # "duckduck" or "google" or "wiki" | check shared_config.py for all available search_engines
benchmark_name = "averitec"
benchmark_kwargs = dict(variant="dev")
n_samples = 5
extract_claims = False
verbose = False
logging = True

evaluate(
    model=model,
    multimodal_model=multimodal_model,
    search_engine=search_engine,
    benchmark_name=benchmark_name,
    benchmark_kwargs=benchmark_kwargs,
    n=n_samples,
    extract_claims=extract_claims,
    verbose=verbose,
    logging=logging
)
