from eval.evaluate import evaluate

evaluate(
    model="huggingface:meta-llama/Meta-Llama-3-70B-Instruct",
    benchmark_name="averitec",
    benchmark_kwargs=dict(variant="dev"),
    search_engine="averitec_kb",
    extract_claims=False,
    verbose=True,
)
