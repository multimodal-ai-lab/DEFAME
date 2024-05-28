from eval.evaluate import evaluate

evaluate(
    model="OPENAI:gpt-3.5-turbo-0125",
    multimodal_model=None,
    search_engine="wiki",
    benchmark_name="fever",
    benchmark_kwargs=dict(variant="dev"),
    n=None,
    extract_claims=False,
    verbose=True,
    logging=True
)
