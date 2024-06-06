from eval.evaluate import evaluate

evaluate(
    model="OPENAI:gpt-3.5-turbo-0125",
    benchmark_name="fever",
    search_engine="wiki_dump",
    benchmark_kwargs=dict(variant="dev"),
    # n_samples=1,
    # sample_ids=[865],
    extract_claims=False,
    verbose=True,
    # random_sampling=True,
)
