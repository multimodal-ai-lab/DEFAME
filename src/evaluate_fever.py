from eval.evaluate import evaluate

evaluate(
    model="gpt_4o",
    model_kwargs=dict(temperature=0.01),
    benchmark_name="fever",
    search_engine="wiki_dump",
    benchmark_kwargs=dict(variant="dev"),
    # sample_ids=[2],
    extract_claims=False,
    verbose=True,
    random_sampling=False,
)
