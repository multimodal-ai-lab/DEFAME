from eval.evaluate import evaluate

evaluate(
    llm="gpt_4o",
    tools_config=dict(searcher=dict(
        search_engines=["wiki_dump"],
        limit_per_search=5
    )),
    llm_kwargs=dict(temperature=0.01),
    benchmark_name="fever",
    benchmark_kwargs=dict(version=1, variant="dev"),
    # sample_ids=[2],
    extract_claims=False,
    verbose=True,
    random_sampling=False,
)
