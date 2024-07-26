from src.eval.evaluate import evaluate

evaluate(
    llm="gpt_4o_mini",
    tools_config=dict(searcher=dict(
        search_engines=["averitec_kb"],
        limit_per_search=5
    )),
    fact_checker_kwargs=dict(
        stop_after_q_and_a=True,
        interpret=False,
        decompose=False,
        decontextualize=False,
        filter_check_worthy=False,
        max_iterations=2,
    ),
    llm_kwargs=dict(temperature=0.01),
    benchmark_name="averitec",
    benchmark_kwargs=dict(variant="dev"),
    # sample_ids=[3,4],
    # n_samples=500,
    random_sampling=False,
    verbose=True,
)
