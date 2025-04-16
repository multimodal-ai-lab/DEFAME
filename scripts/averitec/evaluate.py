if __name__ == '__main__':  # evaluation uses multiprocessing
    from defame.eval.evaluate import evaluate

    variant = "dev"
    evaluate(
        llm="gpt_4o",
        tools_config=dict(searcher=dict(
            search_config=dict(
                averitec_kb=dict(variant=variant),
                # google=dict(),
            ),
            limit_per_search=3
        )),
        fact_checker_kwargs=dict(
            procedure_variant="summary/dynamic",
            interpret=False,
            decompose=False,
            decontextualize=False,
            filter_check_worthy=False,
            max_iterations=3,
            max_result_len=64_000,  # characters
        ),
        llm_kwargs=dict(temperature=0.01),
        benchmark_name="averitec",
        benchmark_kwargs=dict(variant=variant),
        allowed_actions=["search"],
        n_samples=None,
        sample_ids=None, # list of integers
        random_sampling=False,
        print_log_level="log",
        n_workers=8,
    )
