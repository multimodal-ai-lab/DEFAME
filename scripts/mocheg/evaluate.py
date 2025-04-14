if __name__ == '__main__':  # evaluation uses multiprocessing
    from defame.eval.evaluate import evaluate

    evaluate(
        llm="gpt_4o",
        tools_config=dict(searcher=dict(
            search_engine_config=dict(
                google=dict(),
            ),
            limit_per_search=3
        )
        ),
        fact_checker_kwargs=dict(
            procedure_variant="summary/dynamic",
            interpret=True,
            decompose=False,
            decontextualize=False,
            filter_check_worthy=False,
            max_iterations=3,
            max_result_len=64_000,  # characters
        ),
        llm_kwargs=dict(temperature=0.01),
        benchmark_name="mocheg",
        benchmark_kwargs=dict(variant="val"),
        n_samples=1,
        sample_ids=None,
        print_log_level="log",
        random_sampling=True, # Set to True in MOCHEG as samples are ordered in the benchmark csv file according to their label.
        n_workers=1,
    )
