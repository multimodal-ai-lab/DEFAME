if __name__ == '__main__':  # evaluation uses multiprocessing
    from defame.eval.evaluate import evaluate

    evaluate(
        llm="gpt_4o",
        tools_config=dict(searcher=dict(
            search_engines=["wiki_dump"],
            limit_per_search=5
        )),
        fact_checker_kwargs=dict(
            interpret=False,
            decompose=False,
            decontextualize=False,
            filter_check_worthy=False,
            max_iterations=3,
        ),
        llm_kwargs=dict(temperature=0.01),
        benchmark_name="fever",
        benchmark_kwargs=dict(version=1, variant="dev"),
        # sample_ids=[2],
        print_log_level="warning",
        random_sampling=False,
    )
