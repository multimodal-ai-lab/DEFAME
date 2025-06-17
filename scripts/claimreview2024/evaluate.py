if __name__ == '__main__':  # evaluation uses multiprocessing
    from defame.eval.evaluate import evaluate

    evaluate(
        llm="gpt_4o",
        tools_config=dict(
            searcher=dict(
                search_config=dict(
                    google=dict(),
                ),
                limit_per_search=3
            ),
            geolocator=dict()
        ),
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
        benchmark_name="claimreview2024",
        benchmark_kwargs=dict(variant="dev"),
        allowed_actions=["search", "geolocate"],
        n_samples=None,
        sample_ids=None, # list of integers
        random_sampling=False,
        print_log_level="info",
        n_workers=1,
    )
