if __name__ == '__main__':  # evaluation uses multiprocessing
    from defame.eval.evaluate import evaluate

    evaluate(
        llm="gpt_4o_mini",
        tools_config=dict(searcher=dict(
            search_config=dict(
                google=dict(),
            ),
            limit_per_search=3
        ) , deepfake_detector=dict()
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
        benchmark_name="dgm4",
        benchmark_kwargs=dict(variant="val"),
        n_samples=1000,
        print_log_level="log",
        random_sampling=False,
        n_workers=6,
    )