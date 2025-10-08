# scripts/dgm4/evaluate.py

if __name__ == "__main__":
    import os, torch
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    torch.set_num_threads(1)

    from defame.eval.serial_eval import evaluate_serial

    evaluate_serial(
        llm="gpt_4o_mini",
        tools_config=dict(  # keep minimal while debugging
            searcher=dict(search_config=dict(google=dict()), limit_per_search=3) ,

        ),
        fact_checker_kwargs=dict(
            procedure_variant="summary/static",  # <- NOT just 'summary'
            interpret=True,
            decompose=False,
            decontextualize=False,
            filter_check_worthy=False,
            max_iterations=3,
            max_result_len=64_000,
        ),
        llm_kwargs=dict(temperature=0.01),
        benchmark_name="dgm4",
        benchmark_kwargs=dict(variant="val"),
        n_samples=100,            # try with 1 while debugging
        random_sampling=True,
        print_log_level="log",
    )
