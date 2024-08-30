from infact.eval.evaluate import evaluate
from multiprocessing import set_start_method

variant = "dev"

if __name__ == '__main__':  # evaluation uses multiprocessing
    set_start_method("spawn")
    evaluate(
        llm="gpt_4o",
        tools_config=dict(searcher=dict(
            search_engine_config=dict(
                averitec_kb=dict(variant=variant),
            ),
            limit_per_search=5
        )),
        fact_checker_kwargs=dict(
            procedure_variant="infact",
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
        # sample_ids=[381],
        # n_samples=8,
        random_sampling=False,
        print_log_level="info",
        # n_workers=4,
    )
