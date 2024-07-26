from src.eval.evaluate import evaluate
from multiprocessing import set_start_method

if __name__ == '__main__':  # evaluation uses multiprocessing
    set_start_method("spawn")
    evaluate(
        llm="gpt_4o_mini",
        tools_config=dict(searcher=dict(
            search_engine_config=dict(
                averitec_kb=dict(),
            ),
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
        verbose=False,
    )
