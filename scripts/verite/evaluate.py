import warnings
from multiprocessing import set_start_method

from infact.eval.evaluate import evaluate

warnings.filterwarnings("ignore")

if __name__ == '__main__':  # evaluation uses multiprocessing
    set_start_method("spawn")
    evaluate(
        llm="gpt_4o",
        tools_config=dict(searcher=dict(
            search_engine_config=dict(google={})
            ,
            limit_per_search=3
        )),
        fact_checker_kwargs=dict(
            interpret=False,
            decompose=False,
            decontextualize=False,
            filter_check_worthy=False,
            max_iterations=3,
        ),
        llm_kwargs=dict(temperature=0.01),
        benchmark_name="verite",
        benchmark_kwargs=dict(variant="dev"),
        n_samples=2,
        print_log_level="warning",
        random_sampling=False,
        n_workers=1
    )
