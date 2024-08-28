from infact.eval.evaluate import evaluate
import warnings
warnings.filterwarnings("ignore")

evaluate(
    llm="gpt_4o",
    mllm="llava_15",
    tools_config=dict(searcher=dict(
        search_engines=["duckduckgo"],
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
    n_samples=20,
    print_log_level="warning",
    random_sampling=False,
)
